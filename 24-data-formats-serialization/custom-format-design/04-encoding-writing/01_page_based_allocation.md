# Page-Based Memory Allocation: A Deep Dive

## The Problem: Writing Variable-Length Data in Parallel

When creating a dataset, we need to write samples (images, audio, text) into one big file. The challenges are:

1.  **Data sizes vary**: An image might be 50KB compressed, another 500KB. We can't pre-compute exact positions.
2.  **Parallel workers**: For speed, multiple CPU threads encode samples simultaneously. They all want to write to the same file.
3.  **Random-access reads**: After writing, we want to read any sample in O(1), meaning we need an index of where each sample lives.

### Naive Approach (and Why It Fails)

```python
# BAD: Multiple threads calling this will corrupt the file
def write_sample(file, sample_data):
    pos = file.tell()  # Where am I?
    file.write(sample_data)  # Write data
    return pos  # Return position for index
```

**Problem**: If Thread A calls `tell()`, then Thread B calls `tell()`, then A calls `write()`, then B calls `write()`, the file is corrupted because B's data overwrites A's.

### Simple Fix: A Global Lock

```python
lock = threading.Lock()

def write_sample(file, sample_data):
    with lock:
        pos = file.tell()
        file.write(sample_data)
        return pos
```

**Problem**: Only one thread writes at a time. You've serialized your parallel workload. On a 16-core machine, this is 15/16ths wasted.

## The FFCV Solution: Page-Based Allocation

Instead of a global lock on the whole file, FFCV:
1.  Divides the data region into fixed-size **pages** (e.g., 2 MB each).
2.  Each worker thread claims an entire page atomically.
3.  The worker fills its page in RAM (no I/O).
4.  When the page is full, it's written to disk and a new page is claimed.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    THE DATA REGION AS PAGES                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  FILE:                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ HEADER │ METADATA │       DATA REGION (pages)       │ ALLOC TABLE │     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ PAGE 0   │ │ PAGE 1   │ │ PAGE 2   │ │ PAGE 3   │ │ PAGE 4   │ ...       │
│  │ 2 MB     │ │ 2 MB     │ │ 2 MB     │ │ 2 MB     │ │ 2 MB     │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│       ↑             ↑             ↑             ↑             ↑              │
│   Thread 0      Thread 1      Thread 2      Thread 0      Thread 1          │
│   (claimed)     (claimed)     (claimed)     (new page)    (new page)        │
│                                                                                │
│  INSIDE A PAGE:                                                               │
│  ┌─────────────────────────────────────────────────┐                         │
│  │ Sample A  │ Sample B  │ Sample C  │   UNUSED   │                         │
│  │ (300 KB)  │ (150 KB)  │ (800 KB)  │ (remaining)│                         │
│  └─────────────────────────────────────────────────┘                         │
│  ← writes grow this way                                                      │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Why This Works

### 1. Atomic Page Claiming
A shared counter tracks the next available page. Claiming a page is a single atomic increment:

```python
from multiprocessing import Value
import ctypes

next_page = Value(ctypes.c_uint64, 0)

def claim_page():
    with next_page.get_lock():
        my_page = next_page.value
        next_page.value += 1
    return my_page
```

This takes nanoseconds. Once claimed, the page belongs exclusively to that worker.

### 2. RAM Buffering
Each worker has a private RAM buffer the size of one page. All `malloc` calls fill this buffer. There's no contention because each buffer is thread-private.

### 3. Sequential Disk Writes
Pages must be written to disk in order (page 0, then page 1, ...). This ensures the file is always coherent. A "write token" mechanism ensures this:

```python
written_page = Value(ctypes.c_uint64, 0)

def flush_page(my_page, data):
    # Wait for my turn
    while written_page.value != my_page:
        time.sleep(0.0001)  # Spin wait (or use event)
    
    # Write to disk
    file.seek(header_size + my_page * page_size)
    file.write(data)
    
    # Pass the token
    with written_page.get_lock():
        written_page.value += 1
```

## The `malloc` Function: Heart of the Allocator

When a field encodes data, it calls `malloc(size)` to get space:

```python
class PageAllocator:
    """
    Allocates space for variable-length data in pages.
    """
    
    def __init__(self, file_path: str, data_start_offset: int, page_size: int = 2 * 1024 * 1024):
        """
        Args:
            file_path: Path to the output file.
            data_start_offset: Byte offset where the data region begins (after header + metadata).
            page_size: Size of each page in bytes.
        """
        self.file_path = file_path
        self.page_size = page_size
        
        # Align data_start to page boundary for efficient I/O
        self.data_start = self._align_up(data_start_offset, page_size)
        
        # Shared state (for multiprocessing)
        self.next_page_counter = Value(ctypes.c_uint64, 0)
        self.written_page_counter = Value(ctypes.c_uint64, 0)
        
        # Per-worker state
        self._current_page_id = -1      # Which page we own (-1 = none)
        self._page_cursor = 0           # Current position within the page
        self._page_buffer = np.zeros(page_size, dtype=np.uint8)  # RAM buffer
        
        # Allocation records (for building the allocation table)
        self._allocations = []
        self._current_sample_id = None
        self._sample_alloc_start = 0
        
        # File handle
        self._file = None
    
    def _align_up(self, value: int, alignment: int) -> int:
        """Round up to next alignment boundary."""
        remainder = value % alignment
        if remainder == 0:
            return value
        return value + (alignment - remainder)
    
    @property
    def _space_remaining(self) -> int:
        """Bytes left in current page."""
        if self._current_page_id < 0:
            return 0
        return self.page_size - self._page_cursor
    
    def begin_sample(self, sample_id: int):
        """
        Call before encoding a sample.
        Marks the start point for rollback if the sample doesn't fit.
        """
        self._current_sample_id = sample_id
        self._sample_alloc_start = len(self._allocations)
    
    def malloc(self, size: int) -> tuple:
        """
        Allocate 'size' bytes for data.
        
        Args:
            size: Number of bytes needed.
        
        Returns:
            (file_offset, buffer):
                file_offset: The byte position in the file where this data will reside.
                buffer: A writable numpy array (view into page buffer) where you write the data.
        
        Raises:
            ValueError: If size exceeds page size.
            MemoryError: If sample spans page boundary (needs retry).
        """
        # Sanity check
        if size > self.page_size:
            raise ValueError(
                f"Allocation of {size} bytes exceeds page size {self.page_size}. "
                "Either increase page_size or split the data."
            )
        
        # Do we need a new page?
        if size > self._space_remaining:
            # We need a new page. But first, check if current sample spans pages.
            if self._sample_spans_pages():
                # Rollback this sample's allocations and signal retry
                self._rollback_current_sample()
                raise MemoryError("Sample does not fit in remaining page space. Retry on fresh page.")
            
            # Flush current page to disk
            self._flush_page()
            
            # Claim a new page
            self._claim_new_page()
        
        # Allocate from current page
        offset_in_page = self._page_cursor
        self._page_cursor += size
        
        # Calculate the file offset for this allocation
        file_offset = self.data_start + (self._current_page_id * self.page_size) + offset_in_page
        
        # Get a view into the buffer
        buffer = self._page_buffer[offset_in_page : self._page_cursor]
        
        # Record the allocation
        self._allocations.append({
            'sample_id': self._current_sample_id,
            'file_offset': file_offset,
            'size': size,
            'page_id': self._current_page_id,
        })
        
        return file_offset, buffer
    
    def _sample_spans_pages(self) -> bool:
        """
        Check if the current sample has allocations from a previous page.
        If so, those allocations are now invalid because we're moving to a new page.
        """
        if self._current_page_id < 0:
            return False
        
        for i in range(self._sample_alloc_start, len(self._allocations)):
            if self._allocations[i]['page_id'] != self._current_page_id:
                return True
        return False
    
    def _rollback_current_sample(self):
        """Remove allocations for the current sample (to retry on fresh page)."""
        self._allocations = self._allocations[:self._sample_alloc_start]
    
    def _claim_new_page(self):
        """Atomically claim the next available page."""
        with self.next_page_counter.get_lock():
            self._current_page_id = self.next_page_counter.value
            self.next_page_counter.value += 1
        
        # Reset page state
        self._page_cursor = 0
        self._page_buffer.fill(0)  # Zero the buffer (optional, for cleanliness)
    
    def _flush_page(self):
        """Write the current page to disk in order."""
        if self._current_page_id < 0 or self._page_cursor == 0:
            return  # Nothing to flush
        
        # Sequential write ordering: wait for previous pages
        while self.written_page_counter.value != self._current_page_id:
            time.sleep(0.0001)  # 100 microsecond spin
        
        # Write the page
        file_offset = self.data_start + self._current_page_id * self.page_size
        self._file.seek(file_offset)
        # Only write the used portion (or the full page, depending on preference)
        self._file.write(self._page_buffer.tobytes())
        
        # Pass the write token to the next page
        with self.written_page_counter.get_lock():
            self.written_page_counter.value += 1
    
    def open(self):
        """Open the file for writing. Call before any allocations."""
        self._file = open(self.file_path, 'r+b', buffering=0)  # Unbuffered for control
    
    def close(self):
        """Flush remaining data and close the file."""
        self._flush_page()
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def get_allocations(self) -> list:
        """Return all allocation records for building the allocation table."""
        return self._allocations
```

## Handling Samples That Span Pages

A sample might consist of multiple fields (image + label + audio). If the image is 1.8 MB and the audio is 0.5 MB, and the page is 2 MB, the sample won't fit on one page.

**FFCV's rule**: A sample's data must be entirely on one page.

If a sample's first allocation starts near the end of a page, subsequent allocations for the same sample might not fit. The solution:
1.  Detect when an allocation would cross the page boundary.
2.  Rollback all allocations for the current sample.
3.  Flush the current page.
4.  Claim a fresh page.
5.  Retry the entire sample encoding.

```python
def encode_sample_with_retry(sample, fields, allocator, max_attempts=2):
    """
    Encode a sample, retrying if it doesn't fit on current page.
    """
    for attempt in range(max_attempts):
        try:
            allocator.begin_sample(sample['id'])
            
            for name, field in fields.items():
                value = sample[name]
                # This calls allocator.malloc internally
                field.encode(value, allocator)
            
            return  # Success!
        
        except MemoryError as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Sample {sample['id']} cannot fit in a single page. "
                    f"Page size: {allocator.page_size}, sample requires more space."
                ) from e
            # Otherwise, retry. The allocator has claimed a fresh page.
```

## Page Size Selection

The choice of page size significantly impacts performance and disk usage.

### Factors to Consider

| Factor | Small Pages (e.g., 512 KB) | Large Pages (e.g., 8 MB) |
|--------|---------------------------|--------------------------|
| **Wasted space** | Less waste per page | More waste per page |
| **Write efficiency** | Many small writes (bad for HDD) | Fewer large writes (good) |
| **Huge page support** | May not align with OS huge pages (2 MB) | Aligns well |
| **Large samples** | May not fit | Can fit larger samples |
| **Memory usage** | Lower RAM per worker | Higher RAM per worker |

### FFCV's Default

```python
# From ffcv/writer.py
MIN_PAGE_SIZE = 2 * 1024 * 1024  # 2 MiB (Linux huge page size)
DEFAULT_PAGE_SIZE = 4 * MIN_PAGE_SIZE  # 8 MiB
```

**Why 2 MB minimum?** Linux supports "huge pages" which are 2 MB (instead of the standard 4 KB). Using 2 MB pages aligns with this, reducing TLB (Translation Lookaside Buffer) misses during memory-mapped reads.

### Rule of Thumb

```
page_size = max(2 MB, 10 * average_sample_size)
```

For ImageNet (average JPEG ~100 KB), 2-8 MB pages are fine.
For video clips (average 10 MB), you'd want 32-64 MB pages.

## The Allocation Table

After all samples are written, we have a list of allocations: `(sample_id, file_offset, size)`. This becomes the **Allocation Table** written at the end of the file.

```python
def build_allocation_table(allocations: list) -> np.ndarray:
    """
    Build the allocation table from recorded allocations.
    
    The table is sorted by file_offset for efficient binary search during reading.
    """
    DTYPE = np.dtype([
        ('ptr', '<u8'),    # File offset
        ('size', '<u4'),   # Size in bytes
        ('_pad', '<u4'),   # Padding for alignment
    ])
    
    table = np.zeros(len(allocations), dtype=DTYPE)
    for i, alloc in enumerate(allocations):
        table[i]['ptr'] = alloc['file_offset']
        table[i]['size'] = alloc['size']
    
    # Sort by pointer for binary search
    return np.sort(table, order='ptr')
```

## Complete Parallel Writer Example

```python
from concurrent.futures import ThreadPoolExecutor
import threading

def parallel_writer_example(samples, fields, output_path, num_workers=4, page_size=8*1024*1024):
    """
    Write samples in parallel using page-based allocation.
    """
    num_samples = len(samples)
    
    # Calculate layout
    # (Assume header + metadata are already written, and we know data_start_offset)
    data_start_offset = 1024 * 1024  # Example: 1 MB for header + metadata
    
    # Shared allocator
    allocator = PageAllocator(output_path, data_start_offset, page_size)
    
    # Thread-local storage for each worker's allocator instance
    # (In practice, you might use multiprocessing with separate instances)
    
    # ... This example is simplified; real implementation would use
    # multiprocessing.Pool with proper serialization ...
    
    with allocator:
        for sample in samples:
            encode_sample_with_retry(sample, fields, allocator)
    
    # Write allocation table
    alloc_table = build_allocation_table(allocator.get_allocations())
    with open(output_path, 'r+b') as f:
        f.seek(0, 2)  # End of file
        alloc_table_offset = f.tell()
        f.write(alloc_table.tobytes())
        
        # Update header with alloc_table_offset
        # ... (seek to header, update the pointer field)
```

## Exercises

1.  **Multi-Page Samples**: Modify `PageAllocator` to support samples larger than one page by allocating contiguous pages. Hint: Change `_claim_new_page` to optionally claim multiple pages at once.

2.  **Statistics**: Add methods to track and report:
    - Total pages allocated.
    - Average fill ratio per page.
    - Number of sample retries.

3.  **Memory-Mapped Writing**: Instead of `file.write()`, use `mmap` to create a writeable memory-mapped region. Compare performance.

4.  **Defragmentation**: After writing, compute the "wasted space" (unused bytes at end of each page). Implement an optional post-processing step that rewrites the file without gaps.
