# FFCV Memory Internals: The Zero-Copy Architecture

## What "Zero-Copy" Really Means

"Zero-copy" is often misused. In FFCV, it has a precise meaning:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       ZERO-COPY DATA FLOW                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  TRADITIONAL APPROACH (with copies):                                          │
│  ───────────────────────────────────                                           │
│                                                                                │
│  ┌─────────┐  read()  ┌─────────┐  decode  ┌─────────┐  transform ┌─────────┐│
│  │  Disk   │ ──────▶ │ Kernel  │ ──────▶ │  User   │ ──────────▶│ User    ││
│  │  File   │  COPY 1  │ Buffer  │  COPY 2  │ Buffer  │   COPY 3   │ Output  ││
│  └─────────┘         └─────────┘         └─────────┘            └─────────┘│
│                                                                                │
│  3 memory copies per sample!                                                  │
│                                                                                │
│                                                                                │
│  FFCV ZERO-COPY APPROACH:                                                     │
│  ────────────────────────                                                      │
│                                                                                │
│  ┌─────────┐  page   ┌─────────────────────────────────────────────────────┐ │
│  │  Disk   │  fault  │   Page Cache (kernel memory mapped to user space)    │ │
│  │  File   │ ──────▶│   ┌─────────────────────────────────────────────────┐ │ │
│  └─────────┘         │   │  mmap view: direct pointer into file contents   │ │ │
│                      │   │                                                  │ │ │
│                      │   │   ┌──────┐          ┌──────────┐                │ │ │
│                      │   │   │Sample│ ───┬───▶│ Decode   │                │ │ │
│                      │   │   │ 0    │    │     │ (writes  │                │ │ │
│                      │   │   └──────┘    │     │  directly│                │ │ │
│                      │   │               │     │  to pre- │                │ │ │
│                      │   │               │     │  alloc'd │                │ │ │
│                      │   │               │     │  buffer) │                │ │ │
│                      │   │               │     └────┬─────┘                │ │ │
│                      │   │               │          │                      │ │ │
│                      │   │               │          ▼                      │ │ │
│                      │   │               │     ┌──────────┐                │ │ │
│                      │   │               ◀────│ Pre-alloc│                │ │ │
│                      │   │                     │ Output   |                │ │ │
│                      │   │                     └──────────┘                │ │ │
│                      │   └─────────────────────────────────────────────────┘ │ │
│                      └─────────────────────────────────────────────────────────┘ │
│                                                                                │
│  Total copies: 1 (disk → page cache, done by OS/DMA)                          │
│  Then: pointer passed to decoder, which writes directly to output buffer.    │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## The Read Path: OSCacheManager

The `OSCacheManager` provides zero-copy access to file data using `mmap`.

### Memory-Mapped File Access

```python
import numpy as np
import mmap
import os

class OSCacheManager:
    """
    Memory manager that uses OS page cache for caching.
    
    The file is memory-mapped, allowing direct access to file contents
    as if they were in RAM. The OS handles paging data in/out.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file = open(file_path, 'rb')
        self.file_size = os.path.getsize(file_path)
        
        # Memory-map the entire file
        self.mm = mmap.mmap(
            self.file.fileno(),
            0,                      # 0 = entire file
            access=mmap.ACCESS_READ
        )
        
        # Create numpy view (no copy!)
        self.mmap_view = np.frombuffer(self.mm, dtype=np.uint8)
    
    def load_allocation_table(self, header):
        """
        Load the allocation table into RAM for fast lookup.
        
        The allocation table maps sample IDs to (offset, size) pairs.
        We load this once at startup, then use binary search for lookups.
        """
        table_offset = header['allocation_table_offset']
        table_size = header['num_samples'] * 16  # 8 bytes offset + 8 bytes size
        
        # Read allocation table (this is the only "read" we do)
        raw_table = self.mmap_view[table_offset:table_offset + table_size]
        
        # Parse into structured array
        dtype = np.dtype([('offset', '<u8'), ('size', '<u8')])
        self.alloc_table = np.frombuffer(raw_table, dtype=dtype)
        
        # Pre-sort for binary search (usually already sorted)
        # Store separate arrays for Numba compatibility
        self.sorted_offsets = self.alloc_table['offset'].copy()
        self.sorted_sizes = self.alloc_table['size'].copy()
    
    @property
    def state(self):
        """
        Return state tuple for JIT functions.
        
        This tuple is passed to all decoder functions, giving them
        access to the memory-mapped file.
        """
        return (self.mmap_view, self.sorted_offsets, self.sorted_sizes)
    
    def close(self):
        self.mm.close()
        self.file.close()
```

### The Zero-Copy Read Function

```python
import numba as nb

@nb.njit(nogil=True)
def read_sample_data(sample_id: int, storage_state):
    """
    Read raw bytes for a sample without copying.
    
    This function is called from within JIT-compiled pipelines.
    
    Args:
        sample_id: Index of sample to read.
        storage_state: Tuple of (mmap_view, offsets, sizes).
    
    Returns:
        Slice of mmap_view pointing to sample data (NO COPY).
    """
    mmap_view, offsets, sizes = storage_state
    
    # Get offset and size from pre-loaded table
    offset = offsets[sample_id]
    size = sizes[sample_id]
    
    # Return a VIEW into the mmap
    # In Numba, this is just pointer arithmetic, no copy!
    return mmap_view[offset:offset + size]


@nb.njit(nogil=True)
def read_by_pointer(ptr: int, storage_state):
    """
    Read raw bytes by direct pointer.
    
    Used when metadata contains the pointer directly.
    """
    mmap_view, offsets, sizes = storage_state
    
    # Binary search to find size
    # O(log N) lookup
    idx = nb.typed.searchsorted(offsets, ptr)
    size = sizes[idx]
    
    return mmap_view[ptr:ptr + size]
```

### Why This Works

The key insight is that `mmap_view[offset:offset + size]` in Numba returns a **view**, not a copy:

```python
# What happens at the machine level:

# Python version (conceptual):
data = mmap_view[100:200]  # Creates new array with copied data

# Numba version (actual):
data = mmap_view[100:200]  # Returns pointer: mmap_view.data + 100
                           # Length: 100
                           # NO COPY!
```

This is only true for contiguous (`C` or `F`) arrays in Numba's nopython mode.

## The Write Path: PageAllocator

Writing is more complex than reading because we need to:
1. Support parallel writers
2. Minimize random I/O
3. Handle variable-size data

### Page-Based Allocation

```python
class PageAllocator:
    """
    Allocates space for samples in fixed-size pages.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  File Layout                                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  [Header] [Metadata] [Alloc Table] [Page 0] [Page 1] [Page 2] ...       │
    │                                    ▲        ▲        ▲                  │
    │                                    │        │        │                  │
    │                                 2 MB     2 MB     2 MB                  │
    │                                each page is a contiguous block          │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Benefits:
    - Large sequential writes (2 MB at a time)
    - Parallel allocation (each thread gets its own page)
    - Minimal fragmentation
    """
    
    DEFAULT_PAGE_SIZE = 2 * 1024 * 1024  # 2 MB
    
    def __init__(
        self,
        output_path: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        alignment: int = 512,  # Align allocations to sector boundaries
    ):
        self.output_path = output_path
        self.page_size = page_size
        self.alignment = alignment
        
        self.file = open(output_path, 'r+b')
        self.file.seek(0, 2)  # Seek to end
        self.current_file_offset = self.file.tell()
        
        # Thread-local page buffers
        import threading
        self.local = threading.local()
        
        # Lock for allocating new pages
        self.page_lock = threading.Lock()
        
        # Track all allocations for building the table
        self.allocations = []  # List of (sample_id, offset, size)
    
    def _get_page(self) -> 'Page':
        """Get the current thread's page, allocating if needed."""
        if not hasattr(self.local, 'page') or self.local.page is None:
            self.local.page = self._allocate_new_page()
        return self.local.page
    
    def _allocate_new_page(self) -> 'Page':
        """Allocate a new page (thread-safe)."""
        with self.page_lock:
            # Record the file offset for this page
            page_offset = self.current_file_offset
            self.current_file_offset += self.page_size
            
            return Page(
                file_offset=page_offset,
                size=self.page_size,
                alignment=self.alignment,
            )
    
    def malloc(self, size: int):
        """
        Allocate space for data.
        
        Args:
            size: Number of bytes needed.
        
        Returns:
            (file_offset, buffer_view) tuple.
            file_offset: Where this data will be in the final file.
            buffer_view: numpy array to write data into.
        """
        page = self._get_page()
        
        # Check if current page has space
        if not page.can_allocate(size):
            # Flush current page and get a new one
            self._flush_page(page)
            page = self._allocate_new_page()
            self.local.page = page
        
        # Allocate within page
        offset, view = page.allocate(size)
        
        # Calculate final file offset
        file_offset = page.file_offset + offset
        
        return file_offset, view
    
    def _flush_page(self, page: 'Page'):
        """Write a completed page to disk."""
        with self.page_lock:
            self.file.seek(page.file_offset)
            self.file.write(page.buffer[:page.current_offset].tobytes())
    
    def finalize(self):
        """Flush all remaining pages and close."""
        # Flush thread-local pages
        if hasattr(self.local, 'page') and self.local.page is not None:
            self._flush_page(self.local.page)
        
        self.file.close()


class Page:
    """A fixed-size buffer for accumulating data before writing."""
    
    def __init__(self, file_offset: int, size: int, alignment: int):
        self.file_offset = file_offset
        self.size = size
        self.alignment = alignment
        
        self.buffer = np.zeros(size, dtype=np.uint8)
        self.current_offset = 0
    
    def can_allocate(self, size: int) -> bool:
        """Check if this page has room for `size` bytes."""
        aligned_size = self._align(size)
        return self.current_offset + aligned_size <= self.size
    
    def allocate(self, size: int):
        """
        Allocate space within this page.
        
        Returns (offset, view) where:
        - offset is the position within this page
        - view is a numpy array slice to write into
        """
        aligned_size = self._align(size)
        
        offset = self.current_offset
        view = self.buffer[offset:offset + size]
        
        self.current_offset += aligned_size
        
        return offset, view
    
    def _align(self, size: int) -> int:
        """Round up to alignment boundary."""
        return ((size + self.alignment - 1) // self.alignment) * self.alignment
```

## Alignment Considerations

Proper alignment is critical for performance:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       ALIGNMENT MATTERS                                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  MISALIGNED ACCESS:                                                           │
│  ──────────────────                                                            │
│                                                                                │
│  Page Boundaries:    |────PAGE 0────|────PAGE 1────|                          │
│                      0            4096           8192                         │
│                                                                                │
│  Sample Data:               [███████████████]                                 │
│                             3900          4600                                 │
│                                 ▲              ▲                              │
│                                 │              │                              │
│                              crosses page   boundary!                         │
│                                                                                │
│  Result: 2 page faults instead of 1                                          │
│  Result: 2 disk reads instead of 1                                            │
│                                                                                │
│                                                                                │
│  ALIGNED ACCESS:                                                              │
│  ───────────────                                                               │
│                                                                                │
│  Page Boundaries:    |────PAGE 0────|────PAGE 1────|                          │
│                      0            4096           8192                         │
│                                                                                │
│  Sample Data:        [███████████]                                            │
│                      0          700                                           │
│                                 ▲                                             │
│                                 │                                             │
│                              within one page                                  │
│                                                                                │
│  Result: 1 page fault, 1 disk read                                           │
│                                                                                │
│                                                                                │
│  ALIGNMENT LEVELS:                                                            │
│  ─────────────────                                                             │
│                                                                                │
│  Level           Size       Purpose                                           │
│  ─────           ────       ───────                                           │
│  CPU Cache Line  64 B       Avoid false sharing, prefetch efficiency         │
│  Disk Sector     512 B      Direct I/O, SSD block alignment                  │
│  OS Page         4 KB       Page fault granularity                           │
│  Huge Page       2 MB       TLB efficiency for large datasets                │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Pre-Allocation: The Other Half of Zero-Copy

Zero-copy reads are only half the story. We also need to avoid allocations during training:

```python
class BufferPool:
    """
    Pre-allocate all buffers needed for a training run.
    
    At pipeline setup time, we know:
    - Batch size
    - Output shapes for each pipeline stage
    - Output dtypes
    
    We allocate all buffers ONCE and reuse them.
    """
    
    def __init__(
        self,
        batch_size: int,
        pipeline_shapes: dict,  # {'image': (224, 224, 3), 'label': ()}
        pipeline_dtypes: dict,  # {'image': np.uint8, 'label': np.int64}
    ):
        self.batch_size = batch_size
        self.buffers = {}
        
        for name, shape in pipeline_shapes.items():
            dtype = pipeline_dtypes[name]
            
            # Allocate buffer for entire batch
            full_shape = (batch_size,) + shape
            self.buffers[name] = np.zeros(full_shape, dtype=dtype)
    
    def get_buffer(self, name: str) -> np.ndarray:
        """Get the buffer for a pipeline stage."""
        return self.buffers[name]
    
    def get_sample_slice(self, name: str, index: int) -> np.ndarray:
        """Get a view into the buffer for a single sample."""
        return self.buffers[name][index]
```

## The Complete Data Flow

```python
# At setup time:
#   1. Memory-map the file
#   2. Load allocation table
#   3. Pre-allocate output buffers
#   4. Compile the pipeline

# At runtime (per batch):
def load_batch(batch_indices, storage_state, buffers):
    """
    Load a batch with zero copies and zero allocations.
    """
    # This is the compiled JIT function
    for i in range(len(batch_indices)):
        sample_id = batch_indices[i]
        
        # Zero-copy: get VIEW into mmap
        raw_data = read_sample_data(sample_id, storage_state)
        
        # Decode directly into pre-allocated buffer
        # The decoder writes to buffers['image'][i] in-place
        decode_jpeg(raw_data, buffers['image'][i])
        
        # Transform in-place
        normalize(buffers['image'][i], buffers['image'][i])
    
    return buffers['image']  # Same array, now filled with data

# Memory allocations during this loop: ZERO
# Memory copies: 1 (disk → page cache, done by OS DMA)
```

## Exercises

1.  **Benchmark Alignment Impact**: Write samples with 512B vs no alignment; measure random access latency.

2.  **Implement Direct I/O**: Bypass the page cache entirely using `O_DIRECT` for writes.

3.  **Memory Pool**: Implement a thread-safe page pool to avoid repeated allocations.

4.  **Prefetching**: Implement `madvise(MADV_WILLNEED)` prefetching for the next batch.
