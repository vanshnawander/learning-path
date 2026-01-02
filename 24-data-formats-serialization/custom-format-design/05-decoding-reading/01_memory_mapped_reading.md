# Memory-Mapped Reading: The Zero-Copy Paradigm

## What is Memory Mapping?

Memory mapping (`mmap`) is an operating system feature that lets a process access file contents as if they were in RAM, *without explicitly reading the file into a buffer*.

### The Classic Way: `read()`

```python
# Traditional file reading
with open('data.bin', 'rb') as f:
    f.seek(1000000)              # Seek to 1 MB offset
    data = f.read(1024)          # Read 1 KB into a Python bytes object
    # 'data' is a COPY in your process's heap memory
```

What happens under the hood:
1.  Your program issues a `read()` system call.
2.  The OS checks if the data is in the **Page Cache** (a kernel-managed RAM cache of recently accessed file blocks).
3.  If not in cache, the OS reads from disk into the Page Cache.
4.  The OS then **copies** the data from the Page Cache to your user-space buffer.
5.  Your program receives the copied data.

**The problem**: Step 4 is a memory copy that wastes CPU cycles and doubles RAM usage (one copy in Page Cache, one in your buffer).

### The `mmap` Way

```python
import mmap
import numpy as np

# Memory-mapped file access
with open('data.bin', 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    # Access data directly - NO copy!
    data_view = mm[1000000:1001024]  # This is a view, not a copy
    
    # NumPy's memmap is even easier
    arr = np.memmap('data.bin', dtype='u1', mode='r')
    data_view = arr[1000000:1001024]  # Also a view
```

What happens under the hood:
1.  The OS creates a **virtual address range** in your process's address space.
2.  This range "points" to the file on disk, but no data is loaded yet.
3.  When you access `mm[1000000]`, the CPU triggers a **page fault**.
4.  The OS handles the fault by loading the relevant 4 KB page from disk into BOTH the Page Cache AND mapping it directly into your virtual address space.
5.  Your program now reads from the Page Cache directly—**no copy**.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      MEMORY MAPPING ARCHITECTURE                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TRADITIONAL read():                                                            │
│  ────────────────────                                                           │
│                                                                                 │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐             │
│  │     DISK     │ ─────► │  PAGE CACHE  │ ─────► │  USER BUFFER │             │
│  │   (file)     │  I/O   │  (kernel)    │  COPY  │  (your app)  │             │
│  └──────────────┘        └──────────────┘        └──────────────┘             │
│                                                                                 │
│  • Two memory copies: disk → kernel buffer, kernel buffer → user buffer       │
│  • Double RAM usage                                                            │
│  • Syscall overhead for every read                                             │
│                                                                                 │
│                                                                                 │
│  MEMORY MAPPING (mmap):                                                         │
│  ──────────────────────                                                         │
│                                                                                 │
│  ┌──────────────┐        ┌──────────────────────────────────────┐             │
│  │     DISK     │ ─────► │          PAGE CACHE                  │             │
│  │   (file)     │  I/O   │  (kernel, but SHARED with user)      │             │
│  └──────────────┘        └───────────────────┬──────────────────┘             │
│                                              │                                 │
│                                              │ (virtual memory mapping)        │
│                                              ▼                                 │
│                          ┌──────────────────────────────────────┐             │
│                          │       YOUR PROCESS ADDRESS SPACE      │             │
│                          │  (no copy - same physical pages!)     │             │
│                          └──────────────────────────────────────┘             │
│                                                                                 │
│  • ZERO copies after I/O                                                       │
│  • Memory is shared (one physical copy)                                        │
│  • No syscalls after initial mmap                                              │
│  • OS manages caching automatically                                            │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

## How mmap Works at the Hardware Level

### Virtual Memory 101

Every process has a **virtual address space**—a 64-bit range of addresses (on modern systems) that the process can use. But this doesn't map directly to physical RAM. Instead, the Memory Management Unit (MMU) translates virtual addresses to physical addresses using a **page table**.

A **page** is typically 4 KB of contiguous memory. The page table entries (PTEs) map virtual pages to physical pages.

### The Page Fault

When a program accesses a virtual address that isn't currently mapped to physical RAM:

1.  The MMU raises a **page fault** exception.
2.  The CPU traps to the OS kernel.
3.  The kernel checks: Is this a valid address (within a mapped region)?
    - If invalid: `SIGSEGV` (segmentation fault).
    - If valid (like an mmap region): Load the data.
4.  For an mmap'd file, the kernel reads the 4 KB page from disk into a free physical page.
5.  The kernel updates the page table to map the virtual page to the physical page.
6.  The kernel resumes the program at the faulting instruction.
7.  Now the access succeeds!

### Cost of a Page Fault

| Type | What Happens | Latency |
|------|--------------|---------|
| **Soft fault** | Page is in Page Cache but not mapped. Just update PTE. | ~1 microsecond |
| **Hard fault** | Page not in cache. Must read from disk. | 5-50 milliseconds (HDD) or 50-500 microseconds (SSD) |

For training, we want to minimize hard faults by ensuring the dataset fits in RAM (so pages stay cached after the first epoch).

## NumPy's `memmap`: The Pythonic Way

```python
import numpy as np

# Create a memory-mapped array
# The file is NOT loaded into RAM - just mapped
mmap_array = np.memmap('data.bin', dtype=np.uint8, mode='r')

print(f"Virtual size: {mmap_array.nbytes / 1e9:.2f} GB")
print(f"Physical RAM used: ~0 (until accessed)")

# Access a slice - triggers page faults for those pages only
chunk = mmap_array[1_000_000:2_000_000]  # Load 1 MB

# 'chunk' is a VIEW into the mmap, not a copy!
# Modifying 'chunk' (if mode='r+') would modify the file.
```

### Key Properties

1.  **Lazy loading**: Pages are loaded on-demand, not at `memmap()` time.
2.  **Shared cache**: If two processes mmap the same file, they share the same physical pages.
3.  **Automatic eviction**: The OS can evict cached pages under memory pressure and reload them later.
4.  **Zero-copy slicing**: `mmap_array[a:b]` returns a view (pointer arithmetic), not a copy.

## FFCV's OSCacheManager

FFCV uses NumPy memmap for reading. Here's the core logic from `ffcv/memory_managers/os_cache.py`:

```python
import numpy as np
import numba as nb
from numba import njit

class OSCacheContext:
    """
    Context manager that holds the memory map during an epoch.
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.mmap = None
    
    def __enter__(self):
        # Create the memory map if not already done
        if self.mmap is None:
            self.mmap = np.memmap(
                self.manager.file_path,
                dtype=np.uint8,
                mode='r'  # Read-only
            )
        return self
    
    def __exit__(self, *args):
        # We DON'T close the mmap here - keep it alive for next epoch
        # NumPy's memmap doesn't have an explicit close; it's GC'd
        pass
    
    @property
    def state(self):
        """
        Return the tuple passed to JIT-compiled reader functions.
        
        Returns:
            (mmap_array, ptrs, sizes):
                mmap_array: The full mmap as uint8 array.
                ptrs: Sorted array of allocation pointers (for binary search).
                sizes: Corresponding sizes for each pointer.
        """
        return (self.mmap, self.manager.ptrs, self.manager.sizes)


class OSCacheManager:
    """
    Memory manager that relies on OS page cache for file caching.
    
    Best when:
    - Dataset fits in RAM (for repeated epochs).
    - You want simplicity (let OS handle caching).
    - Multiple processes share the same dataset file.
    """
    
    def __init__(self, file_path: str, alloc_table: np.ndarray):
        """
        Args:
            file_path: Path to the dataset file.
            alloc_table: Allocation table array with 'ptr' and 'size' columns.
        """
        self.file_path = file_path
        
        # Extract and sort pointers for binary search
        self.ptrs = np.sort(alloc_table['ptr'])
        
        # Reorder sizes to match sorted pointers
        sort_order = np.argsort(alloc_table['ptr'])
        self.sizes = alloc_table['size'][sort_order]
        
        # The context (created lazily)
        self._context = OSCacheContext(self)
    
    def schedule_epoch(self, sample_schedule: np.ndarray):
        """
        Called at the start of each epoch.
        
        Args:
            sample_schedule: Array of sample IDs in the order they'll be read.
        
        Returns:
            The context manager for memory access.
        """
        # For OSCacheManager, we don't do prefetching here.
        # The OS will handle caching.
        return self._context
    
    def compile_reader(self):
        """
        Return a JIT-compiled function to read data from storage.
        
        Returns:
            A Numba-jitted function: read(ptr, mem_state) -> uint8 array slice
        """
        @njit
        def read(address: np.uint64, mem_state):
            """
            Read data at the given address.
            
            Args:
                address: Byte offset in the file (from metadata).
                mem_state: Tuple (mmap, ptrs, sizes).
            
            Returns:
                View into mmap array (zero-copy).
            """
            mmap_arr, ptrs, sizes = mem_state
            
            # Binary search for the size
            idx = np.searchsorted(ptrs, address)
            size = sizes[idx]
            
            # Return a slice (this is a VIEW, not a copy!)
            return mmap_arr[address:address + size]
        
        return read
```

### The `read` Function: A Critical Detail

The `read` function does two things:
1.  **Finds the size** using binary search on the sorted pointer array.
2.  **Returns a slice** of the mmap array.

In Numba `@njit` mode, `array[start:end]` on a contiguous uint8 array returns a **view** (just pointer arithmetic), not a copy. This is crucial for performance—returning millions of views is fast because no data moves.

## Page Fault Flow During Training

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      A TRAINING ITERATION                                       │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. DataLoader requests batch [sample_id=42, 99, 17, ...]                      │
│                                                                                 │
│  2. For each sample:                                                            │
│     metadata[42]['data_ptr'] = 12345678  (byte offset in file)                 │
│                                                                                 │
│  3. Decoder calls read(12345678, mem_state)                                    │
│                                                                                 │
│  4. read() does:                                                                │
│     idx = searchsorted(ptrs, 12345678)  → fast, in RAM                        │
│     size = sizes[idx]  → e.g., 50000 bytes                                     │
│     return mmap[12345678 : 12345678+50000]                                     │
│                                                                                 │
│  5. mmap[12345678] is accessed:                                                │
│                                                                                 │
│     ┌─── IS THAT PAGE IN RAM? ───┐                                             │
│     │                             │                                             │
│     │  YES (warm cache)           │  NO (cold/first access)                    │
│     │  ↓                          │  ↓                                          │
│     │  Direct access              │  PAGE FAULT!                               │
│     │  (nanoseconds)              │  - OS loads 4KB page from disk             │
│     │                             │  - Updates page table                       │
│     │                             │  - Resumes (microseconds to milliseconds)  │
│     └─────────────────────────────┘                                             │
│                                                                                 │
│  6. Decoder receives the view, decodes (e.g., JPEG → pixels)                   │
│                                                                                 │
│  7. Repeat for all samples in batch                                            │
│                                                                                 │
│  FIRST EPOCH: Many hard faults (loading from disk).                            │
│  LATER EPOCHS: Mostly soft faults or no faults (cached).                       │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Prefetching with `madvise`

The OS doesn't know which pages you'll need next. You can give it hints!

### POSIX `madvise` System Call

```c
int madvise(void *addr, size_t length, int advice);
```

| Advice | Meaning | Use Case |
|--------|---------|----------|
| `MADV_WILLNEED` | "I will need these pages soon" | Prefetch upcoming batches |
| `MADV_DONTNEED` | "I'm done with these pages" | Free RAM after use |
| `MADV_SEQUENTIAL` | "I'll read sequentially" | Linear scan |
| `MADV_RANDOM` | "I'll read randomly" | Disable readahead |

### Python Implementation

```python
import ctypes
import os

# Load libc
if os.name == 'posix':
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    
    MADV_WILLNEED = 3
    MADV_SEQUENTIAL = 2
    MADV_RANDOM = 1
    MADV_DONTNEED = 4
    
    def prefetch_region(mmap_obj, offset: int, length: int):
        """
        Tell the OS to prefetch a region of the mmap.
        
        Args:
            mmap_obj: A Python mmap object or numpy memmap.
            offset: Start byte offset.
            length: Number of bytes to prefetch.
        """
        # Get the base address of the mmap
        if hasattr(mmap_obj, 'ctypes'):
            # numpy memmap
            base_addr = mmap_obj.ctypes.data
        else:
            # Python mmap
            base_addr = ctypes.addressof(ctypes.c_char.from_buffer(mmap_obj))
        
        addr = base_addr + offset
        
        result = libc.madvise(
            ctypes.c_void_p(addr),
            ctypes.c_size_t(length),
            ctypes.c_int(MADV_WILLNEED)
        )
        
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
```

### Using Prefetch in a DataLoader

```python
def prefetch_next_batch(mmap_arr, metadata, next_batch_indices):
    """
    Prefetch pages for the next batch while current batch is training.
    """
    for sample_id in next_batch_indices:
        ptr = metadata[sample_id]['data_ptr']
        size = metadata[sample_id]['data_size']
        
        # Round to page boundaries
        page_size = 4096
        start_page = (ptr // page_size) * page_size
        end_page = ((ptr + size + page_size - 1) // page_size) * page_size
        
        prefetch_region(mmap_arr, start_page, end_page - start_page)
```

## Complete Memory-Mapped Reader

```python
import numpy as np
from typing import Dict, Any

class MemoryMappedReader:
    """
    Complete reader for memory-mapped dataset access.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Parse the file structure
        self._read_header()
        self._read_field_descriptors()
        self._read_metadata()
        self._read_allocation_table()
        
        # Create memory map
        self.mmap = np.memmap(file_path, dtype=np.uint8, mode='r')
        
        # Build pointer lookup
        self._build_pointer_lookup()
    
    def _read_header(self):
        with open(self.file_path, 'rb') as f:
            # Read magic
            magic = f.read(8)
            if magic != MAGIC_BYTES:
                raise ValueError("Invalid file format")
            
            # Read header
            header = np.fromfile(f, dtype=HeaderType, count=1)[0]
            
            self.num_samples = int(header['num_samples'])
            self.num_fields = int(header['num_fields'])
            self.field_desc_ptr = int(header['field_desc_ptr'])
            self.metadata_ptr = int(header['metadata_ptr'])
            self.data_ptr = int(header['data_ptr'])
            self.alloc_table_ptr = int(header['alloc_table_ptr'])
    
    def _read_field_descriptors(self):
        with open(self.file_path, 'rb') as f:
            f.seek(self.field_desc_ptr)
            self.field_descriptors = np.fromfile(
                f, dtype=FieldDescType, count=self.num_fields
            )
        
        # Reconstruct field handlers
        self.fields = {}
        for desc in self.field_descriptors:
            name = bytes(desc['name']).rstrip(b'\x00').decode('utf-8')
            type_id = int(desc['type_id'])
            field_class = get_field_class(type_id)
            self.fields[name] = field_class.from_binary(bytes(desc['arguments']))
    
    def _read_metadata(self):
        # Build composite dtype from fields
        row_dtype = np.dtype([
            (name, field.metadata_type)
            for name, field in self.fields.items()
        ], align=True)
        
        with open(self.file_path, 'rb') as f:
            f.seek(self.metadata_ptr)
            self.metadata = np.fromfile(f, dtype=row_dtype, count=self.num_samples)
    
    def _read_allocation_table(self):
        with open(self.file_path, 'rb') as f:
            f.seek(self.alloc_table_ptr)
            self.alloc_table = np.fromfile(f, dtype=AllocEntryType)
    
    def _build_pointer_lookup(self):
        # Sort by pointer for binary search
        order = np.argsort(self.alloc_table['ptr'])
        self.sorted_ptrs = self.alloc_table['ptr'][order]
        self.sorted_sizes = self.alloc_table['size'][order]
    
    def read_data(self, ptr: int) -> np.ndarray:
        """
        Read variable-length data at the given pointer.
        
        Args:
            ptr: Byte offset (from metadata['data_ptr']).
        
        Returns:
            View into mmap (uint8 array).
        """
        idx = np.searchsorted(self.sorted_ptrs, ptr)
        size = self.sorted_sizes[idx]
        return self.mmap[ptr:ptr + size]
    
    def read_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Read and decode all fields for a sample.
        
        Args:
            sample_id: Index of the sample (0 to num_samples-1).
        
        Returns:
            Dictionary mapping field names to decoded values.
        """
        result = {}
        sample_meta = self.metadata[sample_id]
        
        for name, field in self.fields.items():
            field_meta = sample_meta[name]
            decoder = field.get_decoder()
            result[name] = decoder(field_meta, self.read_data)
        
        return result
    
    def __len__(self):
        return self.num_samples
```

## Exercises

1.  **Measure Page Faults**: Use `getrusage` or `/proc/self/stat` to count page faults during the first vs. second epoch.

2.  **Implement Prefetching**: Add a background thread that prefetches pages for the next batch while the current batch is decoding.

3.  **Huge Pages**: On Linux, use `madvise(MADV_HUGEPAGE)` to request 2 MB pages for the mmap. Measure TLB miss impact with `perf stat`.

4.  **Process Cache Manager**: Implement an alternative to OSCacheManager that explicitly reads pages into process memory (useful when mmap isn't available, e.g., on Windows with very large files).
