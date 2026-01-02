# FFCV Memory Internals: The Zero-Copy Promise

The term "Zero-Copy" is thrown around a lot. In FFCV, it means strictly: **Data is never copied from the OS Page Cache to User Space buffers until the very moment it is transformed.**

## The Read Path: `OSCacheManager`

The `OSCacheManager` is the bridge between the raw file bytes and the JIT pipeline.

### 1. The State Tuple
Passed to every JIT function is a state tuple: `(mmap_view, sorted_ptrs, sorted_sizes)`.

*   `mmap_view`: A numpy memory map of the *entire* file as `uint8`.
*   `sorted_ptrs`: Array of all data pointers in the file, sorted ascending.
*   `sorted_sizes`: Corresponding sizes for those pointers.

### 2. The `read` Function Algorithm

```python
def read(address, mem_state):
    mmap, ptrs, sizes = mem_state
    
    # 1. Find the index of the address in the sorted pointer list
    #    Binary Search: O(log N)
    idx = np.searchsorted(ptrs, address)
    
    # 2. Retrieve the size
    size = sizes[idx]
    
    # 3. Return a Slice (View)
    return mmap[address : address + size]
```

**Critical Implementation Detail**: In Numba, `array[start:end]` on a `uint8[::1]` array returns a **view**, not a copy. This means the returned variable is just a pointer to the address inside the OS file cache (mapped into virtual memory).

### 3. Allocation Table

To make step 1 work, FFCV must load the *entire* allocation table into RAM at startup. For a dataset with 10M samples, this is `10M * 8 bytes (ptr) + 10M * 8 bytes (size) ≈ 160MB`. This is negligible compared to the dataset size.

## The Write Path: `PageAllocator`

Writing efficiently is harder than reading. Naively appending data causes fragmentation and poor IOPS. FFCV uses **Page-Based Allocation**.

### 1. The Page Concept
Data is not written to the file immediately. It is collected into a buffer (a "Page"), typically 2MB or larger.

```
[ HEADER ] [ SPACE for 15 images ] [ ... ]
```

### 2. Parallel Writers, Serial Output
To allow multi-threaded writing without file locking:
1.  A worker thread requests a "Page" from the allocator.
2.  The allocator assigns a file offset range (e.g., bits `100MB` to `102MB`) to that thread.
3.  The thread fills this local buffer completely in RAM.
4.  Once full, the thread asks for a new page.
5.  The filled page is flushed to disk.

**Benefit**: Large sequential writes are friendly to HDDs and SSDs. Small random writes are avoided.

### 3. The `malloc` function

Inside the `DatasetWriter`, fields receive a `malloc` callback.

```python
def malloc(size):
    # Check if current page has space
    if current_page.remaining < size:
        flush(current_page)
        current_page = allocate_new_page()
        
    ptr = current_page.current_ptr
    current_page.current_ptr += size
    
    # Return pointer into the BUFFER, not the file yet
    return current_page_file_offset + ptr, current_page.buffer[ptr : ptr+size]
```

## OS-Level Mechanics

### Memory Mapping (`mmap`)
`mmap` tells the kernel: "Map the file `data.beton` to the virtual address range `0x7f...` to `0x8f...`".
*   **Initial State**: The range is valid but empty.
*   **Access**: When `read()` accesses `mmap[ptr]`, the CPU raises a **Page Fault**.
*   **Resolution**: The Kernel catches the fault, sees it's a mapped file, reads the 4KB block from disk into RAM (Page Cache), and resumes the process.

### `madvise` Hints
FFCV (optionally) calls `madvise` to pre-trigger these page faults.
*   `MADV_WILLNEED`: Triggers the read immediately (Prefetching).
*   `MADV_SEQUENTIAL`: Tells OS to use aggressive readahead (good for linear scans).
*   `MADV_RANDOM`: Tells OS to disable readahead (good for random sampling).

## Alignment

For maximum efficiency with Direct I/O (if used) and SSD block boundaries, FFCV attempts to align allocations.
*   **4KB Alignment**: Matches standard OS page size.
*   **512B Alignment**: Matches standard disk sector size.

If a file format ignores alignment, `read()` calls might span two pages unnecessarily, triggering two page faults instead of one.
