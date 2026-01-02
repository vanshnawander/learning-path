# OS and Hardware I/O Optimization: Deep Dive

## The Memory Hierarchy

Understanding the memory hierarchy is fundamental to building fast data loaders.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY HIERARCHY                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Level         Size          Latency      Bandwidth    Notes                  │
│  ─────         ────          ───────      ─────────    ─────                  │
│                                                                                │
│  L1 Cache      32-64 KB      ~1 ns        ~TB/s        Per-core               │
│       ↓                                                                        │
│  L2 Cache      256 KB-1 MB   ~3 ns        ~500 GB/s    Per-core               │
│       ↓                                                                        │
│  L3 Cache      8-64 MB       ~10 ns       ~200 GB/s    Shared across cores    │
│       ↓                                                                        │
│  RAM (DRAM)    32-512 GB     ~100 ns      ~50 GB/s     Main memory            │
│       ↓                                                                        │
│  NVMe SSD      512 GB-8 TB   ~50 µs       ~7 GB/s      PCIe attached          │
│       ↓                                                                        │
│  SATA SSD      256 GB-4 TB   ~100 µs      ~550 MB/s    SATA attached          │
│       ↓                                                                        │
│  HDD           1-20 TB       ~5-10 ms     ~150 MB/s    Spinning platters      │
│       ↓                                                                        │
│  Network       Unlimited     ~1-100 ms    ~1-100 Gbps  Variable               │
│                                                                                │
│  Key insight: Each level is 10-1000x slower than the one above.              │
│  Goal: Keep data in higher levels as much as possible.                        │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## The Translation Lookaside Buffer (TLB)

The TLB is a small, fast cache that stores recent virtual-to-physical address translations.

### Virtual Memory Primer

When your program accesses `array[1000000]`, you're using a **virtual address**. The CPU must translate this to a **physical address** in RAM. This is done via a **page table**, a large data structure that maps virtual pages to physical pages.

**The problem**: Page tables are huge and live in RAM. Looking them up for every memory access would be catastrophic for performance.

**The solution**: The TLB. It caches recent translations, making repeat accesses to the same pages nearly free.

### TLB Miss Cost

| Scenario | Latency |
|----------|---------|
| TLB hit | ~1 ns (included in L1 cache access) |
| TLB miss (page table walk) | ~10-20 ns (for a 4-level page table) |
| TLB miss + page fault | ~50,000-10,000,000 ns (disk access) |

### TLB Size and Page Size

| CPU | TLB Entries (approximate) |
|-----|---------------------------|
| Intel (Skylake+) | ~1500 for 4KB pages, ~1000 for 2MB pages |
| AMD (Zen 3+) | ~2000 for 4KB pages, ~1000 for 2MB pages |

With 4KB pages and 1500 TLB entries, you can cover 1500 × 4KB = **6 MB** of memory without TLB misses.

With 2MB huge pages and 1000 TLB entries, you can cover 1000 × 2MB = **2 GB** of memory.

## Huge Pages

### What They Are

Standard pages are 4KB. **Huge pages** (or "large pages") are 2MB (on x86) or 1GB.

Using huge pages:
1.  Reduces TLB misses (covers more memory with fewer entries).
2.  Reduces page table overhead.
3.  Can improve performance by 5-15% for large datasets.

### How to Use Huge Pages on Linux

#### Transparent Huge Pages (THP)

Linux can automatically promote 4KB pages to 2MB pages. This is called THP.

Check status:
```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never
```

Enable for specific memory regions with `madvise`:

```python
import ctypes
import numpy as np

libc = ctypes.CDLL("libc.so.6")
MADV_HUGEPAGE = 14

def enable_huge_pages(array: np.ndarray):
    """
    Request the kernel to use huge pages for this array.
    """
    addr = array.ctypes.data
    size = array.nbytes
    
    # Align to 2MB boundary
    page_size = 2 * 1024 * 1024
    aligned_addr = (addr // page_size) * page_size
    aligned_size = ((addr + size - aligned_addr + page_size - 1) // page_size) * page_size
    
    result = libc.madvise(
        ctypes.c_void_p(aligned_addr),
        ctypes.c_size_t(aligned_size),
        ctypes.c_int(MADV_HUGEPAGE)
    )
    
    if result != 0:
        import os
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
```

#### Explicit Huge Pages (HugePages)

For guaranteed huge pages (not subject to kernel decisions):

```bash
# Reserve 1024 huge pages (2GB)
echo 1024 > /proc/sys/vm/nr_hugepages
```

Then use `mmap` with `MAP_HUGETLB`:

```python
import mmap
import os

MAP_HUGETLB = 0x40000

def allocate_huge_page_mmap(size: int):
    """
    Allocate memory backed by huge pages.
    """
    fd = os.open("/dev/hugepages/my_data", os.O_RDWR | os.O_CREAT, 0o644)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return mm
```

## The OS Page Cache

The kernel maintains a **page cache**: recently accessed file data is kept in RAM. When you read a file, you might be reading from RAM, not disk.

### How It Works

1.  **Read**: If not in cache, read from disk → cache → your buffer. If in cache, cache → your buffer (fast!).
2.  **Write**: Write to cache (fast!). Kernel writes to disk later (asynchronously).
3.  **Eviction**: When RAM fills up, the kernel evicts (discards) least-recently-used pages.

### Why mmap Uses the Page Cache

When you `mmap` a file, the kernel maps the file's pages directly into your address space. There's no separate "your buffer"—you read directly from the page cache.

```
Traditional read():
  Disk → Page Cache → Your Buffer (COPY)

mmap():
  Disk → Page Cache = Your Address Space (NO COPY)
```

### Page Cache Thrashing

**Thrashing** occurs when your working set is larger than RAM, and you're constantly evicting pages you'll need again.

For an ML dataset:
- First epoch: All pages are loaded from disk (slow).
- Subsequent epochs: If dataset fits in RAM, all pages are cached (fast).
- If dataset > RAM, you get partial caching. Random shuffle is worst-case (no page is used twice before eviction).

## Quasi-Random Access: The Smart Shuffle

FFCV's key insight: **Global randomness for statistics, local sequentiality for I/O.**

### The Algorithm

```python
import numpy as np
from collections import defaultdict
from typing import List, Dict

def quasi_random_shuffle(
    metadata: np.ndarray,  # With 'data_ptr' field
    page_size: int = 2 * 1024 * 1024,  # 2MB pages
    seed: int = 42,
) -> np.ndarray:
    """
    Shuffle samples while keeping samples on the same page together.
    
    This achieves global randomness (for training) with local sequentiality (for I/O).
    
    Args:
        metadata: Array of sample metadata, each with a 'data_ptr' field.
        page_size: Size of disk/memory pages to group by.
        seed: Random seed for reproducibility.
    
    Returns:
        Array of sample indices in shuffled order.
    """
    rng = np.random.default_rng(seed)
    num_samples = len(metadata)
    
    # Step 1: Group samples by page
    page_to_samples: Dict[int, List[int]] = defaultdict(list)
    
    for sample_id in range(num_samples):
        ptr = metadata[sample_id]['data_ptr']
        page_id = ptr // page_size
        page_to_samples[page_id].append(sample_id)
    
    # Step 2: Get list of pages
    pages = list(page_to_samples.keys())
    
    # Step 3: Shuffle the pages
    rng.shuffle(pages)
    
    # Step 4: For each page, shuffle the samples within it
    order = []
    for page_id in pages:
        samples = page_to_samples[page_id]
        rng.shuffle(samples)
        order.extend(samples)
    
    return np.array(order, dtype=np.int64)


def analyze_locality(order: np.ndarray, metadata: np.ndarray, page_size: int):
    """
    Analyze the locality of a shuffle order.
    
    Returns:
        sequential_runs: Number of consecutive samples on the same page.
        total_page_switches: Number of times we switch between pages.
    """
    if len(order) == 0:
        return 0, 0
    
    current_page = metadata[order[0]]['data_ptr'] // page_size
    sequential_runs = 0
    page_switches = 0
    run_length = 1
    
    for i in range(1, len(order)):
        page = metadata[order[i]]['data_ptr'] // page_size
        if page == current_page:
            run_length += 1
        else:
            sequential_runs += run_length
            run_length = 1
            page_switches += 1
            current_page = page
    
    sequential_runs += run_length
    
    return sequential_runs, page_switches
```

### Why This Works

| Shuffle Type | Page Switches | Cache Efficiency |
|--------------|---------------|------------------|
| Fully random | ~N (worst) | Poor (high eviction) |
| Quasi-random | ~N/P (P = samples/page) | Good (use page fully before moving) |
| Sequential | 0 | Perfect (but biased training) |

For ImageNet:
- 1.2M samples, average ~100KB compressed = ~10 samples per 1MB page
- Quasi-random has ~120K page switches instead of ~1.2M
- First epoch speedup: 2-5x (depending on disk and RAM)

## madvise: Giving Hints to the Kernel

The kernel's default page cache behavior is generic. You can give it hints to optimize for your specific access pattern.

### The madvise Call

```c
int madvise(void *addr, size_t length, int advice);
```

| Advice | Constant | Meaning | Use Case |
|--------|----------|---------|----------|
| `MADV_NORMAL` | 0 | Default behavior | – |
| `MADV_RANDOM` | 1 | Expect random access | Disable readahead |
| `MADV_SEQUENTIAL` | 2 | Expect sequential access | Aggressive readahead |
| `MADV_WILLNEED` | 3 | Will access soon | Prefetch pages |
| `MADV_DONTNEED` | 4 | Don't need anymore | Allow eviction |
| `MADV_HUGEPAGE` | 14 | Use huge pages (THP) | Reduce TLB misses |
| `MADV_NOHUGEPAGE` | 15 | Don't use huge pages | Fine-grained control |

### Python Implementation

```python
import ctypes
import os
import numpy as np

libc = ctypes.CDLL("libc.so.6")

# Constants
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4
MADV_HUGEPAGE = 14

PAGE_SIZE = 4096  # Typical 4KB page

def madvise(mmap_array: np.ndarray, offset: int, length: int, advice: int):
    """
    Give the kernel a hint about memory usage.
    
    Args:
        mmap_array: A memory-mapped numpy array.
        offset: Byte offset within the array.
        length: Number of bytes to advise about.
        advice: One of the MADV_* constants.
    """
    # Get base address of the array
    base_addr = mmap_array.ctypes.data
    
    # Calculate the target address
    target_addr = base_addr + offset
    
    # Align to page boundary (required by madvise)
    aligned_start = (target_addr // PAGE_SIZE) * PAGE_SIZE
    aligned_end = ((target_addr + length + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    aligned_length = aligned_end - aligned_start
    
    result = libc.madvise(
        ctypes.c_void_p(aligned_start),
        ctypes.c_size_t(aligned_length),
        ctypes.c_int(advice)
    )
    
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def prefetch_batch(
    mmap_array: np.ndarray,
    metadata: np.ndarray,
    batch_indices: np.ndarray,
):
    """
    Prefetch pages for a batch of samples.
    
    Call this for NEXT batch while processing CURRENT batch.
    """
    for sample_id in batch_indices:
        ptr = metadata[sample_id]['data_ptr']
        size = metadata[sample_id]['data_size']
        madvise(mmap_array, ptr, size, MADV_WILLNEED)


def release_batch(
    mmap_array: np.ndarray,
    metadata: np.ndarray,
    batch_indices: np.ndarray,
):
    """
    Tell kernel we're done with a batch's pages.
    
    Call this AFTER processing to help with memory pressure.
    """
    for sample_id in batch_indices:
        ptr = metadata[sample_id]['data_ptr']
        size = metadata[sample_id]['data_size']
        madvise(mmap_array, ptr, size, MADV_DONTNEED)
```

## Direct I/O: Bypassing the Page Cache

For datasets much larger than RAM, the page cache can hurt:
1.  You pay the CPU cost of copying data into the cache...
2.  ...only to evict it immediately (no reuse).

**Direct I/O** (`O_DIRECT`) bypasses the page cache entirely:

```python
import os
import numpy as np

def read_direct(path: str, offset: int, size: int) -> np.ndarray:
    """
    Read data using Direct I/O, bypassing the page cache.
    
    Requirements:
    - offset must be aligned to 512 bytes (sector size)
    - size must be aligned to 512 bytes
    - the buffer must be aligned to 512 bytes
    """
    SECTOR_SIZE = 512
    
    # Open with O_DIRECT
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    
    try:
        # Align offset down
        aligned_offset = (offset // SECTOR_SIZE) * SECTOR_SIZE
        prepad = offset - aligned_offset
        
        # Align size up
        aligned_size = ((size + prepad + SECTOR_SIZE - 1) // SECTOR_SIZE) * SECTOR_SIZE
        
        # Allocate aligned buffer
        # Use ctypes or special allocators for alignment
        import ctypes
        buffer = (ctypes.c_char * aligned_size)()
        
        # Seek and read
        os.lseek(fd, aligned_offset, os.SEEK_SET)
        bytes_read = os.read(fd, aligned_size)
        
        # Extract the actual data (skip prepad)
        data = np.frombuffer(bytes_read[prepad:prepad+size], dtype=np.uint8).copy()
        return data
    
    finally:
        os.close(fd)
```

**Trade-offs**:
- ✅ No page cache pollution
- ✅ Lower CPU overhead for large, sequential reads
- ❌ No caching (every read hits disk)
- ❌ Requires aligned buffers and offsets
- ❌ More complex code

FFCV generally doesn't use Direct I/O because the page cache works well for "warm" datasets (multiple epochs).

## NUMA: Non-Uniform Memory Access

On multi-socket servers (2+ CPUs), memory access latency depends on which CPU socket the memory is attached to.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      NUMA TOPOLOGY (2-socket example)                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────┐         ┌─────────────────────┐                      │
│  │  CPU Socket 0       │  QPI    │  CPU Socket 1       │                      │
│  │  Cores 0-15         │ ◄─────► │  Cores 16-31        │                      │
│  │  L3 Cache (shared)  │  ~100ns │  L3 Cache (shared)  │                      │
│  └──────────┬──────────┘         └──────────┬──────────┘                      │
│             │ ~80ns                          │ ~80ns                           │
│             ▼                                ▼                                 │
│  ┌─────────────────────┐         ┌─────────────────────┐                      │
│  │  Memory Node 0      │         │  Memory Node 1      │                      │
│  │  DDR4 (128GB)       │         │  DDR4 (128GB)       │                      │
│  └─────────────────────┘         └─────────────────────┘                      │
│                                                                                │
│  Local access (CPU 0 → Node 0):   ~80ns                                       │
│  Remote access (CPU 0 → Node 1):  ~170ns (2x slower!)                         │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### NUMA-Aware Loading

For multi-GPU training on NUMA systems:
1.  Pin each worker thread to a specific CPU socket.
2.  Allocate buffers on the local memory node.
3.  Ensure each GPU's worker uses nearby CPUs/memory.

```python
import os

def set_numa_affinity(node: int):
    """
    Bind the current process to a NUMA node.
    
    Args:
        node: NUMA node ID (0, 1, ...)
    """
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    
    # Get CPUs for this node (simplified; use libnuma for real code)
    cpus = get_cpus_for_node(node)
    
    # Create CPU mask
    mask = 0
    for cpu in cpus:
        mask |= (1 << cpu)
    
    # Set affinity
    libc.sched_setaffinity(0, ctypes.sizeof(ctypes.c_ulong), ctypes.byref(ctypes.c_ulong(mask)))


def allocate_on_numa_node(size: int, node: int) -> np.ndarray:
    """
    Allocate memory on a specific NUMA node.
    """
    # This requires libnuma
    # pip install numa
    import numa
    
    numa.set_preferred(node)
    arr = np.empty(size, dtype=np.uint8)
    # Touch the memory to actually allocate it
    arr[:] = 0
    return arr
```

## Exercises

1.  **Measure TLB Misses**: Use `perf stat -e dTLB-load-misses` to compare 4KB vs 2MB pages.

2.  **Implement Quasi-Random Shuffle**: Modify the provided code to handle the case where samples span multiple pages.

3.  **Prefetch Benchmark**: Implement a loader with and without `madvise(MADV_WILLNEED)` prefetching. Measure the difference on an SSD and an HDD.

4.  **NUMA Locality Test**: On a NUMA system, measure memory bandwidth with local vs. remote memory access using a simple loop.
