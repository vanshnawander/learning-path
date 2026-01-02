# OS and Hardware Concepts for Data Loading

## Overview

Building a high-performance data format requires deep understanding of how the operating system and hardware work together. This document covers the essential concepts.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE DATA LOADING STACK                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Application Layer                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │   PyTorch DataLoader  ←→  Custom Binary Format  ←→  Transforms  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                    │
│                                    │                                    │
│  OS Layer                          │                                    │
│  ┌────────────────┬────────────────┴────────────────┬──────────────┐   │
│  │  Virtual Mem   │      File System (VFS)          │   Scheduler  │   │
│  │  Management    │      ↓                          │              │   │
│  │                │  Page Cache / Buffer Cache      │              │   │
│  └────────────────┴────────────────┬────────────────┴──────────────┘   │
│                                    │                                    │
│  Hardware Layer                    ▼                                    │
│  ┌────────────────┬────────────────┬────────────────┬──────────────┐   │
│  │     CPU        │    Memory      │    Storage     │    GPU       │   │
│  │  (L1/L2/L3)    │    (DDR4/5)    │   (SSD/NVMe)   │   (HBM)      │   │
│  └────────────────┴────────────────┴────────────────┴──────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Memory Hierarchy

Understanding memory hierarchy is crucial for performance:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEMORY HIERARCHY                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Access                                                                  │
│  Time     Size        Level                                              │
│  ─────    ────        ─────                                              │
│                                                                          │
│  ~1ns     32KB       ┌────────────┐                                     │
│                      │  L1 Cache  │  ← Per core, instruction + data     │
│                      └─────┬──────┘                                     │
│                            │                                             │
│  ~3ns     256KB      ┌─────▼──────┐                                     │
│                      │  L2 Cache  │  ← Per core                         │
│                      └─────┬──────┘                                     │
│                            │                                             │
│  ~12ns    30MB       ┌─────▼──────┐                                     │
│                      │  L3 Cache  │  ← Shared across cores              │
│                      └─────┬──────┘                                     │
│                            │                                             │
│  ~100ns   128GB      ┌─────▼──────┐                                     │
│                      │    RAM     │  ← Main memory (DDR4/5)             │
│                      └─────┬──────┘                                     │
│                            │                                             │
│  ~100µs   2TB        ┌─────▼──────┐                                     │
│                      │   NVMe     │  ← Fast SSD                         │
│                      └─────┬──────┘                                     │
│                            │                                             │
│  ~10ms    10TB       ┌─────▼──────┐                                     │
│                      │    HDD     │  ← Spinning disk                    │
│                      └────────────┘                                     │
│                                                                          │
│  Key Insight: 100µs (SSD) vs 100ns (RAM) = 1000x slower!               │
│  Goal: Keep hot data in RAM, access sequentially                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Page Cache Deep Dive

The OS maintains a page cache that buffers disk data in RAM:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAGE CACHE OPERATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Application: read(fd, buffer, 4096)                                    │
│                       │                                                  │
│                       ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Check Page Cache                                        │   │
│  │                                                                   │   │
│  │   Page Cache (in RAM):                                           │   │
│  │   ┌─────┬─────┬─────┬─────┬─────┐                               │   │
│  │   │ P0  │ P1  │ P2  │ ... │ Pn  │  ← Page = 4KB                 │   │
│  │   │ ✓   │ ✓   │  ?  │     │     │  ← ✓ = in cache               │   │
│  │   └─────┴─────┴─────┴─────┴─────┘                               │   │
│  │                                                                   │   │
│  │   If page in cache: CACHE HIT (fast path)                        │   │
│  │   If page not in cache: CACHE MISS (slow path)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                       │                                                  │
│                       ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: On Cache Miss                                            │   │
│  │                                                                   │   │
│  │   1. Find free page (or evict LRU page)                          │   │
│  │   2. Issue I/O request to storage                                │   │
│  │   3. Block process until I/O completes                           │   │
│  │   4. Copy data to page cache                                     │   │
│  │   5. Copy from cache to user buffer                              │   │
│  │                                                                   │   │
│  │   Time: ~100µs (SSD) to ~10ms (HDD)                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Checking Page Cache Status

```bash
# Linux: Check how much is cached
$ free -h
              total        used        free      shared  buff/cache   available
Mem:          125Gi       45Gi       2.3Gi        12Gi        78Gi        67Gi
                                                              ^^^^
                                                              Page cache!

# Check specific file's cache status
$ vmtouch -v data.beton
data.beton
[OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO] 100%

Files: 1
Directories: 0
Resident Pages: 262144/262144  1G/1G  100%  # Fully cached!

# Watch cache activity
$ sar -B 1
pgpgin/s  pgpgout/s   # Pages read/written per second
```

## Virtual Memory and mmap

### How Memory Mapping Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEMORY MAPPING (mmap)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Without mmap (traditional read):                                        │
│  ────────────────────────────────                                        │
│                                                                          │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                  │
│  │  Disk  │───►│ Page   │───►│ Kernel │───►│  User  │                  │
│  │        │    │ Cache  │    │ Buffer │    │ Buffer │                  │
│  └────────┘    └────────┘    └────────┘    └────────┘                  │
│                                  │              │                        │
│                               memcpy         memcpy                      │
│                              (wasted!)      (wasted!)                    │
│                                                                          │
│  With mmap:                                                              │
│  ──────────                                                              │
│                                                                          │
│  ┌────────┐    ┌────────────────────────────────────┐                  │
│  │  Disk  │───►│     Page Cache = User Memory       │                  │
│  │        │    │     (SAME physical pages!)         │                  │
│  └────────┘    └────────────────────────────────────┘                  │
│                              │                                           │
│                           NO COPY!                                       │
│                                                                          │
│  mmap() creates a mapping where the page cache pages                    │
│  appear directly in the process's address space.                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Python mmap Usage

```python
import mmap
import numpy as np
import os

def map_file(filename, readonly=True):
    """Memory map a file."""
    
    # Open file
    mode = os.O_RDONLY if readonly else os.O_RDWR
    fd = os.open(filename, mode)
    
    try:
        # Get file size
        size = os.fstat(fd).st_size
        
        # Create mapping
        prot = mmap.PROT_READ if readonly else mmap.PROT_READ | mmap.PROT_WRITE
        mm = mmap.mmap(fd, size, prot=prot)
        
        return mm
    finally:
        os.close(fd)  # Can close fd after mmap


def mmap_numpy(filename, dtype, shape):
    """Create a numpy array backed by memory-mapped file."""
    
    return np.memmap(
        filename,
        dtype=dtype,
        mode='r',  # Read-only
        shape=shape
    )
```

### Page Sizes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PAGE SIZES                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Regular Pages: 4 KB                                                     │
│  ───────────────────                                                     │
│  • Default on most systems                                               │
│  • 1 GB file = 262,144 pages                                            │
│  • Each page needs an entry in page tables                              │
│  • Many entries = TLB pressure                                          │
│                                                                          │
│  Huge Pages: 2 MB (Linux) / 1 GB (Linux, limited)                       │
│  ────────────────────────────────────────────────────                   │
│  • 1 GB file = 512 pages (with 2MB pages)                               │
│  • Much less TLB pressure                                               │
│  • Better for large sequential access                                   │
│  • FFCV uses 2 MB as page_size for this reason!                         │
│                                                                          │
│  TLB (Translation Lookaside Buffer):                                     │
│  ──────────────────────────────────                                      │
│  • Cache for virtual → physical address translations                    │
│  • Only ~1024 entries                                                   │
│  • TLB miss = expensive page table walk                                 │
│  • Huge pages = fewer TLB entries needed                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## I/O Patterns

### Sequential vs Random Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    I/O ACCESS PATTERNS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SEQUENTIAL ACCESS                                                       │
│  ─────────────────                                                       │
│                                                                          │
│  File: ████████████████████████████████████████                         │
│        ──────────────────────────────────────►                          │
│        Read 1   Read 2   Read 3   Read 4                                │
│                                                                          │
│  • Predictable: OS can prefetch (read-ahead)                            │
│  • SSDs: ~3-7 GB/s sequential read                                      │
│  • HDDs: ~150-200 MB/s sequential                                       │
│  • Page cache very effective                                            │
│                                                                          │
│  RANDOM ACCESS                                                           │
│  ─────────────                                                           │
│                                                                          │
│  File: ████████████████████████████████████████                         │
│            ↑        ↑   ↑              ↑                                │
│          Read3    Read1 Read4        Read2                              │
│                                                                          │
│  • Unpredictable: no prefetch benefit                                   │
│  • SSDs: ~500K-1M IOPS, but latency adds up                            │
│  • HDDs: ~100-200 IOPS (seek time dominates!)                          │
│  • Page cache misses common                                             │
│                                                                          │
│  QUASI-RANDOM (FFCV's approach)                                         │
│  ──────────────────────────────                                         │
│                                                                          │
│  File: ████████████████████████████████████████                         │
│        ─────►     ─────►          ─────►                                │
│        Batch1     Batch2          Batch3                                │
│        (shuffled within batch, batches from different regions)          │
│                                                                          │
│  • Shuffle within pages, not across entire dataset                      │
│  • Maintains locality benefits                                          │
│  • Better cache utilization                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### I/O Scheduling

```python
"""
Understanding I/O scheduling for data loading.
"""

import os
import time
import numpy as np

def sequential_read(filename, chunk_size=1024*1024):
    """Read file sequentially."""
    total_bytes = 0
    with open(filename, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            total_bytes += len(data)
    return total_bytes


def random_read(filename, num_reads=10000, read_size=4096):
    """Read random positions."""
    file_size = os.path.getsize(filename)
    
    positions = np.random.randint(0, file_size - read_size, size=num_reads)
    
    with open(filename, 'rb') as f:
        for pos in positions:
            f.seek(pos)
            f.read(read_size)


def benchmark_io_pattern(filename):
    """Compare sequential vs random I/O."""
    
    file_size = os.path.getsize(filename)
    
    # Sequential
    start = time.perf_counter()
    sequential_read(filename)
    seq_time = time.perf_counter() - start
    seq_throughput = file_size / seq_time / 1e9
    
    # Random (same total bytes)
    num_reads = file_size // 4096
    start = time.perf_counter()
    random_read(filename, num_reads, 4096)
    rand_time = time.perf_counter() - start
    rand_throughput = file_size / rand_time / 1e9
    
    print(f"Sequential: {seq_throughput:.2f} GB/s")
    print(f"Random:     {rand_throughput:.2f} GB/s")
    print(f"Ratio:      {seq_throughput/rand_throughput:.1f}x faster")
```

## CPU Cache and Memory Alignment

### Why Alignment Matters

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY ALIGNMENT                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  UNALIGNED ACCESS (slow):                                               │
│  ────────────────────────                                                │
│                                                                          │
│  Memory:  [  Cache Line 1  ][  Cache Line 2  ]                          │
│           [0...............63|64..............127]                      │
│                           ↑       ↑                                      │
│           Our 8-byte value spans two cache lines!                       │
│           → Two cache line fetches                                       │
│           → Potential cache line split penalty                          │
│                                                                          │
│  ALIGNED ACCESS (fast):                                                  │
│  ──────────────────────                                                  │
│                                                                          │
│  Memory:  [  Cache Line 1  ][  Cache Line 2  ]                          │
│           [0...............63|64..............127]                      │
│           ↑───────↑                                                      │
│           Our 8-byte value fits in one cache line                       │
│           → Single cache line fetch                                      │
│           → Maximum efficiency                                           │
│                                                                          │
│  Cache line size: 64 bytes (most modern CPUs)                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Alignment in NumPy

```python
import numpy as np

# Check alignment
def check_alignment(arr):
    """Check if array is cache-line aligned."""
    ptr = arr.ctypes.data
    cache_line = 64
    
    alignment = ptr % cache_line
    print(f"Address: 0x{ptr:x}")
    print(f"Offset from cache line: {alignment} bytes")
    print(f"Aligned: {alignment == 0}")

# Create aligned array
def aligned_array(shape, dtype, alignment=64):
    """Create cache-line aligned array."""
    dtype = np.dtype(dtype)
    nbytes = np.prod(shape) * dtype.itemsize
    
    # Allocate with extra space for alignment
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    
    # Find aligned offset
    offset = alignment - (buf.ctypes.data % alignment)
    
    # Create view at aligned offset
    return buf[offset:offset + nbytes].view(dtype).reshape(shape)

# Example
arr = aligned_array((1024, 1024), np.float32)
check_alignment(arr)
```

## Process and Thread Considerations

### Python GIL and Multiprocessing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM STRATEGIES                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  THREADING (limited by GIL):                                            │
│  ───────────────────────────                                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │             Python Process (one GIL)                │                │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │                │
│  │  │ Thread1 │ │ Thread2 │ │ Thread3 │ │ Thread4 │  │                │
│  │  │  wait   │ │  RUN    │ │  wait   │ │  wait   │  │                │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                          │
│  • Only one thread runs Python at a time                                │
│  • BUT: I/O operations release GIL                                      │
│  • Good for I/O-bound (file reading, network)                          │
│  • Bad for CPU-bound (decompression, transforms)                       │
│                                                                          │
│  MULTIPROCESSING (true parallelism):                                    │
│  ──────────────────────────────────                                     │
│                                                                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                 │
│  │  Process 1    │ │  Process 2    │ │  Process 3    │                 │
│  │  (own GIL)    │ │  (own GIL)    │ │  (own GIL)    │                 │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │                 │
│  │  │  RUN    │  │ │  │  RUN    │  │ │  │  RUN    │  │                 │
│  │  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │                 │
│  └───────────────┘ └───────────────┘ └───────────────┘                 │
│                                                                          │
│  • True parallel execution                                               │
│  • Each process has separate memory                                     │
│  • Need IPC (shared memory, pipes) for communication                   │
│  • Higher overhead for process creation                                 │
│                                                                          │
│  NUMBA JIT (GIL release):                                               │
│  ───────────────────────                                                 │
│                                                                          │
│  @nb.njit(parallel=True, nogil=True)                                    │
│  def process_batch(data):                                               │
│      # This runs in parallel without GIL!                               │
│      for i in nb.prange(data.shape[0]):                                 │
│          # Process each sample                                          │
│                                                                          │
│  • Compiled code can release GIL                                        │
│  • True parallel threading                                               │
│  • Best of both worlds                                                   │
│  • FFCV uses this heavily!                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Shared Memory Between Processes

```python
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

class SharedDataset:
    """
    Dataset with data in shared memory.
    Multiple processes can access without copying.
    """
    
    def __init__(self, data: np.ndarray, name: str = None):
        # Create shared memory block
        self.shm = shared_memory.SharedMemory(
            name=name,
            create=True,
            size=data.nbytes
        )
        
        # Create numpy array backed by shared memory
        self.shared_array = np.ndarray(
            data.shape,
            dtype=data.dtype,
            buffer=self.shm.buf
        )
        
        # Copy data to shared memory
        self.shared_array[:] = data
        
        self.shape = data.shape
        self.dtype = data.dtype
        self.shm_name = self.shm.name
    
    def get_worker_view(self):
        """Get a view of the data for a worker process."""
        # Open existing shared memory
        shm = shared_memory.SharedMemory(name=self.shm_name)
        
        # Create numpy view (no copy!)
        return np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=shm.buf
        )
    
    def cleanup(self):
        """Release shared memory."""
        self.shm.close()
        self.shm.unlink()


def worker_function(shm_name, shape, dtype, indices):
    """Worker that processes samples from shared memory."""
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    results = []
    for idx in indices:
        # Process sample (no copying from main process!)
        sample = data[idx]
        results.append(sample.mean())
    
    shm.close()
    return results
```

## Storage Technologies

### NVMe vs SATA SSD vs HDD

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STORAGE PERFORMANCE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                Sequential Read    Random 4K Read    Latency             │
│                ───────────────    ──────────────    ───────             │
│  NVMe SSD      3-7 GB/s          500K-1M IOPS      ~20-100 µs          │
│  SATA SSD      500-550 MB/s      50K-100K IOPS     ~100-200 µs         │
│  HDD           150-200 MB/s      100-200 IOPS      ~5-15 ms            │
│                                                                          │
│  For ML data loading:                                                   │
│  ────────────────────                                                   │
│  • NVMe: Can saturate CPU with data                                     │
│  • SATA SSD: Adequate for most training                                │
│  • HDD: ONLY viable with aggressive caching                            │
│                                                                          │
│  Queue Depth Matters:                                                   │
│  ────────────────────                                                   │
│  NVMe can handle 64K commands in parallel!                             │
│  • io_uring (Linux 5.1+) or libaio for async I/O                       │
│  • Multiple outstanding requests = higher throughput                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Network-Attached Storage

For distributed training, data often comes from network storage:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NETWORK STORAGE                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Local SSD:                                                              │
│  ──────────                                                              │
│  ┌──────────────────┐                                                   │
│  │  GPU Server      │                                                   │
│  │  ┌────────────┐  │                                                   │
│  │  │   NVMe     │  │  Latency: ~20-100µs                              │
│  │  └────────────┘  │  Throughput: 3-7 GB/s                            │
│  └──────────────────┘                                                   │
│                                                                          │
│  Network Storage (NFS/GPFS/Lustre):                                     │
│  ──────────────────────────────────                                     │
│  ┌──────────────────┐         ┌──────────────────┐                     │
│  │  GPU Server      │   ───►  │  Storage Server  │                     │
│  │                  │  100Gb  │  ┌────────────┐  │                     │
│  │                  │  RDMA   │  │ Array/SSD  │  │                     │
│  └──────────────────┘         │  └────────────┘  │                     │
│                               └──────────────────┘                     │
│                                                                          │
│  Latency: ~100µs-1ms (10-100x higher than local)                       │
│  Throughput: Limited by network (100Gb = 12.5 GB/s theoretical)        │
│                                                                          │
│  Strategies:                                                             │
│  ──────────                                                              │
│  1. Cache dataset locally before training                               │
│  2. Use page cache aggressively (large RAM)                            │
│  3. Prefetch next epoch while training                                  │
│  4. Use distributed file system with local caching (BeeGFS, Lustre)    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary: Design Implications

Understanding these concepts leads to key design decisions:

| Concept | Design Decision |
|---------|-----------------|
| Page cache | Use mmap for automatic caching |
| Huge pages | Align data to 2MB boundaries |
| Sequential I/O | Store related data together |
| Cache lines | Align structures to 64 bytes |
| GIL | Use Numba or multiprocessing |
| Memory hierarchy | Keep hot data small, prefetch cold data |
| Storage speeds | Design for page-cache-hit training |

These concepts are applied throughout FFCV's design and should inform your custom format.
