# Storage Interfaces: From Disk to Memory

Understanding storage interfaces is critical for optimizing data loading pipelines.

## Storage Hierarchy for ML

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    STORAGE HIERARCHY FOR ML                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ┌─────────────┐                                                      ║
║  │ GPU HBM     │  3 TB/s, 40-80 GB      ← Keep tensors here!         ║
║  └──────┬──────┘                                                      ║
║         │ PCIe 4.0: 32 GB/s                                           ║
║  ┌──────▼──────┐                                                      ║
║  │ System RAM  │  100 GB/s, 64-512 GB   ← Batch staging              ║
║  └──────┬──────┘                                                      ║
║         │ NVMe/SATA                                                   ║
║  ┌──────▼──────┐                                                      ║
║  │ NVMe SSD    │  7 GB/s, 1-8 TB        ← Dataset storage            ║
║  └──────┬──────┘                                                      ║
║         │ Network                                                      ║
║  ┌──────▼──────┐                                                      ║
║  │ Cloud/NAS   │  1-25 GB/s, Unlimited  ← Large datasets             ║
║  └─────────────┘                                                      ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## Storage Technologies Comparison

| Technology | Sequential Read | Random 4KB Read | Latency | Cost/TB |
|------------|-----------------|-----------------|---------|---------|
| HDD (7200 RPM) | 200 MB/s | 1 MB/s | 10 ms | $20 |
| SATA SSD | 550 MB/s | 50 MB/s | 100 µs | $80 |
| NVMe SSD (Gen3) | 3.5 GB/s | 500 MB/s | 20 µs | $100 |
| NVMe SSD (Gen4) | 7 GB/s | 1 GB/s | 10 µs | $120 |
| NVMe SSD (Gen5) | 14 GB/s | 2 GB/s | 5 µs | $200 |
| RAM (DDR5) | 50 GB/s | 50 GB/s | 100 ns | $3000 |

## NVMe: The ML Storage Standard

### NVMe Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                       NVMe ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU                          NVMe Controller                   │
│  ┌─────────┐                 ┌─────────────────┐               │
│  │ Submit  │─── PCIe ───────▶│ Command Queue   │               │
│  │ Command │                 │ (up to 64K)     │               │
│  └─────────┘                 └────────┬────────┘               │
│                                       │                         │
│  ┌─────────┐                 ┌────────▼────────┐               │
│  │ Receive │◀─── PCIe ───────│ Completion Queue│               │
│  │ Result  │                 │                 │               │
│  └─────────┘                 └────────┬────────┘               │
│                                       │                         │
│                              ┌────────▼────────┐               │
│                              │   NAND Flash    │               │
│                              │   (Parallel)    │               │
│                              └─────────────────┘               │
│                                                                  │
│  Key Features:                                                  │
│  • Multiple queues (parallelism)                                │
│  • Direct PCIe connection (low latency)                         │
│  • No legacy overhead (unlike SATA)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Queue Depth and Parallelism

```c
// NVMe can handle many concurrent I/O operations
// This is why async I/O is so important!

// Queue depth comparison:
// SATA: 32 commands max
// NVMe: 65,535 commands per queue, up to 65,535 queues!

// For ML data loading:
// - Use io_uring or libaio for async I/O
// - Submit many reads in parallel
// - Let NVMe controller optimize access pattern
```

## I/O Patterns for ML

### Pattern 1: Sequential Reads (Best for Training)
```
File: dataset.bin
Access: [0, 1MB, 2MB, 3MB, 4MB, ...]

Throughput: Near maximum (7 GB/s on Gen4)

Use when:
- WebDataset (tar archives)
- FFCV .beton files
- Pre-shuffled data
```

### Pattern 2: Large Random Reads (Acceptable)
```
File: dataset.bin  
Access: [5MB, 100MB, 2MB, 50MB, ...]
Read size: 256KB - 1MB

Throughput: 3-5 GB/s (still good)

Use when:
- Memory-mapped datasets
- Pre-processed samples
```

### Pattern 3: Small Random Reads (Avoid!)
```
Files: image_0001.jpg, image_9999.jpg, image_0500.jpg
Access: Random files, 10-100KB each

Throughput: < 500 MB/s (very slow!)

This is why individual files are bad for ML!
```

## Direct I/O vs Buffered I/O

### Buffered I/O (Default)
```c
// Data flow: Disk → Page Cache → User Buffer
FILE* f = fopen("data.bin", "rb");
fread(buffer, 1, size, f);

// Pros:
// - OS caching helps repeated reads
// - Works with any file system

// Cons:
// - Extra memory copy
// - Page cache pollution
// - Unpredictable latency (cache eviction)
```

### Direct I/O (O_DIRECT)
```c
// Data flow: Disk → User Buffer (DMA)
int fd = open("data.bin", O_RDONLY | O_DIRECT);
read(fd, aligned_buffer, size);  // Buffer must be aligned!

// Pros:
// - No page cache pollution
// - Predictable performance
// - Better for large sequential reads

// Cons:
// - Requires aligned buffers
// - No OS caching (bad for small random reads)
```

## Memory-Mapped I/O (mmap)

### How mmap Works
```
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY MAPPING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Process Virtual Address Space                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 0x00000000  [Code]                                       │   │
│  │ 0x00400000  [Heap]                                       │   │
│  │ ...                                                       │   │
│  │ 0x7F000000  [mmap region] ◄─── Points to file pages      │   │
│  │ 0x7FFFFFFF  [Stack]                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                     │
│            │ Page fault on first access                         │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Page Cache                             │   │
│  │  OS loads pages on demand from file                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                     │
│            ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Disk                                 │   │
│  │  dataset.bin (100 GB)                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Benefits for ML:                                               │
│  • No explicit read() calls                                     │
│  • OS handles caching automatically                             │
│  • Multiple processes can share same mapping                    │
│  • Zero-copy when possible                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### mmap Best Practices for ML

```c
// Good: Large sequential mapping with hints
void* data = mmap(NULL, file_size, PROT_READ, 
                  MAP_SHARED | MAP_POPULATE, fd, 0);

// Tell OS about access pattern
madvise(data, file_size, MADV_SEQUENTIAL);  // Sequential access
madvise(data, file_size, MADV_WILLNEED);    // Prefetch

// For random access (FFCV style):
madvise(data, file_size, MADV_RANDOM);      // Don't prefetch linearly

// Bad: Mapping many small files
// Each mmap has overhead, use single large file instead!
```

## Async I/O: io_uring (Linux)

### Why io_uring?
```
Traditional:
  submit() → syscall → kernel → wait → return
  
io_uring:
  submit() → shared ring buffer → kernel processes async
  
No syscall per I/O operation!
```

### io_uring for ML Data Loading
```c
#include <liburing.h>

// Setup
struct io_uring ring;
io_uring_queue_init(256, &ring, 0);  // 256 queue entries

// Submit multiple reads
for (int i = 0; i < batch_size; i++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, fd, buffers[i], read_size, offsets[i]);
    sqe->user_data = i;  // Track which read this is
}
io_uring_submit(&ring);

// Wait for completions
struct io_uring_cqe *cqe;
for (int i = 0; i < batch_size; i++) {
    io_uring_wait_cqe(&ring, &cqe);
    int idx = cqe->user_data;
    // Process buffers[idx]
    io_uring_cqe_seen(&ring, cqe);
}
```

## Performance Numbers

### Real Benchmark: Loading 256KB Samples

| Method | Throughput | CPU Usage | Notes |
|--------|------------|-----------|-------|
| fread() sequential | 5.5 GB/s | 30% | Simple, good |
| fread() random | 800 MB/s | 80% | Syscall overhead |
| mmap + madvise | 6.2 GB/s | 10% | Best for FFCV |
| O_DIRECT | 6.5 GB/s | 5% | Needs alignment |
| io_uring | 6.8 GB/s | 5% | Best async |
| Individual files | 200 MB/s | 90% | Terrible! |

## ML Framework Storage Patterns

| Framework/Library | Storage Pattern | Why |
|-------------------|-----------------|-----|
| FFCV | mmap + single file | Zero-copy, random access |
| WebDataset | Sequential tar | Streaming, cloud-friendly |
| NVIDIA DALI | Async readers | GPU pipeline |
| HuggingFace | Arrow/Parquet | Columnar, lazy loading |
| TFRecord | Sequential protobuf | TensorFlow native |
