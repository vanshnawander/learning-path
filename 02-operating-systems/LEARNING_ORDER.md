# Operating Systems Learning Order

A suggested order for going through this module.

## Week 1: Memory Mapping (Most Important!)
**Folder**: `01-memory-mapping/`

1. `01_mmap_basics.c` - Memory mapping fundamentals
2. `02_mmap_dataloader.c` - Build a mini FFCV-style loader

**Goal**: Understand how FFCV achieves fast data loading.

## Week 2: Processes and Threads
**Folder**: `02-processes-threads/`

1. `01_fork_basics.c` - Process creation
2. `02_threads_basics.c` - POSIX threads

**Goal**: Understand DataLoader multiprocessing.

## Week 3: I/O and System Calls
**Folders**: `03-file-io/`, `05-system-calls/`

1. `01_io_basics.c` - I/O methods comparison
2. `01_syscall_overhead.c` - Why syscalls are slow

**Goal**: Know when to use which I/O method.

## Week 4: Memory Management
**Folders**: `04-virtual-memory/`, `06-memory-allocators/`

1. `01_virtual_memory.c` - Pages, page faults
2. `01_malloc_internals.c` - Allocator design

**Goal**: Understand memory allocation (PyTorch caching allocator).

## Week 5: Concurrency
**Folders**: `07-synchronization/`, `08-shared-memory-ipc/`

1. `01_atomics.c` - Lock-free programming
2. `01_shared_memory.c` - IPC for DataLoader

**Goal**: Understand parallel data loading.

## Compile and Run

```bash
# Basic
gcc -O2 -o program program.c

# With threads
gcc -O2 -pthread -o program program.c

# With shared memory
gcc -O2 -o program program.c -lrt

# With real-time library (some systems)
gcc -O2 -o program program.c -lrt -lpthread
```

## Connection to ML Systems

| OS Concept | ML System Use |
|------------|---------------|
| mmap | FFCV .beton files |
| fork | DataLoader workers |
| Shared memory | Worker → main data transfer |
| Virtual memory | Model weight loading |
| Allocator caching | PyTorch CUDA allocator |
| Atomics | Gradient accumulation |
