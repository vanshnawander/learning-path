# 02 - Operating Systems

Understanding how the OS manages hardware resources and provides abstractions for programs.

## 📁 Directory Structure

```
02-operating-systems/
├── 01-memory-mapping/        # mmap - foundation of FFCV
│   ├── 01_mmap_basics.c
│   └── 02_mmap_dataloader.c
├── 02-processes-threads/     # fork, pthreads
│   ├── 01_fork_basics.c
│   └── 02_threads_basics.c
├── 03-file-io/               # Buffered, direct, async I/O
│   └── 01_io_basics.c
├── 04-virtual-memory/        # Page tables, demand paging
│   └── 01_virtual_memory.c
├── 05-system-calls/          # Syscall overhead
│   └── 01_syscall_overhead.c
├── 06-memory-allocators/     # malloc, caching allocator
│   └── 01_malloc_internals.c
├── 07-synchronization/       # Atomics, mutexes
│   └── 01_atomics.c
└── 08-shared-memory-ipc/     # IPC for DataLoader
    └── 01_shared_memory.c
```

## 📚 Topics Covered

### Process Management
- **Processes vs Threads**: Creation, lifecycle, states
- **Context Switching**: Overhead and optimization
- **Scheduling**: CFS, priority scheduling, real-time
- **Inter-Process Communication**: Pipes, shared memory, sockets

### Memory Management
- **Virtual Address Space**: Layout, segments
- **Page Tables**: Multi-level, huge pages
- **Memory Allocation**: malloc internals, jemalloc, tcmalloc
- **Memory Mapping**: mmap, file-backed memory
- **Swap**: Paging to disk

### I/O and File Systems
- **Block I/O**: Schedulers, async I/O
- **File Systems**: ext4, XFS, performance characteristics
- **Direct I/O**: Bypassing page cache
- **Memory-Mapped Files**: Performance benefits

### Synchronization
- **Locks**: Mutexes, spinlocks, reader-writer locks
- **Lock-Free Programming**: Atomics, CAS operations
- **Condition Variables**: Wait/signal patterns
- **Barriers**: Thread synchronization

### System Calls
- **Interface to Kernel**: syscall mechanism
- **Common syscalls**: read, write, mmap, clone
- **Overhead**: User/kernel transitions

## 🎯 Learning Objectives

- [ ] Understand process vs thread differences
- [ ] Implement synchronization primitives
- [ ] Use mmap for efficient file access
- [ ] Profile system call overhead
- [ ] Understand memory allocator design

## 💻 Practical Exercises

1. Implement a simple thread pool
2. Write a memory allocator
3. Benchmark different I/O patterns
4. Profile context switch overhead

## 📖 Resources

### Books
- "Operating Systems: Three Easy Pieces" (OSTEP) - FREE online
- "Linux Kernel Development" - Robert Love
- "Understanding the Linux Kernel" - Bovet & Cesati

### Online
- xv6 teaching operating system (MIT)
- Linux kernel source code

## 📁 Structure

```
02-operating-systems/
├── processes-threads/
│   ├── creation/
│   ├── scheduling/
│   └── ipc/
├── memory-management/
│   ├── virtual-memory/
│   ├── allocators/
│   └── mmap/
├── io-filesystems/
│   ├── block-io/
│   ├── async-io/
│   └── direct-io/
└── synchronization/
    ├── locks/
    ├── atomics/
    └── lock-free/
```

## ⏱️ Estimated Time: 4-6 weeks
