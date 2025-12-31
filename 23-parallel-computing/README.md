# 08 - Parallel Computing Fundamentals

Understanding parallelism before diving into GPUs.

## 📚 Topics Covered

### Parallel Programming Models
- **Shared Memory**: Threads, OpenMP
- **Distributed Memory**: MPI basics
- **Data Parallelism**: SIMD, vectorization
- **Task Parallelism**: Fork-join, work stealing

### Threading
- **POSIX Threads**: pthread API
- **C++ std::thread**: Modern threading
- **Thread Pools**: Efficient thread reuse
- **Thread-Local Storage**: Per-thread data

### OpenMP
- **Parallel Regions**: #pragma omp parallel
- **Work Sharing**: for, sections, tasks
- **Synchronization**: critical, atomic, barrier
- **Reductions**: Sum, min, max patterns
- **SIMD Directives**: #pragma omp simd

### Synchronization Primitives
- **Mutexes**: pthread_mutex, std::mutex
- **Spinlocks**: Busy-waiting locks
- **Read-Write Locks**: Multiple readers
- **Condition Variables**: Wait/notify patterns
- **Atomics**: Compare-and-swap, fetch-add

### Parallel Patterns
- **Map**: Apply function to all elements
- **Reduce**: Combine elements
- **Scan**: Prefix sums
- **Stencil**: Neighborhood operations
- **Pipeline**: Streaming parallelism

## 🎯 Learning Objectives

- [ ] Write OpenMP parallel code
- [ ] Implement parallel reduction
- [ ] Design scalable synchronization
- [ ] Measure parallel speedup

## 💻 Practical Exercises

1. Parallelize matrix multiplication
2. Implement parallel merge sort
3. Write a thread pool
4. Benchmark Amdahl's law

## 📖 Resources

### Books
- "An Introduction to Parallel Programming" - Pacheco
- "Programming with POSIX Threads" - Butenhof

## 📁 Structure

```
08-parallel-computing/
├── threading/
│   ├── pthreads/
│   ├── cpp-threads/
│   └── thread-pools/
├── openmp/
│   ├── basics/
│   ├── work-sharing/
│   └── simd/
├── synchronization/
│   ├── mutexes/
│   ├── atomics/
│   └── lock-free/
└── patterns/
    ├── map-reduce/
    ├── scan/
    └── pipeline/
```

## ⏱️ Estimated Time: 3-4 weeks
