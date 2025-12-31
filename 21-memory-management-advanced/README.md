# 06 - Memory Management

Deep dive into how memory works - critical for performance optimization.

## 📚 Topics Covered

### Memory Architecture
- **DRAM Organization**: Banks, rows, columns
- **Memory Controllers**: Scheduling, interleaving
- **NUMA**: Non-Uniform Memory Access
- **Memory Bandwidth**: Theoretical vs achieved

### Cache Systems
- **Cache Organization**: Sets, ways, lines
- **Replacement Policies**: LRU, pseudo-LRU
- **Write Policies**: Write-back, write-through
- **Prefetching**: Hardware and software prefetch
- **Cache Blocking/Tiling**: Matrix algorithms

### Virtual Memory Deep Dive
- **Page Tables**: 4-level paging (x86-64)
- **TLB**: Translation Lookaside Buffer
- **Huge Pages**: 2MB, 1GB pages
- **Memory Mapping**: Anonymous, file-backed
- **Copy-on-Write**: Fork optimization

### Memory Allocators
- **glibc malloc**: Arena-based allocation
- **jemalloc**: Facebook's allocator
- **tcmalloc**: Google's thread-caching allocator
- **mimalloc**: Microsoft's allocator
- **Custom Allocators**: Pools, slabs, arenas

### Memory Optimization
- **Data Layout**: SoA vs AoS
- **Alignment**: Cache line alignment
- **False Sharing**: Multi-threaded pitfall
- **Memory Barriers**: Ordering constraints
- **Memory Profiling**: Cachegrind, perf

## 🎯 Learning Objectives

- [ ] Understand cache behavior deeply
- [ ] Optimize data layouts for cache
- [ ] Use huge pages effectively
- [ ] Profile memory access patterns
- [ ] Implement a custom allocator

## 💻 Practical Exercises

1. Measure cache miss rates
2. Compare SoA vs AoS performance
3. Implement a slab allocator
4. Profile NUMA effects

## 📖 Resources

### Books
- "What Every Programmer Should Know About Memory" - Ulrich Drepper (FREE)
- "Computer Architecture" - Hennessy & Patterson

### Tools
- perf stat, perf mem
- Valgrind Cachegrind
- Intel VTune

## 📁 Structure

```
06-memory-management/
├── memory-architecture/
│   ├── dram/
│   ├── numa/
│   └── bandwidth/
├── caching/
│   ├── cache-organization/
│   ├── cache-blocking/
│   └── prefetching/
├── virtual-memory/
│   ├── page-tables/
│   ├── huge-pages/
│   └── mmap/
├── allocators/
│   ├── malloc-internals/
│   ├── jemalloc/
│   └── custom/
└── optimization/
    ├── data-layout/
    ├── profiling/
    └── false-sharing/
```

## ⏱️ Estimated Time: 3-4 weeks
