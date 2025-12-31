# 01 - Computer Architecture

Understanding how computers work at the hardware level - the foundation for all performance optimization.

## 📁 Directory Structure

```
01-computer-architecture/
├── 01-binary-and-bits/       # Binary, floating point, bit ops
│   ├── 01_binary_basics.c
│   ├── 02_floating_point.c
│   ├── 03_bit_operations.c
│   └── 04_endianness.c
├── 02-memory-hierarchy/      # Caching, blocking, prefetch
│   ├── 01_cache_basics.c
│   ├── 02_cache_blocking.c
│   ├── 03_prefetching.c
│   └── 04_false_sharing.c
├── 03-simd-vectorization/    # SSE, AVX, vectorization
│   └── 01_simd_basics.c
├── 04-memory-alignment/      # Alignment for CPU/GPU
│   └── 01_alignment_basics.c
├── 05-cpu-pipeline/          # ILP, branch prediction
│   └── 01_pipeline_basics.c
├── 06-data-layout/           # AoS vs SoA, tensor layouts
│   └── 01_soa_vs_aos.c
└── 07-benchmarking/          # Correct measurement
    └── 01_benchmark_basics.c
```

## 📚 Topics Covered

### CPU Fundamentals
- **Von Neumann Architecture**: Fetch-decode-execute cycle
- **Pipelining**: Instruction-level parallelism, hazards
- **Superscalar Execution**: Multiple execution units
- **Branch Prediction**: Speculative execution
- **Out-of-Order Execution**: Tomasulo's algorithm

### Memory Hierarchy
- **Registers**: Fastest storage, limited quantity
- **Cache Levels**: L1, L2, L3 cache design
- **Cache Coherency**: MESI protocol, false sharing
- **DRAM**: Row buffers, bank conflicts
- **Virtual Memory**: Page tables, TLB

### Instruction Set Architecture (ISA)
- **RISC vs CISC**: Design philosophies
- **x86-64**: Intel/AMD architecture
- **ARM**: Mobile and server ARM processors
- **RISC-V**: Open-source ISA

### Modern CPU Features
- **SIMD**: SSE, AVX, AVX-512, NEON
- **Vector Processing**: Data parallelism
- **Hardware Prefetching**: Automatic data loading
- **Memory Ordering**: Memory barriers, atomics

## 🎯 Learning Objectives

- [ ] Explain the fetch-decode-execute cycle
- [ ] Understand cache hierarchy and locality
- [ ] Identify pipeline hazards
- [ ] Use SIMD instructions for vectorization
- [ ] Analyze memory access patterns

## 💻 Practical Exercises

1. Write a program that demonstrates cache effects
2. Measure memory bandwidth at different levels
3. Implement matrix multiply with SIMD
4. Profile branch mispredictions

## 📖 Resources

### Books
- "Computer Architecture: A Quantitative Approach" - Hennessy & Patterson
- "Computer Systems: A Programmer's Perspective" - Bryant & O'Hallaron

### Online
- MIT 6.004 Computation Structures
- Computer Architecture course by Onur Mutlu (CMU)

## 📁 Structure

```
01-computer-architecture/
├── cpu-fundamentals/
│   ├── pipelining/
│   ├── branch-prediction/
│   └── superscalar/
├── memory-hierarchy/
│   ├── caching/
│   ├── virtual-memory/
│   └── cache-coherency/
├── isa/
│   ├── x86-64/
│   ├── arm/
│   └── risc-v/
└── simd-vectorization/
    ├── sse-avx/
    └── neon/
```

## ⏱️ Estimated Time: 4-6 weeks
