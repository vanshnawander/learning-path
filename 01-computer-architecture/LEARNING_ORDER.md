# Computer Architecture Learning Order

A suggested order for going through this module.

## Week 1: Binary Foundations
**Folder**: `01-binary-and-bits/`

1. `01_binary_basics.c` - Integer representation
2. `02_floating_point.c` - IEEE 754 (critical for quantization!)
3. `03_bit_operations.c` - Bitwise ops for CUDA/Triton
4. `04_endianness.c` - Byte ordering

**Goal**: Understand exactly how data is represented.

## Week 2: Memory Hierarchy
**Folder**: `02-memory-hierarchy/`

1. `01_cache_basics.c` - Cache behavior experiments
2. `02_cache_blocking.c` - Tiling (Flash Attention foundation!)
3. `03_prefetching.c` - Hiding latency
4. `04_false_sharing.c` - Multi-threaded pitfall

**Goal**: Understand why memory access patterns matter.

## Week 3: CPU Optimization
**Folders**: `03-simd-vectorization/`, `04-memory-alignment/`, `05-cpu-pipeline/`

1. `01_simd_basics.c` - Vector processing
2. `01_alignment_basics.c` - Memory alignment
3. `01_pipeline_basics.c` - Branch prediction

**Goal**: Know CPU optimization techniques.

## Week 4: Data Organization
**Folders**: `06-data-layout/`, `07-benchmarking/`

1. `01_soa_vs_aos.c` - Data layout for performance
2. `01_benchmark_basics.c` - Correct measurement

**Goal**: Organize data for maximum performance.

## Compile and Run

All files are standalone C programs:
```bash
# Basic compilation
gcc -O2 -o program program.c

# With SIMD
gcc -O3 -mavx2 -o program program.c

# With threads
gcc -O2 -pthread -o program program.c

# With math library
gcc -O2 -o program program.c -lm
```

## Connection to ML Systems

| Concept | Used In |
|---------|---------|
| Cache blocking | Flash Attention tiling |
| mmap | FFCV data loading |
| SIMD | NumPy, image decode |
| Data layout | Tensor NCHW vs NHWC |
| Alignment | GPU coalescing |
| Prefetching | DataLoader prefetch |
