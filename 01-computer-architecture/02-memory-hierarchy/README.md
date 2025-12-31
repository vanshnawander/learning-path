# Memory Hierarchy Deep Dive

The #1 factor in performance - understanding how memory works.

## Files in This Directory

| File | Description | Key Concept |
|------|-------------|-------------|
| `01_cache_basics.c` | Cache behavior experiments | Locality, cache lines |
| `02_cache_blocking.c` | Tiling for matrix ops | Flash Attention foundation |
| `03_prefetching.c` | Hardware/software prefetch | Data loading pipelines |
| `04_false_sharing.c` | Multi-threaded pitfall | Cache coherency |

## Key Concepts

### Memory Hierarchy
```
Registers   ~1 cycle    ~KB      (compiler managed)
L1 Cache    ~4 cycles   32-64KB  (per core)
L2 Cache    ~12 cycles  256KB-1MB (per core)
L3 Cache    ~40 cycles  8-64MB   (shared)
RAM         ~100 cycles GBs      (main memory)
SSD         ~10000+     TBs      (storage)
```

### Cache Line
- 64 bytes on most systems
- Smallest unit of transfer
- Accessing 1 byte loads 64 bytes

### Locality
- **Temporal**: Recently used data used again
- **Spatial**: Nearby data used together

## Connection to ML Systems

1. **FFCV**: Quasi-random access exploits spatial locality
2. **Flash Attention**: Tiling keeps data in SRAM
3. **Tensor layouts**: Contiguous access is fastest
4. **Batch size**: Affects working set size

## Exercises

1. Run `01_cache_basics.c` and explain the results
2. Modify `02_cache_blocking.c` with different block sizes
3. Profile with `perf stat -e cache-misses`
