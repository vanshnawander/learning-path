# Memory Hierarchy

The foundation of all performance optimization.

## Hierarchy Levels

```
Registers     ~1 cycle      ~KB
L1 Cache      ~4 cycles     32-64 KB
L2 Cache      ~12 cycles    256 KB - 1 MB
L3 Cache      ~40 cycles    8-64 MB
DRAM          ~100 cycles   GBs
SSD           ~10000 cycles TBs
```

## Cache Concepts

### Locality
- **Temporal**: Recently accessed data accessed again
- **Spatial**: Nearby data accessed together

### Cache Lines
- Typical size: 64 bytes
- Alignment matters
- Prefetching patterns

### Associativity
- Direct-mapped: 1 location
- N-way: N possible locations
- Fully associative: Any location

## Exercises
1. Measure cache line size
2. Demonstrate cache effects with matrix access
3. Profile L1/L2/L3 miss rates
