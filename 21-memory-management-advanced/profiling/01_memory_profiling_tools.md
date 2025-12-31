# Memory Profiling Tools

## Linux perf for Memory Analysis

### Cache Statistics

```bash
# Basic cache miss analysis
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    ./myprogram

# Detailed cache hierarchy
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
L1-dcache-stores,L1-dcache-store-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses \
    ./myprogram

# TLB statistics
perf stat -e dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses \
    ./myprogram
```

### Memory Bandwidth

```bash
# Memory operations
perf stat -e mem-loads,mem-stores ./myprogram

# NUMA statistics
perf stat -e node-loads,node-load-misses,node-stores,node-store-misses \
    ./myprogram
```

### perf mem for Detailed Analysis

```bash
# Record memory accesses
perf mem record ./myprogram

# Analyze by data address
perf mem report --sort=mem

# Show data symbols causing misses
perf mem report --sort=symbol_daddr
```

## Valgrind Cachegrind

Simulates cache behavior (slower but detailed):

```bash
# Run with cachegrind
valgrind --tool=cachegrind ./myprogram

# Output file: cachegrind.out.<pid>
# Analyze with cg_annotate
cg_annotate cachegrind.out.12345

# Show source-level cache misses
cg_annotate --auto=yes cachegrind.out.12345

# Example output:
# I   refs:      1,234,567,890
# I1  misses:        1,234,567  (0.1%)
# LLi misses:          123,456  (0.01%)
# D   refs:        987,654,321
# D1  misses:       12,345,678  (1.2%)
# LLd misses:        1,234,567  (0.1%)
```

### Cachegrind Configuration

```bash
# Custom cache parameters
valgrind --tool=cachegrind \
    --I1=32768,8,64 \    # L1 instruction: 32KB, 8-way, 64B line
    --D1=32768,8,64 \    # L1 data: 32KB, 8-way, 64B line
    --LL=8388608,16,64 \ # LLC: 8MB, 16-way, 64B line
    ./myprogram
```

## Valgrind Massif (Heap Profiler)

```bash
# Profile heap usage over time
valgrind --tool=massif ./myprogram

# With stack profiling
valgrind --tool=massif --stacks=yes ./myprogram

# Visualize with ms_print
ms_print massif.out.12345

# Example output (ASCII graph):
#     MB
# 12.00^                                               #
#      |                                              @#
#      |                                           @@@#
#      |                                        @@@@@@#
#      |                                     @@@@@@@@@#
#      |                                  @@@@@@@@@@@@#
# ...
```

## Intel VTune Memory Analysis

```bash
# Memory access analysis
vtune -collect memory-access ./myprogram

# Memory consumption analysis
vtune -collect memory-consumption ./myprogram

# GUI analysis
vtune-gui r000ma/r000ma.vtune
```

Key VTune metrics:
- **Memory Bound**: % of cycles stalled on memory
- **L1/L2/L3 Bound**: Cache level causing stalls
- **DRAM Bound**: Waiting for main memory
- **Store Bound**: Store buffer full

## AddressSanitizer (ASan)

Finds memory errors at runtime:

```bash
# Compile with ASan
gcc -fsanitize=address -g myprogram.c -o myprogram

# Run
./myprogram

# Detects:
# - Heap buffer overflow
# - Stack buffer overflow
# - Use after free
# - Double free
# - Memory leaks (with -fsanitize=leak)
```

## heaptrack

Modern heap profiler (better than Massif):

```bash
# Profile
heaptrack ./myprogram

# Analyze
heaptrack_gui heaptrack.myprogram.12345.gz

# Or text analysis
heaptrack_print heaptrack.myprogram.12345.gz
```

Shows:
- Peak heap usage
- Allocation hotspots
- Temporary allocations
- Memory leaks

## Custom Memory Tracking

```c
// Simple allocation tracker
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void*) = NULL;
static size_t total_allocated = 0;
static size_t allocation_count = 0;

void* malloc(size_t size) {
    if (!real_malloc) real_malloc = dlsym(RTLD_NEXT, "malloc");
    
    void* ptr = real_malloc(size);
    if (ptr) {
        __atomic_add_fetch(&total_allocated, size, __ATOMIC_RELAXED);
        __atomic_add_fetch(&allocation_count, 1, __ATOMIC_RELAXED);
    }
    return ptr;
}

// Compile: gcc -shared -fPIC -o libmemtrack.so memtrack.c -ldl
// Use: LD_PRELOAD=./libmemtrack.so ./myprogram
```

## Quick Reference

| Tool | Use Case | Overhead |
|------|----------|----------|
| perf stat | Quick cache stats | ~5% |
| perf mem | Detailed memory sampling | ~10% |
| cachegrind | Line-level cache analysis | 20-50x |
| massif | Heap usage over time | 10-20x |
| heaptrack | Allocation profiling | 5-10x |
| ASan | Memory error detection | 2-3x |
| VTune | Comprehensive analysis | 5-20% |
