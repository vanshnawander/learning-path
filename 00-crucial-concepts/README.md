# Crucial Often-Ignored Concepts

These concepts are frequently overlooked but ESSENTIAL for building efficient ML systems.

## Why This Matters

> "Premature optimization is the root of all evil" - Knuth
> 
> But: **Ignorance of fundamentals is the root of all slow code**

Most ML engineers never learn these. Understanding them separates good from great.

## Directory Structure

```
00-crucial-concepts/
├── 01_memory_bandwidth.md      # The real bottleneck
├── 02_data_alignment.md        # Hidden performance killer
├── 03_false_sharing.md         # Multi-threaded trap
├── 04_branch_prediction.md     # Why if-statements cost
├── 05_floating_point.md        # Numerical precision traps
├── 06_endianness.md            # Cross-platform data
├── 07_memory_ordering.md       # Concurrent programming
├── 08_system_call_cost.md      # OS overhead
├── 09_virtual_memory.md        # Page faults kill performance
├── 10_compiler_optimizations.md # What the compiler does
└── 11_profiling_basics.md      # Measure, don't guess
```

## The Top 10 Most Ignored Concepts

### 1. Memory Bandwidth is the Bottleneck
```
GPU Compute: 300 TFLOPS
GPU Memory:  3 TB/s

To fully utilize compute:
  300 TFLOPS ÷ 3 TB/s = 100 ops per byte loaded

Matrix multiply: ~2N ops per N bytes = 2 ops/byte
  → Memory bound by 50x!

This is why Flash Attention matters (recompute > reload)
```

### 2. Cache Lines Change Everything
```
sizeof(int) = 4 bytes
Cache line = 64 bytes

Accessing arr[0] loads arr[0] through arr[15] automatically!

Sequential: Cache hit rate ~100%
Random: Cache hit rate ~0%
Difference: 10-100x performance!
```

### 3. Data Alignment is Free Performance
```c
// BAD: Unaligned struct
struct Bad {
    char a;     // 1 byte
    int b;      // 4 bytes, but starts at offset 1!
    char c;     // 1 byte
};  // Size: 12 bytes (with padding) or 6 bytes (packed, SLOW)

// GOOD: Aligned struct  
struct Good {
    int b;      // 4 bytes at offset 0
    char a;     // 1 byte
    char c;     // 1 byte
    // 2 bytes padding
};  // Size: 8 bytes, naturally aligned
```

### 4. Floating Point is Approximate
```python
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False

# In ML:
# - Gradient accumulation order matters
# - Batch size affects numerical results
# - Mixed precision needs loss scaling
```

### 5. Virtual Memory Has Real Costs
```
First access to new memory:
1. CPU requests address → TLB miss
2. Page table walk (4 levels!) → ~200 cycles
3. Page not in RAM → Page fault
4. OS loads page from disk → ~10ms!

Solution: Touch memory sequentially, use huge pages
```

### 6. System Calls Are Expensive
```
Function call: ~1 ns
System call: ~100-1000 ns

Reading 1 byte 1000 times: 1000 syscalls = 1ms
Reading 1000 bytes once: 1 syscall = 1µs

Batch your I/O!
```

### 7. Branch Misprediction Costs 15-20 Cycles
```c
// BAD: Unpredictable branch
for (int i = 0; i < n; i++) {
    if (data[i] > threshold) {  // Random: 50% misprediction
        sum += data[i];
    }
}

// BETTER: Branchless
for (int i = 0; i < n; i++) {
    sum += (data[i] > threshold) * data[i];
}
```

### 8. False Sharing Kills Parallelism
```c
// BAD: Different threads write adjacent memory
int counters[NUM_THREADS];  // All in same cache line!
// Thread 0 writes counters[0]
// Thread 1 writes counters[1]
// → Cache line bounces between CPUs!

// GOOD: Pad to separate cache lines
struct alignas(64) PaddedCounter {
    int value;
};
PaddedCounter counters[NUM_THREADS];
```

### 9. Compiler Can't Read Your Mind
```c
// Compiler doesn't know these don't alias
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // What if c == a?
    }
}

// Tell the compiler!
void add(float* restrict a, float* restrict b, 
         float* restrict c, int n) {
    // Now compiler can vectorize!
}
```

### 10. Measure, Don't Guess
```python
# WRONG: "I think this is slow"
# RIGHT: Profile it!

import torch.profiler
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())

# CPU: perf, VTune, gprof
# GPU: nsight, nvprof
# Python: cProfile, line_profiler
```

## Quick Reference: Numbers to Know

| Operation | Time | Notes |
|-----------|------|-------|
| L1 cache hit | 1 ns | ~4 cycles |
| L2 cache hit | 4 ns | ~12 cycles |
| L3 cache hit | 12 ns | ~40 cycles |
| DRAM access | 100 ns | ~300 cycles |
| NVMe read | 10 µs | 10,000 ns |
| HDD read | 10 ms | 10,000,000 ns |
| GPU kernel launch | 5 µs | Minimum overhead |
| cudaMemcpy setup | 10 µs | Before data moves |
| Context switch | 1-10 µs | OS scheduling |
| System call | 100 ns - 1 µs | Kernel entry |

## The Optimization Checklist

Before optimizing, ask:

1. **Did I measure?** Profile first!
2. **Am I memory bound?** Usually yes
3. **Is data layout optimal?** Sequential access?
4. **Am I minimizing transfers?** CPU↔GPU, Disk↔RAM
5. **Am I batching operations?** Amortize overhead
6. **Is this the right algorithm?** O(n) vs O(n²)
7. **Can hardware help?** GPU decode, SIMD, etc.
