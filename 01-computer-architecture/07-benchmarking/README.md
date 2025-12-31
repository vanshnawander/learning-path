# Benchmarking and Performance Measurement

Measuring performance correctly is a skill - bad benchmarks lead to wrong conclusions.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_benchmark_basics.c` | Common pitfalls and best practices |

## Benchmarking Pitfalls

### 1. No Warmup
First iteration includes cache cold-start, JIT compilation, etc.
Always discard first run.

### 2. Dead Code Elimination
```c
// BAD: Compiler may remove this
for (int i = 0; i < N; i++) {
    result = compute(data[i]);  // Result unused!
}

// GOOD: Force compiler to keep result
volatile int result;
for (int i = 0; i < N; i++) {
    result = compute(data[i]);
}
```

### 3. Insufficient Runs
One measurement has variance. Report mean ± std deviation.

### 4. Wrong Metric
- Throughput: Operations per second
- Latency: Time per operation
- Don't mix them up!

## Tools

### CPU Profiling
```bash
perf stat ./program          # Basic counters
perf record ./program        # Sampling
perf stat -e cache-misses    # Specific events
```

### GPU Profiling
```bash
nsys profile python train.py     # Timeline
ncu --set full python train.py   # Kernel details
```

## Roofline Model

```
Performance = min(Peak FLOPS, Peak Bandwidth × Arithmetic Intensity)

Arithmetic Intensity = FLOPS / Bytes Accessed
```

If below roofline:
- Left of peak: Memory bound → optimize memory access
- Right of peak: Compute bound → optimize computation
