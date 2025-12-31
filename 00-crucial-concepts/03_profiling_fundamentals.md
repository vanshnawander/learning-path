# Profiling Fundamentals: Measure Everything

**Rule #1: Never optimize without measuring first.**

## The Profiling Mindset

```
Amateur: "I think this is slow, let me optimize it"
Professional: "Let me measure where time is spent, then optimize the bottleneck"

Without profiling:
  - You optimize the wrong thing
  - You make code complex for no gain
  - You waste days on 1% improvements

With profiling:
  - You find the real bottleneck (often surprising!)
  - You get 10-100x improvements
  - You know when to stop optimizing
```

## Time Scales Reference

```
╔═══════════════════════════════════════════════════════════════════╗
║                    TIME SCALE REFERENCE                            ║
╠═══════════════════════════════════════════════════════════════════╣
║  1 nanosecond (ns)     = 0.000000001 seconds                      ║
║  1 microsecond (µs)    = 0.000001 seconds = 1,000 ns              ║
║  1 millisecond (ms)    = 0.001 seconds = 1,000 µs                 ║
║  1 second (s)          = 1,000 ms                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  OPERATION                              TIME          SCALE        ║
╠═══════════════════════════════════════════════════════════════════╣
║  CPU cycle (3 GHz)                      0.3 ns        ████         ║
║  L1 cache access                        1 ns          ████         ║
║  L2 cache access                        4 ns          █████        ║
║  L3 cache access                        12 ns         ██████       ║
║  DRAM access                            100 ns        ████████     ║
║  NVMe SSD read (4KB)                    10 µs         ██████████   ║
║  GPU kernel launch                      5 µs          ██████████   ║
║  cudaMemcpy setup                       10 µs         ██████████   ║
║  PCIe round-trip                        1-5 µs        █████████    ║
║  Context switch                         1-10 µs       █████████    ║
║  HDD seek                               10 ms         ████████████ ║
║  Network RTT (same DC)                  500 µs        ███████████  ║
║  Network RTT (cross-region)             100 ms        █████████████║
╚═══════════════════════════════════════════════════════════════════╝
```

## Profiling Tools by Language

### Python
```python
# 1. cProfile - Function-level profiling
import cProfile
cProfile.run('train_epoch()', 'profile_output')

# View results
import pstats
p = pstats.Stats('profile_output')
p.sort_stats('cumulative').print_stats(20)

# 2. line_profiler - Line-by-line timing
# Add @profile decorator, run with kernprof
@profile
def process_batch(batch):
    x = preprocess(batch)      # See time for this line
    y = model(x)               # And this line
    loss = criterion(y)        # And this line
    return loss

# 3. py-spy - Sampling profiler (no code changes!)
# pip install py-spy
# py-spy record -o profile.svg -- python train.py

# 4. Scalene - CPU + GPU + memory profiler
# pip install scalene
# scalene train.py
```

### PyTorch Specific
```python
# torch.profiler - The gold standard for PyTorch
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### C/C++
```c
// 1. Simple timing with clock_gettime
#include <time.h>

struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);

// Your code here

clock_gettime(CLOCK_MONOTONIC, &end);
double elapsed = (end.tv_sec - start.tv_sec) + 
                 (end.tv_nsec - start.tv_nsec) * 1e-9;
printf("Elapsed: %.6f seconds\n", elapsed);

// 2. RDTSC for cycle-accurate timing
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

uint64_t start = rdtsc();
// Your code
uint64_t end = rdtsc();
printf("Cycles: %lu\n", end - start);
```

### Linux perf
```bash
# Record CPU performance counters
perf stat ./my_program

# Sample call stacks
perf record -g ./my_program

# Generate flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

## Multimodal Training: Where Time Goes

```
╔═══════════════════════════════════════════════════════════════════╗
║          MULTIMODAL TRAINING PIPELINE BREAKDOWN                    ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  ┌─────────────┐   Typical: 20-60% of iteration time              ║
║  │ Data Load   │   - Disk I/O: 2-10 ms per batch                  ║
║  │ & Decode    │   - Image decode: 1-5 ms per image               ║
║  └──────┬──────┘   - Video decode: 5-50 ms per clip               ║
║         │          - Audio decode: 0.5-2 ms per sample            ║
║         ▼                                                          ║
║  ┌─────────────┐   Typical: 5-15% of iteration time               ║
║  │ Preprocess  │   - Resize/crop: 0.5-2 ms                        ║
║  │ & Augment   │   - Normalize: 0.1-0.5 ms                        ║
║  └──────┬──────┘   - Tokenize: 0.1-1 ms                           ║
║         │                                                          ║
║         ▼                                                          ║
║  ┌─────────────┐   Typical: 2-5% of iteration time                ║
║  │ CPU → GPU   │   - PCIe transfer: 1-10 ms per batch             ║
║  │  Transfer   │   - Depends on batch size & precision           ║
║  └──────┬──────┘                                                   ║
║         │                                                          ║
║         ▼                                                          ║
║  ┌─────────────┐   Typical: 30-70% of iteration time              ║
║  │   Forward   │   - Attention: dominant for transformers         ║
║  │    Pass     │   - Convolutions: dominant for vision            ║
║  └──────┬──────┘   - Encoder/decoder: multimodal fusion          ║
║         │                                                          ║
║         ▼                                                          ║
║  ┌─────────────┐   Typical: 30-70% of iteration time              ║
║  │  Backward   │   - Usually 1.5-2x forward time                  ║
║  │    Pass     │   - Memory bound for large models                ║
║  └──────┬──────┘                                                   ║
║         │                                                          ║
║         ▼                                                          ║
║  ┌─────────────┐   Typical: 1-5% of iteration time                ║
║  │  Optimizer  │   - Adam: more ops than SGD                      ║
║  │    Step     │   - Fused optimizers faster                      ║
║  └─────────────┘                                                   ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Identifying Bottlenecks

### GPU Utilization Check
```python
# If GPU util < 80%, you have a bottleneck elsewhere!
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                        '--format=csv,noheader'], capture_output=True)
print(f"GPU Utilization: {result.stdout.decode().strip()}")

# In training loop:
# - GPU util low, CPU high → Data loading bottleneck
# - GPU util low, CPU low → I/O bottleneck  
# - GPU util high → GPU is the bottleneck (good!)
```

### Memory Bandwidth Check
```python
# Check if memory bound
with torch.profiler.profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    output = model(input)

# Look at memory throughput vs compute
# High memory throughput + low SM util = memory bound
```

## Profiling Template for Every File

```python
"""
PROFILING TEMPLATE - Use this in every performance-critical code

Shows:
1. Wall clock time
2. CPU time
3. Memory usage
4. GPU metrics (if applicable)
"""

import time
import tracemalloc
from contextlib import contextmanager

@contextmanager
def profile_block(name):
    """Profile a code block with timing and memory."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    yield
    
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n{'='*60}")
    print(f"PROFILE: {name}")
    print(f"{'='*60}")
    print(f"  Wall time:    {elapsed*1000:.3f} ms")
    print(f"  Memory used:  {current/1024/1024:.2f} MB")
    print(f"  Peak memory:  {peak/1024/1024:.2f} MB")
    print(f"{'='*60}\n")

# Usage:
with profile_block("Matrix Multiplication"):
    result = torch.matmul(a, b)
```
