# Scientific Benchmarking Methodology

## Why Proper Benchmarking Matters

Bad benchmarking leads to:
- Wrong optimization decisions
- Wasted engineering time
- Misleading comparisons
- Non-reproducible results

## The Benchmarking Process

### 1. Define What You're Measuring

```
Questions to answer:
- Throughput? (ops/sec, samples/sec)
- Latency? (mean, p50, p99, p999)
- Memory usage? (peak, average)
- Power efficiency? (FLOPS/Watt)

Be specific:
❌ "How fast is my model?"
✅ "What is the p99 inference latency for batch size 1 on A100?"
```

### 2. Control Variables

```
Fixed variables (document these):
- Hardware (exact GPU model, CPU, memory)
- Software versions (CUDA, PyTorch, drivers)
- Input data (size, distribution, dtype)
- Batch size, sequence length, etc.
- Power/thermal state
- System load

Variable being tested:
- Only change ONE thing at a time
```

### 3. Warmup

Why warmup matters:
- JIT compilation (PyTorch, JAX)
- CUDA context initialization
- GPU boost clock stabilization
- CPU cache warming
- Memory allocator settling

```python
# Warmup iterations
for _ in range(10):
    model(input)
torch.cuda.synchronize()

# Now benchmark
start = time.perf_counter()
for _ in range(iterations):
    model(input)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

### 4. Synchronization

```python
# WRONG: Doesn't wait for GPU
start = time.time()
result = model(input)
elapsed = time.time() - start  # Only measures kernel launch!

# CORRECT: Wait for GPU completion
torch.cuda.synchronize()
start = time.time()
result = model(input)
torch.cuda.synchronize()
elapsed = time.time() - start
```

### 5. Statistical Rigor

#### Multiple Runs

```python
import numpy as np

timings = []
for _ in range(num_runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model(input)
    torch.cuda.synchronize()
    timings.append(time.perf_counter() - start)

# Report statistics
print(f"Mean: {np.mean(timings)*1000:.2f} ms")
print(f"Std:  {np.std(timings)*1000:.2f} ms")
print(f"Min:  {np.min(timings)*1000:.2f} ms")
print(f"Max:  {np.max(timings)*1000:.2f} ms")
print(f"Median: {np.median(timings)*1000:.2f} ms")
```

#### Confidence Intervals

```python
from scipy import stats

# 95% confidence interval
ci = stats.t.interval(0.95, len(timings)-1, 
                      loc=np.mean(timings), 
                      scale=stats.sem(timings))
print(f"95% CI: [{ci[0]*1000:.2f}, {ci[1]*1000:.2f}] ms")
```

#### Coefficient of Variation

```python
cv = np.std(timings) / np.mean(timings) * 100
print(f"CV: {cv:.1f}%")

# Rule of thumb:
# CV < 5%: Good stability
# CV 5-15%: Acceptable
# CV > 15%: Investigate variability
```

### 6. Percentile Metrics

For latency-sensitive applications:

```python
percentiles = [50, 90, 95, 99, 99.9]
for p in percentiles:
    val = np.percentile(timings, p) * 1000
    print(f"p{p}: {val:.2f} ms")
```

## Common Pitfalls

### 1. Not Disabling Gradients

```python
# Training mode (includes gradient computation)
output = model(input)  # Slower

# Inference mode
with torch.no_grad():
    output = model(input)  # Faster

# Even better for inference
model.eval()
with torch.inference_mode():
    output = model(input)
```

### 2. Including Data Transfer

```python
# WRONG: Includes CPU→GPU transfer
data = torch.randn(batch_size, ...).cuda()  # Transfer here!
torch.cuda.synchronize()
start = time.perf_counter()
output = model(data)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# CORRECT: Data already on GPU
data = torch.randn(batch_size, ..., device='cuda')
torch.cuda.synchronize()
# Warmup to ensure data is resident
_ = model(data)
torch.cuda.synchronize()
start = time.perf_counter()
output = model(data)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

### 3. GPU Throttling

```bash
# Check GPU state before benchmarking
nvidia-smi -q -d PERFORMANCE

# Lock clocks for reproducibility (requires root)
sudo nvidia-smi -lgc 1400,1400  # Lock to 1400 MHz
# Remember to reset after
sudo nvidia-smi -rgc
```

### 4. System Interference

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Check for background processes
htop

# Pin process to specific cores
taskset -c 0-7 python benchmark.py
```

### 5. Optimistic Results

```python
# Don't just report the best run
# Report: mean ± std, or median with percentiles

# Avoid:
print(f"Latency: {min(timings)*1000:.2f} ms")  # Cherry-picking

# Prefer:
print(f"Latency: {np.mean(timings)*1000:.2f} ± {np.std(timings)*1000:.2f} ms")
```

## Benchmark Template

```python
import torch
import time
import numpy as np
import json
from datetime import datetime

def benchmark_model(model, input_shape, device='cuda', 
                   warmup=10, iterations=100, runs=5):
    """
    Comprehensive model benchmarking.
    
    Returns dict with timing statistics.
    """
    model = model.to(device).eval()
    input_data = torch.randn(*input_shape, device=device)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
        'input_shape': input_shape,
        'warmup_iterations': warmup,
        'timed_iterations': iterations,
        'runs': runs,
    }
    
    all_timings = []
    
    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            _ = model(input_data)
        torch.cuda.synchronize()
        
        # Benchmark runs
        for run in range(runs):
            timings = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_data)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            all_timings.extend(timings)
    
    # Statistics
    timings_ms = np.array(all_timings) * 1000
    results['statistics'] = {
        'mean_ms': float(np.mean(timings_ms)),
        'std_ms': float(np.std(timings_ms)),
        'min_ms': float(np.min(timings_ms)),
        'max_ms': float(np.max(timings_ms)),
        'median_ms': float(np.median(timings_ms)),
        'p90_ms': float(np.percentile(timings_ms, 90)),
        'p95_ms': float(np.percentile(timings_ms, 95)),
        'p99_ms': float(np.percentile(timings_ms, 99)),
        'cv_percent': float(np.std(timings_ms) / np.mean(timings_ms) * 100),
    }
    
    # Throughput
    batch_size = input_shape[0]
    results['throughput'] = {
        'samples_per_second': batch_size / (results['statistics']['mean_ms'] / 1000),
    }
    
    return results


if __name__ == '__main__':
    import torchvision.models as models
    
    model = models.resnet50(pretrained=False)
    results = benchmark_model(model, (32, 3, 224, 224))
    
    print(json.dumps(results, indent=2))
```

## Reporting Results

### What to Include

1. **Hardware**: GPU model, CPU, RAM, drivers
2. **Software**: Framework versions, CUDA version
3. **Configuration**: Batch size, precision, compilation
4. **Methodology**: Warmup, iterations, runs
5. **Statistics**: Mean, std, percentiles
6. **Reproducibility**: Code/scripts to reproduce

### Example Table

```markdown
| Configuration | Mean (ms) | Std (ms) | p99 (ms) | Throughput |
|--------------|-----------|----------|----------|------------|
| FP32, BS=1   | 5.2 ± 0.3 | 0.3      | 6.1      | 192 img/s  |
| FP16, BS=1   | 3.1 ± 0.2 | 0.2      | 3.5      | 323 img/s  |
| FP32, BS=32  | 45.1 ± 1.2| 1.2      | 48.0     | 709 img/s  |
| FP16, BS=32  | 24.3 ± 0.8| 0.8      | 26.1     | 1317 img/s |

Hardware: NVIDIA A100 80GB
Software: PyTorch 2.1, CUDA 12.1
Warmup: 10 iterations
Measurement: 100 iterations × 5 runs
```

## References

- "A Systematic Approach to HPC Benchmarking" - SC proceedings
- PyTorch Benchmark Utils documentation
- NVIDIA Deep Learning Performance Guide
