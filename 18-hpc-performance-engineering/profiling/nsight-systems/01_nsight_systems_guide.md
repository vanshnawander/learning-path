# NVIDIA Nsight Systems: Timeline Profiling

## What is Nsight Systems?

Nsight Systems is a **system-wide performance analysis tool** that provides:
- Timeline visualization of CPU/GPU activity
- CUDA API tracing
- OS runtime analysis
- Multi-process/multi-GPU profiling

**Use Nsight Systems for**: Finding *what* is slow (overview)
**Use Nsight Compute for**: Understanding *why* a kernel is slow (deep dive)

## Basic Usage

### Command Line

```bash
# Profile Python script
nsys profile -o report python train.py

# Profile with CUDA API trace
nsys profile --trace=cuda,nvtx -o report python train.py

# Profile specific duration
nsys profile --duration=30 -o report python train.py

# Sample CPU backtrace
nsys profile --sample=cpu --backtrace=lbr python train.py

# Profile MPI application
nsys profile --trace=cuda,mpi mpirun -n 4 ./my_mpi_app
```

### Common Options

| Option | Description |
|--------|-------------|
| `-o <name>` | Output file name |
| `--trace=cuda,nvtx,osrt` | What to trace |
| `--sample=cpu` | CPU sampling |
| `--cudabacktrace=true` | CUDA API call stacks |
| `--duration=<sec>` | Profile duration |
| `--delay=<sec>` | Delay before profiling |
| `--capture-range=cudaProfilerApi` | Manual range control |

## Understanding the Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Process: python (PID 12345)                                     │
├─────────────────────────────────────────────────────────────────┤
│ CPU Thread 0 ─────────────────────────────────────────────────  │
│ ██████████░░░░██████████░░░░██████████░░░░                      │
│ Python GC    Launch kernels    Wait                            │
├─────────────────────────────────────────────────────────────────┤
│ CUDA API ────────────────────────────────────────────────────── │
│ ┌──────┐ ┌────┐ ┌──────┐ ┌────┐ ┌──────┐                      │
│ │cudaMem│ │kern│ │cudaMem│ │kern│ │cudaMem│                      │
│ └──────┘ └────┘ └──────┘ └────┘ └──────┘                      │
├─────────────────────────────────────────────────────────────────┤
│ GPU 0 Stream 0 ─────────────────────────────────────────────── │
│      ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│      │ Kernel  │    │ Kernel  │    │ Kernel  │                │
│      └─────────┘    └─────────┘    └─────────┘                │
├─────────────────────────────────────────────────────────────────┤
│ Memory (HtoD) ─────────────────────────────────────────────── │
│ ████████            ████████            ████████              │
│ Memory (DtoH) ─────────────────────────────────────────────── │
│           ████████            ████████            ████████    │
└─────────────────────────────────────────────────────────────────┘
      0ms      10ms      20ms      30ms      40ms      50ms
```

## NVTX Annotations

Add custom markers to your code:

### Python

```python
import torch
import nvtx

# Range annotation
with nvtx.annotate("forward_pass", color="blue"):
    output = model(input)

# Or as decorator
@nvtx.annotate("my_function", color="green")
def my_function():
    pass

# Manual range
rng = nvtx.start_range("manual_range", color="red")
# ... code ...
nvtx.end_range(rng)

# PyTorch profiler integration
with torch.autograd.profiler.emit_nvtx():
    output = model(input)
```

### C++/CUDA

```cpp
#include <nvtx3/nvToolsExt.h>

void my_function() {
    nvtxRangePush("my_function");
    
    // Nested range
    nvtxRangePushA("inner_section");
    // ... code ...
    nvtxRangePop();
    
    nvtxRangePop();
}

// With color
nvtxEventAttributes_t attr = {0};
attr.version = NVTX_VERSION;
attr.colorType = NVTX_COLOR_ARGB;
attr.color = 0xFF00FF00;  // Green
attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
attr.message.ascii = "colored_range";
nvtxRangePushEx(&attr);
```

## Common Patterns to Look For

### 1. GPU Idle Time

```
Bad: CPU launches, then waits
CPU: ██████████████████░░░░░░░░░░░░░░░░██████████████████
GPU:       ██████████████████████████

Good: Overlap CPU work with GPU execution
CPU: ██████████████████░░████████████░░██████████████████
GPU:       ██████████████████████████████████████████████
```

**Diagnosis**: Large gaps between GPU kernels
**Fix**: Overlap computation, use CUDA graphs, async operations

### 2. Memory Transfer Bottleneck

```
Bad: Synchronous transfers
Timeline: [HtoD]────[Kernel]────[DtoH]────[HtoD]────[Kernel]...

Good: Overlap transfers with compute
Timeline: [HtoD]────[Kernel]────[DtoH]
               [HtoD]────[Kernel]────[DtoH]
                    [HtoD]────[Kernel]...
```

**Diagnosis**: Memory copies blocking compute
**Fix**: Use pinned memory, streams, async copies

### 3. Small Kernel Launches

```
Bad: Many tiny kernels
GPU: ┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐

Good: Fused operations
GPU: ┌────────────────────────────────────┐
```

**Diagnosis**: Kernel launch overhead dominates
**Fix**: Kernel fusion, CUDA graphs, torch.compile()

### 4. CPU Bottleneck

```
Bad: CPU processing between GPU ops
CPU: ████████████████████████████████████████████████████
GPU:         ████        ████        ████        ████

Good: CPU just launches kernels
CPU: ██░░░░░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░░░░░██
GPU:   ██████████████████████████████████████████████
```

**Diagnosis**: CPU work between kernel launches
**Fix**: Move preprocessing to GPU, use DataLoader workers

## Analysis Workflow

### Step 1: High-Level Overview

```bash
# Quick profile with minimal overhead
nsys profile --trace=cuda -o overview python train.py
nsys stats overview.nsys-rep
```

Look at:
- Overall GPU utilization
- Time spent in kernels vs. memory ops
- Gaps in GPU activity

### Step 2: Identify Bottlenecks

```bash
# Add NVTX and CPU sampling
nsys profile --trace=cuda,nvtx --sample=cpu -o detailed python train.py
```

Look at:
- Which phase (data loading, forward, backward) is slowest
- CPU call stacks during GPU idle

### Step 3: Deep Dive on Slow Kernels

```bash
# Use Nsight Compute for specific kernels
ncu --kernel-name "volta_sgemm" python train.py
```

## Example Analysis: Training Loop

```python
import torch
import nvtx

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(loader):
        with nvtx.annotate("data_to_gpu"):
            data, target = data.cuda(), target.cuda()
        
        with nvtx.annotate("forward"):
            output = model(data)
            loss = criterion(output, target)
        
        with nvtx.annotate("backward"):
            optimizer.zero_grad()
            loss.backward()
        
        with nvtx.annotate("optimizer_step"):
            optimizer.step()
```

### Expected Timeline (Healthy)

```
┌────────────────────────────────────────────────────────────┐
│ data_to_gpu │ forward │ backward │ step │ data_to_gpu │...│
│ ████████████│█████████│██████████│██████│████████████│   │
└────────────────────────────────────────────────────────────┘
```

### Problem: DataLoader Bottleneck

```
┌────────────────────────────────────────────────────────────┐
│ data_to_gpu (waiting)        │ forward │ backward │ step │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░█│█████████│██████████│██████│
└────────────────────────────────────────────────────────────┘
GPU mostly idle!
```

**Fix**: Increase DataLoader workers, use FFCV, prefetch

## Integration with PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    with_stack=True
) as prof:
    for step, data in enumerate(loader):
        if step >= 5:
            break
        train_step(data)
        prof.step()
```

## References

- NVIDIA Nsight Systems Documentation
- CUDA Profiling Tools Interface (CUPTI)
- PyTorch Profiler documentation
