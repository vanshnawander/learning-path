# NVIDIA Nsight Compute

Deep kernel-level GPU profiling.

## What It Does

- Detailed GPU kernel analysis
- Memory throughput metrics
- Compute utilization
- Roofline analysis
- Warp-level statistics

## Basic Usage

```bash
# Profile a Python script
ncu --set full python train.py

# Profile specific kernels
ncu --kernel-name "flash_attn" python train.py

# Save to file
ncu -o profile.ncu-rep python train.py
```

## Key Metrics

### Memory
- **Memory Throughput**: % of peak
- **L1/L2 Hit Rate**: Cache efficiency
- **Global Load/Store Efficiency**: Coalescing

### Compute
- **SM Occupancy**: Active warps / max warps
- **Achieved Occupancy**: Actual vs theoretical
- **Compute Throughput**: % of peak FLOPS

### Warp
- **Warp Execution Efficiency**: No divergence = 100%
- **Instructions per Cycle (IPC)**
- **Stall Reasons**: Memory, sync, etc.

## Reading the Roofline

```
    Compute     |
    Bound       |    /
              * | * /
                | /
                |/
    Memory      |
    Bound       |
    ────────────┼────────────
                Arithmetic Intensity
                (FLOPS/Byte)
```

If your kernel is below the roofline:
- Left of ridge: Memory bound → Optimize memory access
- Right of ridge: Compute bound → Optimize compute

## Integration with PyTorch

```python
import torch.cuda.profiler as profiler

profiler.start()
# Your code
profiler.stop()
```

## Tips
1. Profile individual kernels, not full training
2. Focus on top time-consuming kernels
3. Compare theoretical vs achieved performance
4. Use roofline to identify bottleneck type
