# FFCV vs WebDataset: Data Loading Performance Comparison

## The Data Loading Problem

```
╔══════════════════════════════════════════════════════════════════════╗
║  TYPICAL TRAINING BOTTLENECK                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  GPU utilization during training:                                     ║
║                                                                       ║
║  Naive DataLoader:                                                    ║
║  GPU: ████░░░░████░░░░████░░░░  ~40% utilization                     ║
║       ▲   ▲   ▲   ▲   ▲   ▲                                          ║
║       │   │   │   │   │   └── Waiting for data                       ║
║       │   └───┴───┴───────── GPU compute                             ║
║                                                                       ║
║  Optimized (FFCV/DALI):                                               ║
║  GPU: ██████████████████████████  ~95% utilization                   ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Performance Comparison (ImageNet, batch=256)

| Method | Images/sec | GPU Util | CPU Load | Memory |
|--------|------------|----------|----------|--------|
| PyTorch DataLoader | 800-1200 | 40-60% | High | Medium |
| + num_workers=8 | 2000-3000 | 60-80% | Very High | High |
| + pin_memory | 2500-3500 | 70-85% | Very High | High |
| FFCV | 5000-8000 | 90-98% | Low | Low |
| NVIDIA DALI | 6000-10000 | 95-99% | Very Low | Medium |
| WebDataset | 3000-5000 | 80-90% | Medium | Low |

## FFCV: How It Achieves Speed

```
┌─────────────────────────────────────────────────────────────────┐
│                    FFCV ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PREPARATION PHASE (once):                                       │
│  ┌──────────────┐                                               │
│  │ Original     │                                               │
│  │ Dataset      │                                               │
│  │ (JPEG files) │                                               │
│  └──────┬───────┘                                               │
│         │ ffcv.writer.DatasetWriter                             │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    .beton FILE                            │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┬──────────┐   │   │
│  │  │ Header  │ Sample0 │ Sample1 │ Sample2 │ ...      │   │   │
│  │  │ (index) │ (raw)   │ (raw)   │ (raw)   │          │   │   │
│  │  └─────────┴─────────┴─────────┴─────────┴──────────┘   │   │
│  │                                                           │   │
│  │  • Pre-resized images (no resize at load time!)          │   │
│  │  • Page-aligned for OS optimization                       │   │
│  │  • Contiguous storage (sequential read)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  TRAINING PHASE:                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Memory-mapped access (mmap)                               │   │
│  │  • No data copy - direct access to file pages             │   │
│  │  • OS handles caching automatically                       │   │
│  │  • Quasi-random sampling for locality                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### FFCV Timing Breakdown

| Operation | Standard | FFCV | Speedup |
|-----------|----------|------|---------|
| File open | 50 µs | 0 (mmap) | ∞ |
| JPEG decode | 5 ms | 0 (pre-decoded) | ∞ |
| Resize | 2 ms | 0 (pre-resized) | ∞ |
| Normalize | 0.5 ms | 0.5 ms | 1x |
| To tensor | 0.3 ms | 0.1 ms | 3x |
| **Total** | **~8 ms** | **~0.6 ms** | **13x** |

## WebDataset: Streaming for Scale

```
┌─────────────────────────────────────────────────────────────────┐
│                   WEBDATASET ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Storage: TAR archives (sequential access)                       │
│                                                                  │
│  dataset-000000.tar                                             │
│  ├── sample0.jpg                                                │
│  ├── sample0.json                                               │
│  ├── sample1.jpg                                                │
│  ├── sample1.json                                               │
│  └── ...                                                         │
│                                                                  │
│  Benefits:                                                       │
│  • Works with cloud storage (S3, GCS) efficiently               │
│  • Sequential reads = maximum disk throughput                    │
│  • Sharding for distributed training                             │
│  • No random access (but shuffling via shard shuffling)          │
│                                                                  │
│  Best for:                                                       │
│  • Very large datasets (petabyte scale)                          │
│  • Cloud/distributed training                                    │
│  • Streaming from remote storage                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## When to Use What

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| ImageNet-scale, local SSD | FFCV | Fastest, low overhead |
| Petabyte-scale, cloud | WebDataset | Streaming, S3-friendly |
| Video training | NVIDIA DALI | GPU decode |
| Quick prototyping | PyTorch DataLoader | Simple, flexible |
| Multi-modal (video+audio+text) | WebDataset | Flexible format |

## Profiling Your DataLoader

```python
import time
import torch
from torch.utils.data import DataLoader

def profile_dataloader(loader, num_batches=100):
    """Profile DataLoader performance."""
    
    times = []
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    
    # Profile
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        batch_start = time.perf_counter()
        
        # Simulate GPU transfer
        if isinstance(batch, (list, tuple)):
            for t in batch:
                if isinstance(t, torch.Tensor):
                    t.cuda(non_blocking=True)
        
        torch.cuda.synchronize()
        times.append(time.perf_counter() - batch_start)
        
        if i >= num_batches:
            break
    
    total = time.perf_counter() - start
    avg_batch = sum(times) / len(times) * 1000
    
    print(f"Total time: {total:.2f}s")
    print(f"Avg batch: {avg_batch:.2f}ms")
    print(f"Throughput: {num_batches * loader.batch_size / total:.0f} samples/sec")
    
    return times

# Usage:
# profile_dataloader(your_dataloader)
```

## Quick Optimization Checklist

```
□ num_workers = CPU cores (usually 8-16)
□ pin_memory = True (for GPU training)
□ prefetch_factor = 2-4
□ persistent_workers = True (PyTorch 1.7+)
□ Pre-resize images to training size
□ Use fast JPEG decoder (TurboJPEG)
□ Consider FFCV for maximum speed
□ Profile before and after each change!
```
