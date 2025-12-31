# CUDA Memory Management

This directory covers GPU memory management - often the #1 source of performance issues.

## 📚 Modules

### 01_memory_management.py
**Deep dive into GPU memory hierarchy and management**

- Global Memory (HBM) characteristics
- Memory allocation patterns and PyTorch caching
- CPU ↔ GPU transfer optimization
- Memory-bound vs compute-bound analysis
- Memory fragmentation issues
- Async transfers and overlap

**Key Profiled Experiments:**
- HBM bandwidth measurement
- Allocation overhead comparison
- Pinned vs pageable memory transfer
- Arithmetic intensity analysis
- Fragmentation demonstration
- Stream-based overlap

**Run:** `python 01_memory_management.py`

## 🎯 Learning Objectives

- [ ] Understand GPU memory hierarchy
- [ ] Optimize memory allocation patterns
- [ ] Minimize CPU-GPU transfer overhead
- [ ] Identify memory-bound operations
- [ ] Avoid memory fragmentation

## 🔗 Connection to Training

| Memory Aspect | Training Impact |
|--------------|-----------------|
| HBM Bandwidth | Limits attention, layer norm speed |
| Allocation | Affects dynamic batch sizes |
| Transfers | Data loading bottleneck |
| Fragmentation | OOM with "enough" memory |

## ⏱️ Expected Time

- Reading + Running: 2-3 hours
- Deep understanding: 1 day
