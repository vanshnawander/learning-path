# CUDA Optimization Techniques

This directory contains practical optimization techniques for GPU code with profiled demonstrations.

## 📚 Modules

### 01_optimization_techniques.py
**Comprehensive optimization guide with profiled examples**

- Batch size optimization
- Mixed precision (AMP) training
- torch.compile for automatic optimization
- Memory-efficient attention (Flash Attention)
- Operation fusion
- Data loading optimization

**Key Profiled Experiments:**
- Batch size vs throughput curves
- FP32 vs FP16 vs BF16 performance
- torch.compile speedup measurement
- Fused vs unfused operation comparison

**Run:** `python 01_optimization_techniques.py`

## 🎯 Learning Objectives

- [ ] Understand impact of batch size on GPU utilization
- [ ] Use mixed precision training effectively
- [ ] Apply torch.compile for automatic optimization
- [ ] Know when to use Flash Attention
- [ ] Optimize data loading pipelines

## ⏱️ Expected Time

- Reading + Running: 2-3 hours
- Deep understanding: 1 day
