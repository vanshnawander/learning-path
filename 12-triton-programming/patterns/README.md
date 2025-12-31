# Triton Kernel Patterns

This directory contains essential Triton kernel patterns with profiled implementations.

## 📚 Modules

### 01_softmax_kernel.py
**Implementing efficient softmax - a memory-bound operation**

- Standard 3-pass softmax
- Fused 1-pass softmax
- Online (streaming) softmax algorithm
- Memory traffic analysis

**Key Profiled Experiments:**
- Correctness verification
- Fused vs unfused performance
- Memory bandwidth analysis
- Online algorithm for long sequences

**Run:** `python 01_softmax_kernel.py`

---

### 02_matmul_kernel.py
**Matrix multiplication - the foundation of deep learning**

- Naive tiled matmul
- Auto-tuned optimized matmul
- Tiling and blocking strategies
- L2 cache optimization

**Key Profiled Experiments:**
- Why tiling reduces memory traffic
- Block size impact on performance
- Comparison with cuBLAS
- Non-square matrix handling

**Run:** `python 02_matmul_kernel.py`

---

## 🎯 Learning Objectives

After completing this module, you will:

- [ ] Implement efficient memory-bound kernels (softmax)
- [ ] Understand tiling for compute-bound kernels (matmul)
- [ ] Use Triton's auto-tuning effectively
- [ ] Know when to use online algorithms

## 🔗 Connection to Deep Learning

| Pattern | Use in Deep Learning |
|---------|---------------------|
| Softmax | Attention scores, classification output |
| Matmul | Linear layers, attention QK^T and PV |
| Online Softmax | Flash Attention, long context models |
| Tiled Matmul | Every matrix operation |

## ⏱️ Expected Time

- Reading + Running: 3-4 hours
- Deep understanding: 1-2 days
