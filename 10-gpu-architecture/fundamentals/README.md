# GPU Architecture Fundamentals

This directory contains comprehensive learning materials for understanding GPU architecture from first principles. Every module includes **profiled experiments** to build intuition about GPU performance.

## 📚 Modules

### 01_gpu_vs_cpu.py
**Understanding the fundamental differences between GPU and CPU architecture**

- Latency vs throughput orientation
- Memory bandwidth comparison
- SIMT parallelism demonstration
- Branch divergence effects
- Kernel launch overhead
- Data transfer costs

**Key Profiled Experiments:**
- Size-dependent crossover between CPU and GPU
- Memory bandwidth measurement
- Matrix multiply scaling
- Divergence penalty measurement

**Run:** `python 01_gpu_vs_cpu.py`

---

### 02_memory_hierarchy.py
**Deep dive into GPU memory hierarchy**

- Registers → Shared Memory → L1/L2 → Global Memory
- Memory access patterns and coalescing
- Memory-bound vs compute-bound operations
- Bandwidth measurement
- Latency hiding through occupancy

**Key Profiled Experiments:**
- Coalesced vs strided access patterns
- Arithmetic intensity analysis
- Memory allocation overhead
- HBM bandwidth measurement

**Run:** `python 02_memory_hierarchy.py`

---

### 03_streaming_multiprocessor.py
**Understanding the SM - the fundamental compute unit**

- SM architecture across generations
- Warp execution model (32 threads)
- Occupancy calculation and limits
- Register pressure effects
- Tensor Cores operation
- Warp schedulers and ILP

**Key Profiled Experiments:**
- SM parallelism scaling
- Warp divergence measurement
- Occupancy impact on performance
- Tensor Core speedup measurement

**Run:** `python 03_streaming_multiprocessor.py`

---

### 04_profiling_fundamentals.py
**Essential GPU profiling techniques**

- Correct GPU timing (CUDA events)
- PyTorch Profiler usage
- Memory profiling
- Training step breakdown
- Bottleneck identification
- Nsight Systems/Compute overview

**Key Profiled Experiments:**
- Correct vs incorrect timing demonstration
- Forward/backward/optimizer breakdown
- Memory tracking through operations
- Bottleneck type identification

**Run:** `python 04_profiling_fundamentals.py`

---

## 🎯 Learning Objectives

After completing this module, you will:

- [ ] Understand why GPUs are faster for ML workloads
- [ ] Know the GPU memory hierarchy and its implications
- [ ] Understand SM architecture and occupancy
- [ ] Be able to profile GPU code correctly
- [ ] Identify compute vs memory vs transfer bottlenecks
- [ ] Understand Tensor Cores and when they help

## 🔗 Connection to Multimodal Training

| Component | GPU Architecture Relevance |
|-----------|---------------------------|
| Image Batch Processing | Parallel processing across pixels |
| Attention Mechanism | Memory-bound, benefits from tiling |
| Linear Layers | Compute-bound, Tensor Core optimized |
| Layer Normalization | Memory-bound, benefits from fusion |
| KV Cache | HBM bandwidth limited |
| Data Loading | Transfer-bound, overlap with compute |

## 📖 Recommended Order

1. `01_gpu_vs_cpu.py` - Start here for fundamentals
2. `02_memory_hierarchy.py` - Critical for optimization
3. `03_streaming_multiprocessor.py` - Deep architecture knowledge
4. `04_profiling_fundamentals.py` - Essential skill

## ⏱️ Expected Time

- Reading + Running: 3-4 hours
- Deep understanding: 1-2 days
- Full internalization: Ongoing practice

## 📚 Additional Resources

- NVIDIA GPU Architecture Whitepapers
- GPU MODE Lecture 1 (profiling)
- "Programming Massively Parallel Processors" - Kirk & Hwu
- NVIDIA Developer Blog
