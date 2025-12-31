# NVIDIA GPU Architectures

This directory covers the evolution of NVIDIA GPU architectures and their implications for deep learning workloads.

## 📚 Modules

### 01_architecture_evolution.py
**Complete guide to NVIDIA GPU architecture evolution**

- Architecture timeline (Pascal → Volta → Ampere → Hopper → Blackwell)
- Tensor Core evolution and capabilities
- Memory bandwidth improvements
- SM architecture changes
- Practical implications for your GPU

**Key Profiled Experiments:**
- Tensor Core speedup by precision
- Memory bandwidth measurement
- Feature availability detection

**Run:** `python 01_architecture_evolution.py`

## 🎯 Learning Objectives

- [ ] Understand GPU architecture evolution
- [ ] Know Tensor Core capabilities by generation
- [ ] Understand memory technology differences
- [ ] Choose appropriate optimizations for your GPU

## 🔗 Architecture Quick Reference

| Architecture | Year | Key ML Feature |
|-------------|------|----------------|
| Volta | 2017 | First Tensor Cores |
| Turing | 2018 | INT8 inference |
| Ampere | 2020 | TF32, BF16, sparsity |
| Hopper | 2022 | FP8, Transformer Engine |
| Blackwell | 2024 | FP4, 8 TB/s bandwidth |

## ⏱️ Expected Time

- Reading + Running: 1-2 hours
