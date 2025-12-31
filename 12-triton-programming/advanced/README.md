# Advanced Triton Programming

This directory covers advanced Triton patterns used in production deep learning systems.

## 📚 Modules

### 01_flash_attention.py
**Understanding and implementing Flash Attention**

- Why O(N²) attention is problematic
- Online softmax algorithm
- Tiled attention computation
- Memory comparison with standard attention
- Performance benchmarks

### Unsloth Kernels (02_unsloth_kernels.py) ★ NEW
The actual optimization patterns used in Unsloth for 2-5x speedup:
- **Fused RMSNorm + Residual**: 2-3x speedup, 3x fewer memory passes
- **Fused Cross-Entropy (Chunked)**: 10x memory reduction for large vocab
- **Fused RoPE**: Single kernel for Q and K rotation
- **Fused SwiGLU**: gate + up + silu in one pass
- **Fused LoRA Forward**: Efficient adapter inference

### Quantization Kernels (03_quantization_kernels.py) ★ NEW
Essential for efficient inference:
- **INT8 Quantization/Dequantization**: 4x memory reduction
- **INT8 Matmul with on-the-fly dequant**: QLoRA-style inference
- **NF4 (4-bit NormalFloat)**: Information-optimal for weights
- **FP8 (E4M3/E5M2)**: Hopper Tensor Cores
- **Dynamic per-row quantization**: Activation quantization

## 📁 Files

| File | Description |
|------|-------------|
| `01_flash_attention.py` | Flash Attention deep dive with experiments |
| `02_unsloth_kernels.py` | Production fused kernels (Unsloth-style) |
| `03_quantization_kernels.py` | INT8/FP8/NF4 quantization kernels |
| `flash-attention/README.md` | Additional Flash Attention resources |

## 🎯 Learning Order

1. **01_flash_attention.py** - Understand memory-efficient attention
2. **02_unsloth_kernels.py** - Learn production optimization patterns
3. **03_quantization_kernels.py** - Master quantization for inference

## 📖 Resources

- Flash Attention paper (Dao et al.)
- Unsloth GitHub: github.com/unslothai/unsloth
- bitsandbytes for INT8/NF4 reference
- NVIDIA FP8 documentation
- Flash Attention paper: "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Flash Attention 2 paper: Improved parallelism
- Triton implementation in flash-attention repository

## ⏱️ Expected Time

- Reading + Running: 2-3 hours
- Deep understanding: 1-2 days
