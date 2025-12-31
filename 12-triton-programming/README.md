# 12 - Triton Programming

High-level GPU programming with OpenAI Triton - the secret weapon for fast LLM training.

## 📚 Topics Covered

### Triton Basics
- **Why Triton**: Easier than CUDA, produces highly optimized code
- **Kernel Definition**: @triton.jit decorator
- **Block-Based Programming**: Tile abstraction (not thread-level like CUDA)
- **Auto-Tuning**: Automatic configuration optimization

### Programming Model
- **Program IDs**: tl.program_id for block indexing
- **Pointers**: Arithmetic, loading, storing
- **Masking**: Boundary handling for safety
- **Block Operations**: Vectorized load, store, compute

### Core Operations
- **Vector Operations**: Element-wise (add, mul, etc.)
- **Reductions**: tl.sum, tl.max, tl.min
- **Matrix Operations**: tl.dot (Tensor Core accelerated)
- **Atomic Operations**: tl.atomic_add

### Memory Management
- **Global Memory Access**: Automatic coalescing
- **Shared Memory**: Automatic management
- **Memory Layout**: Row/column major considerations
- **Prefetching**: Software pipelining

### Optimization
- **Auto-Tuning**: @triton.autotune decorator
- **Block Sizes**: Performance impact
- **Num Warps**: Occupancy tuning
- **Num Stages**: Software pipelining depth

### Production Kernels (Unsloth Style)
- **Fused RMSNorm + Residual**: Single kernel for normalization
- **Fused Cross-Entropy**: Chunked for large vocabularies
- **Fused RoPE**: Rotary position embeddings
- **Fused SwiGLU MLP**: Gate + up + activation
- **Quantization Kernels**: INT8, FP8, NF4

## 🎯 Learning Objectives

- [x] Write Triton kernels from scratch
- [x] Implement memory-efficient attention
- [x] Use auto-tuning effectively
- [x] Understand Unsloth-style optimizations
- [x] Write quantization kernels

## 💻 Practical Exercises

### Puzzles (puzzles/01_triton_puzzles.py)
1. Vector addition (warmup)
2. Fused add + ReLU
3. Row-wise sum (2D reduction)
4. Softmax implementation
5. Layer normalization
6. Tiled matrix multiplication
7. GELU activation
8. Online maximum algorithm
9. RMSNorm challenge

### Advanced Projects
1. Flash Attention from scratch
2. Unsloth-style fused kernels
3. INT8/NF4 quantization

## 📖 Resources

### Official
- Triton Documentation: triton-lang.org
- OpenAI Triton blog post

### Community
- GPU Mode lectures (Lecture 14, 29)
- Triton Puzzles by Sasha Rush
- triton-resources GitHub repo

### Code References
- `unsloth/unsloth/kernels/` - Production Triton kernels
- `flash-attention/flash_attn/` - Flash Attention
- `bitsandbytes/` - Quantization kernels

## 📁 Structure

```
12-triton-programming/
├── README.md
├── basics/
│   ├── README.md
│   └── 01_triton_fundamentals.py      # Core concepts, profiling
├── patterns/
│   ├── README.md
│   ├── 01_softmax_kernel.py           # Softmax + online algorithm
│   └── 02_matmul_kernel.py            # Tiled matmul, auto-tuning
├── advanced/
│   ├── README.md
│   ├── 01_flash_attention.py          # Flash Attention deep dive
│   ├── 02_unsloth_kernels.py          # ★ Production kernels (NEW)
│   ├── 03_quantization_kernels.py     # ★ INT8/FP8/NF4 (NEW)
│   └── flash-attention/
│       └── README.md
├── puzzles/                            # ★ NEW
│   └── 01_triton_puzzles.py           # 9 practice problems
└── triton_programming_notebook.ipynb
```

## 🔥 Unsloth Kernel Coverage (NEW)

### Covered in 02_unsloth_kernels.py:
- **Fused RMSNorm + Residual**: 2-3x speedup
- **Fused Cross-Entropy (Chunked)**: 10x memory reduction
- **Fused RoPE**: 2x speedup
- **Fused SwiGLU**: 1.5-2x speedup
- **Fused LoRA Forward**: Efficient adapter inference

### Covered in 03_quantization_kernels.py:
- **INT8 Quantization/Dequantization**
- **INT8 Matmul with on-the-fly dequant**
- **NF4 (QLoRA) dequantization concept**
- **FP8 (E4M3/E5M2) for Hopper**
- **Dynamic per-row quantization**

## ⏱️ Estimated Time: 4-6 weeks

## 🎓 Recommended Order

1. `basics/01_triton_fundamentals.py` - Core concepts
2. `puzzles/01_triton_puzzles.py` - Practice problems (do all 9!)
3. `patterns/01_softmax_kernel.py` - Online softmax
4. `patterns/02_matmul_kernel.py` - Tiling, auto-tuning
5. `advanced/01_flash_attention.py` - Memory-efficient attention
6. `advanced/02_unsloth_kernels.py` - Production optimizations
7. `advanced/03_quantization_kernels.py` - Quantization for inference
