# Triton Puzzles - Practice Problems

Hands-on exercises to master Triton kernel development.

## 🎯 Overview

These puzzles progress from basic to advanced, covering all essential Triton concepts.

## 📚 Puzzles in 01_triton_puzzles.py

| # | Puzzle | Concepts | Difficulty |
|---|--------|----------|------------|
| 1 | Vector Add | Basic kernel structure, program IDs, masking | ⭐ |
| 2 | Fused Add+ReLU | Kernel fusion, tl.maximum | ⭐ |
| 3 | Row Sum | 2D indexing, reductions, strides | ⭐⭐ |
| 4 | Softmax | Multi-pass algorithm, numerical stability | ⭐⭐ |
| 5 | Layer Norm | Statistics (mean, variance), parameters | ⭐⭐⭐ |
| 6 | Matmul | Tiling, tl.dot, 2D grid | ⭐⭐⭐ |
| 7 | GELU | Math operations, tl.libdevice | ⭐⭐ |
| 8 | Online Max | Streaming algorithms, state management | ⭐⭐ |
| 9 | RMSNorm | Production kernel (Unsloth-style) | ⭐⭐⭐ |

## 🚀 How to Use

1. **Read the puzzle description** - Understand what to implement
2. **Look at the hints** - Each puzzle has inline hints
3. **Write your solution** - Fill in the `# YOUR CODE HERE` sections
4. **Run the tests** - Verify correctness

```bash
python 01_triton_puzzles.py
```

## 💡 Key Concepts by Puzzle

### Puzzle 1-2: Basics
- `@triton.jit` decorator
- `tl.program_id(axis)` for block index
- `tl.arange(start, end)` for offsets
- `tl.load(ptr, mask=mask)` and `tl.store(ptr, val, mask=mask)`

### Puzzle 3-4: Reductions
- `tl.sum(x, axis=0)` for summation
- `tl.max(x, axis=0)` for maximum
- Row-wise processing patterns
- Numerical stability (subtract max before exp)

### Puzzle 5-6: Advanced
- Computing statistics (mean, variance)
- Loading parameters (gamma, beta)
- 2D program grids for matrix operations
- `tl.dot(a, b)` for Tensor Core acceleration

### Puzzle 7-9: Production
- Math functions via `tl.libdevice`
- Streaming/online algorithms
- RMSNorm (simpler than LayerNorm)

## 📖 After Completing

Once you've mastered these puzzles, move on to:
1. `advanced/02_unsloth_kernels.py` - Production optimizations
2. `advanced/03_quantization_kernels.py` - INT8/NF4 kernels
3. Write your own fused kernels!

## 🔗 Resources

- [Triton Puzzles by Sasha Rush](https://github.com/srush/Triton-Puzzles)
- [GPU Mode Triton Lectures](https://www.youtube.com/@GPUMODE)
- [Official Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
