# Tensor Cores

Specialized matrix multiply accelerators in NVIDIA GPUs.

## What Are Tensor Cores?

Fixed-function units that compute:
```
D = A × B + C
```
Where A, B, C, D are small matrices (typically 16×16 or 8×8).

## Evolution

| Gen | Architecture | Precision | Matrix Size |
|-----|-------------|-----------|-------------|
| 1st | Volta (V100) | FP16→FP32 | 4×4×4 |
| 2nd | Turing | FP16, INT8 | 8×8×4 |
| 3rd | Ampere (A100) | FP16, BF16, TF32, INT8 | 16×8×16 |
| 4th | Hopper (H100) | FP8, FP16, BF16, TF32 | 16×8×32 |

## Speedup

- 8-16x faster than CUDA cores for matrix multiply
- Essential for Transformer training
- Used by cuBLAS, cuDNN automatically

## Programming Tensor Cores

### CUDA (WMMA)
```cuda
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, a, 16);
wmma::load_matrix_sync(b_frag, b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### Triton
Triton uses Tensor Cores automatically with `tl.dot()`.

## Requirements
- Aligned memory
- Specific matrix dimensions
- Supported precisions
