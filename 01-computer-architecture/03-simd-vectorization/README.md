# SIMD and Vectorization

Processing multiple data elements with single instructions.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_simd_basics.c` | AVX intrinsics examples |

## SIMD Extensions

| Extension | Width | Floats | Era |
|-----------|-------|--------|-----|
| SSE | 128-bit | 4 | 1999 |
| AVX | 256-bit | 8 | 2011 |
| AVX-512 | 512-bit | 16 | 2016 |

## Key Concepts

### Vector Registers
```
XMM0-15: 128-bit (SSE)
YMM0-15: 256-bit (AVX)
ZMM0-31: 512-bit (AVX-512)
```

### Intrinsics
```c
#include <immintrin.h>

__m256 a = _mm256_load_ps(ptr);      // Load 8 floats
__m256 b = _mm256_load_ps(ptr + 8);
__m256 c = _mm256_add_ps(a, b);       // Add 8 pairs
_mm256_store_ps(result, c);           // Store 8 floats
```

### Auto-vectorization
Compilers can vectorize loops:
```bash
gcc -O3 -march=native -ftree-vectorize
```

## ML Applications

1. **NumPy**: Uses SIMD in BLAS
2. **Image decoding**: libjpeg-turbo uses SIMD
3. **FFCV transforms**: CPU augmentations
4. **Tokenizers**: Fast text processing

## GPU Parallel: SIMT

GPUs use **SIMT** (Single Instruction Multiple Threads):
- 32 threads in a warp execute same instruction
- Similar concept, massive scale

## Exercises

1. Write SIMD matrix-vector multiply
2. Compare auto-vectorized vs manual SIMD
3. Measure speedup for different operations
