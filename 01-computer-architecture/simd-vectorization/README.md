# SIMD and Vectorization

Single Instruction, Multiple Data - the CPU's parallel processing.

## x86 SIMD Extensions

| Extension | Width | Types |
|-----------|-------|-------|
| SSE | 128-bit | 4x float |
| AVX | 256-bit | 8x float |
| AVX-512 | 512-bit | 16x float |

## Key Concepts

### Vector Registers
- xmm0-xmm15 (SSE, 128-bit)
- ymm0-ymm15 (AVX, 256-bit)
- zmm0-zmm31 (AVX-512, 512-bit)

### Intrinsics
```c
#include <immintrin.h>

// Load 8 floats
__m256 a = _mm256_load_ps(ptr);

// Multiply
__m256 c = _mm256_mul_ps(a, b);

// Store
_mm256_store_ps(result, c);
```

### Auto-vectorization
Compiler can auto-vectorize loops with:
- `-O3 -march=native`
- Proper alignment
- No dependencies

## ARM NEON
Similar concepts for ARM processors.

## Exercises
1. Write SIMD dot product
2. Compare scalar vs vectorized
3. Analyze compiler vectorization
