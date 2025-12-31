# SIMD/AVX Assembly for LLM Inference

This is what makes llama.cpp fast on CPU - hand-optimized SIMD code.

## Why SIMD for LLM Inference?

LLM inference is dominated by:
1. **Matrix-vector multiply** (attention, FFN)
2. **Dot products** (Q·K scoring)
3. **Element-wise operations** (activations)

All are perfect for SIMD!

## SIMD Register Hierarchy

```
XMM0-15:  128-bit = 4 floats  (SSE)
YMM0-15:  256-bit = 8 floats  (AVX)
ZMM0-31:  512-bit = 16 floats (AVX-512)
```

## Key AVX Instructions for ML

| Instruction | Purpose | ML Use |
|-------------|---------|--------|
| `vaddps` | Add packed singles | Residual connections |
| `vmulps` | Multiply packed singles | Scaling |
| `vfmadd231ps` | Fused multiply-add | Matrix multiply |
| `vdpps` | Dot product | Attention scores |
| `vbroadcastss` | Broadcast scalar | Weight broadcast |

## Quantization Instructions (INT8/INT4)

| Instruction | Purpose |
|-------------|---------|
| `vpmaddubsw` | Multiply-add unsigned/signed bytes |
| `vpmaddwd` | Multiply-add words to dwords |
| `vpshufb` | Shuffle bytes (dequantization) |

## Files in This Directory

| File | Description |
|------|-------------|
| `01_avx_basics.c` | AVX intrinsics from C |
| `02_avx_dotproduct.s` | Hand-written AVX dot product |
| `03_avx_matmul.c` | Matrix multiply with AVX |
| `04_quantized_matmul.c` | INT8 matrix multiply |

## Compile Flags

```bash
# Enable AVX2
gcc -mavx2 -mfma -O3 program.c

# Enable AVX-512
gcc -mavx512f -mavx512bw -O3 program.c

# Check CPU support
cat /proc/cpuinfo | grep -E "avx|avx2|avx512"
```
