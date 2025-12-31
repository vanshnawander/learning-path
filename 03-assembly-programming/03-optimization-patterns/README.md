# Assembly Optimization Patterns for LLM Inference

These are the techniques used in llama.cpp, GGML, and other high-performance inference engines.

## Key Optimization Areas

### 1. Quantized Matrix Multiply
- INT4/INT8 dot products
- Dequantization on the fly
- Packed weight formats

### 2. Memory Access Patterns
- Prefetching
- Cache blocking
- Non-temporal stores

### 3. SIMD Utilization
- AVX-512 for 16 floats at once
- FMA chains for high throughput
- Horizontal reductions

## Files in This Directory

| File | Description |
|------|-------------|
| `01_quantized_dot.c` | INT8/INT4 quantized dot product |
| `02_prefetch_patterns.c` | Software prefetching |
| `03_cache_blocking_asm.c` | Tiled matrix multiply |
| `04_llm_kernels.c` | LLM-specific optimizations |

## llama.cpp Optimization Techniques

From studying the GGML source:

```c
// Q4_0 format: 4-bit quantization
// 32 weights packed into 16 bytes + 2 byte scale
typedef struct {
    ggml_fp16_t d;         // scale (2 bytes)
    uint8_t qs[QK4_0/2];   // 4-bit weights (16 bytes)
} block_q4_0;

// Dot product uses:
// 1. Load 32 4-bit values
// 2. Unpack to 8-bit
// 3. Multiply-add with AVX vnni
// 4. Scale by d factor
```

## Key Instructions for Quantization

| Instruction | Use |
|-------------|-----|
| `vpshufb` | Unpack 4-bit to 8-bit |
| `vpmaddubsw` | Multiply u8 × s8, add pairs |
| `vpmaddwd` | Multiply s16 × s16, add pairs |
| `vpdpbusd` | 8-bit dot product (AVX-VNNI) |

## Performance Tips

1. **Keep weights in cache** - Block by L2 cache size
2. **Unroll loops** - Hide instruction latency
3. **Use FMA chains** - Maximize ALU utilization
4. **Minimize branches** - Use conditional moves
5. **Prefetch next block** - Hide memory latency
