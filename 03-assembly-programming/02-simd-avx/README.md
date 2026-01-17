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

Each register can be accessed at different widths:
- XMM (128-bit): `_mm_*` intrinsics / `xmm` in assembly
- YMM (256-bit): `_mm256_*` intrinsics / `ymm` in assembly  
- ZMM (512-bit): `_mm512_*` intrinsics / `zmm` in assembly

## AVX Instruction Categories

### Vector Load/Store
| Instruction | Description | Bytes |
|-------------|-------------|-------|
| `vmovups` | Unaligned packed single | 32 |
| `vmovaps` | Aligned packed single | 32 |
| `vmovss` | Scalar single | 4 |
| `vbroadcastss` | Broadcast scalar to vector | 32 |

### Vector Arithmetic
| Instruction | Description | FLOPs |
|-------------|-------------|-------|
| `vaddps` | Packed single add | 8 |
| `vsubps` | Packed single subtract | 8 |
| `vmulps` | Packed single multiply | 8 |
| `vfmadd231ps` | Fused multiply-add | 16 |

### Horizontal Operations
| Instruction | Description |
|-------------|-------------|
| `vhaddps` | Horizontal add adjacent pairs |
| `vextractf128` | Extract 128 bits to XMM |
| `vpermilps` | Permute floats within 128-bit lanes |

## FMA - The Heart of Matrix Multiply

FMA (Fused Multiply-Add) computes `a * b + c` in a single instruction.

### FMA Syntax
```
vfmadd231ps ymm1, ymm2, ymm0   # ymm0 = ymm1 * ymm2 + ymm0
vfmadd132ps ymm1, ymm2, ymm0   # ymm0 = ymm1 * ymm0 + ymm2
vfmadd213ps ymm1, ymm2, ymm0   # ymm0 = ymm1 * ymm2 + ymm0
```

The number encodes operand order:
- 1 = first source (multiplicand)
- 2 = second source (multiplier)
- 3 = destination (accumulator)

## Quantization Instructions (INT8/INT4)

For quantized inference (llama.cpp uses this!):

| Instruction | Description |
|-------------|-------------|
| `vpmaddubsw` | Multiply unsigned byte × signed byte → signed word |
| `vpmaddwd` | Multiply signed words → add pairs to dwords |
| `vpshufb` | Shuffle bytes (lookup table pattern) |
| `vpmovsxbw` | Sign-extend bytes to words |
| `vpmovzxbw` | Zero-extend bytes to words |

## Files in This Directory

| File | Description |
|------|-------------|
| `01_avx_basics.c` | AVX intrinsics from C (start here) |
| `02_avx_dotproduct.s` | Hand-written AVX assembly dot product |
| `03_avx_matmul.c` | Matrix multiply with AVX |
| `04_quantized_matmul.c` | INT8 matrix multiply |

## Compile Flags

```bash
# Enable AVX2 (256-bit)
gcc -mavx2 -mfma -O3 program.c

# Enable AVX-512 (512-bit)
gcc -mavx512f -mavx512bw -O3 program.c

# Check CPU support (Linux)
cat /proc/cpuinfo | grep -E "avx|avx2|avx512"

# Check CPU support (macOS)
sysctl machdep.cpu.features
```

## AVX Assembly Template

```asm
# Function: void avx_operation(float* out, float* a, float* b, int n)
# Args: RDI=out, RSI=a, RDX=b, RCX=n
avx_operation:
    push %rbp
    mov %rsp, %rbp
    
    vxorps %ymm0, %ymm0, %ymm0    # Zero accumulator
    
    cmp $8, %ecx
    jl scalar_loop
    
    shr $3, %ecx                  # n / 8
    
vector_loop:
    vmovups (%rsi), %ymm1         # Load a
    vmovups (%rdx), %ymm2         # Load b
    vfmadd231ps %ymm1, %ymm2, %ymm0  # FMA
    
    add $32, %rsi                 # 8 floats
    add $32, %rdx
    dec %ecx
    jnz vector_loop
    
    # Horizontal sum to scalar...
    
    pop %rbp
    ret
```

## Performance Tips

1. **Use FMA** - 2 FLOPs per cycle instead of 1
2. **Unroll loops** - 4x or 8x for better ILP
3. **Multiple accumulators** - Break dependency chains
4. **Align data** - 32-byte alignment for AVX loads
5. **Prefetch** - For large working sets
6. **vzeroupper** - Clear upper YMM bits after AVX code

## Common Patterns

### Dot Product
```asm
vxorps %ymm0, %ymm0, %ymm0        # sum = 0
loop:
    vmovups (%rsi), %ymm1
    vmovups (%rdi), %ymm2
    vfmadd231ps %ymm1, %ymm2, %ymm0
    add $32, %rsi
    add $32, %rdi
    dec %ecx
    jnz loop
# Horizontal sum...
```

### Broadcast + Multiply
```asm
vbroadcastss (%rdi), %ymm0       # ymm0 = [s,s,s,s,s,s,s,s]
vmulps %ymm0, %ymm1, %ymm2        # ymm2 = ymm1 * broadcast
```

### Masked Operation (AVX-512)
```asm
kmovw $0b1111, %k1                # Mask for 4 active lanes
vmovups (%rsi), %ymm1 {%k1}      # Load only masked elements
