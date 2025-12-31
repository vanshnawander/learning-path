# The Roofline Model: Performance Analysis Framework

## What is the Roofline Model?

The Roofline model is a visual performance model that helps you:
- Understand if code is **compute-bound** or **memory-bound**
- Identify **optimization opportunities**
- Set **realistic performance expectations**
- Compare **achieved vs. theoretical** performance

## The Model

```
Performance                      Compute Ceiling
(GFLOPS)                         ┌────────────────────────────
    ^                           /
    │                          /
    │              Achievable /
    │              Region    /
    │                       /
    │                      / ← Roofline
    │                     /
    │         Memory     /
    │         Ceiling   /
    │                  /
    │                 /
    │                /
    │               /
    └──────────────/─────────────────────────►
                  ^                    Arithmetic Intensity
           Ridge Point              (FLOPS / Byte)
```

### Key Equations

**Arithmetic Intensity (AI):**
```
AI = Total FLOPs / Total Bytes Transferred
   = (Floating point operations) / (Memory traffic)
```

**Attainable Performance:**
```
P = min(Peak_Compute, Peak_Bandwidth × AI)

Where:
- Peak_Compute = Peak GFLOPS of processor
- Peak_Bandwidth = Memory bandwidth (GB/s)
- AI = Arithmetic Intensity (FLOP/Byte)
```

**Ridge Point:**
```
Ridge_AI = Peak_Compute / Peak_Bandwidth

Operations with AI < Ridge_AI are memory-bound
Operations with AI > Ridge_AI are compute-bound
```

## Example: NVIDIA A100 GPU

```
A100 80GB SXM Specifications:
- Peak FP32: 19.5 TFLOPS
- Peak FP16: 312 TFLOPS (Tensor Cores)
- HBM2e Bandwidth: 2039 GB/s

Ridge Points:
- FP32: 19500 / 2039 = 9.6 FLOP/Byte
- FP16 (TC): 312000 / 2039 = 153 FLOP/Byte
```

### Roofline for A100

```
TFLOPS
   ^
312│                              ════════════════  FP16 Tensor Core
   │                             /
   │                            /
   │                           /
19.5│         ════════════════/─────────────────  FP32
   │        /
   │       /
   │      /
   │     /  Memory Bound Region
   │    /
   │   /
   │  /
   │ /
   └─┴─────────────────────────────────────────►
     1    9.6        100    153                    AI (FLOP/Byte)
```

## Common Operations Analysis

### Matrix Multiplication (GEMM)

```
C[M,N] = A[M,K] × B[K,N]

FLOPs = 2 × M × N × K  (multiply-add per element)

Bytes = (M×K + K×N + M×N) × sizeof(element)
      ≈ 2×M×N×K / min(M,N,K) for large matrices

AI = 2×M×N×K / (2×M×N×K / min(M,N,K) × sizeof)
   = min(M,N,K) / sizeof

For M=N=K=4096, FP32:
AI = 4096 / 4 = 1024 FLOP/Byte  ← Compute-bound!
```

### Element-wise Operations (Vector Add)

```
C[N] = A[N] + B[N]

FLOPs = N  (one add per element)
Bytes = 3 × N × sizeof(element)  (read A, B; write C)

AI = N / (3×N×sizeof) = 1 / (3×sizeof)
   = 1/12 for FP32 = 0.083 FLOP/Byte  ← Memory-bound!
```

### Convolution

```
For Conv2D with input [N,C,H,W], kernel [K,C,R,S]:
FLOPs = 2 × N × K × H_out × W_out × C × R × S

AI depends heavily on:
- Batch size (larger = higher AI)
- Channel count
- Kernel size
- Implementation (Winograd, FFT, etc.)

Typical AI range: 10-100 FLOP/Byte
```

### Attention (Transformer)

```
Standard Attention: Q×K^T / sqrt(d) → Softmax → ×V

For sequence length S, head dim D:
QK^T: FLOPs = 2×S×S×D, Bytes = 4×S×D (read Q,K)
AI = S×D / (2×D) = S/2

For S=2048: AI = 1024 FLOP/Byte ← Compute-bound

But softmax and memory reads dominate in practice
Flash Attention addresses this by fusing operations
```

## Hierarchical Roofline

Real systems have multiple memory levels:

```
GFLOPS
   ^
   │                              ══════════════ Peak Compute
   │                             /
   │               ─────────────/───────────────  L1 Cache Roofline
   │              /
   │      ───────/──────────────────────────────  L2 Cache Roofline
   │     /
   │  ──/───────────────────────────────────────  L3/HBM Roofline
   │ /
   │/
   └────────────────────────────────────────────►
                                                 AI (FLOP/Byte)
```

### GPU Memory Hierarchy (A100)

| Level | Bandwidth | Capacity |
|-------|-----------|----------|
| Registers | ~20 TB/s | 256 KB/SM |
| Shared Memory | ~19 TB/s | 164 KB/SM |
| L2 Cache | ~5 TB/s | 40 MB |
| HBM | ~2 TB/s | 80 GB |

## Using Nsight Compute for Roofline

```bash
# Collect roofline data
ncu --set roofline -o roofline.ncu-rep python train.py

# Analyze specific kernel
ncu --kernel-name "gemm" --set roofline python train.py
```

### Interpreting Results

```
Kernel: volta_sgemm_128x64_nn

Achieved Performance: 8.5 TFLOPS
Peak Performance: 19.5 TFLOPS
Efficiency: 43.6%

Arithmetic Intensity: 128 FLOP/Byte
Memory Throughput: 1200 GB/s (58% of peak)

Analysis:
- AI > Ridge Point (9.6) → Compute-bound
- Low compute efficiency → Check occupancy, instruction mix
```

## Optimization Strategies by Region

### Memory-Bound (AI < Ridge)

1. **Reduce memory traffic**
   - Kernel fusion
   - Cache blocking
   - Data reuse

2. **Increase AI**
   - Batch operations
   - Use higher precision compute

3. **Maximize bandwidth**
   - Coalesced access
   - Avoid bank conflicts
   - Prefetching

### Compute-Bound (AI > Ridge)

1. **Increase ILP/TLP**
   - More threads
   - Unroll loops
   - Reduce dependencies

2. **Use specialized units**
   - Tensor Cores
   - SIMD/AVX

3. **Algorithm improvements**
   - FFT for convolutions
   - Strassen for GEMM

## Practical Example: Optimizing Vector Add

```c
// Baseline: AI = 0.083
__global__ void vector_add(float* c, float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Fused operations: AI = 0.25 (3x better!)
__global__ void fused_ops(float* d, float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i] + b[i];  // Reuse loaded values
        d[i] = x * c[i];        // More compute per byte
    }
}

// Kernel fusion is key to improving AI for memory-bound ops
```

## Tools for Roofline Analysis

| Tool | Platform | Features |
|------|----------|----------|
| Nsight Compute | NVIDIA GPU | Automatic roofline |
| Intel Advisor | Intel CPU/GPU | Roofline + vectorization |
| AMD μProf | AMD CPU/GPU | Roofline analysis |
| likwid | Linux/x86 | CLI roofline |
| Empirical Roofline Tool | CPU | Measures actual ceilings |

## References

- "Roofline: An Insightful Visual Performance Model" - Williams, Waterman, Patterson
- NERSC Roofline documentation
- NVIDIA Nsight Compute documentation
