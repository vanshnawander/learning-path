# Memory Bandwidth: The Real Bottleneck

This is the #1 most important concept for ML performance.

## The Fundamental Problem

```
Moore's Law (compute): 2x every 18 months
Memory bandwidth growth: ~1.1x every 18 months

Gap doubles every 2 years!
```

## Compute vs Memory Bound

### Compute Bound
```
Time = Operations / Compute_Speed
Example: Complex math with data already in registers
```

### Memory Bound (Most ML Operations!)
```
Time = Data_Size / Memory_Bandwidth
Example: Element-wise operations, attention, most layers
```

## Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Loaded

High AI (compute bound):
  Matrix multiply: O(N³) ops / O(N²) bytes = O(N) AI
  As N grows, becomes compute bound

Low AI (memory bound):
  ReLU: N ops / N bytes = 1 AI
  Always memory bound!
  
  Softmax: ~5N ops / 2N bytes = 2.5 AI
  Memory bound!
  
  Layer Norm: ~10N ops / 3N bytes = 3.3 AI  
  Memory bound!
```

## The Roofline Model

```
Performance (FLOPS)
     │
     │                    ╱ Compute ceiling
     │              ╱────────────────────
     │         ╱
     │    ╱         Ridge point
     │╱─────────────────────────────────
     │   Memory bandwidth ceiling
     │
     └─────────────────────────────────── Arithmetic Intensity
           1    10   100   1000
           
Operations below the roof are memory bound
Operations on the compute ceiling are compute bound
```

## Real Numbers: A100 GPU

```
Compute: 312 TFLOPS (FP16 Tensor Core)
Memory:  2 TB/s (HBM2e)

Ridge Point = 312 TFLOPS / 2 TB/s = 156 FLOPS/byte

To be compute bound, need AI > 156!

Matrix Multiply (large N): AI ≈ N/2
  N=1024: AI=512 → Compute bound ✓
  N=64: AI=32 → Memory bound ✗
  
Attention: AI ≈ 2-4 → Memory bound (why Flash Attention!)
Elementwise: AI = 1 → Memory bound (fuse operations!)
```

## Why Flash Attention Works

### Standard Attention
```
Q, K, V: [B, H, N, D]

# Step 1: Compute S = Q @ K^T
S = Q @ K.transpose(-2, -1)  # [B, H, N, N]
# Load Q,K: 2*B*H*N*D bytes
# Store S: B*H*N*N bytes

# Step 2: Softmax
P = softmax(S, dim=-1)  # [B, H, N, N]
# Load S, Store P: 2*B*H*N*N bytes

# Step 3: Output
O = P @ V  # [B, H, N, D]
# Load P,V, Store O

Total memory: O(N²) for intermediate S, P
Arithmetic Intensity: Low (many loads/stores)
```

### Flash Attention
```
# Tile computation to fit in SRAM
# Never materialize full N×N matrix!

for each tile of Q:
    for each tile of K, V:
        # Compute local attention in SRAM
        # Running softmax normalization
        # Accumulate output

Total memory: O(N) - no N×N storage!
Arithmetic Intensity: Much higher (recompute > reload)
```

## Kernel Fusion: Reducing Memory Traffic

### Unfused (Memory Bound)
```python
# Each operation reads/writes to global memory
x = linear(x)    # Load x, compute, store x
x = relu(x)      # Load x, compute, store x  
x = dropout(x)   # Load x, compute, store x

Memory traffic: 6× the tensor size!
```

### Fused (Better)
```python
# Single kernel does all three
x = fused_linear_relu_dropout(x)

Memory traffic: 2× the tensor size (load once, store once)
3× less memory traffic!
```

## Practical Guidelines

### 1. Batch Operations
```python
# BAD: Small operations don't saturate bandwidth
for item in batch:
    process(item)

# GOOD: Large operations better utilize bandwidth
process(batch)
```

### 2. Avoid Materialization
```python
# BAD: Creates intermediate tensor
y = x.exp()
z = y.sum()

# GOOD: Fused operation
z = x.logsumexp()
```

### 3. Use Appropriate Precision
```
FP32: 4 bytes per element
FP16: 2 bytes per element
INT8: 1 byte per element

Lower precision = 2-4× more compute per byte loaded!
```

### 4. Optimize Memory Layout
```python
# Contiguous is faster
x = x.contiguous()  # Ensure contiguous before compute

# channels_last can be faster for convolutions
x = x.to(memory_format=torch.channels_last)
```

## Measuring Memory Boundedness

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    output = model(input)

# Look for:
# - High memory throughput
# - Low SM utilization
# - These indicate memory bound!
```
