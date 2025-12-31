# Floating Point Precision: What Every ML Engineer Must Know

Floating point is NOT real numbers. Understanding this prevents bugs and improves training.

## IEEE 754 Representation

```
┌─────────────────────────────────────────────────────────────┐
│                  FLOATING POINT FORMAT                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  FP32 (32 bits):                                            │
│  ┌────┬─────────────┬───────────────────────────────────┐   │
│  │Sign│  Exponent   │           Mantissa                 │   │
│  │ 1  │     8       │              23                    │   │
│  └────┴─────────────┴───────────────────────────────────┘   │
│                                                              │
│  FP16 (16 bits):                                            │
│  ┌────┬─────────┬───────────────────┐                       │
│  │Sign│Exponent │     Mantissa      │                       │
│  │ 1  │    5    │        10         │                       │
│  └────┴─────────┴───────────────────┘                       │
│                                                              │
│  BF16 (16 bits):                                            │
│  ┌────┬─────────────┬───────────┐                           │
│  │Sign│  Exponent   │ Mantissa  │                           │
│  │ 1  │      8      │     7     │                           │
│  └────┴─────────────┴───────────┘                           │
│                                                              │
│  Value = (-1)^sign × 2^(exp-bias) × (1 + mantissa)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Format Comparison

| Format | Exp Bits | Mantissa | Range | Precision |
|--------|----------|----------|-------|-----------|
| FP32 | 8 | 23 | ±3.4×10³⁸ | ~7 decimals |
| FP16 | 5 | 10 | ±65504 | ~3 decimals |
| BF16 | 8 | 7 | ±3.4×10³⁸ | ~2 decimals |
| TF32 | 8 | 10 | ±3.4×10³⁸ | ~3 decimals |

## The Classic Gotcha

```python
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False

# Why? 0.1 cannot be exactly represented in binary!
# 0.1 (decimal) = 0.0001100110011... (binary, repeating)
```

## Precision Loss in ML

### 1. Large + Small Problem
```python
>>> large = 1e7
>>> small = 1e-7
>>> large + small == large
True  # small was lost!

# In ML: Learning rate * gradient << weights
# Gradients can be "swallowed"!
```

### 2. Catastrophic Cancellation
```python
>>> a = 1.0000001
>>> b = 1.0000000
>>> (a - b)  # Should be 0.0000001
1.0000000861326987e-07  # Lost precision!

# In ML: Happens in variance computation, softmax
```

### 3. Accumulation Order Matters
```python
import numpy as np

# Random numbers
np.random.seed(42)
values = np.random.randn(1000000).astype(np.float32)

# Different order, different results!
sum_forward = np.sum(values)
sum_reverse = np.sum(values[::-1])
sum_sorted = np.sum(np.sort(values))

print(f"Forward: {sum_forward}")
print(f"Reverse: {sum_reverse}")  # Different!
print(f"Sorted:  {sum_sorted}")   # More accurate

# In ML: Batch size affects gradient accumulation order
# Different batch sizes can give different results!
```

## FP16 vs BF16: The Tradeoff

### FP16 (Half Precision)
```
Range: ±65504 (SMALL!)
Precision: 10 mantissa bits

Problems:
- Gradients can overflow (> 65504)
- Gradients can underflow (< 2^-24 ≈ 6×10⁻⁸)
- Need loss scaling!
```

### BF16 (Brain Float)
```
Range: ±3.4×10³⁸ (same as FP32!)
Precision: 7 mantissa bits (less than FP16)

Benefits:
- Same range as FP32 → no overflow
- Drop-in replacement (truncate FP32)
- No loss scaling needed usually
```

## Mixed Precision Training

```python
# PyTorch AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # Backward pass with loss scaling
    scaler.scale(loss).backward()
    
    # Unscale before optimizer step
    scaler.step(optimizer)
    scaler.update()
```

### Why Loss Scaling?
```
Problem: Small gradients (< 2^-24) become zero in FP16

Solution:
1. Scale loss by large factor (e.g., 1024)
2. Gradients scaled up proportionally
3. Now in representable range!
4. Unscale before weight update

GradScaler does this automatically
```

## Numerical Stability Tricks

### 1. LogSumExp (Softmax Stability)
```python
# BAD: Can overflow
def softmax_unstable(x):
    exp_x = np.exp(x)  # exp(1000) = Inf!
    return exp_x / exp_x.sum()

# GOOD: Subtract max first
def softmax_stable(x):
    x_max = x.max()
    exp_x = np.exp(x - x_max)  # Now max is 0, exp(0)=1
    return exp_x / exp_x.sum()
```

### 2. Log-space Computation
```python
# BAD: Product underflows
prob = p1 * p2 * p3 * p4  # → 0.0

# GOOD: Sum in log-space
log_prob = log_p1 + log_p2 + log_p3 + log_p4  # Works!
```

### 3. Kahan Summation (Compensated Sum)
```python
def kahan_sum(values):
    total = 0.0
    compensation = 0.0
    
    for value in values:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    
    return total

# Much more accurate for large sums!
```

## ML-Specific Issues

### Gradient Checkpointing
```
FP16 forward → FP16 activations saved
FP16 recompute → DIFFERENT values (numerical noise)
Can cause training instability!

Solution: Use deterministic operations where possible
```

### Batch Normalization
```python
# Running mean/var accumulates error
# Reset periodically or use larger dtype for statistics
model.bn.running_mean.dtype  # Keep as FP32!
```

### Attention Scores
```python
# Q @ K^T can have large values
# Softmax on large values → numerical issues

# Solution: Scale by sqrt(d_k)
scores = (Q @ K.T) / math.sqrt(d_k)
```

## Debugging Precision Issues

```python
# Check for NaN/Inf
torch.isnan(tensor).any()
torch.isinf(tensor).any()

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```
