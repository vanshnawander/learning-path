"""
Floating Point Formats for Deep Learning
==========================================

This module provides a comprehensive understanding of floating-point
formats used in deep learning: FP32, FP16, BF16, FP8, and their
trade-offs for training and inference.

Key Topics:
1. IEEE 754 Floating Point Representation
2. FP32 (Single Precision) - The baseline
3. FP16 (Half Precision) - Memory efficient but limited range
4. BF16 (Brain Float 16) - Best of both worlds
5. FP8 (8-bit Floats) - Next frontier
6. Numerical Stability Considerations
7. When to Use Each Format
"""

import torch
import numpy as np
import struct
from typing import Tuple
import math

# =============================================================================
# SECTION 1: IEEE 754 FLOATING POINT FUNDAMENTALS
# =============================================================================
"""
IEEE 754 FLOATING POINT FORMAT:
═══════════════════════════════

A floating-point number is represented as:
    
    value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa)

Components:
┌──────┬────────────┬────────────────────────────────────────┐
│ Sign │  Exponent  │              Mantissa/Fraction         │
└──────┴────────────┴────────────────────────────────────────┘
  1 bit   E bits                    M bits

The FORMAT determines:
- Dynamic Range: Determined by exponent bits
- Precision: Determined by mantissa bits

COMPARISON OF FORMATS:
┌─────────┬──────┬──────────┬──────────┬────────────────┬─────────────────┐
│ Format  │ Bits │ Exponent │ Mantissa │ Dynamic Range  │ Decimal Digits  │
├─────────┼──────┼──────────┼──────────┼────────────────┼─────────────────┤
│ FP32    │ 32   │ 8 bits   │ 23 bits  │ ±3.4 × 10^38   │ ~7.2 digits     │
│ FP16    │ 16   │ 5 bits   │ 10 bits  │ ±65,504        │ ~3.3 digits     │
│ BF16    │ 16   │ 8 bits   │ 7 bits   │ ±3.4 × 10^38   │ ~2.4 digits     │
│ FP8 E4M3│ 8    │ 4 bits   │ 3 bits   │ ±448           │ ~1.2 digits     │
│ FP8 E5M2│ 8    │ 5 bits   │ 2 bits   │ ±57,344        │ ~0.9 digits     │
└─────────┴──────┴──────────┴──────────┴────────────────┴─────────────────┘

KEY INSIGHT:
- More exponent bits = larger dynamic range
- More mantissa bits = higher precision
- Trade-off between range and precision for fixed bit-width
"""


def float_to_binary(value: float, format_type: str = 'fp32') -> str:
    """
    Convert a float to its binary representation.
    
    Args:
        value: Float value to convert
        format_type: 'fp32', 'fp16', 'bf16'
    
    Returns:
        Binary string with format breakdown
    """
    if format_type == 'fp32':
        # Pack as 32-bit float, unpack as 32-bit int
        packed = struct.pack('>f', value)
        bits = struct.unpack('>I', packed)[0]
        binary = format(bits, '032b')
        sign = binary[0]
        exponent = binary[1:9]
        mantissa = binary[9:]
        return f"Sign: {sign} | Exp: {exponent} | Mantissa: {mantissa}"
    
    elif format_type == 'fp16':
        # Use numpy for fp16
        fp16_val = np.float16(value)
        bits = fp16_val.view(np.uint16)
        binary = format(bits, '016b')
        sign = binary[0]
        exponent = binary[1:6]
        mantissa = binary[6:]
        return f"Sign: {sign} | Exp: {exponent} | Mantissa: {mantissa}"
    
    elif format_type == 'bf16':
        # BF16 is top 16 bits of FP32
        packed = struct.pack('>f', value)
        bits = struct.unpack('>I', packed)[0]
        bf16_bits = bits >> 16  # Take top 16 bits
        binary = format(bf16_bits, '016b')
        sign = binary[0]
        exponent = binary[1:9]
        mantissa = binary[9:]
        return f"Sign: {sign} | Exp: {exponent} | Mantissa: {mantissa}"
    
    else:
        raise ValueError(f"Unknown format: {format_type}")


def demonstrate_float_formats():
    """
    Demonstrate the binary representation of different float formats.
    """
    print("\n" + "="*70)
    print("FLOATING POINT FORMAT DEMONSTRATION")
    print("="*70)
    
    test_values = [1.0, 0.1, 3.14159, 1000.5, 0.00001, 65504.0]
    
    for val in test_values:
        print(f"\nValue: {val}")
        print("-" * 50)
        print(f"  FP32: {float_to_binary(val, 'fp32')}")
        print(f"  FP16: {float_to_binary(val, 'fp16')}")
        print(f"  BF16: {float_to_binary(val, 'bf16')}")


# =============================================================================
# SECTION 2: FP32 - THE BASELINE
# =============================================================================
"""
FP32 (Single Precision):
════════════════════════

Structure: 1 sign + 8 exponent + 23 mantissa = 32 bits

Properties:
- Dynamic range: ±1.18 × 10^-38 to ±3.4 × 10^38
- Precision: ~7.2 decimal digits
- Smallest positive normal: 1.18 × 10^-38
- Machine epsilon: 2^-23 ≈ 1.19 × 10^-7

Memory per parameter: 4 bytes

In Deep Learning:
- Standard format for model weights
- Accumulation in mixed precision
- Master weights in AMP training
- Highest precision, highest memory
"""


def fp32_properties():
    """
    Demonstrate FP32 properties.
    """
    print("\n" + "="*70)
    print("FP32 (SINGLE PRECISION) PROPERTIES")
    print("="*70)
    
    # Get machine properties
    finfo = torch.finfo(torch.float32)
    
    print(f"""
Format: 1 sign + 8 exponent + 23 mantissa = 32 bits

Numerical Properties:
- Smallest positive normal: {finfo.tiny:.2e}
- Largest finite value: {finfo.max:.2e}
- Machine epsilon: {finfo.eps:.2e}
- Decimal precision: {finfo.resolution:.2e} ({int(-np.log10(finfo.resolution))} digits)

Memory Usage:
- Per parameter: 4 bytes
- 1B parameters: 4 GB
- 7B parameters: 28 GB
- 70B parameters: 280 GB

Use Cases:
✓ Master weights in mixed precision
✓ Loss computation
✓ Gradient accumulation
✓ Final model for highest precision inference
""")
    
    # Demonstrate precision limits
    print("\nPrecision Demonstration:")
    print("-" * 50)
    
    # Adding small to large
    large = torch.tensor(1e7, dtype=torch.float32)
    small = torch.tensor(1.0, dtype=torch.float32)
    
    result = large + small
    print(f"1e7 + 1.0 = {result.item()}")  # Works in FP32
    
    # Precision loss example
    a = torch.tensor(1.0000001, dtype=torch.float32)
    b = torch.tensor(1.0000002, dtype=torch.float32)
    print(f"1.0000001 stored as: {a.item():.10f}")
    print(f"1.0000002 stored as: {b.item():.10f}")
    print(f"Difference: {(b - a).item():.2e}")


# =============================================================================
# SECTION 3: FP16 - MEMORY EFFICIENT
# =============================================================================
"""
FP16 (Half Precision):
══════════════════════

Structure: 1 sign + 5 exponent + 10 mantissa = 16 bits

Properties:
- Dynamic range: ±6.1 × 10^-5 to ±65,504
- Precision: ~3.3 decimal digits
- Machine epsilon: 2^-10 ≈ 9.77 × 10^-4

Memory per parameter: 2 bytes (50% of FP32)

CRITICAL LIMITATIONS:

1. Limited Range (Overflow/Underflow):
   - Max value: 65,504 (not enough for loss values!)
   - Min normal: 6.1 × 10^-5 (gradients can underflow!)
   
2. Limited Precision:
   - Only ~3.3 decimal digits
   - Gradient accumulation can lose information
   
3. Common Issues in Training:
   - Loss overflow (loss > 65504)
   - Gradient underflow (small gradients become 0)
   - Weight update vanishing (weight + small_grad = weight)

SOLUTION: Mixed Precision with Loss Scaling (covered in next section)
"""


def fp16_properties():
    """
    Demonstrate FP16 properties and limitations.
    """
    print("\n" + "="*70)
    print("FP16 (HALF PRECISION) PROPERTIES")
    print("="*70)
    
    finfo = torch.finfo(torch.float16)
    
    print(f"""
Format: 1 sign + 5 exponent + 10 mantissa = 16 bits

Numerical Properties:
- Smallest positive normal: {finfo.tiny:.2e}
- Largest finite value: {finfo.max:.2e}
- Machine epsilon: {finfo.eps:.2e}
- Decimal precision: ~3.3 digits

Memory Savings:
- Per parameter: 2 bytes (50% of FP32)
- 7B parameters: 14 GB (vs 28 GB FP32)
- 70B parameters: 140 GB (vs 280 GB FP32)
""")
    
    print("\nLIMITATION 1: Overflow")
    print("-" * 50)
    
    # Overflow demonstration
    large_fp32 = torch.tensor(100000.0, dtype=torch.float32)
    large_fp16 = large_fp32.to(torch.float16)
    print(f"FP32 value: {large_fp32.item()}")
    print(f"Converted to FP16: {large_fp16.item()}")  # Will be inf
    print(f"Is inf: {torch.isinf(large_fp16).item()}")
    
    print("\nLIMITATION 2: Underflow")
    print("-" * 50)
    
    # Underflow demonstration (gradients often this small)
    small_fp32 = torch.tensor(1e-5, dtype=torch.float32)
    small_fp16 = small_fp32.to(torch.float16)
    print(f"FP32 gradient: {small_fp32.item():.2e}")
    print(f"Converted to FP16: {small_fp16.item():.2e}")
    
    # Very small gradients become zero
    tiny_fp32 = torch.tensor(1e-6, dtype=torch.float32)
    tiny_fp16 = tiny_fp32.to(torch.float16)
    print(f"\nFP32 tiny gradient: {tiny_fp32.item():.2e}")
    print(f"Converted to FP16: {tiny_fp16.item():.2e}")  # May become 0
    
    print("\nLIMITATION 3: Precision Loss in Updates")
    print("-" * 50)
    
    # Weight update precision loss
    weight = torch.tensor(1.0, dtype=torch.float16)
    small_update = torch.tensor(0.0001, dtype=torch.float16)
    
    updated = weight + small_update
    print(f"Weight: {weight.item()}")
    print(f"Update: {small_update.item()}")
    print(f"Weight + Update: {updated.item()}")
    print(f"Update applied: {(updated - weight).item() != 0}")
    
    # Even smaller update
    tiny_update = torch.tensor(0.00001, dtype=torch.float16)
    updated2 = weight + tiny_update
    print(f"\nTiny update: {tiny_update.item()}")
    print(f"Weight + Tiny Update: {updated2.item()}")
    print(f"Update lost: {(updated2 - weight).item() == 0}")


# =============================================================================
# SECTION 4: BF16 - BEST OF BOTH WORLDS
# =============================================================================
"""
BF16 (Brain Floating Point 16):
════════════════════════════════

Structure: 1 sign + 8 exponent + 7 mantissa = 16 bits

Key Insight: Same exponent bits as FP32!

Properties:
- Dynamic range: Same as FP32 (±3.4 × 10^38)
- Precision: ~2.4 decimal digits (less than FP16!)
- Machine epsilon: 2^-7 ≈ 0.0078

Trade-off:
┌──────────────────────────────────────────────────────────┐
│  BF16 trades PRECISION for RANGE compared to FP16       │
│                                                          │
│  FP16: Better precision (10 mantissa) but limited range │
│  BF16: Same range as FP32 but less precision (7 mant)   │
└──────────────────────────────────────────────────────────┘

Why BF16 is Better for Deep Learning:

1. No overflow for typical loss values
2. Same gradient range as FP32 (no underflow)
3. Lower precision is acceptable for neural networks
4. Simpler conversion: just truncate FP32's lower 16 bits

Hardware Support:
- NVIDIA Ampere (A100) and newer
- Google TPUs (original inventor)
- AMD MI200+
- Intel Sapphire Rapids
"""


def bf16_properties():
    """
    Demonstrate BF16 properties and advantages.
    """
    print("\n" + "="*70)
    print("BF16 (BRAIN FLOAT 16) PROPERTIES")
    print("="*70)
    
    finfo = torch.finfo(torch.bfloat16)
    
    print(f"""
Format: 1 sign + 8 exponent + 7 mantissa = 16 bits

Numerical Properties:
- Smallest positive normal: {finfo.tiny:.2e}
- Largest finite value: {finfo.max:.2e}
- Machine epsilon: {finfo.eps:.2e}
- Decimal precision: ~2.4 digits

Key Advantage: SAME DYNAMIC RANGE AS FP32!
""")
    
    print("\nComparison: FP16 vs BF16 Range")
    print("-" * 50)
    
    # Large value comparison
    large = torch.tensor(100000.0, dtype=torch.float32)
    fp16_large = large.to(torch.float16)
    bf16_large = large.to(torch.bfloat16)
    
    print(f"Original FP32: {large.item()}")
    print(f"Converted to FP16: {fp16_large.item()} {'(OVERFLOW!)' if torch.isinf(fp16_large) else ''}")
    print(f"Converted to BF16: {bf16_large.item()}")
    
    print("\nComparison: FP16 vs BF16 Precision")
    print("-" * 50)
    
    # Precision comparison
    precise = torch.tensor(1.234567, dtype=torch.float32)
    fp16_precise = precise.to(torch.float16)
    bf16_precise = precise.to(torch.bfloat16)
    
    print(f"Original FP32: {precise.item():.7f}")
    print(f"Converted to FP16: {fp16_precise.item():.7f}")
    print(f"Converted to BF16: {bf16_precise.item():.7f}")
    print("\nNote: BF16 has LESS precision but this is acceptable for DL")
    
    print("\nBF16 Conversion is Simple")
    print("-" * 50)
    print("""
BF16 is literally the upper 16 bits of FP32:

FP32: [Sign(1)][Exponent(8)][Mantissa(23)]
                    ↓
BF16: [Sign(1)][Exponent(8)][Mantissa(7)] + [Truncated(16)]

This makes conversion very fast - just bit manipulation!
""")
    
    print("\nWhen to Use BF16 vs FP16:")
    print("-" * 50)
    print("""
Use BF16 when:
✓ Training (loss values need range)
✓ Large gradients possible
✓ Hardware supports it (Ampere+)
✓ You want simpler code (no loss scaling needed)

Use FP16 when:
✓ Inference only (no overflow from training)
✓ Older hardware (pre-Ampere)
✓ Need slightly better precision
✓ Memory is critical (same as BF16, but works on older GPUs)
""")


# =============================================================================
# SECTION 5: FP8 - THE NEXT FRONTIER
# =============================================================================
"""
FP8 (8-bit Floating Point):
════════════════════════════

Two main variants introduced in NVIDIA Hopper (H100):

1. E4M3 (4 exponent, 3 mantissa):
   - Range: ±448
   - Better precision
   - Typically used for weights and activations
   
2. E5M2 (5 exponent, 2 mantissa):
   - Range: ±57,344
   - Better range
   - Typically used for gradients

Memory: 1 byte per parameter (75% less than FP32!)

CRITICAL: FP8 requires careful handling:
- Per-tensor or per-channel scaling required
- Dynamic scaling during training
- Usually only for matrix multiplications
- Accumulation still in higher precision
"""


def fp8_properties():
    """
    Demonstrate FP8 properties and variants.
    """
    print("\n" + "="*70)
    print("FP8 (8-BIT FLOATING POINT) PROPERTIES")
    print("="*70)
    
    # Check if FP8 is available
    fp8_available = hasattr(torch, 'float8_e4m3fn')
    
    print(f"""
FP8 Availability: {'Yes' if fp8_available else 'No'} (Requires PyTorch 2.1+, CUDA 12+)

Two Variants:
┌─────────────────────────────────────────────────────────────┐
│  E4M3 (float8_e4m3fn)                                       │
│  ├── Structure: 1 sign + 4 exponent + 3 mantissa            │
│  ├── Range: ±448                                            │
│  ├── Use: Weights, forward activations                      │
│  └── Better precision for stable values                     │
├─────────────────────────────────────────────────────────────┤
│  E5M2 (float8_e5m2)                                         │
│  ├── Structure: 1 sign + 5 exponent + 2 mantissa            │
│  ├── Range: ±57,344                                         │
│  ├── Use: Gradients, backward pass                          │
│  └── Better range for varying gradients                     │
└─────────────────────────────────────────────────────────────┘

Memory Savings:
- Per parameter: 1 byte (75% reduction from FP32)
- 7B parameters: 7 GB (vs 28 GB FP32)
- 70B parameters: 70 GB (vs 280 GB FP32)
""")
    
    if fp8_available:
        print("\nFP8 Type Properties:")
        print("-" * 50)
        
        # E4M3
        e4m3_info = torch.finfo(torch.float8_e4m3fn)
        print(f"E4M3: max={e4m3_info.max}, min={e4m3_info.tiny}")
        
        # E5M2
        e5m2_info = torch.finfo(torch.float8_e5m2)
        print(f"E5M2: max={e5m2_info.max}, min={e5m2_info.tiny}")
        
        print("\nFP8 Conversion Example:")
        print("-" * 50)
        
        # Create FP32 tensor
        x = torch.tensor([1.0, 2.0, 100.0, 0.001], dtype=torch.float32)
        print(f"Original FP32: {x.tolist()}")
        
        # Convert to E4M3
        x_e4m3 = x.to(torch.float8_e4m3fn)
        x_back = x_e4m3.to(torch.float32)
        print(f"E4M3 -> FP32: {x_back.tolist()}")
    
    print("\nFP8 Training Strategy:")
    print("-" * 50)
    print("""
┌────────────────────────────────────────────────────────────┐
│                   FP8 Training Pipeline                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Weights (E4M3)  ─┐                                        │
│                   ├──► MatMul (Tensor Cores) ──► FP16/BF16 │
│  Activations     ─┘        ↓                               │
│  (E4M3)                    │                               │
│                            ▼                               │
│                    Accumulation (FP32)                     │
│                            │                               │
│                            ▼                               │
│                    Gradients (E5M2)                        │
│                            │                               │
│                            ▼                               │
│                  Optimizer Step (FP32)                     │
│                                                             │
└────────────────────────────────────────────────────────────┘

Key Requirements:
1. Per-tensor scaling factors (amax tracking)
2. Delayed scaling (use previous iteration's scale)
3. History-based scaling for stability
4. Accumulation in higher precision
""")


# =============================================================================
# SECTION 6: NUMERICAL STABILITY DEEP DIVE
# =============================================================================
"""
NUMERICAL STABILITY IN MIXED PRECISION:
═══════════════════════════════════════

Common Issues and Solutions:

1. OVERFLOW (value too large):
   - Loss explosion in FP16
   - Solution: Use BF16 or loss scaling
   
2. UNDERFLOW (value too small becomes 0):
   - Small gradients vanish
   - Solution: Loss scaling (multiply by large constant)
   
3. PRECISION LOSS (updates ignored):
   - weight + tiny_grad = weight
   - Solution: Master weights in FP32
   
4. NaN/Inf PROPAGATION:
   - One bad value corrupts everything
   - Solution: Gradient clipping, skip bad steps
"""


def demonstrate_numerical_issues():
    """
    Demonstrate common numerical stability issues.
    """
    print("\n" + "="*70)
    print("NUMERICAL STABILITY ISSUES IN MIXED PRECISION")
    print("="*70)
    
    print("\n1. GRADIENT UNDERFLOW")
    print("-" * 50)
    
    # Simulate small gradients in training
    small_grads = torch.tensor([1e-5, 1e-6, 1e-7, 1e-8], dtype=torch.float32)
    print(f"FP32 gradients: {small_grads.tolist()}")
    
    fp16_grads = small_grads.to(torch.float16)
    print(f"FP16 gradients: {fp16_grads.tolist()}")
    print("Note: Very small gradients become 0 in FP16!")
    
    # Solution: Loss scaling
    scale = 1024.0
    scaled_grads = small_grads * scale
    fp16_scaled = scaled_grads.to(torch.float16)
    unscaled = fp16_scaled / scale
    print(f"\nWith loss scaling (scale={scale}):")
    print(f"  Scaled then FP16: {fp16_scaled.tolist()}")
    print(f"  After unscaling: {unscaled.tolist()}")
    
    print("\n2. WEIGHT UPDATE PRECISION LOSS")
    print("-" * 50)
    
    # Large weight + small update = original weight (bad!)
    weight_fp16 = torch.tensor(100.0, dtype=torch.float16)
    update_fp16 = torch.tensor(0.001, dtype=torch.float16)
    
    new_weight = weight_fp16 + update_fp16
    print(f"Weight: {weight_fp16.item()}")
    print(f"Update: {update_fp16.item()}")
    print(f"Weight + Update: {new_weight.item()}")
    print(f"Update lost: {new_weight.item() == weight_fp16.item()}")
    
    # Solution: FP32 master weights
    weight_fp32 = torch.tensor(100.0, dtype=torch.float32)
    update_fp32 = torch.tensor(0.001, dtype=torch.float32)
    
    new_weight_fp32 = weight_fp32 + update_fp32
    print(f"\nWith FP32 master weights:")
    print(f"Weight + Update: {new_weight_fp32.item()}")
    print(f"Update preserved: {new_weight_fp32.item() != weight_fp32.item()}")
    
    print("\n3. SOFTMAX NUMERICAL STABILITY")
    print("-" * 50)
    
    # Unstable softmax
    logits = torch.tensor([1000.0, 1001.0, 1002.0], dtype=torch.float16)
    
    # Naive softmax overflows
    try:
        naive_exp = torch.exp(logits)
        print(f"Naive exp({logits.tolist()}) = {naive_exp.tolist()}")
    except:
        print("Naive exp overflows!")
    
    # Stable softmax (subtract max)
    logits_stable = logits - logits.max()
    stable_exp = torch.exp(logits_stable)
    softmax_result = stable_exp / stable_exp.sum()
    print(f"Stable softmax: {softmax_result.tolist()}")
    
    print("\n4. LAYER NORM STABILITY")
    print("-" * 50)
    print("""
Layer Normalization can be unstable in FP16:

    y = (x - mean) / sqrt(var + eps)

Issues:
- Variance calculation loses precision
- Division by small values
- Subtraction of similar values

Solution:
- Compute mean/var in FP32
- Or use RMSNorm (simpler, more stable)
""")


# =============================================================================
# SECTION 7: FORMAT SELECTION GUIDE
# =============================================================================

def format_selection_guide():
    """
    Comprehensive guide for choosing the right format.
    """
    print("\n" + "="*70)
    print("FLOATING POINT FORMAT SELECTION GUIDE")
    print("="*70)
    
    print("""
DECISION TREE:
══════════════

                        ┌─────────────────┐
                        │   What's your   │
                        │     use case?   │
                        └────────┬────────┘
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌─────────┐        ┌─────────┐        ┌─────────┐
        │Training │        │Inference│        │  Both   │
        └────┬────┘        └────┬────┘        └────┬────┘
             │                  │                  │
             ▼                  ▼                  ▼
    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │  GPU type?     │  │ Latency/Memory │  │  BF16 if       │
    │                │  │ priority?      │  │  hardware      │
    └───────┬────────┘  └───────┬────────┘  │  supports      │
    ┌───────┴───────┐   ┌───────┴───────┐   └────────────────┘
    ▼               ▼   ▼               ▼
┌───────┐     ┌───────┐┌───────┐   ┌───────┐
│Ampere+│     │Older  ││Memory │   │Latency│
│ Use   │     │ Use   ││ Use   │   │ Use   │
│ BF16  │     │FP16+  ││ FP8   │   │FP16/  │
│       │     │ AMP   ││       │   │ BF16  │
└───────┘     └───────┘└───────┘   └───────┘


DETAILED RECOMMENDATIONS:
═════════════════════════

TRAINING:
┌──────────────────┬────────────────────────────────────────────┐
│ Scenario         │ Recommendation                              │
├──────────────────┼────────────────────────────────────────────┤
│ A100/H100        │ BF16 pure (no loss scaling needed)         │
│ V100/older       │ FP16 + AMP with dynamic loss scaling       │
│ Memory critical  │ FP8 (H100) or gradient checkpointing       │
│ Unstable training│ FP32 for debugging, then switch to mixed   │
└──────────────────┴────────────────────────────────────────────┘

INFERENCE:
┌──────────────────┬────────────────────────────────────────────┐
│ Scenario         │ Recommendation                              │
├──────────────────┼────────────────────────────────────────────┤
│ Max throughput   │ FP8 (if hardware supports) or FP16         │
│ Quality critical │ BF16 or FP32                                │
│ Edge deployment  │ INT8 quantization (not float)              │
│ Memory limited   │ FP16 + quantized weights (INT4/8)          │
└──────────────────┴────────────────────────────────────────────┘

MIXED PRECISION STRATEGY:
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Master Weights: FP32 (optimizer states)                     │
│        ↓                                                      │
│  Forward Pass: BF16/FP16 (compute + activations)             │
│        ↓                                                      │
│  Loss: FP32 (for stability)                                  │
│        ↓                                                      │
│  Backward Pass: BF16/FP16 (gradients)                        │
│        ↓                                                      │
│  Gradient Accumulation: FP32 (precision)                     │
│        ↓                                                      │
│  Weight Update: FP32 (master weights)                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# SECTION 8: PRACTICAL EXAMPLES
# =============================================================================

def practical_conversion_examples():
    """
    Practical examples of format conversion and usage.
    """
    print("\n" + "="*70)
    print("PRACTICAL FORMAT CONVERSION EXAMPLES")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n1. MODEL WEIGHT CONVERSION")
    print("-" * 50)
    
    # Simulate model weights
    weights_fp32 = torch.randn(1000, 1000, device=device)
    
    # Convert to different formats
    weights_fp16 = weights_fp32.to(torch.float16)
    weights_bf16 = weights_fp32.to(torch.bfloat16)
    
    # Memory comparison
    print(f"FP32 memory: {weights_fp32.element_size() * weights_fp32.numel() / 1e6:.2f} MB")
    print(f"FP16 memory: {weights_fp16.element_size() * weights_fp16.numel() / 1e6:.2f} MB")
    print(f"BF16 memory: {weights_bf16.element_size() * weights_bf16.numel() / 1e6:.2f} MB")
    
    # Accuracy comparison
    print(f"\nReconstruction error (FP16): {(weights_fp32 - weights_fp16.float()).abs().mean():.2e}")
    print(f"Reconstruction error (BF16): {(weights_fp32 - weights_bf16.float()).abs().mean():.2e}")
    
    print("\n2. MATMUL PRECISION")
    print("-" * 50)
    
    # Matrix multiplication in different precisions
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    
    # FP32 matmul (reference)
    c_fp32 = torch.matmul(a, b)
    
    # FP16 matmul
    c_fp16 = torch.matmul(a.half(), b.half())
    
    # BF16 matmul
    c_bf16 = torch.matmul(a.bfloat16(), b.bfloat16())
    
    print(f"Matmul error (FP16 vs FP32): {(c_fp32 - c_fp16.float()).abs().mean():.2e}")
    print(f"Matmul error (BF16 vs FP32): {(c_fp32 - c_bf16.float()).abs().mean():.2e}")
    
    print("\n3. CHECKING HARDWARE SUPPORT")
    print("-" * 50)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        
        # BF16 support (SM 80+)
        bf16_support = props.major >= 8
        print(f"BF16 Tensor Core support: {'Yes' if bf16_support else 'No'}")
        
        # FP8 support (SM 89+, Hopper)
        fp8_support = props.major >= 9 or (props.major == 8 and props.minor >= 9)
        print(f"FP8 support: {'Yes' if fp8_support else 'No'}")
    else:
        print("CUDA not available")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FLOATING POINT FORMATS FOR DEEP LEARNING")
    print("="*70)
    
    # Demonstrate all formats
    demonstrate_float_formats()
    
    # FP32 properties
    fp32_properties()
    
    # FP16 properties and limitations
    fp16_properties()
    
    # BF16 advantages
    bf16_properties()
    
    # FP8 next generation
    fp8_properties()
    
    # Numerical stability issues
    demonstrate_numerical_issues()
    
    # Format selection guide
    format_selection_guide()
    
    # Practical examples
    practical_conversion_examples()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
