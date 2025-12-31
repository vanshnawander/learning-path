"""
03_quantization_kernels.py - Quantization Kernels in Triton

This module covers Triton implementations of quantization kernels
for efficient LLM inference and training.

Key Topics:
1. INT8 Quantization Kernels
2. FP8 (E4M3/E5M2) Kernels
3. INT4/NF4 Dequantization (QLoRA style)
4. Dynamic vs Static Quantization
5. Per-channel vs Per-tensor Quantization

These are essential for:
- Fast inference with quantized models
- QLoRA training with on-the-fly dequantization
- Memory-efficient serving

Run: python 03_quantization_kernels.py
"""

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional

# ============================================================================
# PROFILING
# ============================================================================

def profile_kernel(func, warmup=25, iterations=100):
    """Profile a kernel."""
    if not torch.cuda.is_available():
        return 0.0
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations

# ============================================================================
# SECTION 1: INT8 QUANTIZATION FUNDAMENTALS
# ============================================================================
"""
INT8 QUANTIZATION:
==================

Affine (asymmetric) quantization:
    x_int8 = round((x - zero_point) / scale)
    x_dequant = x_int8 * scale + zero_point

Symmetric quantization (common for weights):
    x_int8 = round(x / scale)
    x_dequant = x_int8 * scale

Scale computation:
    scale = (max(x) - min(x)) / 255  # Affine
    scale = max(|x|) / 127           # Symmetric

WHY INT8?
- 4x memory reduction vs FP32
- 2x memory reduction vs FP16
- Tensor Core INT8 operations (2x throughput vs FP16)
"""

@triton.jit
def quantize_int8_kernel(
    output_ptr,
    scale_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-tensor symmetric INT8 quantization.
    
    x_int8 = round(x / scale)
    where scale = max(|x|) / 127
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load precomputed scale
    scale = tl.load(scale_ptr)
    
    # Quantize: round to nearest
    x_scaled = x / scale
    x_int8 = tl.libdevice.rint(x_scaled)
    
    # Clamp to INT8 range [-127, 127] (symmetric)
    x_int8 = tl.maximum(tl.minimum(x_int8, 127.0), -127.0)
    
    # Store as int8
    tl.store(output_ptr + offsets, x_int8.to(tl.int8), mask=mask)


@triton.jit
def dequantize_int8_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    INT8 dequantization.
    
    x_float = x_int8 * scale
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load quantized values
    x_int8 = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Load scale
    scale = tl.load(scale_ptr)
    
    # Dequantize
    x_float = x_int8.to(tl.float32) * scale
    
    tl.store(output_ptr + offsets, x_float, mask=mask)


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8."""
    # Compute scale
    scale = x.abs().max() / 127.0
    scale = scale.unsqueeze(0)  # Make it a tensor
    
    # Allocate output
    output = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    quantize_int8_kernel[grid](
        output, scale, x.view(-1), n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, scale


def dequantize_int8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor."""
    output = torch.empty(x_int8.shape, dtype=torch.float32, device=x_int8.device)
    
    n_elements = x_int8.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    dequantize_int8_kernel[grid](
        output, x_int8.view(-1), scale, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# SECTION 2: INT8 MATMUL WITH ON-THE-FLY DEQUANTIZATION
# ============================================================================
"""
INT8 MATMUL FOR INFERENCE:
==========================

For quantized inference:
    1. Weights stored as INT8
    2. Activations quantized dynamically
    3. Matmul in INT8 or with dequantization
    
Two approaches:
    A. True INT8 matmul (needs Tensor Cores)
    B. Dequantize-on-the-fly (more flexible)

We implement B here - useful for QLoRA-style inference.
"""

@triton.jit
def int8_matmul_dequant_kernel(
    output_ptr,
    a_ptr,  # FP16 activations
    b_int8_ptr,  # INT8 weights
    scale_ptr,  # Per-column scales
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Matrix multiply with INT8 weights and on-the-fly dequantization.
    
    C = A @ dequant(B_int8)
    
    A: (M, K) FP16
    B: (K, N) INT8 with per-column scales
    C: (M, N) FP16
    
    This is used in QLoRA inference!
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load scales for this N block (per-column quantization)
    scales = tl.load(scale_ptr + n_offs, mask=n_offs < N, other=1.0)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        # Load A block (FP16)
        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block (INT8) and dequantize
        b_ptrs = b_int8_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0)
        
        # Dequantize: multiply by scale
        b_fp32 = b_int8.to(tl.float32) * scales[None, :]
        
        # Accumulate
        acc += tl.dot(a.to(tl.float32), b_fp32)
    
    # Store result
    out_ptrs = output_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


# ============================================================================
# SECTION 3: INT4/NF4 DEQUANTIZATION (QLoRA Style)
# ============================================================================
"""
4-BIT QUANTIZATION (QLoRA):
===========================

NF4 (NormalFloat4):
    - 16 quantization levels optimized for normal distribution
    - Used in QLoRA for base model weights
    
INT4:
    - Standard 4-bit integer quantization
    - Range: [-8, 7] or [0, 15]

Storage: 2 values packed per byte
    byte = (val1 << 4) | (val2 & 0xF)

Dequantization lookup table:
    NF4_VALUES = [-1.0, -0.6962, ..., 0.7230, 1.0]
    x_float = NF4_VALUES[x_4bit] * scale
"""

# NF4 quantization levels (information-theoretically optimal for N(0,1))
NF4_VALUES = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


@triton.jit
def dequantize_nf4_kernel(
    output_ptr,
    packed_ptr,  # INT8 with 2 NF4 values per byte
    scale_ptr,
    lut_ptr,  # NF4 lookup table
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantize NF4 (4-bit NormalFloat) values.
    
    Each byte contains 2 NF4 values:
        byte = (high_nibble << 4) | low_nibble
    
    Dequantization:
        x_float = LUT[x_4bit] * scale
    """
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE output elements
    # But packed data has BLOCK_SIZE/2 bytes
    output_offset = pid * BLOCK_SIZE
    packed_offset = pid * (BLOCK_SIZE // 2)
    
    # Process pairs
    for i in range(0, BLOCK_SIZE, 2):
        out_idx = output_offset + i
        pack_idx = packed_offset + i // 2
        
        if out_idx >= n_elements:
            break
        
        # Load packed byte
        packed_byte = tl.load(packed_ptr + pack_idx)
        
        # Extract nibbles
        low_nibble = packed_byte & 0xF
        high_nibble = (packed_byte >> 4) & 0xF
        
        # Load scale
        scale = tl.load(scale_ptr)
        
        # Lookup and dequantize
        # Note: In real implementation, LUT would be in shared memory
        low_val = tl.load(lut_ptr + low_nibble) * scale
        high_val = tl.load(lut_ptr + high_nibble) * scale
        
        # Store
        tl.store(output_ptr + out_idx, low_val)
        if out_idx + 1 < n_elements:
            tl.store(output_ptr + out_idx + 1, high_val)


@triton.jit
def fused_nf4_linear_kernel(
    output_ptr,
    input_ptr,  # FP16 input
    weight_packed_ptr,  # NF4 packed weights
    scale_ptr,  # Per-block scales
    M, N, K,
    block_size,  # Quantization block size (typically 64)
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused NF4 dequantization and linear layer.
    
    output = input @ dequant(weight_nf4)
    
    This is the core operation for QLoRA inference!
    Weights are dequantized on-the-fly, never stored in FP16.
    """
    # NF4 lookup table (embedded as constants)
    # In practice, would use shared memory
    nf4_lut_0 = -1.0
    nf4_lut_1 = -0.6962
    nf4_lut_2 = -0.5251
    nf4_lut_3 = -0.3949
    nf4_lut_4 = -0.2844
    nf4_lut_5 = -0.1848
    nf4_lut_6 = -0.0911
    nf4_lut_7 = 0.0
    nf4_lut_8 = 0.0796
    nf4_lut_9 = 0.1609
    nf4_lut_10 = 0.2461
    nf4_lut_11 = 0.3379
    nf4_lut_12 = 0.4407
    nf4_lut_13 = 0.5626
    nf4_lut_14 = 0.7230
    nf4_lut_15 = 1.0
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Simplified: actual implementation would handle packed format
    # This shows the concept of fused dequant + matmul
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        # Load input
        a_ptrs = input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # In real implementation: load packed weights and dequantize
        # For now, assume weights are already unpacked
        
        # This is a placeholder - real impl would:
        # 1. Load packed INT4 bytes
        # 2. Unpack to get indices
        # 3. Use LUT to get float values
        # 4. Multiply by block-wise scales
        
    # Store result
    out_ptrs = output_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


# ============================================================================
# SECTION 4: FP8 QUANTIZATION (HOPPER+)
# ============================================================================
"""
FP8 QUANTIZATION (H100/Hopper):
===============================

Two FP8 formats:
    E4M3: 4 exponent bits, 3 mantissa bits
        - Range: ~±240
        - Good for forward pass (weights, activations)
    
    E5M2: 5 exponent bits, 2 mantissa bits  
        - Range: ~±57000
        - Good for backward pass (gradients)

FP8 vs INT8:
    - FP8 handles dynamic range better
    - No need for separate scaling (mostly)
    - Native Tensor Core support on Hopper

Usage in training:
    - Forward: FP8 E4M3 for weights and activations
    - Backward: FP8 E5M2 for gradients
"""

@triton.jit
def quantize_fp8_e4m3_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize FP32/FP16 to FP8 E4M3.
    
    Note: Triton's native FP8 support varies by version.
    This shows the conceptual approach.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load scale (for amax scaling)
    scale = tl.load(scale_ptr)
    
    # Scale to FP8 range
    x_scaled = x / scale
    
    # Clamp to E4M3 range (approximately ±240)
    x_clamped = tl.maximum(tl.minimum(x_scaled, 240.0), -240.0)
    
    # In Triton with FP8 support:
    # tl.store(output_ptr + offsets, x_clamped.to(tl.float8e4m3fn), mask=mask)
    
    # For now, store as float (actual impl depends on Triton version)
    tl.store(output_ptr + offsets, x_clamped, mask=mask)


# ============================================================================
# SECTION 5: DYNAMIC QUANTIZATION KERNEL
# ============================================================================
"""
DYNAMIC QUANTIZATION:
=====================

For activations, we quantize dynamically at runtime:
1. Compute min/max of the tensor
2. Compute scale and zero point
3. Quantize

This is fused into a single kernel for efficiency.
"""

@triton.jit
def dynamic_quantize_int8_kernel(
    output_ptr,
    scale_ptr,
    input_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dynamic per-row INT8 quantization.
    
    For each row:
    1. Find max(|x|)
    2. scale = max(|x|) / 127
    3. x_int8 = round(x / scale)
    
    Used for activation quantization in inference.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    
    # First pass: find max absolute value
    max_val = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
        block_max = tl.max(tl.abs(x), axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Compute scale
    scale = max_val / 127.0
    scale = tl.maximum(scale, 1e-8)  # Avoid division by zero
    
    # Store scale
    tl.store(scale_ptr + row_idx, scale)
    
    # Second pass: quantize
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
        x_scaled = x / scale
        x_int8 = tl.libdevice.rint(x_scaled)
        x_int8 = tl.maximum(tl.minimum(x_int8, 127.0), -127.0)
        
        tl.store(output_ptr + row_start + col_offsets, x_int8.to(tl.int8), mask=mask)


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_int8_quantization():
    """Test INT8 quantization accuracy and speed."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: INT8 QUANTIZATION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Test tensor
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    
    # Quantize
    x_int8, scale = quantize_int8(x)
    
    # Dequantize
    x_reconstructed = dequantize_int8(x_int8, scale)
    
    # Error analysis
    abs_error = (x - x_reconstructed).abs()
    rel_error = abs_error / (x.abs() + 1e-8)
    
    print(f"\n Tensor shape: {x.shape}")
    print(f" Original dtype: {x.dtype}, size: {x.numel() * 4 / 1e6:.1f} MB")
    print(f" Quantized dtype: {x_int8.dtype}, size: {x_int8.numel() * 1 / 1e6:.1f} MB")
    print(f" Compression: {x.numel() * 4 / (x_int8.numel() * 1):.1f}x")
    
    print(f"\n Reconstruction error:")
    print(f"   Max absolute error: {abs_error.max().item():.4f}")
    print(f"   Mean absolute error: {abs_error.mean().item():.4f}")
    print(f"   Mean relative error: {rel_error.mean().item() * 100:.2f}%")
    
    # Benchmark
    time_quant = profile_kernel(lambda: quantize_int8(x), iterations=50)
    time_dequant = profile_kernel(lambda: dequantize_int8(x_int8, scale), iterations=50)
    
    print(f"\n Performance:")
    print(f"   Quantization: {time_quant:.3f} ms")
    print(f"   Dequantization: {time_dequant:.3f} ms")


def experiment_nf4_concept():
    """Demonstrate NF4 quantization concept."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: NF4 (4-BIT NORMALFLOAT) CONCEPT")
    print("="*70)
    
    print("""
    NF4 QUANTIZATION (Used in QLoRA):
    ══════════════════════════════════
    
    Standard INT4 levels: -8, -7, ..., 0, ..., 6, 7
        - Uniformly spaced
        - Wastes bits on unlikely values
    
    NF4 levels (optimized for N(0,1)):
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000
    
    These values minimize quantization error for normally
    distributed data (which neural network weights typically are!).
    
    MEMORY SAVINGS:
    ═══════════════
    
    7B parameter model:
        FP32: 28 GB
        FP16: 14 GB
        INT8: 7 GB
        INT4/NF4: 3.5 GB (+ scales)
    
    QLoRA uses NF4 for frozen base weights!
    """)
    
    if not torch.cuda.is_available():
        return
    
    # Simulate NF4 quantization
    nf4_levels = NF4_VALUES.cuda()
    
    # Create normally distributed tensor
    x = torch.randn(1000000, device='cuda')
    
    # Find nearest NF4 level for each element
    x_normalized = x / x.abs().max()  # Normalize to [-1, 1]
    
    # Compute distances to all NF4 levels
    distances = (x_normalized.unsqueeze(1) - nf4_levels.unsqueeze(0)).abs()
    indices = distances.argmin(dim=1)
    
    # Reconstruct
    x_nf4 = nf4_levels[indices] * x.abs().max()
    
    # Error
    error = (x - x_nf4).abs()
    
    print(f"\n NF4 quantization simulation:")
    print(f"   Max error: {error.max().item():.4f}")
    print(f"   Mean error: {error.mean().item():.4f}")
    print(f"   Memory reduction: {32/4:.0f}x (FP32 → 4-bit)")


def print_quantization_summary():
    """Print quantization kernels summary."""
    print("\n" + "="*70)
    print(" QUANTIZATION KERNELS SUMMARY")
    print("="*70)
    
    print("""
    QUANTIZATION FORMATS FOR LLMs:
    ══════════════════════════════
    
    ┌──────────────────────────────────────────────────────────────────┐
    │ Format │ Bits │ Range        │ Use Case                         │
    ├──────────────────────────────────────────────────────────────────┤
    │ FP32   │ 32   │ ±3.4e38      │ Training (master weights)        │
    │ FP16   │ 16   │ ±65504       │ Training (mixed precision)       │
    │ BF16   │ 16   │ ±3.4e38      │ Training (better for LLMs)       │
    │ FP8    │ 8    │ ±240/57000   │ Training (Hopper), inference     │
    │ INT8   │ 8    │ -128 to 127  │ Inference (good quality)         │
    │ INT4   │ 4    │ -8 to 7      │ Inference (more compression)     │
    │ NF4    │ 4    │ -1 to 1 (LUT)│ QLoRA (optimal for weights)      │
    └──────────────────────────────────────────────────────────────────┘
    
    TRITON QUANTIZATION PATTERNS:
    ═════════════════════════════
    
    1. Static Quantization (weights):
       - Quantize once, store quantized
       - Dequantize on-the-fly during inference
    
    2. Dynamic Quantization (activations):
       - Quantize at runtime
       - Per-tensor or per-row scaling
    
    3. Fused Dequant + Matmul:
       - Never materialize dequantized weights
       - Memory-efficient inference
    
    KERNEL OPTIMIZATION TIPS:
    ═════════════════════════
    
    1. Use lookup tables for NF4/INT4
       - Store LUT in shared memory
       - Fast random access
    
    2. Pack 4-bit values
       - 2 values per byte
       - Use bit manipulation to unpack
    
    3. Block-wise quantization
       - Different scale per block (32-128 elements)
       - Better accuracy than per-tensor
    
    4. Fuse dequantization with compute
       - Don't store intermediate FP16
       - Saves memory bandwidth
    
    USING QUANTIZATION:
    ═══════════════════
    
    # QLoRA inference (Unsloth)
    from unsloth import FastLanguageModel
    model, _ = FastLanguageModel.from_pretrained(
        "unsloth/llama-2-7b-bnb-4bit",  # 4-bit model
        load_in_4bit=True,
    )
    
    # bitsandbytes INT8
    import bitsandbytes as bnb
    linear_8bit = bnb.nn.Linear8bitLt(in_features, out_features)
    
    # GPTQ/AWQ quantization
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized("model-4bit-gptq")
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " QUANTIZATION KERNELS IN TRITON ".center(68) + "║")
    print("║" + " INT8, FP8, INT4/NF4 for efficient inference ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_int8_quantization()
    experiment_nf4_concept()
    print_quantization_summary()
    
    print("\n" + "="*70)
    print(" Quantization is key to efficient LLM deployment!")
    print("="*70)
