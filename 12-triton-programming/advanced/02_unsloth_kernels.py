"""
02_unsloth_kernels.py - Production Triton Kernels (Unsloth Style)

This module covers the actual Triton kernel patterns used in production
libraries like Unsloth for efficient LLM training and inference.

Unsloth achieves 2-5x speedups through:
1. Fused RMSNorm with residual
2. Fused Cross-Entropy Loss (chunked)
3. Fused RoPE (Rotary Position Embedding)
4. Fused LoRA forward/backward
5. Fused SwiGLU MLP

These kernels are the SECRET SAUCE of fast LLM training!

Run: python 02_unsloth_kernels.py
Requirements: triton, torch
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

def profile_kernel(func, warmup=25, iterations=100):
    """Profile a CUDA kernel."""
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
# KERNEL 1: FUSED RMSNORM + RESIDUAL (Unsloth Style)
# ============================================================================
"""
UNSLOTH RMSNORM OPTIMIZATION:
=============================

Standard RMSNorm (3 kernels):
    1. residual_add: hidden = hidden + residual
    2. rms_compute: rms = sqrt(mean(hidden²))
    3. normalize: output = hidden / rms * weight

Fused RMSNorm (1 kernel):
    - Load hidden and residual once
    - Add, compute RMS, normalize, scale - all in registers
    - Single write to output
    
Memory savings: 3x fewer memory passes
"""

@triton.jit
def fused_rmsnorm_residual_kernel(
    output_ptr,
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    hidden_stride,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm with residual connection.
    
    output = RMSNorm(hidden + residual) * weight
    
    This is exactly what Unsloth does for each transformer layer!
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Pointer to start of this row
    row_start = row_idx * hidden_stride
    
    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load hidden and residual (fused memory access)
    hidden = tl.load(hidden_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Fused add (in registers, no memory write!)
    x = hidden + residual
    
    # Compute RMS in FP32 for numerical stability
    x_fp32 = x.to(tl.float32)
    variance = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(variance + eps)
    
    # Normalize
    x_normed = x_fp32 * inv_rms
    
    # Load weight and apply
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = x_normed * weight
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, output.to(hidden.dtype), mask=mask)


@triton.jit
def fused_rmsnorm_residual_backward_kernel(
    grad_output_ptr,
    grad_hidden_ptr,
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    hidden_stride,
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for fused RMSNorm + residual.
    
    Computes gradients with respect to hidden (before residual add).
    The gradient flows back to both hidden and residual paths.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load all needed values
    grad_out = tl.load(grad_output_ptr + row_start + col_offsets, mask=mask, other=0.0)
    hidden = tl.load(hidden_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    
    # Forward recomputation
    x = (hidden + residual).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(variance + eps)
    x_normed = x * inv_rms
    
    # Backward through scale
    grad_normed = grad_out.to(tl.float32) * weight
    
    # Backward through normalization
    # d(x/rms)/dx = 1/rms - x²/(rms³ * n_cols)
    grad_var = -0.5 * tl.sum(grad_normed * x_normed, axis=0) * inv_rms
    grad_x = grad_normed * inv_rms + 2.0 * grad_var * x / n_cols
    
    # Store gradient (same grad goes to both hidden and residual)
    tl.store(grad_hidden_ptr + row_start + col_offsets, grad_x.to(hidden.dtype), mask=mask)


def fused_rmsnorm_residual(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Wrapper for fused RMSNorm + residual."""
    n_rows, n_cols = hidden.shape
    output = torch.empty_like(hidden)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    
    grid = (n_rows,)
    
    fused_rmsnorm_residual_kernel[grid](
        output, hidden, residual, weight,
        hidden.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# KERNEL 2: FUSED CROSS-ENTROPY LOSS (Chunked/Online)
# ============================================================================
"""
UNSLOTH CROSS-ENTROPY OPTIMIZATION:
===================================

Problem: For LLMs, logits tensor is HUGE
    batch=4, seq=2048, vocab=32000
    logits = 4 * 2048 * 32000 * 2 bytes = 512 MB!

Standard approach:
    1. Compute full logits = hidden @ lm_head.T  (512 MB)
    2. Apply softmax                             (512 MB)
    3. Compute loss                              (scalar)
    
Unsloth's chunked approach:
    1. Process vocabulary in chunks of 65536
    2. Never materialize full logits
    3. Use online logsumexp for stability
    
Memory savings: Up to 10x for large vocabularies!
"""

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    logits_ptr,
    labels_ptr,
    logits_row_stride,
    n_cols,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy forward pass.
    
    For each row:
    1. Compute logsumexp (online algorithm for stability)
    2. Get logit at label position
    3. loss = logsumexp - logit_at_label
    
    This avoids materializing the full softmax!
    """
    row_idx = tl.program_id(0)
    
    # Load label for this row
    label = tl.load(labels_ptr + row_idx)
    
    # Check for ignored index
    if label == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        return
    
    # Pointer to logits row
    row_start = logits_ptr + row_idx * logits_row_stride
    
    # Online logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    # Process in chunks to handle arbitrary vocab size
    
    # First pass: find max
    max_val = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        block = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(block, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        block = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(block - max_val), axis=0)
    
    # logsumexp = max + log(sum_exp)
    logsumexp = max_val + tl.log(sum_exp)
    
    # Load logit at label position
    logit_at_label = tl.load(row_start + label)
    
    # Cross-entropy loss = logsumexp - logit_at_label
    loss = logsumexp - logit_at_label
    
    tl.store(loss_ptr + row_idx, loss)


@triton.jit
def cross_entropy_bwd_kernel(
    grad_logits_ptr,
    logits_ptr,
    labels_ptr,
    grad_loss,
    logits_row_stride,
    n_cols,
    ignore_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy backward pass.
    
    grad_logits = softmax(logits) - one_hot(label)
    grad_logits *= grad_loss
    
    Again, never materializes full softmax - computes on-the-fly!
    """
    row_idx = tl.program_id(0)
    
    label = tl.load(labels_ptr + row_idx)
    
    if label == ignore_index:
        # Zero gradient for ignored labels
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            tl.store(grad_logits_ptr + row_idx * logits_row_stride + col_offsets, 
                    0.0, mask=mask)
        return
    
    row_start = logits_ptr + row_idx * logits_row_stride
    
    # Find max for numerical stability
    max_val = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(block, axis=0))
    
    # Compute sum of exp
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(block - max_val), axis=0)
    
    # Compute gradients: softmax - one_hot
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        block = tl.load(row_start + col_offsets, mask=mask, other=-float('inf'))
        softmax = tl.exp(block - max_val) / sum_exp
        
        # Subtract 1 at label position
        is_label = col_offsets == label
        grad = (softmax - tl.where(is_label, 1.0, 0.0)) * grad_loss
        
        tl.store(grad_logits_ptr + row_idx * logits_row_stride + col_offsets,
                grad, mask=mask)


def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Fused cross-entropy loss.
    
    This is similar to Unsloth's fast_cross_entropy_loss.
    """
    n_rows, n_cols = logits.shape
    loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    
    BLOCK_SIZE = min(4096, triton.next_power_of_2(n_cols))
    
    grid = (n_rows,)
    
    cross_entropy_fwd_kernel[grid](
        loss, logits, labels,
        logits.stride(0), n_cols,
        ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return loss.mean()


# ============================================================================
# KERNEL 3: FUSED ROPE (Rotary Position Embedding)
# ============================================================================
"""
UNSLOTH ROPE OPTIMIZATION:
==========================

RoPE applies rotation to Q and K based on position:
    q_rot = q * cos(theta) + rotate_half(q) * sin(theta)

Standard (3+ kernels):
    1. Compute theta = position * frequencies
    2. Compute cos/sin
    3. Apply rotation

Fused (1 kernel):
    - Compute everything in one pass
    - Apply to both Q and K simultaneously
"""

@triton.jit
def fused_rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    out_q_ptr, out_k_ptr,
    seq_len, head_dim,
    q_stride_batch, q_stride_head, q_stride_seq, q_stride_dim,
    k_stride_batch, k_stride_head, k_stride_seq, k_stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RoPE for both Q and K.
    
    q_rot = q * cos - q_rotated * sin
    where q_rotated swaps pairs: [q0, q1, q2, q3] -> [-q1, q0, -q3, q2]
    """
    # Program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Dimension offsets (process pairs)
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < head_dim
    
    # Load cos and sin for this position
    cos = tl.load(cos_ptr + seq_idx * head_dim + dim_offsets, mask=mask, other=1.0)
    sin = tl.load(sin_ptr + seq_idx * head_dim + dim_offsets, mask=mask, other=0.0)
    
    # Q offsets
    q_offset = (batch_idx * q_stride_batch + 
                head_idx * q_stride_head + 
                seq_idx * q_stride_seq)
    
    # K offsets
    k_offset = (batch_idx * k_stride_batch + 
                head_idx * k_stride_head + 
                seq_idx * k_stride_seq)
    
    # Load Q and K
    q = tl.load(q_ptr + q_offset + dim_offsets * q_stride_dim, mask=mask, other=0.0)
    k = tl.load(k_ptr + k_offset + dim_offsets * k_stride_dim, mask=mask, other=0.0)
    
    # Rotate: swap pairs and negate alternating
    # For [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    half_dim = head_dim // 2
    
    # Even indices: take from odd positions with negative
    # Odd indices: take from even positions
    is_even = (dim_offsets % 2) == 0
    
    # For even index i, get value at i+1 and negate
    # For odd index i, get value at i-1
    partner_idx = tl.where(is_even, dim_offsets + 1, dim_offsets - 1)
    partner_idx = tl.minimum(partner_idx, head_dim - 1)
    
    q_partner = tl.load(q_ptr + q_offset + partner_idx * q_stride_dim, mask=mask, other=0.0)
    k_partner = tl.load(k_ptr + k_offset + partner_idx * k_stride_dim, mask=mask, other=0.0)
    
    # Apply sign: negative for even indices
    sign = tl.where(is_even, -1.0, 1.0)
    q_rotated = sign * q_partner
    k_rotated = sign * k_partner
    
    # Apply RoPE: x_rot = x * cos + x_rotated * sin
    q_out = q * cos + q_rotated * sin
    k_out = k * cos + k_rotated * sin
    
    # Store
    tl.store(out_q_ptr + q_offset + dim_offsets * q_stride_dim, q_out, mask=mask)
    tl.store(out_k_ptr + k_offset + dim_offsets * k_stride_dim, k_out, mask=mask)


def fused_rope(
    q: torch.Tensor,  # (batch, heads, seq, head_dim)
    k: torch.Tensor,
    cos: torch.Tensor,  # (seq, head_dim)
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for fused RoPE."""
    batch, heads, seq_len, head_dim = q.shape
    
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    grid = (batch, heads, seq_len)
    
    fused_rope_kernel[grid](
        q, k, cos, sin, out_q, out_k,
        seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_q, out_k


# ============================================================================
# KERNEL 4: FUSED SWIGLU MLP
# ============================================================================
"""
UNSLOTH MLP OPTIMIZATION:
=========================

Llama MLP (SwiGLU):
    gate = silu(x @ W_gate)
    up = x @ W_up
    output = (gate * up) @ W_down

Standard (5+ kernels):
    1. gate = x @ W_gate
    2. gate = silu(gate)
    3. up = x @ W_up
    4. hidden = gate * up
    5. output = hidden @ W_down

Fused approach:
    1. gate_up = x @ W_gate_up  (fused projection)
    2. hidden = fused_silu_mul(gate_up)  (fused silu + multiply)
    3. output = hidden @ W_down
"""

@triton.jit
def fused_silu_mul_kernel(
    output_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SiLU activation and element-wise multiplication.
    
    output = silu(gate) * up = gate * sigmoid(gate) * up
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU = x * sigmoid(x)
    gate_fp32 = gate.to(tl.float32)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_fp32))
    silu_gate = gate_fp32 * sigmoid_gate
    
    # Multiply with up projection
    output = silu_gate * up.to(tl.float32)
    
    tl.store(output_ptr + offsets, output.to(gate.dtype), mask=mask)


@triton.jit
def fused_silu_mul_backward_kernel(
    grad_gate_ptr,
    grad_up_ptr,
    grad_output_ptr,
    gate_ptr,
    up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for fused SiLU + multiply.
    
    d(silu(g) * u)/dg = u * d(silu(g))/dg = u * (sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g)))
    d(silu(g) * u)/du = silu(g)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sigmoid
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sigmoid_gate
    
    # Gradient w.r.t. gate: d(silu)/dg = sigmoid + g * sigmoid * (1 - sigmoid)
    dsilu_dgate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = grad_out * up * dsilu_dgate
    
    # Gradient w.r.t. up: d(silu(g) * u)/du = silu(g)
    grad_up = grad_out * silu_gate
    
    tl.store(grad_gate_ptr + offsets, grad_gate.to(tl.float16), mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up.to(tl.float16), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation and multiplication."""
    output = torch.empty_like(gate)
    n_elements = gate.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_silu_mul_kernel[grid](
        output, gate, up, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# ============================================================================
# KERNEL 5: FUSED LORA FORWARD
# ============================================================================
"""
UNSLOTH LORA OPTIMIZATION:
==========================

Standard LoRA forward:
    base_out = x @ W_base.T
    lora_out = (x @ A.T) @ B.T * scale
    output = base_out + lora_out

This requires:
    1. Base matmul
    2. LoRA A matmul
    3. LoRA B matmul
    4. Scale
    5. Add

Unsloth fuses LoRA computations and can batch multiple adapters!
"""

@triton.jit
def fused_lora_forward_kernel(
    output_ptr,
    base_output_ptr,
    x_ptr,
    lora_A_ptr,
    lora_B_ptr,
    scale,
    M, N, K, R,  # M=batch*seq, N=out_dim, K=in_dim, R=rank
    stride_xm, stride_xk,
    stride_am, stride_ak,  # A is (R, K)
    stride_bm, stride_bn,  # B is (N, R)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Fused LoRA forward: output = base_output + scale * (x @ A.T @ B.T)
    
    This kernel fuses the LoRA computation with the addition to base output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # First: compute x @ A.T -> (M, R)
    # This is a reduction over K
    lora_hidden = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        # Load x block (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load A block (BLOCK_R, BLOCK_K) - note A is (R, K)
        r_offs = tl.arange(0, BLOCK_R)
        a_ptrs = lora_A_ptr + r_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (r_offs[:, None] < R) & (k_offs[None, :] < K)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Accumulate: x @ A.T
        lora_hidden += tl.dot(x_block, tl.trans(a_block))
    
    # Second: multiply by B.T -> (M, N)
    lora_output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r in range(0, R, BLOCK_R):
        r_offs_inner = r + tl.arange(0, BLOCK_R)
        
        # Load B block (BLOCK_N, BLOCK_R) - B is (N, R)
        b_ptrs = lora_B_ptr + n_offs[:, None] * stride_bm + r_offs_inner[None, :] * stride_bn
        b_mask = (n_offs[:, None] < N) & (r_offs_inner[None, :] < R)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Get corresponding hidden slice
        hidden_slice = lora_hidden[:, r_offs_inner - r_offs_inner[0]]
        
        # Accumulate: lora_hidden @ B.T
        lora_output += tl.dot(hidden_slice, tl.trans(b_block))
    
    # Scale LoRA output
    lora_output = lora_output * scale
    
    # Load base output and add
    base_ptrs = base_output_ptr + m_offs[:, None] * N + n_offs[None, :]
    base_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    base_output = tl.load(base_ptrs, mask=base_mask, other=0.0)
    
    # Final output
    output = base_output + lora_output
    
    # Store
    out_ptrs = output_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, output, mask=base_mask)


# ============================================================================
# EXPERIMENTS AND BENCHMARKS
# ============================================================================

def experiment_rmsnorm_fusion():
    """Benchmark fused vs unfused RMSNorm."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: FUSED RMSNORM + RESIDUAL")
    print(" (This is what Unsloth does for every transformer layer)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch, seq, hidden = 32, 2048, 4096
    
    hidden_states = torch.randn(batch * seq, hidden, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch * seq, hidden, device='cuda', dtype=torch.float16)
    weight = torch.ones(hidden, device='cuda', dtype=torch.float16)
    eps = 1e-6
    
    # Unfused version
    def unfused_rmsnorm_residual():
        x = hidden_states + residual
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + eps)
        return (weight * x_normed).half()
    
    # Verify correctness
    ref = unfused_rmsnorm_residual()
    out = fused_rmsnorm_residual(hidden_states, residual, weight, eps)
    
    max_error = (ref - out).abs().max().item()
    print(f"\n Correctness: max error = {max_error:.2e}")
    
    # Benchmark
    time_unfused = profile_kernel(unfused_rmsnorm_residual)
    time_fused = profile_kernel(lambda: fused_rmsnorm_residual(hidden_states, residual, weight, eps))
    
    print(f"\n Performance ({batch}×{seq}×{hidden}):")
    print(f" Unfused (3 ops): {time_unfused:.3f} ms")
    print(f" Fused (1 kernel): {time_fused:.3f} ms")
    print(f" Speedup: {time_unfused/time_fused:.2f}x")


def experiment_cross_entropy_fusion():
    """Benchmark fused vs unfused cross-entropy."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: FUSED CROSS-ENTROPY LOSS")
    print(" (Unsloth's chunked cross-entropy for large vocabularies)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch_seq = 4096
    vocab_sizes = [8192, 32000, 65536, 128256]
    
    print(f"\n Performance for different vocabulary sizes:")
    print(f"{'Vocab Size':<15} {'PyTorch (ms)':<15} {'Fused (ms)':<15} {'Speedup'}")
    print("-" * 60)
    
    for vocab in vocab_sizes:
        logits = torch.randn(batch_seq, vocab, device='cuda', dtype=torch.float32)
        labels = torch.randint(0, vocab, (batch_seq,), device='cuda')
        
        # PyTorch reference
        time_pytorch = profile_kernel(
            lambda: F.cross_entropy(logits, labels),
            iterations=50
        )
        
        # Fused version
        time_fused = profile_kernel(
            lambda: fused_cross_entropy(logits, labels),
            iterations=50
        )
        
        speedup = time_pytorch / time_fused
        print(f"{vocab:<15} {time_pytorch:<15.3f} {time_fused:<15.3f} {speedup:.2f}x")
    
    # Verify correctness
    logits = torch.randn(1024, 32000, device='cuda', dtype=torch.float32)
    labels = torch.randint(0, 32000, (1024,), device='cuda')
    
    ref = F.cross_entropy(logits, labels)
    out = fused_cross_entropy(logits, labels)
    
    print(f"\n Correctness check (vocab=32000):")
    print(f" PyTorch loss: {ref.item():.4f}")
    print(f" Fused loss: {out.item():.4f}")
    print(f" Difference: {abs(ref.item() - out.item()):.2e}")


def experiment_silu_mul_fusion():
    """Benchmark fused vs unfused SiLU + multiply."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: FUSED SILU + MULTIPLY")
    print(" (Part of Unsloth's MLP optimization)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Typical MLP dimensions for Llama-7B
    batch_seq = 32 * 2048
    intermediate = 11008
    
    gate = torch.randn(batch_seq, intermediate, device='cuda', dtype=torch.float16)
    up = torch.randn(batch_seq, intermediate, device='cuda', dtype=torch.float16)
    
    # Unfused
    def unfused_silu_mul():
        return F.silu(gate) * up
    
    # Verify correctness
    ref = unfused_silu_mul()
    out = fused_silu_mul(gate, up)
    
    max_error = (ref - out).abs().max().item()
    print(f"\n Correctness: max error = {max_error:.2e}")
    
    # Benchmark
    time_unfused = profile_kernel(unfused_silu_mul)
    time_fused = profile_kernel(lambda: fused_silu_mul(gate, up))
    
    print(f"\n Performance ({batch_seq}×{intermediate}):")
    print(f" Unfused (silu + mul): {time_unfused:.3f} ms")
    print(f" Fused (1 kernel): {time_fused:.3f} ms")
    print(f" Speedup: {time_unfused/time_fused:.2f}x")


def print_unsloth_summary():
    """Print summary of Unsloth-style optimizations."""
    print("\n" + "="*70)
    print(" UNSLOTH KERNEL OPTIMIZATION SUMMARY")
    print("="*70)
    
    print("""
    UNSLOTH'S KEY OPTIMIZATIONS:
    ════════════════════════════
    
    1. FUSED RMSNORM + RESIDUAL
       ┌────────────────────────────────────────────────────┐
       │ Standard: 3 kernels, 6 memory passes               │
       │ Fused: 1 kernel, 2 memory passes                   │
       │ Speedup: ~2-3x                                     │
       └────────────────────────────────────────────────────┘
    
    2. FUSED CROSS-ENTROPY (CHUNKED)
       ┌────────────────────────────────────────────────────┐
       │ Standard: Materialize full logits (vocab × batch)  │
       │ Fused: Process in chunks, online logsumexp         │
       │ Memory savings: Up to 10x for large vocabularies   │
       └────────────────────────────────────────────────────┘
    
    3. FUSED ROPE (ROTARY POSITION EMBEDDING)
       ┌────────────────────────────────────────────────────┐
       │ Standard: Separate cos/sin computation and apply   │
       │ Fused: Single kernel for Q and K                   │
       │ Speedup: ~2x                                       │
       └────────────────────────────────────────────────────┘
    
    4. FUSED SWIGLU MLP
       ┌────────────────────────────────────────────────────┐
       │ Standard: gate_proj, silu, up_proj, mul, down_proj │
       │ Fused: gate_up_proj, fused_silu_mul, down_proj     │
       │ Speedup: ~1.5-2x                                   │
       └────────────────────────────────────────────────────┘
    
    5. FUSED LORA
       ┌────────────────────────────────────────────────────┐
       │ Standard: Base + A matmul + B matmul + scale + add │
       │ Fused: Single kernel with minimal memory traffic   │
       │ Speedup: ~2x for LoRA forward pass                 │
       └────────────────────────────────────────────────────┘
    
    WHY THESE OPTIMIZATIONS MATTER:
    ═══════════════════════════════
    
    For a Llama-7B forward pass:
    - ~60% time in attention (Flash Attention helps)
    - ~30% time in MLP (SwiGLU fusion helps)
    - ~10% time in normalization (RMSNorm fusion helps)
    
    Combined Unsloth speedup: 2-5x faster training!
    
    MEMORY HIERARCHY REMINDER:
    ══════════════════════════
    
    ┌─────────────────────────────────────────────────────────┐
    │ Level      │ Size    │ Bandwidth │ Latency              │
    ├─────────────────────────────────────────────────────────┤
    │ Registers  │ ~256KB  │ ~20 TB/s  │ 0 cycles             │
    │ L1/Shared  │ 128KB   │ ~10 TB/s  │ ~30 cycles           │
    │ L2         │ 40MB    │ ~5 TB/s   │ ~200 cycles          │
    │ HBM        │ 80GB    │ 2 TB/s    │ ~400 cycles          │
    └─────────────────────────────────────────────────────────┘
    
    Fusion keeps data in registers/L1 → massive speedup!
    
    USING IN YOUR CODE:
    ═══════════════════
    
    Option 1: Use Unsloth directly
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(...)
    
    Option 2: Use these patterns in custom code
        - Understand the fusion patterns
        - Write custom Triton kernels
        - Or use torch.compile with max-autotune
    
    Option 3: Contribute optimizations
        - Profile your model
        - Identify memory-bound operations
        - Fuse them with Triton!
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " UNSLOTH-STYLE TRITON KERNELS ".center(68) + "║")
    print("║" + " Production optimizations for LLM training ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" Triton version: {triton.__version__}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_rmsnorm_fusion()
    experiment_cross_entropy_fusion()
    experiment_silu_mul_fusion()
    print_unsloth_summary()
    
    print("\n" + "="*70)
    print(" These kernels are the foundation of Unsloth's 2-5x speedup!")
    print("="*70)
