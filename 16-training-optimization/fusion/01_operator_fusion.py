"""
Operator Fusion for Deep Learning
==================================

This module covers operator fusion techniques that combine multiple
operations into single optimized kernels, reducing memory bandwidth
and improving performance.

Key Topics:
1. Why Operator Fusion Matters
2. Fused LayerNorm and RMSNorm
3. Fused Attention (QKV + Softmax + Output)
4. Fused MLP (Gate + Up + Activation + Down)
5. Cross-Entropy Loss Fusion
6. Triton Implementation Examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# =============================================================================
# SECTION 1: WHY OPERATOR FUSION MATTERS
# =============================================================================
"""
THE MEMORY BANDWIDTH BOTTLENECK:
════════════════════════════════

Modern GPUs have:
    - Massive compute: 312 TFLOPS (A100 FP16)
    - Limited bandwidth: 2 TB/s (A100 HBM)

Arithmetic Intensity = FLOPs / Bytes Transferred

Many operations are MEMORY-BOUND, not compute-bound:
┌─────────────────────────────────────────────────────────────────┐
│ Operation      │ FLOPs    │ Memory     │ Intensity │ Bound     │
├─────────────────────────────────────────────────────────────────┤
│ MatMul (large) │ O(N³)    │ O(N²)      │ O(N)      │ Compute   │
│ Softmax        │ O(N)     │ O(N)       │ O(1)      │ Memory    │
│ LayerNorm      │ O(N)     │ O(N)       │ O(1)      │ Memory    │
│ GELU           │ O(N)     │ O(N)       │ O(1)      │ Memory    │
│ Add/Residual   │ O(N)     │ O(N)       │ O(1)      │ Memory    │
└─────────────────────────────────────────────────────────────────┘

FUSION BENEFIT:
    Unfused:
        x = layer_norm(x)       # Read x, Write x
        x = linear(x)           # Read x, Write x  
        x = gelu(x)             # Read x, Write x
        
        Total: 6 memory passes!
    
    Fused:
        x = fused_linear_gelu_norm(x)  # Read x once, Write x once
        
        Total: 2 memory passes!
        
    Speedup: Up to 3x for memory-bound operations
"""


def demonstrate_fusion_benefit():
    """Demonstrate the benefit of operator fusion."""
    print("\n" + "="*70)
    print("OPERATOR FUSION BENEFIT DEMONSTRATION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required for timing demonstration")
        return
    
    import time
    device = torch.device("cuda")
    
    # Setup
    batch, seq, hidden = 32, 2048, 4096
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)
    
    # Create layers
    norm = nn.LayerNorm(hidden).to(device).half()
    linear = nn.Linear(hidden, hidden).to(device).half()
    
    # Warmup
    for _ in range(10):
        y = F.gelu(linear(norm(x)))
    torch.cuda.synchronize()
    
    # Unfused timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        y = norm(x)
        y = linear(y)
        y = F.gelu(y)
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) * 10  # ms
    
    print(f"\nConfiguration: batch={batch}, seq={seq}, hidden={hidden}")
    print(f"\nUnfused (3 separate ops): {unfused_time:.2f} ms")
    print(f"  - LayerNorm: read+write = {2 * x.numel() * 2 / 1e9:.2f} GB")
    print(f"  - Linear: read+write = {2 * x.numel() * 2 / 1e9:.2f} GB")
    print(f"  - GELU: read+write = {2 * x.numel() * 2 / 1e9:.2f} GB")
    print(f"  Total memory: {6 * x.numel() * 2 / 1e9:.2f} GB")
    
    print(f"\nWith fusion (theoretical):")
    print(f"  - Fused: read+write = {2 * x.numel() * 2 / 1e9:.2f} GB")
    print(f"  Potential speedup: ~3x for memory-bound portion")


# =============================================================================
# SECTION 2: RMSNORM - SIMPLER AND FASTER NORMALIZATION
# =============================================================================
"""
RMSNorm vs LayerNorm:
═════════════════════

LayerNorm:
    y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    
    Requires: mean, variance, 2 passes or online algorithm
    Parameters: gamma (scale), beta (shift)

RMSNorm (Root Mean Square Normalization):
    y = x / sqrt(mean(x²) + eps) * gamma
    
    Requires: Only mean of squares, single pass
    Parameters: gamma only (no beta/shift)

Why RMSNorm for LLMs:
    - Simpler computation (no mean subtraction)
    - Fewer parameters (no beta)
    - Empirically similar quality
    - Used in Llama, Mistral, most modern LLMs
"""


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Used in Llama, Mistral, and most modern LLMs.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation.
    
    In practice, this would be a Triton kernel that:
    1. Loads x in one pass
    2. Computes sum of squares in registers
    3. Normalizes and multiplies by weight
    4. Writes output
    
    All in a single kernel launch!
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is the fused computation pattern
        # In Triton, this would be a single kernel
        
        orig_dtype = x.dtype
        x = x.float()  # Compute in FP32 for stability
        
        # Single pass: compute RMS and normalize
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        return (self.weight * x).to(orig_dtype)


def compare_norms():
    """Compare LayerNorm vs RMSNorm."""
    print("\n" + "="*70)
    print("LAYERNORM vs RMSNORM COMPARISON")
    print("="*70)
    
    hidden = 4096
    
    layer_norm = nn.LayerNorm(hidden)
    rms_norm = RMSNorm(hidden)
    
    # Parameter count
    ln_params = sum(p.numel() for p in layer_norm.parameters())
    rms_params = sum(p.numel() for p in rms_norm.parameters())
    
    print(f"\nParameter Comparison (hidden={hidden}):")
    print(f"  LayerNorm: {ln_params:,} (gamma + beta)")
    print(f"  RMSNorm:   {rms_params:,} (gamma only)")
    
    print(f"\nComputation Comparison:")
    print(f"  LayerNorm: mean, variance, subtract, divide, scale, shift")
    print(f"  RMSNorm:   square, mean, rsqrt, multiply")
    
    # Output comparison
    x = torch.randn(2, 10, hidden)
    
    with torch.no_grad():
        y_ln = layer_norm(x)
        y_rms = rms_norm(x)
    
    print(f"\nOutput Statistics:")
    print(f"  LayerNorm output mean: {y_ln.mean():.4f}")
    print(f"  RMSNorm output mean: {y_rms.mean():.4f}")
    print(f"  LayerNorm output std: {y_ln.std():.4f}")
    print(f"  RMSNorm output std: {y_rms.std():.4f}")


# =============================================================================
# SECTION 3: FUSED ATTENTION
# =============================================================================
"""
FUSED ATTENTION:
════════════════

Standard Attention (many kernels):
    1. Q = X @ W_q           # Kernel 1
    2. K = X @ W_k           # Kernel 2
    3. V = X @ W_v           # Kernel 3
    4. scores = Q @ K^T      # Kernel 4
    5. scores = scores/√d    # Kernel 5
    6. attn = softmax(scores)# Kernel 6
    7. out = attn @ V        # Kernel 7
    8. out = out @ W_o       # Kernel 8

Fused Attention (Flash Attention):
    1. QKV = X @ W_qkv       # Fused QKV projection
    2. out = flash_attn(Q,K,V)  # Fused attention kernel
    3. out = out @ W_o       # Output projection

Memory Savings:
    - Never materialize N×N attention matrix
    - Single pass through Q, K, V
    - Recompute in backward instead of storing
"""


class FusedQKVProjection(nn.Module):
    """
    Fused QKV projection.
    
    Instead of 3 separate linear layers, use one large linear
    and split the output.
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Single fused projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        
        # Single matmul for Q, K, V
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        return q, k, v


def fused_attention_explanation():
    """Explain fused attention benefits."""
    print("\n" + "="*70)
    print("FUSED ATTENTION EXPLANATION")
    print("="*70)
    
    print("""
FUSED QKV PROJECTION:
═════════════════════

Unfused:
    Q = X @ W_q  # Kernel 1
    K = X @ W_k  # Kernel 2  
    V = X @ W_v  # Kernel 3
    
    3 kernel launches, 3× read of X
    
Fused:
    QKV = X @ W_qkv  # Single kernel
    Q, K, V = split(QKV)
    
    1 kernel launch, 1× read of X
    

FLASH ATTENTION FUSION:
═══════════════════════

Standard attention stores O(N²) intermediate tensors:
    S = Q @ K^T     # N × N matrix stored
    P = softmax(S)  # N × N matrix stored
    O = P @ V       # Finally output
    
Flash Attention fuses everything:
    O = flash_attention(Q, K, V)  # Never stores N × N
    
    Uses tiling and online softmax to:
    - Keep data in SRAM
    - Never write full attention matrix to HBM
    - Recompute during backward


PYTORCH SDPA (Scaled Dot-Product Attention):
════════════════════════════════════════════

PyTorch 2.0+ provides F.scaled_dot_product_attention which
automatically selects the best backend:

    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )

Backends (selected automatically):
    1. Flash Attention (if available)
    2. Memory-efficient attention (xformers)
    3. Math backend (fallback)
""")


# =============================================================================
# SECTION 4: FUSED MLP
# =============================================================================
"""
FUSED MLP (SwiGLU/GeGLU):
═════════════════════════

Modern LLM MLP structure (SwiGLU):
    gate = X @ W_gate
    up = X @ W_up
    hidden = SiLU(gate) * up
    output = hidden @ W_down

Fusion opportunities:
    1. Fuse gate and up projections
    2. Fuse SiLU with multiplication
    3. In backward, fuse gradient computations
"""


class SwiGLU(nn.Module):
    """
    SwiGLU activation used in Llama and modern LLMs.
    
    SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU with combined gate+up projection.
    
    Fuses gate_proj and up_proj into single matmul.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Fused gate+up projection
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul for both projections
        gate_up = self.gate_up_proj(x)
        
        # Split and apply activation
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        
        return self.down_proj(hidden)


def compare_mlp_implementations():
    """Compare unfused vs fused MLP."""
    print("\n" + "="*70)
    print("FUSED MLP COMPARISON")
    print("="*70)
    
    hidden_size = 4096
    intermediate_size = 11008  # Llama-7B intermediate size
    
    unfused = SwiGLU(hidden_size, intermediate_size)
    fused = FusedSwiGLU(hidden_size, intermediate_size)
    
    # Count parameters
    unfused_params = sum(p.numel() for p in unfused.parameters())
    fused_params = sum(p.numel() for p in fused.parameters())
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    
    print(f"\nParameter count:")
    print(f"  Unfused: {unfused_params:,}")
    print(f"  Fused: {fused_params:,}")
    print(f"  (Same parameters, different organization)")
    
    print(f"\nKernel launches:")
    print(f"  Unfused: 4 (gate_proj, up_proj, silu*up, down_proj)")
    print(f"  Fused: 3 (gate_up_proj, silu*up, down_proj)")
    
    print(f"\nMemory accesses for x read:")
    print(f"  Unfused: 2× (read for gate_proj AND up_proj)")
    print(f"  Fused: 1× (read once for gate_up_proj)")


# =============================================================================
# SECTION 5: FUSED CROSS-ENTROPY LOSS
# =============================================================================
"""
FUSED CROSS-ENTROPY:
════════════════════

Standard Cross-Entropy:
    1. logits = hidden @ lm_head    # Shape: (batch*seq, vocab)
    2. log_probs = log_softmax(logits)
    3. loss = -log_probs[target]
    
Problem: logits tensor is HUGE!
    - 7B model: vocab = 32000
    - Sequence = 2048, batch = 4
    - logits = 4 * 2048 * 32000 * 4 bytes = 1 GB!

Fused Cross-Entropy (Chunked):
    - Never materialize full logits tensor
    - Compute loss in chunks
    - Use online log-sum-exp for stability

This is what Unsloth's fused cross-entropy does!
"""


def chunked_cross_entropy_explanation():
    """Explain chunked/fused cross-entropy."""
    print("\n" + "="*70)
    print("FUSED CROSS-ENTROPY LOSS")
    print("="*70)
    
    print("""
MEMORY PROBLEM:
═══════════════

For loss computation:
    logits = hidden_states @ lm_head.weight.T
    loss = F.cross_entropy(logits, labels)

Memory usage for logits:
    - batch=4, seq=2048, vocab=32000
    - logits: 4 × 2048 × 32000 × 2 bytes = 512 MB (FP16)
    
    This is stored just for the loss computation!


CHUNKED CROSS-ENTROPY:
══════════════════════

Instead of computing all logits at once:

for chunk in chunks(hidden_states):
    chunk_logits = chunk @ lm_head.weight.T
    chunk_loss = cross_entropy(chunk_logits, chunk_labels)
    accumulate(chunk_loss)

Memory: Only one chunk of logits at a time!


ONLINE LOGSUMEXP:
═════════════════

Cross-entropy needs log-softmax:
    log_softmax(x)_i = x_i - log(sum(exp(x)))

For numerical stability:
    log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))

Online computation (chunked):
    For each chunk, update running:
    - Maximum
    - Log-sum-exp
    
Then compute final loss.


UNSLOTH'S IMPLEMENTATION:
═════════════════════════

Unsloth's fast_cross_entropy_loss:
1. Processes vocabulary in chunks of 65536
2. Uses Triton kernel for parallel reduction
3. Supports logit softcapping (Gemma 2)
4. Supports logit scaling (Cohere)

Memory savings: Up to 10× for large vocabularies!


USAGE:
══════

# Standard (high memory)
loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))

# Chunked (low memory) - from Unsloth
from unsloth.kernels import fast_cross_entropy_loss
loss = fast_cross_entropy_loss(logits, labels)
""")


# =============================================================================
# SECTION 6: TRITON KERNEL EXAMPLE
# =============================================================================

def triton_fusion_example():
    """Show Triton kernel fusion example."""
    print("\n" + "="*70)
    print("TRITON FUSION EXAMPLE")
    print("="*70)
    
    print("""
TRITON: GPU Kernel Language
════════════════════════════

Triton allows writing fused kernels in Python-like syntax.
Here's a fused RMSNorm + residual add:

```python
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_residual_kernel(
    output_ptr, input_ptr, residual_ptr, weight_ptr,
    stride, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Load pointers for this row
    row_start = row_idx * stride
    
    # Load input and residual in one pass
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask)
    res = tl.load(residual_ptr + row_start + col_offsets, mask=mask)
    
    # Add residual (fused!)
    x = x + res
    
    # Compute RMS
    x_fp32 = x.to(tl.float32)
    variance = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(variance + eps)
    
    # Normalize
    x_normed = x_fp32 * inv_rms
    
    # Load weight and apply
    w = tl.load(weight_ptr + col_offsets, mask=mask)
    output = x_normed * w
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
```

This kernel:
1. Loads input and residual (1 read each)
2. Adds them (no intermediate write!)
3. Computes RMS norm
4. Writes output (1 write)

Total memory ops: 3 (vs 5+ for unfused)


BENEFITS OF TRITON:
═══════════════════

1. Python-like syntax (easier than CUDA)
2. Automatic memory coalescing
3. Automatic shared memory management
4. Easy fusion of operations
5. Portable across GPU architectures

WHEN TO USE TRITON:
═══════════════════

✓ Memory-bound operations (element-wise, reductions)
✓ Custom fusion patterns
✓ Operations not in PyTorch

✗ Complex algorithms (stick to CUDA)
✗ Operations already optimized (matmul, conv)
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("OPERATOR FUSION FOR DEEP LEARNING")
    print("="*70)
    
    # Demonstrate fusion benefit
    demonstrate_fusion_benefit()
    
    # Compare normalization layers
    compare_norms()
    
    # Fused attention explanation
    fused_attention_explanation()
    
    # Compare MLP implementations
    compare_mlp_implementations()
    
    # Fused cross-entropy
    chunked_cross_entropy_explanation()
    
    # Triton example
    triton_fusion_example()
    
    print("\n" + "="*70)
    print("OPERATOR FUSION MODULE COMPLETE")
    print("="*70)
