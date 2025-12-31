"""
01_flash_attention.py - Understanding Flash Attention

Flash Attention is THE breakthrough algorithm for efficient transformers.
It achieves O(N) memory instead of O(N²) for attention.

This module explains the algorithm and provides a simplified implementation
for learning purposes. For production, use the official Flash Attention library.

Key Ideas:
1. Tiling: Process attention in blocks
2. Online Softmax: Never materialize full attention matrix
3. Recomputation: Trade compute for memory in backward

Run: python 01_flash_attention.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# ============================================================================
# PROFILING
# ============================================================================

def profile_triton(func, warmup=25, iterations=100):
    """Profile a function."""
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
# STANDARD ATTENTION (FOR COMPARISON)
# ============================================================================

def standard_attention(Q, K, V, causal=False):
    """
    Standard attention: O(N²) memory
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    
    # QK^T: (B, H, N, N) - O(N²) memory!
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    if causal:
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Output
    output = torch.matmul(attn_weights, V)
    
    return output

# ============================================================================
# FLASH ATTENTION KERNEL (SIMPLIFIED)
# ============================================================================

@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, D,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Simplified Flash Attention forward kernel.
    
    For each query block:
    1. Load Q block to SRAM
    2. For each K,V block:
       a. Load K,V block
       b. Compute local attention scores
       c. Update running max and sum (online softmax)
       d. Update output
    
    This is simplified - real Flash Attention has more optimizations.
    """
    # Batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Query block index
    q_block_idx = tl.program_id(2)
    
    # Query positions for this block
    q_start = q_block_idx * BLOCK_N
    q_offs = q_start + tl.arange(0, BLOCK_N)
    
    # Initialize output accumulator
    o_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Initialize online softmax statistics
    m_prev = tl.zeros([BLOCK_N], dtype=tl.float32) - float('inf')  # Running max
    l_prev = tl.zeros([BLOCK_N], dtype=tl.float32)  # Running sum
    
    # Load Q block [BLOCK_N, BLOCK_D]
    q_ptrs = (Q_ptr + 
              batch_idx * stride_qb + 
              head_idx * stride_qh + 
              q_offs[:, None] * stride_qn + 
              tl.arange(0, BLOCK_D)[None, :] * stride_qd)
    q_mask = q_offs[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Iterate over K,V blocks
    for kv_block_idx in range(tl.cdiv(N, BLOCK_N)):
        kv_start = kv_block_idx * BLOCK_N
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        
        # Load K block [BLOCK_N, BLOCK_D]
        k_ptrs = (K_ptr + 
                  batch_idx * stride_kb + 
                  head_idx * stride_kh + 
                  kv_offs[:, None] * stride_kn + 
                  tl.arange(0, BLOCK_D)[None, :] * stride_kd)
        k_mask = kv_offs[:, None] < N
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load V block [BLOCK_N, BLOCK_D]
        v_ptrs = (V_ptr + 
                  batch_idx * stride_vb + 
                  head_idx * stride_vh + 
                  kv_offs[:, None] * stride_vn + 
                  tl.arange(0, BLOCK_D)[None, :] * stride_vd)
        v_mask = kv_offs[:, None] < N
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute attention scores: Q @ K^T [BLOCK_N, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale
        
        # Mask invalid positions
        s_mask = (q_offs[:, None] < N) & (kv_offs[None, :] < N)
        s = tl.where(s_mask, s, float('-inf'))
        
        # Online softmax update
        m_curr = tl.max(s, axis=1)  # Row-wise max
        m_new = tl.maximum(m_prev, m_curr)
        
        # Compute exp(s - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(s_mask, p, 0.0)
        
        # Update running sum with correction
        l_curr = tl.sum(p, axis=1)
        alpha = tl.exp(m_prev - m_new)
        l_new = l_prev * alpha + l_curr
        
        # Update output accumulator
        # o_new = (o_prev * l_prev * alpha + p @ v) / l_new
        o_acc = o_acc * (l_prev * alpha)[:, None]
        o_acc += tl.dot(p.to(v.dtype), v)
        
        # Update statistics for next iteration
        m_prev = m_new
        l_prev = l_new
    
    # Final normalization
    o_acc = o_acc / l_prev[:, None]
    
    # Store output
    o_ptrs = (O_ptr + 
              batch_idx * stride_ob + 
              head_idx * stride_oh + 
              q_offs[:, None] * stride_on + 
              tl.arange(0, BLOCK_D)[None, :] * stride_od)
    o_mask = q_offs[:, None] < N
    tl.store(o_ptrs, o_acc, mask=o_mask)


def triton_flash_attention(Q, K, V):
    """Wrapper for Flash Attention kernel."""
    B, H, N, D = Q.shape
    
    O = torch.empty_like(Q)
    
    # Block sizes
    BLOCK_N = min(64, N)
    BLOCK_D = D  # Assume D fits in SRAM
    
    # Grid: (batch, heads, query_blocks)
    grid = (B, H, triton.cdiv(N, BLOCK_N))
    
    scale = 1.0 / math.sqrt(D)
    
    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, D,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return O

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_why_flash_attention():
    """Explain why Flash Attention is needed."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: WHY FLASH ATTENTION?")
    print(" The O(N²) memory problem")
    print("="*70)
    
    print("""
    STANDARD ATTENTION MEMORY USAGE:
    
    Q, K, V: each (B, H, N, D)
    
    Step 1: S = Q @ K^T
            Shape: (B, H, N, N) ← O(N²) memory!
    
    Step 2: P = softmax(S)
            Shape: (B, H, N, N) ← Still O(N²)!
    
    Step 3: O = P @ V
            Shape: (B, H, N, D)
    
    MEMORY EXAMPLE:
    B=1, H=32, N=8192, D=128 (typical LLM)
    
    Q, K, V: 32 × 8192 × 128 × 4 bytes = 128 MB each
    S, P:    32 × 8192 × 8192 × 4 bytes = 8.6 GB each!
    
    For N=16384: S, P would be 34.4 GB each!
    → OOM on most GPUs
    
    FLASH ATTENTION SOLUTION:
    
    Never materialize full S or P matrix!
    
    1. Process in blocks (tiles)
    2. Use online softmax (maintain running max/sum)
    3. Output each block independently
    
    Memory: O(N) instead of O(N²)
    """)


def experiment_memory_comparison():
    """Compare memory usage of standard vs flash attention."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY USAGE COMPARISON")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    B, H, D = 2, 8, 64
    
    print(f"\n Memory usage for different sequence lengths:")
    print(f" Batch={B}, Heads={H}, HeadDim={D}")
    print(f"\n{'Seq Len':<12} {'Standard (MB)':<18} {'SDPA (MB)':<18} {'Ratio'}")
    print("-" * 60)
    
    for N in [256, 512, 1024, 2048, 4096]:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        # Standard attention memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            _ = standard_attention(Q, K, V)
            mem_standard = torch.cuda.max_memory_allocated() / 1e6
        except RuntimeError:
            mem_standard = float('inf')
        
        # SDPA (uses Flash Attention when available)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = F.scaled_dot_product_attention(Q, K, V)
        mem_sdpa = torch.cuda.max_memory_allocated() / 1e6
        
        if mem_standard != float('inf'):
            ratio = mem_standard / mem_sdpa
            print(f"{N:<12} {mem_standard:<18.1f} {mem_sdpa:<18.1f} {ratio:.1f}x")
        else:
            print(f"{N:<12} {'OOM':<18} {mem_sdpa:<18.1f} ∞")
        
        del Q, K, V
        torch.cuda.empty_cache()


def experiment_flash_attention_correctness():
    """Verify Flash Attention correctness."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: FLASH ATTENTION CORRECTNESS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print(f"\n Comparing our simplified Flash Attention with reference:")
    print(f"{'Shape (B,H,N,D)':<25} {'Max Error':<20} {'Correct?'}")
    print("-" * 60)
    
    shapes = [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (2, 8, 256, 64),
        (1, 4, 512, 64),
    ]
    
    for B, H, N, D in shapes:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        ref = standard_attention(Q, K, V)
        
        try:
            out = triton_flash_attention(Q, K, V)
            max_error = torch.max(torch.abs(out - ref)).item()
            correct = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
            print(f"{str((B,H,N,D)):<25} {max_error:<20.2e} {correct}")
        except Exception as e:
            print(f"{str((B,H,N,D)):<25} Error: {str(e)[:30]}")


def experiment_flash_attention_performance():
    """Benchmark Flash Attention performance."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: FLASH ATTENTION PERFORMANCE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    B, H, D = 2, 8, 64
    
    print(f"\n Performance comparison:")
    print(f" Batch={B}, Heads={H}, HeadDim={D}")
    print(f"\n{'Seq Len':<12} {'Standard (ms)':<18} {'SDPA (ms)':<18} {'Speedup'}")
    print("-" * 60)
    
    for N in [256, 512, 1024, 2048]:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        
        try:
            time_standard = profile_triton(
                lambda: standard_attention(Q, K, V), 
                iterations=50
            )
        except:
            time_standard = float('inf')
        
        time_sdpa = profile_triton(
            lambda: F.scaled_dot_product_attention(Q, K, V),
            iterations=50
        )
        
        if time_standard != float('inf'):
            speedup = time_standard / time_sdpa
            print(f"{N:<12} {time_standard:<18.3f} {time_sdpa:<18.3f} {speedup:.2f}x")
        else:
            print(f"{N:<12} {'OOM':<18} {time_sdpa:<18.3f} ∞")


def experiment_online_softmax_algorithm():
    """Explain the online softmax algorithm in detail."""
    print("\n" + "="*70)
    print(" EXPERIMENT 5: ONLINE SOFTMAX ALGORITHM")
    print(" The key enabling technique for Flash Attention")
    print("="*70)
    
    print("""
    ONLINE SOFTMAX RECURRENCE:
    
    Standard softmax needs two passes:
    Pass 1: m = max(x)
    Pass 2: softmax(x) = exp(x - m) / sum(exp(x - m))
    
    Online softmax processes incrementally:
    
    Given previous statistics (m_prev, l_prev) and new block x_curr:
    
    1. m_curr = max(x_curr)                    # Block max
    2. m_new = max(m_prev, m_curr)             # Update global max
    3. l_new = l_prev * exp(m_prev - m_new)    # Correct previous sum
            + sum(exp(x_curr - m_new))         # Add new contribution
    
    This allows processing arbitrarily long sequences in blocks!
    
    FLASH ATTENTION ALGORITHM:
    
    For each query block Q_i:
        Initialize: O_i = 0, m_i = -∞, l_i = 0
        
        For each key-value block (K_j, V_j):
            1. S_ij = Q_i @ K_j^T / sqrt(d)       # Local scores
            2. m_ij = max(S_ij)                   # Local max
            3. P_ij = exp(S_ij - m_ij)            # Local softmax numerator
            4. l_ij = sum(P_ij)                   # Local sum
            
            # Update running statistics
            5. m_new = max(m_i, m_ij)
            6. l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
            
            # Update output
            7. O_i = O_i * (l_i * exp(m_i - m_new) / l_new)
                   + P_ij * exp(m_ij - m_new) @ V_j / l_new
            
            8. m_i, l_i = m_new, l_new
        
        Store O_i
    
    MEMORY COMPLEXITY:
    - Standard: O(N²) for attention matrix
    - Flash:    O(N) - only store blocks at a time
    """)


def print_flash_attention_summary():
    """Print Flash Attention summary."""
    print("\n" + "="*70)
    print(" FLASH ATTENTION SUMMARY")
    print("="*70)
    
    print("""
    KEY INNOVATIONS:
    
    1. TILING
       - Process attention in blocks
       - Never materialize full N×N matrix
       - Memory: O(N) instead of O(N²)
    
    2. ONLINE SOFTMAX
       - Compute softmax incrementally
       - Maintain running max and sum
       - Correct for max changes
    
    3. IO-AWARENESS
       - Minimize HBM reads/writes
       - Keep data in SRAM as long as possible
       - Fuse all operations
    
    4. RECOMPUTATION IN BACKWARD
       - Don't store attention matrix for backward
       - Recompute during backward pass
       - Trade compute for memory
    
    VARIANTS:
    
    - Flash Attention 1: Original algorithm
    - Flash Attention 2: Better parallelism, ~2x faster
    - Flash Attention 3: Hopper-optimized, hardware-aware
    - xFormers: Similar algorithm, different implementation
    
    USAGE IN PYTORCH:
    
    # Automatic (uses Flash Attention when applicable)
    output = F.scaled_dot_product_attention(Q, K, V)
    
    # Force specific implementation
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        output = F.scaled_dot_product_attention(Q, K, V)
    
    WHEN FLASH ATTENTION IS USED:
    ✓ Head dimension ≤ 256
    ✓ Sequence length > 64
    ✓ FP16 or BF16 data type
    ✓ No custom attention mask (or causal mask)
    
    MULTIMODAL IMPLICATIONS:
    - Enables long context for images (many patches)
    - Enables long context for video (many frames)
    - Critical for multimodal models with mixed sequence lengths
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " FLASH ATTENTION DEEP DIVE ".center(68) + "║")
    print("║" + " O(N) memory attention for transformers ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_why_flash_attention()
    experiment_memory_comparison()
    experiment_flash_attention_correctness()
    experiment_flash_attention_performance()
    experiment_online_softmax_algorithm()
    print_flash_attention_summary()
    
    print("\n" + "="*70)
    print(" Flash Attention is essential for modern transformers!")
    print(" Use F.scaled_dot_product_attention for automatic optimization")
    print("="*70)
