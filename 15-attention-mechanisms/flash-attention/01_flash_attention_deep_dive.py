"""
Flash Attention Deep Dive: From Theory to Implementation
=========================================================

This module provides a comprehensive understanding of Flash Attention,
the breakthrough algorithm that made transformer training practical at scale.

Key Topics:
1. Memory Hierarchy and IO-Awareness
2. Standard Attention Memory Problem
3. Online Softmax Algorithm
4. Tiling Strategy
5. Flash Attention Forward Pass
6. Flash Attention Backward Pass
7. Flash Attention 2 & 3 Improvements

Prerequisites:
- Understanding of scaled dot-product attention
- Basic GPU memory hierarchy knowledge
- PyTorch autograd fundamentals
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import time

# =============================================================================
# SECTION 1: GPU MEMORY HIERARCHY - THE FOUNDATION
# =============================================================================
"""
Understanding GPU Memory Hierarchy is CRITICAL for Flash Attention.

GPU Memory Types (NVIDIA A100 example):
┌─────────────────────────────────────────────────────────────────┐
│  HBM (High Bandwidth Memory) - 80GB                             │
│  ├── Bandwidth: 2TB/s                                           │
│  ├── Latency: ~400 cycles                                       │
│  └── Stores: Model weights, activations, optimizer states       │
├─────────────────────────────────────────────────────────────────┤
│  L2 Cache - 40MB                                                │
│  ├── Bandwidth: ~5TB/s                                          │
│  └── Latency: ~50 cycles                                        │
├─────────────────────────────────────────────────────────────────┤
│  SRAM (Shared Memory per SM) - 192KB per SM                     │
│  ├── Bandwidth: ~19TB/s                                         │
│  ├── Latency: ~20 cycles                                        │
│  └── Programmer-controlled scratchpad                           │
├─────────────────────────────────────────────────────────────────┤
│  Registers - 256KB per SM                                       │
│  ├── Bandwidth: ~infinite (same cycle)                          │
│  └── Fastest but most limited                                   │
└─────────────────────────────────────────────────────────────────┘

The Key Insight:
- SRAM is ~10x faster than HBM but ~1000x smaller
- Standard attention stores N×N matrices in HBM
- Flash Attention keeps data in SRAM as much as possible
"""


def demonstrate_memory_hierarchy():
    """
    Demonstrate the memory access patterns and their costs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate different access patterns
    n = 4096
    d = 64
    
    # Create tensors
    Q = torch.randn(n, d, device=device)
    K = torch.randn(n, d, device=device)
    V = torch.randn(n, d, device=device)
    
    # Standard attention: Materializes N×N matrix in HBM
    # Memory: O(N²) for attention scores
    # HBM Accesses: O(N²d + N²) reads + O(N²) writes
    
    # Flash Attention: Never materializes full N×N
    # Memory: O(N) for running statistics
    # HBM Accesses: O(N²d² / M) where M is SRAM size
    
    print("Memory Analysis for N={}, d={}:".format(n, d))
    print("-" * 50)
    
    # Standard attention memory
    attention_matrix_bytes = n * n * 4  # float32
    print(f"Standard Attention Matrix: {attention_matrix_bytes / 1e6:.2f} MB")
    
    # Flash attention only needs running stats
    flash_memory_bytes = n * 4 * 2  # m (max) and l (sum) per row
    print(f"Flash Attention Stats: {flash_memory_bytes / 1e6:.4f} MB")
    
    print(f"Memory Reduction: {attention_matrix_bytes / flash_memory_bytes:.0f}x")
    
    return Q, K, V


# =============================================================================
# SECTION 2: THE PROBLEM WITH STANDARD ATTENTION
# =============================================================================
"""
Standard Scaled Dot-Product Attention:

    Attention(Q, K, V) = softmax(QK^T / √d) V

Step-by-step breakdown:
1. S = QK^T / √d     # Shape: (N, N) - STORED IN HBM
2. P = softmax(S)     # Shape: (N, N) - STORED IN HBM  
3. O = PV             # Shape: (N, d) - Output

MEMORY PROBLEM:
- S and P are both N×N matrices
- For N=8192, d=64: S alone is 256MB (float32)
- For N=32768: S is 4GB!
- This QUADRATIC memory is the bottleneck

IO PROBLEM:
- We read Q, K (N×d each) from HBM
- We write S (N×N) to HBM
- We read S from HBM, compute softmax
- We write P (N×N) to HBM
- We read P, V from HBM
- We write O to HBM

Total HBM accesses: O(N²) reads + O(N²) writes
This is MEMORY-BOUND, not compute-bound!
"""


def standard_attention_naive(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              scale: Optional[float] = None) -> torch.Tensor:
    """
    Standard attention implementation - shows the memory problem.
    
    Args:
        Q: Query tensor (batch, seq_len, head_dim) or (seq_len, head_dim)
        K: Key tensor (batch, seq_len, head_dim) or (seq_len, head_dim)
        V: Value tensor (batch, seq_len, head_dim) or (seq_len, head_dim)
        scale: Scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor same shape as Q
    """
    d = Q.shape[-1]
    scale = scale or (1.0 / math.sqrt(d))
    
    # Step 1: Compute attention scores - O(N²) memory allocation!
    # This is where standard attention fails for long sequences
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (N, N) matrix
    
    # Step 2: Softmax - reads and writes O(N²) 
    P = F.softmax(S, dim=-1)  # Another (N, N) matrix
    
    # Step 3: Weighted sum
    O = torch.matmul(P, V)
    
    return O


def analyze_attention_memory(seq_lengths: list):
    """
    Show how memory scales quadratically with sequence length.
    """
    print("\nAttention Memory Scaling Analysis:")
    print("=" * 60)
    print(f"{'Seq Length':<15} {'Attention Matrix':<20} {'Practical?':<15}")
    print("-" * 60)
    
    for n in seq_lengths:
        # Attention matrix size in bytes (float16)
        mem_bytes = n * n * 2  # float16
        mem_gb = mem_bytes / (1024**3)
        
        practical = "✓" if mem_gb < 16 else "✗ (OOM likely)"
        print(f"{n:<15} {mem_gb:.2f} GB{' '*10} {practical}")
    
    print("-" * 60)


# =============================================================================
# SECTION 3: ONLINE SOFTMAX - THE KEY MATHEMATICAL INSIGHT
# =============================================================================
"""
ONLINE SOFTMAX: Computing softmax without materializing the full vector

Standard Softmax:
    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

Problem: We need to see ALL values to compute the sum!

Solution: Online/Streaming Softmax Algorithm

Key Insight: We can update softmax incrementally using:
    1. Running maximum m (for numerical stability)
    2. Running sum l (for normalization)

The Algorithm:
    For each new block of values x_new:
    
    m_new = max(m_old, max(x_new))           # Update max
    l_new = l_old * exp(m_old - m_new) +     # Rescale old sum
            sum(exp(x_new - m_new))           # Add new terms
    
    # Output can be updated incrementally too:
    o_new = o_old * (l_old * exp(m_old - m_new) / l_new) +
            softmax(x_new, m_new) @ V_block / l_new

This is EXACT, not an approximation!
"""


def online_softmax_demonstration():
    """
    Demonstrate online softmax algorithm step by step.
    """
    print("\n" + "="*70)
    print("ONLINE SOFTMAX DEMONSTRATION")
    print("="*70)
    
    # Full vector we want to compute softmax on
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    
    # Standard softmax (ground truth)
    standard_result = F.softmax(x, dim=0)
    print(f"\nFull vector: {x.tolist()}")
    print(f"Standard softmax: {standard_result.tolist()}")
    
    # Online softmax - process in chunks
    chunk_size = 2
    chunks = x.split(chunk_size)
    
    # Initialize running statistics
    m = float('-inf')  # Running max
    l = 0.0            # Running sum of exp(x - m)
    
    print(f"\nProcessing in chunks of {chunk_size}:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks):
        # Update running max
        m_chunk = chunk.max().item()
        m_new = max(m, m_chunk)
        
        # Update running sum with correction factor
        # Old sum needs to be rescaled: l_old * exp(m_old - m_new)
        l_new = l * math.exp(m - m_new) + torch.exp(chunk - m_new).sum().item()
        
        print(f"Chunk {i}: {chunk.tolist()}")
        print(f"  m: {m:.4f} -> {m_new:.4f}")
        print(f"  l: {l:.4f} -> {l_new:.4f}")
        
        m = m_new
        l = l_new
    
    # Verify: compute softmax using our running stats
    online_result = torch.exp(x - m) / l
    print(f"\nOnline softmax result: {online_result.tolist()}")
    print(f"Max difference: {(standard_result - online_result).abs().max().item():.2e}")
    
    return standard_result, online_result


def online_softmax_with_output():
    """
    Full online softmax with weighted output computation.
    This is what Flash Attention actually does.
    """
    print("\n" + "="*70)
    print("ONLINE SOFTMAX WITH OUTPUT ACCUMULATION")
    print("="*70)
    
    # Simulated attention setup
    seq_len = 8
    head_dim = 4
    block_size = 2
    
    # Create Q (single query for simplicity), K, V
    torch.manual_seed(42)
    q = torch.randn(1, head_dim)  # Single query
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)
    
    scale = 1.0 / math.sqrt(head_dim)
    
    # Ground truth: standard attention
    scores = (q @ K.T) * scale  # (1, seq_len)
    attn_weights = F.softmax(scores, dim=-1)
    standard_output = attn_weights @ V
    
    print(f"Sequence length: {seq_len}, Head dim: {head_dim}, Block size: {block_size}")
    
    # Online computation
    m = torch.tensor([float('-inf')])  # Running max
    l = torch.tensor([0.0])             # Running sum
    o = torch.zeros(1, head_dim)        # Running output
    
    num_blocks = seq_len // block_size
    
    print("\nBlock-by-block computation:")
    print("-" * 50)
    
    for j in range(num_blocks):
        # Get current block of K, V
        start = j * block_size
        end = start + block_size
        K_block = K[start:end]  # (block_size, head_dim)
        V_block = V[start:end]  # (block_size, head_dim)
        
        # Compute scores for this block
        s_block = (q @ K_block.T) * scale  # (1, block_size)
        
        # Update running max
        m_block = s_block.max()
        m_new = torch.max(m, m_block)
        
        # Compute exp(scores - new_max) for this block
        p_block = torch.exp(s_block - m_new)  # (1, block_size)
        
        # Update running sum with correction for max change
        l_new = l * torch.exp(m - m_new) + p_block.sum()
        
        # Update output with correction
        # Old output was: o = sum(exp(s_i - m_old) * v_i) / l_old
        # Need to rescale to new max: o_new = o_old * l_old * exp(m_old - m_new) / l_new
        o = o * (l * torch.exp(m - m_new) / l_new) + (p_block @ V_block) / l_new
        
        print(f"Block {j} (indices {start}:{end}):")
        print(f"  m: {m.item():.4f} -> {m_new.item():.4f}")
        print(f"  l: {l.item():.4f} -> {l_new.item():.4f}")
        
        m = m_new
        l = l_new
    
    print(f"\nOnline output: {o.squeeze().tolist()}")
    print(f"Standard output: {standard_output.squeeze().tolist()}")
    print(f"Max difference: {(o - standard_output).abs().max().item():.2e}")


# =============================================================================
# SECTION 4: TILING STRATEGY
# =============================================================================
"""
TILING: Divide and Conquer for Memory Efficiency

The Flash Attention tiling strategy:

1. Divide Q into blocks of size B_r (rows)
2. Divide K, V into blocks of size B_c (columns)
3. Process blocks to fit in SRAM

Block Sizes (chosen based on SRAM size M):
    B_c = ceil(M / (4d))      # K, V blocks
    B_r = min(B_c, d)         # Q blocks

Memory in SRAM during computation:
    - Q block: B_r × d
    - K block: B_c × d  
    - V block: B_c × d
    - Output block: B_r × d
    - Statistics: B_r (max) + B_r (sum)
    
Total SRAM: O(B_r × d + B_c × d) = O(M)

Outer loop: Iterate over K, V blocks (j = 0 to T_c)
Inner loop: Iterate over Q blocks (i = 0 to T_r)

Why this order matters:
- K, V blocks are loaded ONCE
- Q blocks and outputs are loaded/stored ONCE per inner iteration
- Minimizes HBM traffic
"""


def visualize_tiling():
    """
    Visualize the tiling pattern used in Flash Attention.
    """
    print("\n" + "="*70)
    print("FLASH ATTENTION TILING VISUALIZATION")
    print("="*70)
    
    seq_len = 16
    B_r = 4  # Query block size
    B_c = 4  # Key/Value block size
    
    print(f"\nSequence length: {seq_len}")
    print(f"Query block size (B_r): {B_r}")
    print(f"Key/Value block size (B_c): {B_c}")
    
    num_q_blocks = seq_len // B_r
    num_kv_blocks = seq_len // B_c
    
    print(f"\nNumber of Q blocks: {num_q_blocks}")
    print(f"Number of K,V blocks: {num_kv_blocks}")
    
    print("\nAttention Matrix Tiling Pattern:")
    print("  " + " ".join([f"K{j}" for j in range(num_kv_blocks)]))
    
    for i in range(num_q_blocks):
        row = f"Q{i}"
        for j in range(num_kv_blocks):
            row += f" [{i},{j}]"
        print(row)
    
    print("\nProcessing Order (outer loop on KV, inner on Q):")
    step = 0
    for j in range(num_kv_blocks):
        print(f"\n  Load K[{j}], V[{j}] to SRAM")
        for i in range(num_q_blocks):
            step += 1
            print(f"    Step {step}: Process Q[{i}] × K[{j}]^T, update O[{i}]")


# =============================================================================
# SECTION 5: FLASH ATTENTION FORWARD PASS - COMPLETE IMPLEMENTATION
# =============================================================================
"""
Flash Attention Forward Pass Algorithm:

Input: Q, K, V ∈ R^{N×d}, Block sizes B_r, B_c
Output: O ∈ R^{N×d}

1. Initialize O = 0, l = 0, m = -∞ (all vectors of length N)
2. Divide Q into T_r blocks, K, V into T_c blocks
3. For j = 1 to T_c (outer loop over K, V):
   a. Load K_j, V_j from HBM to SRAM
   b. For i = 1 to T_r (inner loop over Q):
      i.   Load Q_i, O_i, l_i, m_i from HBM to SRAM
      ii.  Compute S_ij = Q_i K_j^T (in SRAM)
      iii. Compute m̃_ij = rowmax(S_ij)
      iv.  Compute P̃_ij = exp(S_ij - m̃_ij) (in SRAM)
      v.   Compute l̃_ij = rowsum(P̃_ij)
      vi.  Compute m_i^new = max(m_i, m̃_ij)
      vii. Compute l_i^new = exp(m_i - m_i^new) * l_i + exp(m̃_ij - m_i^new) * l̃_ij
      viii.Update O_i = diag(l_i^new)^{-1} * (
               diag(l_i) * exp(m_i - m_i^new) * O_i + 
               exp(m̃_ij - m_i^new) * P̃_ij * V_j
           )
      ix.  Write O_i, l_i^new, m_i^new to HBM
4. Return O
"""


class FlashAttentionForward(torch.autograd.Function):
    """
    Flash Attention forward pass implementation in pure PyTorch.
    
    This is a pedagogical implementation - real Flash Attention uses CUDA kernels.
    The algorithm is correct but won't achieve the same speed as the CUDA version.
    """
    
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                block_size: int = 64, causal: bool = False) -> torch.Tensor:
        """
        Flash Attention forward pass.
        
        Args:
            Q: (batch, heads, seq_len, head_dim)
            K: (batch, heads, seq_len, head_dim)
            V: (batch, heads, seq_len, head_dim)
            block_size: Size of blocks for tiling
            causal: Whether to use causal masking
        
        Returns:
            O: (batch, heads, seq_len, head_dim)
        """
        batch, heads, seq_len, head_dim = Q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Determine block sizes
        B_r = block_size  # Query block size
        B_c = block_size  # Key/Value block size
        
        # Number of blocks
        T_r = math.ceil(seq_len / B_r)
        T_c = math.ceil(seq_len / B_c)
        
        # Initialize output and statistics
        O = torch.zeros_like(Q)
        l = torch.zeros(batch, heads, seq_len, 1, device=Q.device, dtype=Q.dtype)
        m = torch.full((batch, heads, seq_len, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
        
        # Outer loop: iterate over K, V blocks
        for j in range(T_c):
            # K, V block indices
            kv_start = j * B_c
            kv_end = min(kv_start + B_c, seq_len)
            
            # Load K, V blocks (simulating SRAM load)
            K_j = K[:, :, kv_start:kv_end, :]  # (batch, heads, B_c, head_dim)
            V_j = V[:, :, kv_start:kv_end, :]
            
            # Inner loop: iterate over Q blocks
            for i in range(T_r):
                # Q block indices
                q_start = i * B_r
                q_end = min(q_start + B_r, seq_len)
                
                # Load Q block and current O, l, m
                Q_i = Q[:, :, q_start:q_end, :]  # (batch, heads, B_r, head_dim)
                O_i = O[:, :, q_start:q_end, :]
                l_i = l[:, :, q_start:q_end, :]
                m_i = m[:, :, q_start:q_end, :]
                
                # Compute attention scores for this block
                # S_ij = Q_i @ K_j^T / sqrt(d)
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                # Shape: (batch, heads, B_r, B_c)
                
                # Apply causal mask if needed
                if causal:
                    # Create mask where query position >= key position
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(kv_start, kv_end, device=Q.device)
                    causal_mask = q_indices.unsqueeze(1) < k_indices.unsqueeze(0)
                    S_ij = S_ij.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Compute block-wise max and exp
                m_ij = S_ij.max(dim=-1, keepdim=True).values  # (batch, heads, B_r, 1)
                P_ij = torch.exp(S_ij - m_ij)  # (batch, heads, B_r, B_c)
                l_ij = P_ij.sum(dim=-1, keepdim=True)  # (batch, heads, B_r, 1)
                
                # Update running max
                m_i_new = torch.maximum(m_i, m_ij)
                
                # Update running sum with correction factors
                # l_i_new = l_i * exp(m_i - m_i_new) + l_ij * exp(m_ij - m_i_new)
                alpha = torch.exp(m_i - m_i_new)
                beta = torch.exp(m_ij - m_i_new)
                l_i_new = l_i * alpha + l_ij * beta
                
                # Update output
                # O_i_new = (O_i * l_i * alpha + P_ij * beta @ V_j) / l_i_new
                O_i_new = (O_i * l_i * alpha + torch.matmul(P_ij * beta, V_j)) / l_i_new
                
                # Handle NaN from 0/0 when l_i_new is 0 (all masked)
                O_i_new = torch.nan_to_num(O_i_new, nan=0.0)
                
                # Write back to "HBM" (our output tensors)
                O[:, :, q_start:q_end, :] = O_i_new
                l[:, :, q_start:q_end, :] = l_i_new
                m[:, :, q_start:q_end, :] = m_i_new
        
        # Save for backward pass
        ctx.save_for_backward(Q, K, V, O, l, m)
        ctx.block_size = block_size
        ctx.causal = causal
        ctx.scale = scale
        
        return O
    
    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """
        Flash Attention backward pass.
        
        Key insight: We DON'T save P (the attention weights)!
        Instead, we recompute P during backward using saved Q, K, and statistics.
        This is the memory saving trick.
        """
        Q, K, V, O, l, m = ctx.saved_tensors
        block_size = ctx.block_size
        causal = ctx.causal
        scale = ctx.scale
        
        batch, heads, seq_len, head_dim = Q.shape
        B_r = block_size
        B_c = block_size
        T_r = math.ceil(seq_len / B_r)
        T_c = math.ceil(seq_len / B_c)
        
        # Initialize gradients
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # Compute D = rowsum(dO ⊙ O) for the backward pass
        D = (dO * O).sum(dim=-1, keepdim=True)
        
        # Backward pass: similar structure to forward
        for j in range(T_c):
            kv_start = j * B_c
            kv_end = min(kv_start + B_c, seq_len)
            
            K_j = K[:, :, kv_start:kv_end, :]
            V_j = V[:, :, kv_start:kv_end, :]
            
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)
            
            for i in range(T_r):
                q_start = i * B_r
                q_end = min(q_start + B_r, seq_len)
                
                Q_i = Q[:, :, q_start:q_end, :]
                O_i = O[:, :, q_start:q_end, :]
                dO_i = dO[:, :, q_start:q_end, :]
                l_i = l[:, :, q_start:q_end, :]
                m_i = m[:, :, q_start:q_end, :]
                D_i = D[:, :, q_start:q_end, :]
                
                # Recompute attention scores and weights
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                
                if causal:
                    q_indices = torch.arange(q_start, q_end, device=Q.device)
                    k_indices = torch.arange(kv_start, kv_end, device=Q.device)
                    causal_mask = q_indices.unsqueeze(1) < k_indices.unsqueeze(0)
                    S_ij = S_ij.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                # Recompute P_ij = softmax(S_ij)
                P_ij = torch.exp(S_ij - m_i) / l_i
                P_ij = torch.nan_to_num(P_ij, nan=0.0)
                
                # Gradient of V: dV_j += P_ij^T @ dO_i
                dV_j += torch.matmul(P_ij.transpose(-2, -1), dO_i)
                
                # Gradient of P: dP_ij = dO_i @ V_j^T
                dP_ij = torch.matmul(dO_i, V_j.transpose(-2, -1))
                
                # Gradient of S through softmax: dS_ij = P_ij ⊙ (dP_ij - D_i)
                dS_ij = P_ij * (dP_ij - D_i) * scale
                
                # Gradient of Q: dQ_i += dS_ij @ K_j
                dQ[:, :, q_start:q_end, :] += torch.matmul(dS_ij, K_j)
                
                # Gradient of K: dK_j += dS_ij^T @ Q_i
                dK_j += torch.matmul(dS_ij.transpose(-2, -1), Q_i)
            
            dK[:, :, kv_start:kv_end, :] = dK_j
            dV[:, :, kv_start:kv_end, :] = dV_j
        
        return dQ, dK, dV, None, None


def flash_attention(Q, K, V, block_size=64, causal=False):
    """Wrapper function for Flash Attention."""
    return FlashAttentionForward.apply(Q, K, V, block_size, causal)


# =============================================================================
# SECTION 6: IO COMPLEXITY ANALYSIS
# =============================================================================
"""
IO Complexity Comparison:

Standard Attention:
- Load Q, K: 2Nd from HBM
- Write S: N² to HBM
- Load S: N² from HBM  
- Write P: N² to HBM
- Load P, V: N² + Nd from HBM
- Write O: Nd to HBM
- Total: O(Nd + N²) = O(N²) for N >> d

Flash Attention:
- Q, K, V are loaded in blocks
- For each outer iteration (T_c blocks of K, V):
  - Load K_j, V_j: 2 × B_c × d
  - For each inner iteration (T_r blocks of Q):
    - Load Q_i, O_i, l_i, m_i: B_r × (2d + 2)
    - Write O_i, l_i, m_i: B_r × (d + 2)
    
- Total HBM accesses: O(N²d² / M) where M = SRAM size

Since M is ~100KB-200KB and d is typically 64-128:
- Flash: O(N²d / M) ≈ O(N²/1000) 
- Standard: O(N²)
- Speedup: ~1000x fewer HBM accesses!

This makes Flash Attention COMPUTE-BOUND instead of MEMORY-BOUND.
"""


def analyze_io_complexity():
    """
    Analyze and compare IO complexity.
    """
    print("\n" + "="*70)
    print("IO COMPLEXITY ANALYSIS")
    print("="*70)
    
    # Parameters
    seq_lengths = [1024, 2048, 4096, 8192, 16384]
    d = 64  # head dimension
    M_sram = 192 * 1024  # 192KB SRAM per SM
    bytes_per_element = 2  # float16
    
    print(f"\nHead dimension: {d}")
    print(f"SRAM size: {M_sram / 1024:.0f} KB")
    print(f"Precision: float16 ({bytes_per_element} bytes)")
    
    # Block sizes (simplified calculation)
    B = int(math.sqrt(M_sram / (4 * d * bytes_per_element)))
    print(f"Approximate block size: {B}")
    
    print("\n" + "-"*70)
    print(f"{'Seq Len':<10} {'Standard IO':<20} {'Flash IO':<20} {'Speedup':<15}")
    print("-"*70)
    
    for N in seq_lengths:
        # Standard attention IO (bytes)
        standard_io = (2 * N * d + 3 * N * N) * bytes_per_element
        
        # Flash attention IO (bytes) - simplified
        # O(N²d² / M) accesses
        flash_io = (4 * N * d + N * N * d * d / M_sram) * bytes_per_element
        
        speedup = standard_io / flash_io
        
        print(f"{N:<10} {standard_io/1e6:.1f} MB{' '*10} {flash_io/1e6:.1f} MB{' '*10} {speedup:.1f}x")


# =============================================================================
# SECTION 7: VERIFICATION AND TESTING
# =============================================================================

def test_flash_attention_correctness():
    """
    Verify Flash Attention produces correct results.
    """
    print("\n" + "="*70)
    print("FLASH ATTENTION CORRECTNESS TEST")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Test parameters
    batch = 2
    heads = 4
    seq_len = 128
    head_dim = 64
    block_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random inputs
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, requires_grad=True)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, requires_grad=True)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, requires_grad=True)
    
    # Standard attention
    scale = 1.0 / math.sqrt(head_dim)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = F.softmax(S, dim=-1)
    O_standard = torch.matmul(P, V)
    
    # Flash attention
    Q2 = Q.detach().clone().requires_grad_(True)
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    O_flash = flash_attention(Q2, K2, V2, block_size=block_size, causal=False)
    
    # Compare forward pass
    forward_diff = (O_standard - O_flash).abs().max().item()
    print(f"\nForward pass max difference: {forward_diff:.2e}")
    print(f"Forward pass correct: {forward_diff < 1e-4}")
    
    # Test backward pass
    grad_output = torch.randn_like(O_standard)
    
    O_standard.backward(grad_output, retain_graph=True)
    O_flash.backward(grad_output)
    
    dQ_diff = (Q.grad - Q2.grad).abs().max().item()
    dK_diff = (K.grad - K2.grad).abs().max().item()
    dV_diff = (V.grad - V2.grad).abs().max().item()
    
    print(f"\nBackward pass gradients:")
    print(f"  dQ max difference: {dQ_diff:.2e}")
    print(f"  dK max difference: {dK_diff:.2e}")
    print(f"  dV max difference: {dV_diff:.2e}")
    
    # Test causal masking
    print("\n" + "-"*50)
    print("Testing Causal Masking:")
    
    Q3 = Q.detach().clone()
    K3 = K.detach().clone()
    V3 = V.detach().clone()
    
    # Standard causal attention
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    S_causal = torch.matmul(Q3, K3.transpose(-2, -1)) * scale
    S_causal = S_causal.masked_fill(causal_mask, float('-inf'))
    P_causal = F.softmax(S_causal, dim=-1)
    O_standard_causal = torch.matmul(P_causal, V3)
    
    # Flash attention causal
    O_flash_causal = flash_attention(Q3, K3, V3, block_size=block_size, causal=True)
    
    causal_diff = (O_standard_causal - O_flash_causal).abs().max().item()
    print(f"Causal attention max difference: {causal_diff:.2e}")
    print(f"Causal attention correct: {causal_diff < 1e-4}")


# =============================================================================
# SECTION 8: FLASH ATTENTION 2 AND 3 IMPROVEMENTS
# =============================================================================
"""
FLASH ATTENTION 2 IMPROVEMENTS:

1. Better Parallelism:
   - FA1: Parallelizes over batch and heads
   - FA2: Also parallelizes over sequence length
   - Better GPU utilization for long sequences

2. Reduced Non-matmul FLOPs:
   - FA1: Many element-wise operations
   - FA2: Restructured to maximize matmul efficiency
   - Matmuls use Tensor Cores, element-wise ops don't

3. Better Work Partitioning:
   - FA2: Sequences split across thread blocks
   - Each block handles a contiguous chunk
   - Reduces warp divergence

4. Loop Order Change:
   - FA1: Outer loop over KV, inner over Q
   - FA2: Outer loop over Q, inner over KV
   - Better for parallelization

FLASH ATTENTION 3 IMPROVEMENTS (H100/Hopper):

1. TMA (Tensor Memory Accelerator):
   - Hardware unit for async memory transfers
   - Overlaps compute with memory access
   - Reduces memory stalls

2. Warp Specialization:
   - Different warps do different tasks
   - Producer warps: Load data
   - Consumer warps: Compute
   - Continuous pipeline

3. FP8 Support:
   - 4x memory reduction from FP16
   - With careful handling of precision

4. Persistent Kernels:
   - Keep thread blocks alive across iterations
   - Reduce launch overhead
"""


def flash_attention_versions_comparison():
    """
    Compare theoretical performance of Flash Attention versions.
    """
    print("\n" + "="*70)
    print("FLASH ATTENTION VERSION COMPARISON")
    print("="*70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    Flash Attention Evolution                      │")
    print("├─────────────┬───────────────┬───────────────┬───────────────────┤")
    print("│ Feature     │ FA1 (2022)    │ FA2 (2023)    │ FA3 (2024)        │")
    print("├─────────────┼───────────────┼───────────────┼───────────────────┤")
    print("│ Parallelism │ Batch, Heads  │ + Seq length  │ + Warp special    │")
    print("│ Loop Order  │ KV outer      │ Q outer       │ Optimized         │")
    print("│ Memory      │ O(N)          │ O(N)          │ O(N)              │")
    print("│ Speed       │ 2-4x vs std   │ 2x vs FA1     │ 1.5-2x vs FA2     │")
    print("│ FP8 Support │ No            │ No            │ Yes               │")
    print("│ TMA         │ No            │ No            │ Yes (Hopper)      │")
    print("│ Target GPU  │ A100          │ A100, H100    │ H100, H200        │")
    print("└─────────────┴───────────────┴───────────────┴───────────────────┘")
    
    print("\nKey Architectural Changes:")
    print("-" * 50)
    print("""
FA1 → FA2:
  • Changed loop order (Q outer, KV inner)
  • Better thread block scheduling  
  • Reduced register pressure
  • ~2x speedup

FA2 → FA3:
  • Warp specialization (producer/consumer)
  • TMA for async memory
  • FP8 compute path
  • Persistent kernels
  • ~1.5-2x speedup on H100
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FLASH ATTENTION DEEP DIVE")
    print("="*70)
    
    # Section 1: Memory hierarchy demonstration
    if torch.cuda.is_available():
        demonstrate_memory_hierarchy()
    
    # Section 2: Memory scaling analysis
    analyze_attention_memory([512, 1024, 2048, 4096, 8192, 16384, 32768])
    
    # Section 3: Online softmax
    online_softmax_demonstration()
    online_softmax_with_output()
    
    # Section 4: Tiling visualization
    visualize_tiling()
    
    # Section 5: IO complexity
    analyze_io_complexity()
    
    # Section 6: Correctness test
    test_flash_attention_correctness()
    
    # Section 7: Version comparison
    flash_attention_versions_comparison()
    
    print("\n" + "="*70)
    print("DEEP DIVE COMPLETE")
    print("="*70)
