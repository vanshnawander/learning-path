"""
03_efficient_attention.py - Efficient Attention Mechanisms

The quadratic O(N²) complexity of attention is THE major bottleneck.
This module covers solutions: exact efficient (Flash Attention) and
approximate (Linear Attention, Sparse Attention).

Complexity Comparison:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Method              │ Time      │ Memory    │ Exact? │ Notes               │
├─────────────────────┼───────────┼───────────┼────────┼─────────────────────┤
│ Standard Attention  │ O(N²d)    │ O(N²)     │ Yes    │ Baseline            │
│ Flash Attention     │ O(N²d)    │ O(N)      │ Yes    │ IO-aware tiling     │
│ Flash Attention 2   │ O(N²d)    │ O(N)      │ Yes    │ Better parallelism  │
│ Linear Attention    │ O(Nd²)    │ O(Nd)     │ No     │ Remove softmax      │
│ Sparse Attention    │ O(N√N)    │ O(N√N)    │ No     │ Fixed patterns      │
│ Sliding Window      │ O(Nw)     │ O(Nw)     │ No     │ Local attention     │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 03_efficient_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple

# ============================================================================
# PROFILING
# ============================================================================

def profile_fn(func, warmup=5, iterations=20):
    """Profile with CUDA timing."""
    if torch.cuda.is_available():
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
    else:
        for _ in range(warmup):
            func()
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        return (time.perf_counter() - start) * 1000 / iterations

# ============================================================================
# THE QUADRATIC PROBLEM
# ============================================================================

def explain_quadratic_problem():
    """Explain why O(N²) is problematic."""
    print("\n" + "="*70)
    print(" THE QUADRATIC PROBLEM")
    print(" Why O(N²) limits transformers")
    print("="*70)
    
    print("""
    STANDARD ATTENTION COMPLEXITY:
    ─────────────────────────────────────────────────────────────────
    
    Attention(Q, K, V) = softmax(QK^T / √d) · V
    
    Q, K, V ∈ ℝ^(N×d)
    
    Step 1: QK^T           → O(N²d) compute, O(N²) memory
    Step 2: softmax(...)   → O(N²) compute
    Step 3: ... · V        → O(N²d) compute
    
    Total: O(N²d) compute, O(N²) memory
    
    THE MEMORY PROBLEM (MORE CRITICAL):
    ─────────────────────────────────────────────────────────────────
    
    Attention matrix: N × N floats
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Sequence Length │ Attention Matrix │ Memory (FP16)              │
    ├─────────────────┼──────────────────┼────────────────────────────┤
    │ 512             │ 262K elements    │ 0.5 MB                     │
    │ 2,048           │ 4.2M elements    │ 8 MB                       │
    │ 8,192           │ 67M elements     │ 134 MB                     │
    │ 32,768          │ 1.07B elements   │ 2.1 GB                     │
    │ 131,072 (128K)  │ 17.2B elements   │ 34 GB ← Single layer!     │
    └─────────────────────────────────────────────────────────────────┘
    
    With batch size B and H heads:
    Memory = B × H × N² × sizeof(float)
    
    Example: B=8, H=32, N=8192, FP16
    Memory = 8 × 32 × 8192² × 2 = 34 GB just for attention!
    
    WHY THIS MATTERS FOR LLMS:
    ─────────────────────────────────────────────────────────────────
    
    1. CONTEXT LENGTH LIMITED
       - GPT-2: 1024 tokens
       - GPT-3: 2048 tokens
       - GPT-4: 8K-128K tokens (needs efficient attention!)
    
    2. BATCH SIZE LIMITED
       - Can't fit many sequences
       - Lower throughput
    
    3. INFERENCE BOTTLENECK
       - KV cache grows with sequence length
       - Long conversations become slow
    
    THE TWO APPROACHES:
    ─────────────────────────────────────────────────────────────────
    
    1. EXACT EFFICIENT: Keep exact attention, optimize implementation
       - Flash Attention: IO-aware algorithm
       - Still O(N²) compute but O(N) memory
    
    2. APPROXIMATE: Change the attention mechanism
       - Linear Attention: Remove softmax
       - Sparse Attention: Only attend to subset
       - Sliding Window: Local attention only
    """)

# ============================================================================
# FLASH ATTENTION
# ============================================================================

def explain_flash_attention():
    """Explain Flash Attention in detail."""
    print("\n" + "="*70)
    print(" FLASH ATTENTION")
    print(" IO-aware exact attention (Dao et al. 2022)")
    print("="*70)
    
    print("""
    THE KEY INSIGHT: MEMORY HIERARCHY
    ─────────────────────────────────────────────────────────────────
    
    GPU Memory Hierarchy:
    ┌─────────────────────────────────────────────────────────────────┐
    │ HBM (High Bandwidth Memory)    │ 40-80 GB │ ~2 TB/s           │
    │ └── Where Q, K, V, Output live │          │ (relatively slow) │
    ├─────────────────────────────────────────────────────────────────┤
    │ SRAM (Shared Memory)           │ 20-200 KB │ ~19 TB/s         │
    │ └── Per SM, very fast          │ per SM    │ (10x faster)     │
    └─────────────────────────────────────────────────────────────────┘
    
    Standard Attention: Memory-Bound!
    - Write N×N attention matrix to HBM
    - Read it back for softmax
    - Read it back again for V multiplication
    - Each step limited by HBM bandwidth
    
    FLASH ATTENTION ALGORITHM:
    ─────────────────────────────────────────────────────────────────
    
    Key idea: NEVER materialize the full N×N matrix!
    
    Instead:
    1. Load blocks of Q, K, V into SRAM
    2. Compute attention for that block in SRAM
    3. Accumulate output incrementally
    4. Use online softmax to avoid storing full attention matrix
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ TILING STRATEGY                                                 │
    │                                                                 │
    │ K and V: Split into blocks of size B_c                         │
    │ Q: Split into blocks of size B_r                               │
    │                                                                 │
    │     K₁    K₂    K₃    K₄                                       │
    │   ┌─────┬─────┬─────┬─────┐                                    │
    │ Q₁│  *  │  *  │  *  │  *  │ → Compute block by block          │
    │   ├─────┼─────┼─────┼─────┤    in SRAM, never store full      │
    │ Q₂│  *  │  *  │  *  │  *  │    N×N matrix in HBM              │
    │   ├─────┼─────┼─────┼─────┤                                    │
    │ Q₃│  *  │  *  │  *  │  *  │                                    │
    │   └─────┴─────┴─────┴─────┘                                    │
    └─────────────────────────────────────────────────────────────────┘
    
    ONLINE SOFTMAX (The Clever Trick):
    ─────────────────────────────────────────────────────────────────
    
    Problem: softmax needs the full row to normalize
    Solution: Incremental softmax with running statistics
    
    For each block j:
    1. Compute local scores: S_j = Q_i @ K_j^T
    2. Compute local max: m_j = max(S_j)
    3. Update running max: m_new = max(m_old, m_j)
    4. Rescale previous output: O *= exp(m_old - m_new)
    5. Add current contribution: O += exp(S_j - m_new) @ V_j
    6. Update normalizer similarly
    
    Result: EXACT same output as standard attention!
    
    MEMORY SAVINGS:
    ─────────────────────────────────────────────────────────────────
    
    Standard: O(N²) - store full attention matrix
    Flash:    O(N)  - only store output and running statistics
    
    Speedup comes from:
    - Fewer HBM reads/writes (IO bound → compute bound)
    - Better GPU utilization
    - Fused kernel (no intermediate allocations)
    """)
    
    # Demonstrate Flash Attention if available
    print("\n FLASH ATTENTION vs STANDARD:")
    print("-" * 50)
    
    if torch.cuda.is_available():
        device = 'cuda'
        batch_size = 4
        num_heads = 8
        head_dim = 64
        
        for seq_len in [512, 1024, 2048, 4096]:
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                          device=device, dtype=torch.float16)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                          device=device, dtype=torch.float16)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                          device=device, dtype=torch.float16)
            
            # Standard attention
            def standard_attn():
                scale = 1.0 / math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = F.softmax(scores, dim=-1)
                return torch.matmul(attn, v)
            
            # Flash attention (via SDPA)
            def flash_attn():
                return F.scaled_dot_product_attention(q, k, v)
            
            try:
                time_std = profile_fn(standard_attn)
                time_flash = profile_fn(flash_attn)
                speedup = time_std / time_flash
                print(f" seq_len={seq_len:4d}: Standard={time_std:.2f}ms, "
                      f"Flash={time_flash:.2f}ms, Speedup={speedup:.2f}x")
            except Exception as e:
                print(f" seq_len={seq_len}: Error - {e}")
    else:
        print(" CUDA not available")

# ============================================================================
# LINEAR ATTENTION
# ============================================================================

class LinearAttention(nn.Module):
    """
    Linear Attention - O(N) complexity by removing softmax.
    
    Key insight: softmax(QK^T)V can be rewritten if we remove softmax
    
    Standard: softmax(QK^T) @ V
    Linear:   φ(Q) @ (φ(K)^T @ V)
    
    By associativity: compute K^T @ V first → O(d²) per position
    
    φ is a feature map (e.g., elu(x) + 1)
    """
    
    def __init__(self, d_model: int, num_heads: int, feature_map: str = 'elu'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.feature_map = feature_map
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map to make values positive."""
        if self.feature_map == 'elu':
            return F.elu(x) + 1
        elif self.feature_map == 'relu':
            return F.relu(x)
        elif self.feature_map == 'softmax':
            return F.softmax(x, dim=-1)
        else:
            return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Apply feature maps
        Q = self._feature_map(Q)  # (B, H, N, d)
        K = self._feature_map(K)
        
        # Linear attention: Q @ (K^T @ V)
        # Key insight: compute K^T @ V first → (B, H, d, d)
        KV = torch.einsum('bhnd,bhnv->bhdv', K, V)  # (B, H, d_k, d_v)
        
        # Then Q @ KV → (B, H, N, d_v)
        output = torch.einsum('bhnd,bhdv->bhnv', Q, KV)
        
        # Normalize by sum of K
        K_sum = K.sum(dim=2, keepdim=True)  # (B, H, 1, d)
        normalizer = torch.einsum('bhnd,bhkd->bhnk', Q, K_sum)  # (B, H, N, 1)
        output = output / (normalizer + 1e-6)
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(output)


def explain_linear_attention():
    """Explain linear attention mechanisms."""
    print("\n" + "="*70)
    print(" LINEAR ATTENTION")
    print(" O(N) complexity by removing softmax")
    print("="*70)
    
    print("""
    THE CORE IDEA:
    ─────────────────────────────────────────────────────────────────
    
    Standard attention:
    Attention(Q, K, V) = softmax(QK^T) @ V
    
    The softmax couples all positions → must compute N×N matrix
    
    Linear attention insight:
    If we use φ(Q)φ(K)^T instead of softmax(QK^T):
    
    φ(Q) @ (φ(K)^T @ V)
    
    By matrix associativity:
    - Compute φ(K)^T @ V first: O(Nd²) 
    - Result is d×d matrix!
    - Then φ(Q) @ result: O(Nd²)
    
    Total: O(Nd²) instead of O(N²d)
    
    For d << N, this is a HUGE win!
    
    FEATURE MAPS φ:
    ─────────────────────────────────────────────────────────────────
    
    Requirements:
    - φ(x) ≥ 0 (for valid attention weights)
    - Approximate softmax behavior
    
    Common choices:
    1. elu(x) + 1  (simple, effective)
    2. ReLU (sparse)
    3. Random features (theoretical guarantees)
    4. Performer: exp(x - max(x)) with random projections
    
    CAUSAL LINEAR ATTENTION:
    ─────────────────────────────────────────────────────────────────
    
    For autoregressive models, maintain running state:
    
    S_t = S_{t-1} + k_t ⊗ v_t   (state update)
    z_t = z_{t-1} + k_t          (normalizer update)
    o_t = q_t @ S_t / (q_t @ z_t) (output)
    
    This is like an RNN! O(1) per step inference.
    
    PROS AND CONS:
    ─────────────────────────────────────────────────────────────────
    
    ✓ O(N) complexity - scales to very long sequences
    ✓ RNN-like inference - O(1) per token
    ✓ Constant memory during generation
    
    ✗ Quality gap vs standard attention
    ✗ Loses some expressiveness
    ✗ Feature map choice is tricky
    """)
    
    # Demonstrate linear attention
    print("\n LINEAR vs STANDARD ATTENTION:")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 256
    num_heads = 8
    batch_size = 4
    
    linear_attn = LinearAttention(d_model, num_heads).to(device)
    
    for seq_len in [256, 512, 1024, 2048, 4096]:
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        time_ms = profile_fn(lambda: linear_attn(x))
        
        # Compare scaling
        ratio = time_ms / profile_fn(lambda: linear_attn(
            torch.randn(batch_size, 256, d_model, device=device)))
        
        print(f" seq_len={seq_len:4d}: {time_ms:.3f}ms (ratio: {ratio:.1f}x)")

# ============================================================================
# SPARSE ATTENTION
# ============================================================================

def explain_sparse_attention():
    """Explain various sparse attention patterns."""
    print("\n" + "="*70)
    print(" SPARSE ATTENTION")
    print(" Attend to subset of positions")
    print("="*70)
    
    print("""
    SPARSE ATTENTION PATTERNS:
    ─────────────────────────────────────────────────────────────────
    
    1. SLIDING WINDOW (Local Attention)
    ─────────────────────────────────────────────────────────────────
    
    Each position attends only to w neighbors:
    
    ┌───────────────────────────────────────┐
    │ Attention pattern (window=3):         │
    │                                       │
    │     1  2  3  4  5  6  7  8           │
    │  1 [■  ■  ■  ·  ·  ·  ·  ·]          │
    │  2 [■  ■  ■  ■  ·  ·  ·  ·]          │
    │  3 [■  ■  ■  ■  ■  ·  ·  ·]          │
    │  4 [·  ■  ■  ■  ■  ■  ·  ·]          │
    │  5 [·  ·  ■  ■  ■  ■  ■  ·]          │
    │  ...                                  │
    │                                       │
    │ Complexity: O(N × w)                  │
    │ Used in: Longformer, Mistral         │
    └───────────────────────────────────────┘
    
    2. STRIDED/DILATED ATTENTION
    ─────────────────────────────────────────────────────────────────
    
    Attend to every k-th position:
    
    ┌───────────────────────────────────────┐
    │ Strided pattern (stride=2):           │
    │                                       │
    │     1  2  3  4  5  6  7  8           │
    │  1 [■  ·  ■  ·  ■  ·  ■  ·]          │
    │  2 [·  ■  ·  ■  ·  ■  ·  ■]          │
    │  3 [■  ·  ■  ·  ■  ·  ■  ·]          │
    │  ...                                  │
    │                                       │
    │ Captures long-range with fewer ops   │
    └───────────────────────────────────────┘
    
    3. GLOBAL + LOCAL (Longformer, BigBird)
    ─────────────────────────────────────────────────────────────────
    
    Combine local window + global tokens:
    
    ┌───────────────────────────────────────┐
    │ [CLS] attends to all, all attend to [CLS]
    │                                       │
    │      CLS  1  2  3  4  5  6  7        │
    │ CLS [ ■   ■  ■  ■  ■  ■  ■  ■ ]     │ ← Global
    │  1  [ ■   ■  ■  ■  ·  ·  ·  · ]      │
    │  2  [ ■   ■  ■  ■  ■  ·  ·  · ]      │ ← Local window
    │  3  [ ■   ·  ■  ■  ■  ■  ·  · ]      │
    │  ...                                  │
    │                                       │
    │ O(N × (w + g)) where g = global tokens│
    └───────────────────────────────────────┘
    
    4. BLOCKWISE/CHUNKED
    ─────────────────────────────────────────────────────────────────
    
    Divide into blocks, full attention within blocks:
    
    ┌───────────────────────────────────────┐
    │      1  2  3  4 | 5  6  7  8         │
    │   1 [■  ■  ■  ■ | ·  ·  ·  ·]        │
    │   2 [■  ■  ■  ■ | ·  ·  ·  ·]        │
    │   3 [■  ■  ■  ■ | ·  ·  ·  ·]        │
    │   4 [■  ■  ■  ■ | ·  ·  ·  ·]        │
    │   ─────────────────────────          │
    │   5 [·  ·  ·  · | ■  ■  ■  ■]        │
    │   6 [·  ·  ·  · | ■  ■  ■  ■]        │
    │   7 [·  ·  ·  · | ■  ■  ■  ■]        │
    │   8 [·  ·  ·  · | ■  ■  ■  ■]        │
    │                                       │
    │ O(N × B) where B = block size        │
    └───────────────────────────────────────┘
    
    COMPARISON:
    ─────────────────────────────────────────────────────────────────
    
    ┌──────────────────┬───────────┬─────────────┬─────────────────┐
    │ Pattern          │ Complexity│ Long-Range  │ Used In         │
    ├──────────────────┼───────────┼─────────────┼─────────────────┤
    │ Full             │ O(N²)     │ Full        │ BERT, GPT       │
    │ Sliding Window   │ O(Nw)     │ Limited     │ Mistral, LLaMA  │
    │ Global + Local   │ O(N(w+g)) │ Via globals │ Longformer      │
    │ Strided          │ O(N×N/s)  │ Good        │ Sparse Trans.   │
    │ Random           │ O(Nk)     │ Probabilistic│ BigBird         │
    └──────────────────┴───────────┴─────────────┴─────────────────┘
    """)

# ============================================================================
# GROUPED QUERY ATTENTION (GQA)
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - Used in LLaMA 2, Mistral.
    
    Key insight: Share K and V heads across multiple Q heads.
    
    MHA: Each Q head has its own K, V heads
    MQA: All Q heads share ONE K, V head (extreme)
    GQA: Groups of Q heads share K, V heads (balanced)
    
    Benefits:
    - Smaller KV cache (important for inference!)
    - Similar quality to MHA with fewer parameters
    """
    
    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,  # num_q_heads must be divisible by num_kv_heads
        dropout: float = 0.0
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        Q = self.W_q(x).reshape(batch_size, seq_len, self.num_q_heads, self.head_dim)
        K = self.W_k(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.W_v(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # (B, num_q_heads, N, d)
        K = K.transpose(1, 2)  # (B, num_kv_heads, N, d)
        V = V.transpose(1, 2)
        
        # Expand K, V to match Q heads (repeat for each group)
        K = K.repeat_interleave(self.num_groups, dim=1)  # (B, num_q_heads, N, d)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)


def explain_gqa():
    """Explain Grouped Query Attention."""
    print("\n" + "="*70)
    print(" GROUPED QUERY ATTENTION (GQA)")
    print(" Efficient KV cache for inference")
    print("="*70)
    
    print("""
    THE KV CACHE PROBLEM:
    ─────────────────────────────────────────────────────────────────
    
    During autoregressive generation:
    - Must store K, V for all previous tokens
    - KV cache size = 2 × layers × heads × seq_len × head_dim
    
    For LLaMA 70B at 4K context:
    - 80 layers, 64 heads, head_dim=128
    - KV cache = 2 × 80 × 64 × 4096 × 128 × 2 bytes = 10.7 GB!
    
    GQA SOLUTION:
    ─────────────────────────────────────────────────────────────────
    
    Share K, V heads across groups of Q heads:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Multi-Head Attention (MHA):                                     │
    │                                                                 │
    │ Q heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8                       │
    │           │   │   │   │   │   │   │   │                        │
    │ K heads:  K1  K2  K3  K4  K5  K6  K7  K8  (8 KV heads)         │
    │ V heads:  V1  V2  V3  V4  V5  V6  V7  V8                       │
    ├─────────────────────────────────────────────────────────────────┤
    │ Grouped Query Attention (GQA with 2 KV heads):                  │
    │                                                                 │
    │ Q heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8                       │
    │           └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                       │
    │              │       │       │       │                          │
    │ K heads:     K1      K1      K2      K2    (2 KV heads)         │
    │ V heads:     V1      V1      V2      V2                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ Multi-Query Attention (MQA - extreme):                          │
    │                                                                 │
    │ Q heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8                       │
    │           └───────────┴───────────────┘                         │
    │                       │                                         │
    │ K heads:              K1                  (1 KV head)           │
    │ V heads:              V1                                        │
    └─────────────────────────────────────────────────────────────────┘
    
    KV CACHE SAVINGS:
    ─────────────────────────────────────────────────────────────────
    
    ┌───────────────┬────────────┬───────────────────────────────────┐
    │ Method        │ KV Heads   │ Cache Size (relative to MHA)      │
    ├───────────────┼────────────┼───────────────────────────────────┤
    │ MHA           │ 8          │ 100%                              │
    │ GQA (groups=4)│ 2          │ 25%                               │
    │ MQA           │ 1          │ 12.5%                             │
    └───────────────┴────────────┴───────────────────────────────────┘
    
    USED IN:
    - LLaMA 2 (70B uses GQA with 8 KV heads for 64 Q heads)
    - Mistral 7B (GQA)
    - Falcon (MQA in some versions)
    """)

# ============================================================================
# SLIDING WINDOW ATTENTION
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention - Local attention only.
    
    Each position attends to window_size neighbors.
    Used in Mistral, Longformer.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.head_dim)
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Create sliding window mask
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, :start] = False
            mask[i, end:] = False
        
        # Standard attention with mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        return self.W_o(output)


# ============================================================================
# SUMMARY
# ============================================================================

def print_efficient_summary():
    """Print efficient attention summary."""
    print("\n" + "="*70)
    print(" EFFICIENT ATTENTION SUMMARY")
    print("="*70)
    
    print("""
    CHOOSING THE RIGHT APPROACH:
    
    1. FLASH ATTENTION (Recommended Default)
       When: Standard transformer with exact attention needed
       Pros: Exact, faster, less memory
       Cons: Still O(N²) compute
    
    2. SLIDING WINDOW
       When: Local context is sufficient
       Pros: O(Nw) complexity, simple
       Cons: No long-range attention
       Used: Mistral, many modern LLMs
    
    3. LINEAR ATTENTION
       When: Very long sequences, can tolerate approximation
       Pros: O(N) complexity, RNN-like inference
       Cons: Quality gap, feature map choice
    
    4. GROUPED QUERY ATTENTION
       When: Inference efficiency matters, large KV cache
       Pros: Smaller cache, similar quality
       Cons: Slightly reduced expressiveness
       Used: LLaMA 2, Mistral
    
    PRACTICAL RECOMMENDATIONS:
    ─────────────────────────────────────────────────────────────────
    
    Training:
    - Use Flash Attention 2 (via SDPA)
    - Gradient checkpointing for memory
    
    Inference:
    - GQA for smaller KV cache
    - Sliding window for very long sequences
    - Speculative decoding for latency
    
    Very Long Context (100K+):
    - Sliding window + global tokens
    - Linear attention variants
    - Consider RWKV/Mamba (next section!)
    
    NEXT: 04_advanced_attention.py - RWKV, Mamba, State Space Models
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " EFFICIENT ATTENTION MECHANISMS ".center(68) + "║")
    print("║" + " Scaling attention to long sequences ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    explain_quadratic_problem()
    explain_flash_attention()
    explain_linear_attention()
    explain_sparse_attention()
    explain_gqa()
    print_efficient_summary()
    
    print("\n" + "="*70)
    print(" Efficient attention enables long-context LLMs!")
    print("="*70)
