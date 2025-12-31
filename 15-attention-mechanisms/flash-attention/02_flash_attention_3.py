"""
02_flash_attention_3.py - Flash Attention 3 Deep Dive

Flash Attention 3 (FA3) brings significant improvements on Hopper GPUs (H100)
by exploiting new hardware features: WGMMA, TMA, and FP8.

Key Innovations:
1. Warp-specialization for overlapping compute and memory
2. Pingpong scheduling between warpgroups
3. Intra-warpgroup GEMM-softmax overlapping
4. FP8 support with incoherent processing

Performance: Up to 740 TFLOPS (75% of H100 peak) vs 350 TFLOPS in FA2

Reference: https://tridao.me/blog/2024/flash3/
Paper: https://arxiv.org/abs/2407.08608

Run: python 02_flash_attention_3.py
"""

import torch
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# SECTION 1: HOPPER GPU FEATURES
# ============================================================================
"""
NEW HOPPER (H100) FEATURES:
===========================

1. WGMMA (Warpgroup Matrix Multiply-Accumulate)
   - 4 warps (128 threads) work together
   - Much higher throughput than mma.sync (Ampere)
   - Asynchronous execution
   
2. TMA (Tensor Memory Accelerator)
   - Hardware unit for memory transfers
   - Handles indexing and bounds checking
   - Frees up registers for computation
   
3. FP8 Tensor Cores
   - E4M3: 4 exponent, 3 mantissa (forward pass)
   - E5M2: 5 exponent, 2 mantissa (gradients)
   - 2x throughput vs FP16 (1978 TFLOPS vs 989 TFLOPS)

PERFORMANCE COMPARISON:
=======================

| GPU     | FP16 TFLOPS | FP8 TFLOPS | Memory BW |
|---------|-------------|------------|-----------|
| A100    | 312         | N/A        | 2.0 TB/s  |
| H100    | 989         | 1978       | 3.35 TB/s |

Flash Attention Performance:
| Version | GPU   | TFLOPS | % of Peak |
|---------|-------|--------|-----------|
| FA2     | A100  | 220    | 70%       |
| FA2     | H100  | 350    | 35%       |
| FA3     | H100  | 740    | 75%       |
"""

@dataclass
class HopperSpecs:
    """H100 SXM5 specifications."""
    fp16_tflops: float = 989.0
    fp8_tflops: float = 1978.0
    memory_bandwidth_tb: float = 3.35
    sm_count: int = 132
    shared_memory_per_sm_kb: int = 228
    l2_cache_mb: int = 50
    
    # Multi-function unit (for exp, etc.)
    special_func_tflops: float = 3.9  # Much slower than matmul!


def print_hopper_overview():
    """Print Hopper GPU overview."""
    print("\n" + "="*70)
    print(" HOPPER (H100) GPU FEATURES FOR FLASH ATTENTION 3")
    print("="*70)
    
    specs = HopperSpecs()
    
    print(f"""
    H100 SXM5 KEY SPECS:
    ════════════════════
    
    Compute:
        FP16 Tensor Cores: {specs.fp16_tflops} TFLOPS
        FP8 Tensor Cores:  {specs.fp8_tflops} TFLOPS (2x FP16!)
        Special Functions: {specs.special_func_tflops} TFLOPS (exp, tanh, etc.)
        
    Memory:
        HBM3 Bandwidth: {specs.memory_bandwidth_tb} TB/s
        L2 Cache: {specs.l2_cache_mb} MB
        Shared Memory: {specs.shared_memory_per_sm_kb} KB per SM
        
    Architecture:
        SMs: {specs.sm_count}
        Warpgroups: 4 warps = 128 threads
        
    
    WHY SPECIAL FUNCTIONS MATTER:
    ═════════════════════════════
    
    Softmax requires exp() - a special function
    
    For head_dim=128:
        Matmul FLOPS: 2 × seq × seq × 128 = 256 × seq² FLOPS
        Exp FLOPS: seq² FLOPS
        
    Ratio: 256:1 matmul:exp
    
    But throughput ratio: {specs.fp16_tflops / specs.special_func_tflops:.0f}:1
    
    This means exp can take ~50% of time vs matmul!
    Solution: Overlap matmul and softmax (FA3's key insight)
    """)


# ============================================================================
# SECTION 2: WGMMA AND TMA
# ============================================================================
"""
WGMMA (Warpgroup Matrix Multiply-Accumulate):
=============================================

Old (Ampere - mma.sync):
    - Single warp (32 threads)
    - Synchronous barrier after each instruction
    - ~2/3 peak throughput

New (Hopper - WGMMA):
    - Warpgroup (4 warps = 128 threads)
    - Asynchronous execution
    - Full peak throughput
    - Larger tile sizes

WGMMA Tile Sizes:
    M × N × K = 64 × 256 × 16 (FP16)
    M × N × K = 64 × 256 × 32 (FP8)


TMA (Tensor Memory Accelerator):
================================

Without TMA:
    1. Compute indices in registers
    2. Load addresses
    3. Handle bounds checking
    4. Issue load instructions
    
With TMA:
    1. Set up descriptor once
    2. Issue single TMA instruction
    3. Hardware handles everything
    
Benefits:
    - Fewer registers used (more for tiles!)
    - Automatic coalescing
    - Async execution
"""

def print_wgmma_tma_details():
    """Print WGMMA and TMA details."""
    print("\n" + "="*70)
    print(" WGMMA AND TMA DETAILS")
    print("="*70)
    
    print("""
    WGMMA PROGRAMMING MODEL:
    ════════════════════════
    
    // Old (Ampere) - mma.sync
    for (int k = 0; k < K; k += 16) {
        __syncwarp();  // Barrier!
        mma_sync(acc, A_frag, B_frag);
    }
    
    // New (Hopper) - wgmma
    wgmma_fence();           // Start async region
    for (int k = 0; k < K; k += 16) {
        wgmma(acc, A_desc, B_desc);  // Async!
    }
    wgmma_commit_group();    // Commit all
    wgmma_wait_group();      // Wait when needed
    
    
    TMA PROGRAMMING MODEL:
    ══════════════════════
    
    // Old - Manual loads
    int idx = blockIdx.x * TILE + threadIdx.x;
    if (idx < N) {
        shared[threadIdx.x] = global[idx];
    }
    __syncthreads();
    
    // New - TMA
    // Setup (once):
    CUtensorMap tensor_map;
    cuTensorMapEncode(&tensor_map, ...);
    
    // Load (async):
    cp_async_bulk_tensor_2d_global_to_shared(
        &shared[0], &tensor_map, coords);
    cp_async_bulk_commit_group();
    
    
    REGISTER SAVINGS:
    ═════════════════
    
    Index calculation typically uses:
        - 8-16 registers for addresses
        - 4-8 registers for bounds
        
    TMA frees these for:
        - Larger tiles
        - More accumulators
        - Better occupancy
    """)


# ============================================================================
# SECTION 3: ASYNCHRONOUS OVERLAPPING
# ============================================================================
"""
THE OVERLAPPING PROBLEM:
========================

Attention = softmax(QK^T / √d) × V

Operations:
    1. S = Q × K^T        (GEMM - Tensor Cores)
    2. P = softmax(S)     (Exp, Sum - Multi-function unit)
    3. O = P × V          (GEMM - Tensor Cores)

Without overlapping:
    [GEMM1]----[Softmax]----[GEMM2]----
    
    Tensor Cores idle during softmax!

FLASH ATTENTION 3 SOLUTIONS:
============================

Solution 1: Inter-warpgroup Pingpong
    Warpgroup 1: [GEMM1]--------[GEMM2]--------
    Warpgroup 2: --------[GEMM1]--------[GEMM2]
    
    While WG1 does softmax, WG2 does GEMM!

Solution 2: Intra-warpgroup Pipelining
    Same warpgroup overlaps:
    [GEMM-iter-i]----
         [Softmax-iter-i]----
              [GEMM-iter-i+1]----
    
    Start next GEMM while softmax runs!
"""

def print_overlapping_strategy():
    """Print the overlapping strategy details."""
    print("\n" + "="*70)
    print(" ASYNCHRONOUS OVERLAPPING IN FLASH ATTENTION 3")
    print("="*70)
    
    print("""
    PINGPONG SCHEDULING (Inter-warpgroup):
    ══════════════════════════════════════
    
    Two warpgroups (WG1, WG2) alternate:
    
    Time →
    WG1: |--GEMM0--|--Softmax--|--GEMM1--|--Softmax--|--GEMM0--|...
    WG2: |--Softmax--|--GEMM0--|--Softmax--|--GEMM1--|--Softmax--|...
         
    Synchronization via bar.sync barriers
    
    Result: Softmax "hidden" under GEMM of other warpgroup
    
    Speedup: ~570 → 620 TFLOPS (FP16, head_dim=128)
    
    
    INTRA-WARPGROUP PIPELINING:
    ═══════════════════════════
    
    Within ONE warpgroup, pipeline stages:
    
    Iteration i:
        WGMMA: ████████░░░░░░░░░░░░
        Softmax:    ░░░░████████░░░░
        
    Iteration i+1:
        WGMMA:          ░░░░████████░░░░
        Softmax:            ░░░░████████
    
    The async nature of WGMMA allows this!
    
    Result: Softmax partially hidden within same warpgroup
    
    Speedup: ~620 → 660 TFLOPS (FP16)
    
    Trade-off: Higher register pressure (need both accumulators)
    
    
    COMBINED EFFECT:
    ════════════════
    
    FA2 on H100:  ~350 TFLOPS (35% peak)
    + Pingpong:   ~620 TFLOPS (63% peak)
    + Pipeline:   ~660 TFLOPS (67% peak)
    + Other opts: ~740 TFLOPS (75% peak)
    
    2x improvement from FA2 to FA3 on same hardware!
    """)


# ============================================================================
# SECTION 4: FP8 ATTENTION WITH INCOHERENT PROCESSING
# ============================================================================
"""
FP8 ATTENTION:
==============

FP8 doubles throughput: 1978 vs 989 TFLOPS

BUT: Lower precision causes quality degradation

Problem: QK^T can have outliers
    - Some attention scores much larger than others
    - FP8 can't represent the range well

INCOHERENT PROCESSING:
======================

Insight: Outliers often in same positions across heads

Solution:
    1. Per-head random orthogonal transformation
    2. "Spreads" outliers across all positions
    3. Makes values more uniform → better for FP8

Math:
    Q' = Q × R_q    (R_q is random orthogonal)
    K' = K × R_k
    
    Attention is the same because:
    softmax(Q'K'^T) = softmax(Q × R_q × R_k^T × K^T)
    
    If R_q = R_k, this equals softmax(QK^T) exactly!

Result: FP8 with incoherent processing ≈ BF16 quality
"""

def demonstrate_incoherent_processing():
    """Demonstrate incoherent processing concept."""
    print("\n" + "="*70)
    print(" FP8 WITH INCOHERENT PROCESSING")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Simulate Q, K with outliers
    seq_len, head_dim = 128, 64
    
    Q = torch.randn(seq_len, head_dim)
    K = torch.randn(seq_len, head_dim)
    
    # Add outliers (common in real attention)
    Q[0, :8] = 10.0  # Large values in first row
    K[0, :8] = 10.0
    
    # Original attention scores
    scores = Q @ K.T / math.sqrt(head_dim)
    
    print(f"\n Original Attention Scores:")
    print(f"   Max: {scores.max().item():.2f}")
    print(f"   Min: {scores.min().item():.2f}")
    print(f"   Std: {scores.std().item():.2f}")
    print(f"   Dynamic range: {scores.max() - scores.min():.2f}")
    
    # Create random orthogonal matrix (Hadamard-like for efficiency)
    R = torch.randn(head_dim, head_dim)
    R, _ = torch.linalg.qr(R)  # Make orthogonal
    
    # Transform Q and K
    Q_transformed = Q @ R
    K_transformed = K @ R
    
    # Transformed attention scores
    scores_transformed = Q_transformed @ K_transformed.T / math.sqrt(head_dim)
    
    print(f"\n After Incoherent Processing:")
    print(f"   Max: {scores_transformed.max().item():.2f}")
    print(f"   Min: {scores_transformed.min().item():.2f}")
    print(f"   Std: {scores_transformed.std().item():.2f}")
    print(f"   Dynamic range: {scores_transformed.max() - scores_transformed.min():.2f}")
    
    # Verify correctness (should be identical)
    diff = (scores - scores_transformed).abs().max()
    print(f"\n Correctness Check:")
    print(f"   Max difference: {diff.item():.2e}")
    print(f"   Mathematically equivalent: {diff < 1e-5}")
    
    print(f"""
    WHY THIS HELPS FP8:
    ═══════════════════
    
    FP8 E4M3 range: ~[-240, 240]
    
    Original scores:
        Range: {scores.max() - scores.min():.1f}
        Outliers cause clipping/overflow
        
    Transformed scores:
        Range: {scores_transformed.max() - scores_transformed.min():.1f}
        More uniform → better FP8 representation
        
    Result: FP8 quality ≈ BF16 quality with incoherent processing!
    """)


# ============================================================================
# SECTION 5: FLASH ATTENTION 3 ALGORITHM
# ============================================================================
"""
FLASH ATTENTION 3 ALGORITHM (Simplified):
=========================================

Input: Q, K, V ∈ R^{N×d}
Output: O ∈ R^{N×d}

1. Divide Q into blocks of size Br
2. Divide K, V into blocks of size Bc

3. For each Q block i:
   a. Initialize O_i = 0, l_i = 0, m_i = -inf  (online softmax state)
   
   b. For each K, V block j (PIPELINED):
      - Load K_j, V_j via TMA (async)
      - S_ij = Q_i × K_j^T / √d          (WGMMA)
      - Update online softmax state
      - O_i = update(O_i, S_ij, V_j)     (WGMMA)
      - Overlap: Start next K,V load while computing
      
   c. Store O_i to HBM via TMA

Key Differences from FA2:
- WGMMA instead of mma.sync
- TMA for all memory operations
- Pingpong between warpgroups
- Intra-warpgroup pipelining
- Optional FP8 with incoherent processing
"""

def print_fa3_algorithm():
    """Print FA3 algorithm details."""
    print("\n" + "="*70)
    print(" FLASH ATTENTION 3 ALGORITHM")
    print("="*70)
    
    print("""
    PSEUDO-CODE (Hopper-specific):
    ══════════════════════════════
    
    # Setup
    tensor_map_Q = create_tma_descriptor(Q)
    tensor_map_K = create_tma_descriptor(K)
    tensor_map_V = create_tma_descriptor(V)
    tensor_map_O = create_tma_descriptor(O)
    
    # Kernel launch (2 warpgroups)
    for q_block in range(num_q_blocks):
        
        # Each warpgroup handles different blocks
        wg_id = get_warpgroup_id()  # 0 or 1
        
        if wg_id == 0:
            # Warpgroup 0: Start with GEMM
            init_accumulators()
            
        # Pingpong loop
        for kv_block in range(num_kv_blocks):
            
            # TMA prefetch (async)
            if next_block_exists:
                tma_prefetch(K_next, V_next)
            
            # WGMMA for Q×K^T
            wgmma_fence()
            for k in range(0, head_dim, WGMMA_K):
                wgmma(S_acc, Q_tile, K_tile)
            wgmma_wait()
            
            # Softmax (while other warpgroup does WGMMA)
            compute_online_softmax(S_acc, m, l)
            
            # WGMMA for P×V
            wgmma_fence()
            for k in range(0, head_dim, WGMMA_K):
                wgmma(O_acc, P_tile, V_tile)
            wgmma_wait()
            
            # Synchronize warpgroups (pingpong)
            bar_sync(wg_id)
        
        # Write output via TMA
        tma_store(O_block, tensor_map_O)
    
    
    BLOCK SIZES (Typical):
    ══════════════════════
    
    Br (Q block):  128
    Bc (K/V block): 128-256
    Head dim:      64, 128
    
    Shared memory: ~228 KB per SM on H100
    Registers:     ~256 per thread
    
    
    MEMORY ACCESS PATTERN:
    ══════════════════════
    
    Read from HBM:
        Q: Once per row (streamed)
        K, V: Once total (cached in L2)
        
    Write to HBM:
        O: Once per row
        
    Memory complexity: O(N) - same as FA1/FA2
    Compute complexity: O(N²d) - same as standard attention
    """)


# ============================================================================
# SECTION 6: BENCHMARKS AND USAGE
# ============================================================================

def print_benchmarks():
    """Print FA3 benchmarks."""
    print("\n" + "="*70)
    print(" FLASH ATTENTION 3 BENCHMARKS")
    print("="*70)
    
    print("""
    FORWARD PASS (H100 SXM5):
    ═════════════════════════
    
    FP16, causal=False:
    
    | Head Dim | Seq Len | FA2 TFLOPS | FA3 TFLOPS | Speedup |
    |----------|---------|------------|------------|---------|
    | 64       | 1K      | 310        | 580        | 1.87x   |
    | 64       | 4K      | 340        | 630        | 1.85x   |
    | 64       | 16K     | 355        | 660        | 1.86x   |
    | 128      | 1K      | 320        | 620        | 1.94x   |
    | 128      | 4K      | 350        | 680        | 1.94x   |
    | 128      | 16K     | 360        | 740        | 2.06x   |
    
    
    FP8 (with incoherent processing):
    
    | Head Dim | Seq Len | FP16 TFLOPS | FP8 TFLOPS | Speedup |
    |----------|---------|-------------|------------|---------|
    | 128      | 4K      | 680         | 1200       | 1.76x   |
    | 128      | 16K     | 740         | 1300       | 1.76x   |
    
    FP8 achieves ~1.2-1.3 PFLOPS (66% of FP8 peak)!
    
    
    BACKWARD PASS:
    ══════════════
    
    | Head Dim | Seq Len | FA2 TFLOPS | FA3 TFLOPS | Speedup |
    |----------|---------|------------|------------|---------|
    | 128      | 4K      | 280        | 510        | 1.82x   |
    | 128      | 16K     | 300        | 560        | 1.87x   |
    
    
    VS CUDNN AND OTHERS:
    ════════════════════
    
    cuDNN 9.0:  ~580 TFLOPS (FP16)
    FA3:        ~740 TFLOPS (FP16)
    
    FA3 is ~27% faster than cuDNN!
    """)


def print_usage():
    """Print FA3 usage instructions."""
    print("\n" + "="*70)
    print(" USING FLASH ATTENTION 3")
    print("="*70)
    
    print("""
    INSTALLATION:
    ═════════════
    
    # From GitHub (requires H100 GPU)
    pip install flash-attn --no-build-isolation
    
    # Or build from source for FA3 features
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    python setup.py install
    
    
    BASIC USAGE:
    ════════════
    
    from flash_attn import flash_attn_func
    
    # Standard FP16
    output = flash_attn_func(
        q, k, v,                    # (batch, seqlen, heads, head_dim)
        causal=True,                # Causal masking for LLMs
        softmax_scale=1/sqrt(d),    # Optional custom scale
    )
    
    # FP8 (Hopper only, requires FA3)
    output = flash_attn_func(
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        v.to(torch.float8_e4m3fn),
        causal=True,
    )
    
    
    PYTORCH SDPA INTEGRATION:
    ═════════════════════════
    
    import torch.nn.functional as F
    
    # PyTorch automatically uses Flash Attention when available
    output = F.scaled_dot_product_attention(
        query, key, value,
        is_causal=True,
        enable_flash=True,   # Hint to use Flash Attention
    )
    
    
    CHECKING FA3 AVAILABILITY:
    ══════════════════════════
    
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
    
    # Check if Hopper features available
    import torch
    if torch.cuda.get_device_capability()[0] >= 9:
        print("Hopper GPU detected - FA3 features available")
    else:
        print("Pre-Hopper GPU - using FA2")
    
    
    MEMORY USAGE:
    ═════════════
    
    Standard Attention: O(N²) memory
    Flash Attention:    O(N) memory
    
    For seq_len=16K, batch=8, heads=32, d=128:
        Standard: 16K × 16K × 8 × 32 × 4 bytes = 256 GB (!)
        Flash:    ~2 GB (just Q, K, V, O)
    """)


# ============================================================================
# MAIN
# ============================================================================

def print_summary():
    """Print FA3 summary."""
    print("\n" + "="*70)
    print(" FLASH ATTENTION 3 SUMMARY")
    print("="*70)
    
    print("""
    KEY INNOVATIONS:
    ════════════════
    
    1. WGMMA + TMA
       - New Hopper instructions
       - Higher throughput, async execution
       - Register savings from TMA
    
    2. PINGPONG SCHEDULING
       - Two warpgroups alternate
       - Softmax hidden under GEMM
       - ~15% speedup
    
    3. INTRA-WARPGROUP PIPELINING
       - GEMM and softmax overlap
       - Additional ~5% speedup
    
    4. FP8 SUPPORT
       - 2x throughput potential
       - Incoherent processing for quality
       - ~1.3 PFLOPS achieved
    
    
    WHEN TO USE FA3:
    ════════════════
    
    ✓ H100 (or newer Hopper) GPUs
    ✓ Long sequences (4K+ tokens)
    ✓ Training or inference
    ✓ Need maximum performance
    
    For older GPUs (A100, etc.):
        Use Flash Attention 2
    
    
    LIMITATIONS:
    ════════════
    
    - Hopper only (compute capability 9.0+)
    - Head dimensions: 64, 128 (optimized)
    - Some custom attention patterns not supported
    
    
    FUTURE DIRECTIONS:
    ══════════════════
    
    - Flash Attention 4? (Blackwell GPUs)
    - More FP8 optimizations
    - Custom attention patterns
    - Ring Attention integration
    """)


if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " FLASH ATTENTION 3 ".center(68) + "║")
    print("║" + " Hopper Optimizations: WGMMA, TMA, FP8 ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print_hopper_overview()
    print_wgmma_tma_details()
    print_overlapping_strategy()
    demonstrate_incoherent_processing()
    print_fa3_algorithm()
    print_benchmarks()
    print_usage()
    print_summary()
