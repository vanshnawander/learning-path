"""
02_matmul_kernel.py - Matrix Multiplication in Triton

Matrix multiplication is THE most important operation in deep learning.
Understanding how to implement efficient matmul teaches:
- Tiling and blocking
- Memory hierarchy utilization
- Software pipelining
- Auto-tuning

C = A @ B
Where A is (M, K), B is (K, N), C is (M, N)

Run: python 02_matmul_kernel.py
"""

import torch
import triton
import triton.language as tl
import time

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
# NAIVE MATMUL KERNEL
# ============================================================================

@triton.jit
def matmul_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Naive tiled matrix multiplication.
    
    Each program computes a BLOCK_M x BLOCK_N tile of C.
    
    Key insight: Tiling reduces memory traffic!
    Without tiling: Each element of A and B read M*N times
    With tiling: Each element read M*N / (BLOCK_M * BLOCK_N) times
    """
    # Which tile of C are we computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Starting indices for this tile
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Offsets within our tile
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    
    # Iterate over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # Load A tile: (BLOCK_M, BLOCK_K)
        # A[m, k] = a_ptr + m * stride_am + k * stride_ak
        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile: (BLOCK_K, BLOCK_N)
        # B[k, n] = b_ptr + k * stride_bk + n * stride_bn
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate: C += A @ B
        acc += tl.dot(a_tile, b_tile)
    
    # Store result
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive matmul."""
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    # Grid: one program per output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel_naive[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return c

# ============================================================================
# OPTIMIZED MATMUL WITH AUTO-TUNING
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized matmul with:
    1. Auto-tuned block sizes
    2. Software pipelining (num_stages)
    3. L2 cache optimization via tile grouping
    """
    # Program ID with grouped ordering for L2 cache efficiency
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Offsets
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    
    # Pointers to first tiles
    a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
    b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # Load with masking
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < k_remaining)
        b_mask = (k_offs[:, None] < k_remaining) & (n_offs[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_offs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_offs, acc, mask=c_mask)


def triton_matmul_optimized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized matmul."""
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Grid: 1D for optimized ordering
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    matmul_kernel_optimized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_tiling_importance():
    """Demonstrate why tiling is essential for matmul."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: WHY TILING MATTERS")
    print(" Reducing memory traffic through data reuse")
    print("="*70)
    
    print("""
    MATRIX MULTIPLY MEMORY ANALYSIS:
    
    C[M,N] = A[M,K] @ B[K,N]
    
    Without Tiling (element by element):
    - To compute C[i,j], need row i of A and column j of B
    - Each element of A read N times
    - Each element of B read M times
    - Total reads: M*K*N + K*N*M = 2*M*N*K
    
    With Tiling (BLOCK_M × BLOCK_N tiles):
    - Load tile of A (BLOCK_M × BLOCK_K)
    - Load tile of B (BLOCK_K × BLOCK_N)
    - Compute BLOCK_M × BLOCK_N outputs
    - Reuse factor: BLOCK_N for A, BLOCK_M for B
    - Total reads: M*K/BLOCK_M + K*N/BLOCK_N (much smaller!)
    
    Example: M=N=K=4096, BLOCK=128
    - Without tiling: ~137 billion reads
    - With tiling: ~2 billion reads (68x reduction!)
    """)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return


def experiment_matmul_correctness():
    """Verify matmul implementations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MATMUL CORRECTNESS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    shapes = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (1024, 2048, 512),  # Non-square
    ]
    
    print(f"\n{'Shape (M,K,N)':<25} {'Max Error':<20} {'Correct?'}")
    print("-" * 60)
    
    for M, K, N in shapes:
        a = torch.randn(M, K, device='cuda')
        b = torch.randn(K, N, device='cuda')
        
        ref = a @ b
        
        try:
            out = triton_matmul_optimized(a, b)
            max_error = torch.max(torch.abs(out - ref)).item()
            correct = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
            print(f"{str((M,K,N)):<25} {max_error:<20.2e} {correct}")
        except Exception as e:
            print(f"{str((M,K,N)):<25} Error: {str(e)[:30]}")


def experiment_matmul_performance():
    """Benchmark matmul implementations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: MATMUL PERFORMANCE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    sizes = [512, 1024, 2048, 4096]
    
    print(f"\n Square matrices (N×N @ N×N):")
    print(f"{'N':<10} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Triton TFLOPS':<15} {'% of PyTorch'}")
    print("-" * 70)
    
    for N in sizes:
        a = torch.randn(N, N, device='cuda')
        b = torch.randn(N, N, device='cuda')
        
        # FLOPs for matmul: 2 * M * N * K
        flops = 2 * N * N * N
        
        time_pytorch = profile_triton(lambda: a @ b, iterations=50)
        time_triton = profile_triton(lambda: triton_matmul_optimized(a, b), iterations=50)
        
        tflops_triton = flops / (time_triton / 1000) / 1e12
        pct_pytorch = time_pytorch / time_triton * 100
        
        print(f"{N:<10} {time_pytorch:<15.3f} {time_triton:<15.3f} {tflops_triton:<15.1f} {pct_pytorch:.0f}%")
    
    # Different shapes
    print(f"\n Non-square matrices:")
    print(f"{'Shape':<25} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Ratio'}")
    print("-" * 70)
    
    shapes = [
        (4096, 4096, 1024),
        (1024, 4096, 4096),
        (4096, 1024, 4096),
        (8192, 1024, 1024),
        (1024, 1024, 8192),
    ]
    
    for M, K, N in shapes:
        a = torch.randn(M, K, device='cuda')
        b = torch.randn(K, N, device='cuda')
        
        time_pytorch = profile_triton(lambda: a @ b, iterations=50)
        time_triton = profile_triton(lambda: triton_matmul_optimized(a, b), iterations=50)
        
        ratio = time_pytorch / time_triton
        print(f"{str((M,K,N)):<25} {time_pytorch:<15.3f} {time_triton:<15.3f} {ratio:.2f}x")


def experiment_autotuning_impact():
    """Show impact of auto-tuning on performance."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: AUTO-TUNING IMPACT")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    AUTO-TUNING PARAMETERS:
    
    1. BLOCK_M, BLOCK_N, BLOCK_K - Tile sizes
       - Larger tiles = more data reuse
       - But limited by shared memory and registers
    
    2. num_warps - Parallelism within a block
       - More warps = better latency hiding
       - But may reduce per-warp resources
    
    3. num_stages - Software pipelining depth
       - More stages = better overlap of compute/memory
       - But requires more shared memory
    
    4. GROUP_M - L2 cache optimization
       - Groups tiles to improve L2 hit rate
       - Particularly important for large matrices
    """)
    
    N = 2048
    a = torch.randn(N, N, device='cuda')
    b = torch.randn(N, N, device='cuda')
    
    print(f"\n Performance with different block sizes ({N}x{N}):")
    print(f"{'Block Size':<20} {'Time (ms)':<15} {'TFLOPS'}")
    print("-" * 50)
    
    block_sizes = [(32, 32), (64, 64), (128, 128), (64, 128), (128, 64)]
    flops = 2 * N * N * N
    
    for bm, bn in block_sizes:
        @triton.jit
        def test_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_K):
                k_offs = k + tl.arange(0, BLOCK_K)
                a = tl.load(a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
                           mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0.0)
                b = tl.load(b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn,
                           mask=(k_offs[:, None] < K) & (n_offs[None, :] < N), other=0.0)
                acc += tl.dot(a, b)
            c_offs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
            tl.store(c_offs, acc, mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))
        
        def run_kernel():
            c = torch.empty(N, N, device='cuda')
            grid = (triton.cdiv(N, bm), triton.cdiv(N, bn))
            test_kernel[grid](a, b, c, N, N, N,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            bm, bn, 32)
            return c
        
        try:
            time_ms = profile_triton(run_kernel, iterations=50)
            tflops = flops / (time_ms / 1000) / 1e12
            print(f"{str((bm, bn)):<20} {time_ms:<15.3f} {tflops:.1f}")
        except:
            print(f"{str((bm, bn)):<20} Failed")


def print_matmul_summary():
    """Print matmul implementation summary."""
    print("\n" + "="*70)
    print(" MATRIX MULTIPLICATION SUMMARY")
    print("="*70)
    
    print("""
    TRITON MATMUL TEMPLATE:
    
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        
        # 1. Determine which tile we're computing
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # 2. Calculate tile offsets
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 3. Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # 4. Loop over K tiles
        for k in range(0, K, BLOCK_K):
            k_offs = k + tl.arange(0, BLOCK_K)
            
            # Load A tile [BLOCK_M, BLOCK_K]
            a = tl.load(a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
                       mask=..., other=0.0)
            
            # Load B tile [BLOCK_K, BLOCK_N]
            b = tl.load(b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn,
                       mask=..., other=0.0)
            
            # Accumulate
            acc += tl.dot(a, b)  # Hardware accelerated!
        
        # 5. Store result
        tl.store(c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn,
                acc, mask=...)
    
    OPTIMIZATION TECHNIQUES:
    
    1. TILING: Reduce memory traffic through data reuse
    2. AUTO-TUNING: Find optimal block sizes
    3. SOFTWARE PIPELINING: Overlap memory and compute
    4. L2 OPTIMIZATION: Group tiles for cache locality
    5. TENSOR CORES: Use tl.dot for hardware acceleration
    
    MULTIMODAL IMPLICATIONS:
    - Every linear layer is a matmul
    - Attention QK^T and PV are batched matmuls
    - Efficient matmul = efficient training
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " TRITON MATRIX MULTIPLICATION ".center(68) + "║")
    print("║" + " The most important operation in deep learning ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_tiling_importance()
    experiment_matmul_correctness()
    experiment_matmul_performance()
    experiment_autotuning_impact()
    print_matmul_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Flash Attention implementation")
    print("="*70)
