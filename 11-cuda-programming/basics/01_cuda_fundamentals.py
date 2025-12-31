"""
01_cuda_fundamentals.py - CUDA Programming from First Principles

This module teaches CUDA programming using PyTorch's torch.cuda and
custom CUDA kernels via torch.utils.cpp_extension.load_inline.

CUDA = Compute Unified Device Architecture
It's the programming model for NVIDIA GPUs.

Key Concepts:
- Kernels: Functions that run on GPU
- Threads: Individual execution units
- Blocks: Groups of threads that share memory
- Grids: Groups of blocks
- Memory: Global, shared, registers

For multimodal training, CUDA knowledge helps:
- Understand why operations are fast/slow
- Write custom fused kernels
- Debug performance issues
- Optimize data movement

Run: python 01_cuda_fundamentals.py
Requirements: torch with CUDA, ninja (for compilation)
"""

import torch
import torch.nn.functional as F
import time
import math
from typing import Tuple, Optional

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

def profile_cuda(func, warmup=10, iterations=100, name=""):
    """Profile a CUDA function with proper warmup and synchronization."""
    if not torch.cuda.is_available():
        return 0.0
    
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations

# ============================================================================
# EXPERIMENT 1: THREAD HIERARCHY
# ============================================================================

def experiment_thread_hierarchy():
    """
    CUDA Thread Hierarchy:
    
    Grid (all blocks)
    └── Block (group of threads, share memory)
        └── Warp (32 threads, execute together)
            └── Thread (individual execution unit)
    
    Each thread has:
    - threadIdx: Position within block (x, y, z)
    - blockIdx: Which block this thread is in
    - blockDim: Size of the block
    - gridDim: Size of the grid
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: CUDA THREAD HIERARCHY")
    print(" Understanding threadIdx, blockIdx, blockDim, gridDim")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    CUDA EXECUTION MODEL:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                         GRID                                     │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
    │  │ Block   │ │ Block   │ │ Block   │ │ Block   │  ...          │
    │  │ (0,0)   │ │ (1,0)   │ │ (2,0)   │ │ (3,0)   │               │
    │  │┌──┬──┬─┐│ │         │ │         │ │         │               │
    │  ││T0│T1│..││ │         │ │         │ │         │               │
    │  │└──┴──┴─┘│ │         │ │         │ │         │               │
    │  │ Shared  │ │ Shared  │ │ Shared  │ │ Shared  │               │
    │  │ Memory  │ │ Memory  │ │ Memory  │ │ Memory  │               │
    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
    └─────────────────────────────────────────────────────────────────┘
    
    INDEXING:
    - threadIdx.x: Thread's position in block (0 to blockDim.x-1)
    - blockIdx.x: Block's position in grid (0 to gridDim.x-1)
    - Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
    """)
    
    # Demonstrate with different configurations
    print(f"\n Common configurations:")
    print(f"{'Total Threads':<18} {'Block Size':<15} {'Num Blocks':<15} {'Use Case'}")
    print("-" * 70)
    
    configs = [
        (1024, 256, "Small vector"),
        (65536, 256, "Medium tensor"),
        (1048576, 256, "Large tensor (1M)"),
        (16777216, 256, "Very large (16M)"),
        (1024, 1024, "Max threads/block"),
        (1024, 32, "One warp per block"),
    ]
    
    for total, block_size, use_case in configs:
        num_blocks = (total + block_size - 1) // block_size
        print(f"{total:<18} {block_size:<15} {num_blocks:<15} {use_case}")
    
    # Demonstrate how PyTorch launches kernels
    print(f"\n PyTorch kernel launch (invisible to user):")
    
    sizes = [1024, 65536, 1048576]
    for size in sizes:
        x = torch.randn(size, device='cuda')
        
        time_ms = profile_cuda(lambda: x + 1.0, iterations=1000)
        
        # Estimate launch configuration
        block_size = 256  # Common default
        num_blocks = (size + block_size - 1) // block_size
        
        print(f" Size {size:>10}: ~{num_blocks:>6} blocks × {block_size} threads = {time_ms:.4f} ms")

# ============================================================================
# EXPERIMENT 2: VECTOR ADDITION (Hello World of CUDA)
# ============================================================================

def experiment_vector_addition():
    """
    Vector addition is the "Hello World" of CUDA.
    
    Each thread computes: C[i] = A[i] + B[i]
    Perfectly parallel - no dependencies between elements.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: VECTOR ADDITION")
    print(" The 'Hello World' of CUDA programming")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    VECTOR ADDITION KERNEL LOGIC:
    
    __global__ void vector_add(float* A, float* B, float* C, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
        if (i < N) {                                     // Bounds check
            C[i] = A[i] + B[i];                         // Actual work
        }
    }
    
    Key Points:
    1. Each thread computes ONE element
    2. Global index = blockIdx * blockDim + threadIdx
    3. Always check bounds (blocks may have extra threads)
    """)
    
    # Compare PyTorch implementation with manual
    sizes = [1024, 65536, 1048576, 16777216]
    
    print(f"\n Performance comparison:")
    print(f"{'Size':<15} {'PyTorch (ms)':<18} {'Bandwidth (GB/s)':<18} {'Efficiency'}")
    print("-" * 70)
    
    for size in sizes:
        A = torch.randn(size, device='cuda')
        B = torch.randn(size, device='cuda')
        
        # Profile
        def vec_add():
            return A + B
        
        time_ms = profile_cuda(vec_add)
        
        # Calculate bandwidth
        # Read A, Read B, Write C = 3 * size * 4 bytes
        bytes_moved = 3 * size * 4
        bandwidth = bytes_moved / (time_ms / 1000) / 1e9
        
        # Theoretical max (varies by GPU, assume ~900 GB/s)
        theoretical_max = 900  # GB/s for RTX 3090
        efficiency = bandwidth / theoretical_max * 100
        
        print(f"{size:<15} {time_ms:<18.4f} {bandwidth:<18.1f} ~{efficiency:.0f}%")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Vector addition is MEMORY-BOUND")
    print(f" - Limited by how fast we can read/write data")
    print(f" - Arithmetic intensity = 1 FLOP / 12 bytes = 0.08")
    print(f" - Even simple ops can achieve high bandwidth efficiency")

# ============================================================================
# EXPERIMENT 3: MEMORY COALESCING
# ============================================================================

def experiment_memory_coalescing():
    """
    Memory coalescing is CRITICAL for GPU performance.
    
    When threads in a warp access consecutive memory addresses,
    the GPU can combine these into fewer memory transactions.
    
    32 threads × 4 bytes = 128 bytes = 1 memory transaction (ideal)
    32 threads × 4 bytes scattered = up to 32 transactions (worst)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: MEMORY COALESCING")
    print(" Aligned access patterns for maximum bandwidth")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    COALESCED ACCESS (GOOD):
    Threads: T0  T1  T2  T3  ...  T31
    Memory:  [0] [1] [2] [3] ...  [31]  → 1 transaction
    
    STRIDED ACCESS (BAD):
    Threads: T0    T1    T2    T3    ...
    Memory:  [0]   [32]  [64]  [96]  ...  → Many transactions!
    
    RANDOM ACCESS (WORST):
    Threads: T0      T1       T2      T3     ...
    Memory:  [1024]  [7]      [999]   [42]   ...  → 32 transactions!
    """)
    
    # Create a 2D tensor to demonstrate row vs column access
    rows, cols = 4096, 4096  # 64 MB
    x = torch.randn(rows, cols, device='cuda')
    
    print(f"\n Tensor shape: {rows} × {cols} ({rows * cols * 4 / 1e6:.0f} MB)")
    print(f"{'Access Pattern':<30} {'Time (ms)':<15} {'Bandwidth (GB/s)':<15}")
    print("-" * 60)
    
    # Row-major access (coalesced for row reduction)
    def row_access():
        return x.sum(dim=1)  # Sum along rows
    
    time_row = profile_cuda(row_access)
    bw_row = rows * cols * 4 / (time_row / 1000) / 1e9
    print(f"{'Row-major (coalesced)':<30} {time_row:<15.3f} {bw_row:<15.1f}")
    
    # Column-major access (strided)
    def col_access():
        return x.sum(dim=0)  # Sum along columns
    
    time_col = profile_cuda(col_access)
    bw_col = rows * cols * 4 / (time_col / 1000) / 1e9
    print(f"{'Column-major (strided)':<30} {time_col:<15.3f} {bw_col:<15.1f}")
    
    # Transposed then row access
    x_t = x.t().contiguous()
    def transposed_access():
        return x_t.sum(dim=1)
    
    time_trans = profile_cuda(transposed_access)
    bw_trans = rows * cols * 4 / (time_trans / 1000) / 1e9
    print(f"{'Transposed + row access':<30} {time_trans:<15.3f} {bw_trans:<15.1f}")
    
    print(f"\n Strided access is {time_col/time_row:.1f}x slower!")
    
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Image tensors (B,C,H,W): W is contiguous, iterate over W")
    print(f" - Attention (B,H,S,D): D is contiguous")
    print(f" - Always check tensor.is_contiguous() before operations")
    print(f" - Use .contiguous() to fix layout (costs memory copy)")

# ============================================================================
# EXPERIMENT 4: PARALLEL REDUCTION
# ============================================================================

def experiment_parallel_reduction():
    """
    Reduction operations (sum, max, etc.) are fundamental building blocks.
    
    Sequential reduction: O(N) - one element at a time
    Parallel reduction: O(log N) - halve the problem each step
    
    But parallel reduction needs synchronization!
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: PARALLEL REDUCTION")
    print(" How to efficiently compute sum/max/min on GPU")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    PARALLEL REDUCTION ALGORITHM:
    
    Step 0: [1, 2, 3, 4, 5, 6, 7, 8]
    Step 1: [3,    7,    11,   15  ]  (pairs add)
    Step 2: [10,         26        ]  (pairs add)
    Step 3: [36                    ]  (final result)
    
    Each step: N/2 threads do work, then synchronize
    Total steps: log2(N)
    
    In CUDA:
    - Use shared memory for fast access
    - __syncthreads() between steps
    - Handle non-power-of-2 sizes
    """)
    
    sizes = [1024, 65536, 1048576, 16777216, 67108864]
    
    print(f"\n Reduction performance:")
    print(f"{'Size':<15} {'Time (ms)':<15} {'Throughput (Gelem/s)':<20} {'Steps (log2)'}")
    print("-" * 70)
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        
        def reduce_sum():
            return x.sum()
        
        time_ms = profile_cuda(reduce_sum)
        throughput = size / (time_ms / 1000) / 1e9
        steps = int(math.log2(size))
        
        print(f"{size:<15} {time_ms:<15.4f} {throughput:<20.2f} {steps}")
    
    # Compare different reduction methods
    print(f"\n Reduction method comparison (16M elements):")
    print(f"{'Method':<30} {'Time (ms)':<15}")
    print("-" * 45)
    
    x = torch.randn(16777216, device='cuda')
    
    methods = [
        ("x.sum()", lambda: x.sum()),
        ("torch.sum(x)", lambda: torch.sum(x)),
        ("x.mean()", lambda: x.mean()),
        ("x.max()", lambda: x.max()),
        ("x.min()", lambda: x.min()),
        ("x.std()", lambda: x.std()),
    ]
    
    for name, func in methods:
        time_ms = profile_cuda(func)
        print(f"{name:<30} {time_ms:<15.4f}")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Reductions are memory-bound (read all data)")
    print(f" - PyTorch uses optimized multi-level reduction")
    print(f" - Block-level reduce in shared memory")
    print(f" - Then reduce across blocks")

# ============================================================================
# EXPERIMENT 5: ELEMENT-WISE OPERATIONS
# ============================================================================

def experiment_elementwise_ops():
    """
    Element-wise operations are the simplest CUDA kernels.
    Each thread processes one or more elements independently.
    
    They are memory-bound - limited by bandwidth, not compute.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: ELEMENT-WISE OPERATIONS")
    print(" Memory-bound operations - bandwidth is the limit")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    size = 16777216  # 16M elements = 64 MB
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    print(f"\n Tensor size: {size} elements ({size * 4 / 1e6:.0f} MB)")
    print(f"\n{'Operation':<30} {'Time (ms)':<15} {'BW (GB/s)':<15} {'FLOPs'}")
    print("-" * 70)
    
    operations = [
        # (name, function, bytes_per_element, flops_per_element)
        ("x + 1.0 (add scalar)", lambda: x + 1.0, 8, 1),  # read + write
        ("x + y (add tensor)", lambda: x + y, 12, 1),  # read x, y, write
        ("x * y (multiply)", lambda: x * y, 12, 1),
        ("x * y + x (FMA)", lambda: x * y + x, 12, 2),
        ("torch.relu(x)", lambda: torch.relu(x), 8, 1),
        ("torch.sigmoid(x)", lambda: torch.sigmoid(x), 8, 4),
        ("torch.exp(x)", lambda: torch.exp(x), 8, 8),
        ("torch.sin(x)", lambda: torch.sin(x), 8, 8),
        ("x.pow(2)", lambda: x.pow(2), 8, 1),
        ("torch.sqrt(x.abs())", lambda: torch.sqrt(x.abs()), 8, 2),
    ]
    
    for name, func, bytes_per_elem, flops_per_elem in operations:
        time_ms = profile_cuda(func)
        total_bytes = size * bytes_per_elem
        bandwidth = total_bytes / (time_ms / 1000) / 1e9
        total_flops = size * flops_per_elem
        gflops = total_flops / (time_ms / 1000) / 1e9
        
        print(f"{name:<30} {time_ms:<15.4f} {bandwidth:<15.1f} {gflops:.1f} GFLOPS")
    
    print(f"\n KEY INSIGHT:")
    print(f" - All these operations achieve similar bandwidth")
    print(f" - More complex ops (exp, sin) still memory-bound!")
    print(f" - Arithmetic intensity too low to be compute-bound")
    print(f"\n OPTIMIZATION OPPORTUNITY:")
    print(f" - Fuse multiple operations to reduce memory traffic")
    print(f" - x * y + x done fused uses same bandwidth as x + 1")

# ============================================================================
# EXPERIMENT 6: KERNEL FUSION BENEFIT
# ============================================================================

def experiment_kernel_fusion():
    """
    Kernel fusion combines multiple operations into one kernel.
    This reduces memory traffic - intermediate results stay in registers.
    
    Without fusion: Write intermediate to memory, read it back
    With fusion: Keep intermediate in registers
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: KERNEL FUSION")
    print(" Fusing operations reduces memory traffic")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    size = 16777216  # 16M elements
    x = torch.randn(size, device='cuda')
    
    print(f"\n Comparing fused vs unfused operations:")
    print(f"{'Operation':<40} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 70)
    
    # Unfused: GELU computed in parts
    def unfused_gelu():
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x3)
        tanh_inner = torch.tanh(inner)
        return 0.5 * x * (1 + tanh_inner)
    
    time_unfused = profile_cuda(unfused_gelu)
    
    # Fused: PyTorch's optimized GELU
    def fused_gelu():
        return F.gelu(x)
    
    time_fused = profile_cuda(fused_gelu)
    
    print(f"{'Unfused GELU (manual)':<40} {time_unfused:<15.4f} 1.0x")
    print(f"{'Fused GELU (F.gelu)':<40} {time_fused:<15.4f} {time_unfused/time_fused:.2f}x")
    
    # Another example: LayerNorm
    x_ln = torch.randn(32, 1024, 1024, device='cuda')
    
    def unfused_layernorm():
        mean = x_ln.mean(dim=-1, keepdim=True)
        var = x_ln.var(dim=-1, keepdim=True, unbiased=False)
        return (x_ln - mean) / torch.sqrt(var + 1e-5)
    
    time_unfused_ln = profile_cuda(unfused_layernorm)
    
    ln = torch.nn.LayerNorm(1024).cuda()
    def fused_layernorm():
        return ln(x_ln)
    
    time_fused_ln = profile_cuda(fused_layernorm)
    
    print(f"{'Unfused LayerNorm (manual)':<40} {time_unfused_ln:<15.4f} 1.0x")
    print(f"{'Fused LayerNorm (nn.LayerNorm)':<40} {time_fused_ln:<15.4f} {time_unfused_ln/time_fused_ln:.2f}x")
    
    # torch.compile fusion
    try:
        compiled_gelu = torch.compile(unfused_gelu)
        # Warmup compile
        _ = compiled_gelu()
        torch.cuda.synchronize()
        
        time_compiled = profile_cuda(compiled_gelu)
        print(f"{'torch.compile GELU':<40} {time_compiled:<15.4f} {time_unfused/time_compiled:.2f}x")
    except Exception as e:
        print(f" torch.compile not available: {e}")
    
    print(f"\n WHY FUSION HELPS:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │ Unfused: x → [DRAM] → op1 → [DRAM] → op2 → [DRAM] → y    │")
    print(f" │          3 memory round trips                             │")
    print(f" │                                                           │")
    print(f" │ Fused:   x → [DRAM] → op1 → [REG] → op2 → [DRAM] → y     │")
    print(f" │          1 memory round trip (registers for intermediate) │")
    print(f" └────────────────────────────────────────────────────────────┘")

# ============================================================================
# EXPERIMENT 7: SYNCHRONIZATION CONCEPTS
# ============================================================================

def experiment_synchronization():
    """
    CUDA operations are asynchronous by default.
    Understanding synchronization is crucial for:
    - Correct timing
    - Avoiding race conditions
    - Overlapping compute and transfer
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 7: SYNCHRONIZATION")
    print(" CUDA operations are asynchronous!")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    CUDA SYNCHRONIZATION POINTS:
    
    1. torch.cuda.synchronize()
       - Waits for ALL CUDA operations to complete
       - Most common, safest, but can be slow
    
    2. CUDA Events
       - Fine-grained synchronization
       - Can wait for specific operations
       - Best for accurate timing
    
    3. cudaStreamSynchronize (via PyTorch streams)
       - Wait for operations in specific stream
       - Enables overlapping operations
    
    4. Implicit synchronization
       - .cpu() or .item() syncs automatically
       - print() on CUDA tensor syncs
    """)
    
    x = torch.randn(10000, 10000, device='cuda')
    
    # Demonstrate async nature
    print(f"\n Demonstrating asynchronous execution:")
    
    # Without sync - measures only launch time
    start = time.perf_counter()
    for _ in range(10):
        y = x @ x
    no_sync_time = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()  # Actually complete the work
    
    # With sync - measures actual execution
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    sync_time = (time.perf_counter() - start) * 1000
    
    print(f" Without sync (just launch): {no_sync_time:.2f} ms")
    print(f" With sync (actual work):    {sync_time:.2f} ms")
    print(f" Difference: {sync_time/no_sync_time:.1f}x ← This is why sync matters!")
    
    # Implicit sync examples
    print(f"\n Operations that cause implicit sync:")
    print(f" - tensor.item()    : Copies scalar to CPU")
    print(f" - tensor.cpu()     : Copies tensor to CPU")
    print(f" - print(tensor)    : Needs data on CPU to print")
    print(f" - tensor.numpy()   : Requires CPU tensor")

# ============================================================================
# SUMMARY
# ============================================================================

def print_cuda_fundamentals_summary():
    """Print summary of CUDA fundamentals."""
    print("\n" + "="*70)
    print(" CUDA FUNDAMENTALS SUMMARY")
    print("="*70)
    
    print("""
    KEY CONCEPTS LEARNED:
    
    1. THREAD HIERARCHY
       Grid → Blocks → Warps → Threads
       Global ID = blockIdx * blockDim + threadIdx
    
    2. MEMORY ACCESS
       Coalesced access: 10-100x faster than random
       Always access consecutive addresses within a warp
    
    3. MEMORY BOUNDEDNESS
       Most operations limited by memory bandwidth
       Arithmetic intensity = FLOPs / Bytes
       Low AI → Memory-bound, High AI → Compute-bound
    
    4. SYNCHRONIZATION
       GPU operations are ASYNC
       Always sync for accurate timing
       Use CUDA events for best precision
    
    5. KERNEL FUSION
       Reduces memory traffic
       Keeps intermediates in registers
       torch.compile does this automatically
    
    PERFORMANCE CHECKLIST:
    ✓ Access memory in coalesced patterns
    ✓ Minimize memory traffic (fuse operations)
    ✓ Use enough threads to hide latency
    ✓ Profile before optimizing
    ✓ Sync properly for timing
    
    MULTIMODAL TRAINING IMPLICATIONS:
    - Batch dimension enables parallelism
    - Attention is memory-bound → Flash Attention helps
    - Linear layers are compute-bound → Tensor Cores help
    - Data loading should overlap with compute
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " CUDA FUNDAMENTALS: First Principles ".center(68) + "║")
    print("║" + " Every experiment is profiled for intuition ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" CUDA: {torch.version.cuda}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_thread_hierarchy()
    experiment_vector_addition()
    experiment_memory_coalescing()
    experiment_parallel_reduction()
    experiment_elementwise_ops()
    experiment_kernel_fusion()
    experiment_synchronization()
    print_cuda_fundamentals_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Memory management and optimization techniques")
    print("="*70)
