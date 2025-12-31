"""
02_memory_hierarchy.py - GPU Memory Hierarchy Deep Dive

Understanding GPU memory hierarchy is CRITICAL for performance.
Memory access patterns often determine whether your code achieves
10% or 90% of theoretical GPU performance.

Memory Hierarchy (fastest to slowest):
1. Registers: ~0 cycles, ~256KB per SM
2. Shared Memory: ~20-30 cycles, 48-164KB per SM  
3. L1 Cache: ~30 cycles, combined with shared memory
4. L2 Cache: ~200 cycles, 40-50MB total
5. Global Memory (HBM): ~400-600 cycles, 40-80GB

For multimodal training:
- Attention KV cache lives in global memory (huge!)
- Tiled matmul uses shared memory
- Intermediate values in registers
- Flash Attention minimizes memory traffic

Run: python 02_memory_hierarchy.py
"""

import torch
import torch.nn.functional as F
import time
from typing import List, Tuple
import math

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_gpu_operation(func, warmup=5, iterations=100, name=""):
    """Profile a GPU operation with proper warmup and synchronization."""
    if not torch.cuda.is_available():
        print(f" {name}: CUDA not available")
        return 0.0
    
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    # Profile
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations

def calculate_memory_bandwidth(bytes_moved: int, time_ms: float) -> float:
    """Calculate achieved memory bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    return bytes_moved / (time_ms / 1000) / 1e9

def calculate_arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    """Calculate arithmetic intensity (FLOPs per byte)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved

# ============================================================================
# EXPERIMENT 1: MEMORY ACCESS PATTERNS
# ============================================================================

def experiment_access_patterns():
    """
    GPU memory access should be COALESCED for best performance.
    
    Coalesced access: Threads in a warp access consecutive memory addresses.
    This allows the GPU to combine requests into fewer memory transactions.
    
    32 threads × 4 bytes = 128-byte transaction (ideal)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: MEMORY ACCESS PATTERNS (COALESCING)")
    print(" Coalesced = threads access consecutive addresses")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Create test data - 2D tensor to demonstrate row vs column access
    rows, cols = 8192, 8192  # 256 MB for float32
    x = torch.randn(rows, cols, device='cuda')
    bytes_total = rows * cols * 4
    
    print(f"\n Tensor shape: {rows} x {cols} = {bytes_total/1e6:.0f} MB")
    print(f"\n{'Access Pattern':<30} {'Time (ms)':<15} {'BW (GB/s)':<15} {'Efficiency'}")
    print("-" * 75)
    
    # Row-wise sum (coalesced - threads access consecutive elements)
    def row_sum():
        return x.sum(dim=1)
    
    time_row = profile_gpu_operation(row_sum, name="Row sum")
    bw_row = calculate_memory_bandwidth(bytes_total, time_row)
    
    # Column-wise sum (strided - threads access elements cols apart)
    def col_sum():
        return x.sum(dim=0)
    
    time_col = profile_gpu_operation(col_sum, name="Column sum")
    bw_col = calculate_memory_bandwidth(bytes_total, time_col)
    
    # Transposed column sum (coalesced on transposed data)
    x_t = x.t().contiguous()
    def col_sum_contig():
        return x_t.sum(dim=1)
    
    time_col_contig = profile_gpu_operation(col_sum_contig, name="Column sum (contig)")
    bw_col_contig = calculate_memory_bandwidth(bytes_total, time_col_contig)
    
    print(f"{'Row-wise sum (coalesced)':<30} {time_row:<15.3f} {bw_row:<15.1f} {'OPTIMAL'}")
    print(f"{'Column-wise sum (strided)':<30} {time_col:<15.3f} {bw_col:<15.1f} {bw_col/bw_row*100:.0f}%")
    print(f"{'Column sum (contiguous)':<30} {time_col_contig:<15.3f} {bw_col_contig:<15.1f} {bw_col_contig/bw_row*100:.0f}%")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Strided access is {time_col/time_row:.1f}x slower!")
    print(f" - Memory layout matters: PyTorch tensors are row-major (last dim contiguous)")
    print(f" - Always reduce/iterate over the LAST dimension when possible")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Image tensors: (B, C, H, W) - W is contiguous, iterate over W")
    print(f" - Attention: (B, H, S, D) - D is contiguous")
    print(f" - Use .contiguous() before operations that need specific layout")

# ============================================================================
# EXPERIMENT 2: MEMORY BANDWIDTH VS COMPUTE
# ============================================================================

def experiment_bandwidth_vs_compute():
    """
    Operations are either MEMORY-BOUND or COMPUTE-BOUND.
    
    Arithmetic Intensity = FLOPs / Bytes moved
    
    Memory-bound (low AI): Element-wise ops, softmax, layer norm
    Compute-bound (high AI): Matrix multiplication, convolution
    
    The ROOFLINE MODEL helps understand this:
    - Low AI: Performance limited by memory bandwidth
    - High AI: Performance limited by compute (FLOPS)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY-BOUND VS COMPUTE-BOUND")
    print(" Arithmetic Intensity = FLOPs / Bytes")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    
    operations = []
    
    # 1. Vector addition - extremely memory bound
    x = torch.randn(N * N, device='cuda')
    y = torch.randn(N * N, device='cuda')
    
    def vector_add():
        return x + y
    
    time_add = profile_gpu_operation(vector_add)
    flops_add = N * N  # 1 add per element
    bytes_add = N * N * 4 * 3  # Read x, read y, write z
    ai_add = calculate_arithmetic_intensity(flops_add, bytes_add)
    bw_add = calculate_memory_bandwidth(bytes_add, time_add)
    operations.append(("Vector Add", time_add, ai_add, bw_add, flops_add/(time_add/1000)/1e9))
    
    # 2. Element-wise multiply-add (FMA)
    def vector_fma():
        return x * y + x
    
    time_fma = profile_gpu_operation(vector_fma)
    flops_fma = N * N * 2  # 1 mul + 1 add
    bytes_fma = N * N * 4 * 3
    ai_fma = calculate_arithmetic_intensity(flops_fma, bytes_fma)
    bw_fma = calculate_memory_bandwidth(bytes_fma, time_fma)
    operations.append(("Vector FMA", time_fma, ai_fma, bw_fma, flops_fma/(time_fma/1000)/1e9))
    
    # 3. Matrix multiplication - compute bound
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    def matmul():
        return A @ B
    
    time_mm = profile_gpu_operation(matmul)
    flops_mm = 2 * N * N * N  # 2*N^3 for matmul
    bytes_mm = N * N * 4 * 3  # Read A, read B, write C
    ai_mm = calculate_arithmetic_intensity(flops_mm, bytes_mm)
    tflops_mm = flops_mm / (time_mm / 1000) / 1e12
    operations.append(("Matrix Multiply", time_mm, ai_mm, 0, tflops_mm * 1000))
    
    # 4. Softmax - memory bound
    x_soft = torch.randn(N, N, device='cuda')
    
    def softmax():
        return F.softmax(x_soft, dim=-1)
    
    time_soft = profile_gpu_operation(softmax)
    # Softmax: exp, sum, div - roughly 5 ops per element
    flops_soft = N * N * 5
    bytes_soft = N * N * 4 * 2  # Read + write
    ai_soft = calculate_arithmetic_intensity(flops_soft, bytes_soft)
    bw_soft = calculate_memory_bandwidth(bytes_soft, time_soft)
    operations.append(("Softmax", time_soft, ai_soft, bw_soft, flops_soft/(time_soft/1000)/1e9))
    
    # 5. Layer Norm - memory bound
    x_ln = torch.randn(N, N, device='cuda')
    ln = torch.nn.LayerNorm(N).cuda()
    
    def layer_norm():
        return ln(x_ln)
    
    time_ln = profile_gpu_operation(layer_norm)
    flops_ln = N * N * 10  # mean, var, normalize, scale, shift
    bytes_ln = N * N * 4 * 2
    ai_ln = calculate_arithmetic_intensity(flops_ln, bytes_ln)
    bw_ln = calculate_memory_bandwidth(bytes_ln, time_ln)
    operations.append(("Layer Norm", time_ln, ai_ln, bw_ln, flops_ln/(time_ln/1000)/1e9))
    
    print(f"\n{'Operation':<20} {'Time(ms)':<12} {'AI (F/B)':<12} {'BW(GB/s)':<12} {'Bound'}")
    print("-" * 70)
    
    for name, t, ai, bw, gflops in operations:
        bound = "MEMORY" if ai < 50 else "COMPUTE"
        if "Matrix" in name:
            print(f"{name:<20} {t:<12.3f} {ai:<12.1f} {'-':<12} {bound} ({gflops:.1f} GFLOPS)")
        else:
            print(f"{name:<20} {t:<12.3f} {ai:<12.2f} {bw:<12.1f} {bound}")
    
    print(f"\n ROOFLINE MODEL:")
    print(f" ┌─────────────────────────────────────────────────────────────┐")
    print(f" │  Performance                                                │")
    print(f" │      ▲                    ___________Compute Ceiling        │")
    print(f" │      │                  /                                   │")
    print(f" │      │                /                                     │")
    print(f" │      │              /  ← Roofline                           │")
    print(f" │      │            /                                         │")
    print(f" │      │          /   Memory Bandwidth Ceiling                │")
    print(f" │      │        /                                             │")
    print(f" │      └────────────────────────────────────────────────▶     │")
    print(f" │                 Arithmetic Intensity (FLOPs/Byte)           │")
    print(f" └─────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Attention (QK^T, softmax) is memory-bound → Flash Attention helps")
    print(f" - Linear layers (matmul) are compute-bound → Tensor Cores help")
    print(f" - Fusing ops (e.g., bias+gelu) reduces memory traffic")

# ============================================================================
# EXPERIMENT 3: SHARED MEMORY DEMONSTRATION
# ============================================================================

def experiment_shared_memory_concept():
    """
    Shared Memory is programmer-managed cache per thread block.
    
    - 48-164 KB per SM (configurable)
    - ~20-30 cycle latency (vs 400+ for global memory)
    - Shared among all threads in a block
    - Used for: Tiling, reductions, thread communication
    
    PyTorch doesn't directly expose shared memory, but understanding
    it is crucial for writing/understanding CUDA kernels.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SHARED MEMORY CONCEPT")
    print(" Fast on-chip memory shared by threads in a block")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print(f"\n GPU Shared Memory Info:")
    props = torch.cuda.get_device_properties(0)
    print(f" - Shared memory per block: {props.max_shared_memory_per_block / 1024:.0f} KB")
    print(f" - Shared memory per SM: {props.max_shared_memory_per_multiprocessor / 1024:.0f} KB")
    
    # Demonstrate benefit of data reuse (what shared memory enables)
    N = 2048
    tile_sizes = [32, 64, 128, 256, 512]
    
    print(f"\n Simulating tiled vs naive matrix multiply:")
    print(f" Matrix size: {N} x {N}")
    print(f"\n{'Tile Size':<15} {'Naive (ms)':<15} {'Benefit Estimate'}")
    print("-" * 50)
    
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    # Baseline
    def naive_mm():
        return A @ B
    
    time_naive = profile_gpu_operation(naive_mm)
    
    for tile in tile_sizes:
        # Theoretical memory traffic reduction from tiling
        # Naive: 2*N^3 memory accesses
        # Tiled: 2*N^3 / tile + N^2 (tiles loaded once to shared memory)
        naive_traffic = 2 * N * N * N
        tiled_traffic = 2 * N * N * N / tile + N * N * tile
        reduction = naive_traffic / tiled_traffic
        
        print(f"{tile:<15} {time_naive:<15.2f} ~{reduction:.1f}x less memory traffic")
    
    print(f"\n HOW SHARED MEMORY TILING WORKS:")
    print(f" ┌────────────────────────────────────────────────────────────────┐")
    print(f" │ 1. Load tile of A and tile of B from global → shared memory   │")
    print(f" │ 2. Compute partial results using data in shared memory        │")
    print(f" │ 3. Repeat for all tiles, accumulating results                 │")
    print(f" │ 4. Write final result to global memory                        │")
    print(f" │                                                               │")
    print(f" │ BENEFIT: Each element of A and B loaded once per tile,        │")
    print(f" │          not once per output element!                         │")
    print(f" └────────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Flash Attention uses shared memory for Q, K tiles")
    print(f" - Fused kernels keep intermediate results in shared memory")
    print(f" - cuBLAS/Tensor Core matmul use sophisticated tiling")

# ============================================================================
# EXPERIMENT 4: MEMORY LATENCY HIDING
# ============================================================================

def experiment_latency_hiding():
    """
    GPUs hide memory latency by keeping MANY threads in flight.
    
    When one thread waits for memory, another thread runs.
    This is why OCCUPANCY matters - more threads = better latency hiding.
    
    Occupancy = Active warps / Max warps per SM
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: LATENCY HIDING THROUGH OCCUPANCY")
    print(" More threads in flight = better memory latency hiding")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    props = torch.cuda.get_device_properties(0)
    print(f"\n GPU Info:")
    print(f" - SMs: {props.multi_processor_count}")
    print(f" - Max threads per SM: {props.max_threads_per_multi_processor}")
    print(f" - Max warps per SM: {props.max_threads_per_multi_processor // 32}")
    
    # Demonstrate with varying tensor sizes (affects occupancy)
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
    
    print(f"\n Memory-bound operation (add) at different sizes:")
    print(f"{'Size':<15} {'Threads':<15} {'Time (μs)':<15} {'BW (GB/s)':<15}")
    print("-" * 60)
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        def add_op():
            return x + y
        
        time_ms = profile_gpu_operation(add_op, iterations=1000)
        bytes_moved = size * 4 * 3  # x, y read; z write
        bw = calculate_memory_bandwidth(bytes_moved, time_ms)
        
        # Each thread processes ~1-4 elements typically
        approx_threads = size  # Simplified
        
        print(f"{size:<15} {approx_threads:<15} {time_ms*1000:<15.2f} {bw:<15.1f}")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Small sizes: Not enough threads to hide memory latency")
    print(f" - Large sizes: GPU is 'saturated' - peak bandwidth achieved")
    print(f" - Memory latency ~400 cycles, need ~1000+ threads to hide it")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Batch size directly affects occupancy")
    print(f" - Small batch = wasted GPU potential")
    print(f" - Gradient accumulation: many small batches ≠ one large batch (for efficiency)")

# ============================================================================
# EXPERIMENT 5: HBM (HIGH BANDWIDTH MEMORY)
# ============================================================================

def experiment_hbm_bandwidth():
    """
    Modern AI GPUs use HBM (High Bandwidth Memory).
    
    HBM2e (A100): 2039 GB/s
    HBM3 (H100): 3350 GB/s
    HBM3e (B200): 8000 GB/s
    
    Compare to:
    - DDR5 (CPU): 50-100 GB/s
    - PCIe 4.0: 32 GB/s
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: HBM BANDWIDTH MEASUREMENT")
    print(" Testing actual achieved memory bandwidth")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Get theoretical bandwidth
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    
    # Try to estimate theoretical bandwidth (not always accurate from props)
    print(f"\n GPU: {gpu_name}")
    print(f" Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Measure actual bandwidth with various operations
    sizes_gb = [0.1, 0.5, 1.0, 2.0, 4.0]
    
    print(f"\n Bandwidth test (copy operation):")
    print(f"{'Size (GB)':<15} {'Time (ms)':<15} {'BW (GB/s)':<15} {'Operation'}")
    print("-" * 60)
    
    for size_gb in sizes_gb:
        num_elements = int(size_gb * 1e9 / 4)
        if num_elements * 4 * 2 > props.total_memory * 0.8:
            continue
            
        x = torch.randn(num_elements, device='cuda')
        bytes_moved = num_elements * 4 * 2  # Read + write
        
        def copy_op():
            return x.clone()
        
        time_ms = profile_gpu_operation(copy_op, iterations=50)
        bw = calculate_memory_bandwidth(bytes_moved, time_ms)
        
        print(f"{size_gb:<15.1f} {time_ms:<15.2f} {bw:<15.1f} clone()")
    
    print(f"\n TYPICAL PEAK BANDWIDTHS:")
    print(f" ┌──────────────────┬─────────────────┬──────────────────┐")
    print(f" │ GPU              │ Memory Type     │ Peak BW (GB/s)   │")
    print(f" ├──────────────────┼─────────────────┼──────────────────┤")
    print(f" │ RTX 3090         │ GDDR6X          │ 936              │")
    print(f" │ RTX 4090         │ GDDR6X          │ 1008             │")
    print(f" │ A100             │ HBM2e           │ 2039             │")
    print(f" │ H100             │ HBM3            │ 3350             │")
    print(f" │ B200             │ HBM3e           │ 8000             │")
    print(f" └──────────────────┴─────────────────┴──────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - KV cache for long context is memory-bound")
    print(f" - Higher bandwidth = longer context possible")
    print(f" - Memory bandwidth determines attention speed")

# ============================================================================
# EXPERIMENT 6: MEMORY ALLOCATION PATTERNS
# ============================================================================

def experiment_memory_allocation():
    """
    GPU memory allocation has overhead. Understanding allocation
    patterns helps avoid performance pitfalls.
    
    PyTorch uses a caching allocator to reduce allocation overhead.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: MEMORY ALLOCATION OVERHEAD")
    print(" Allocation is expensive - PyTorch caches to help")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    size = 1024 * 1024  # 4MB tensors
    iterations = 100
    
    # Test 1: Fresh allocations (cold cache)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        x = torch.randn(size, device='cuda')
        del x
        torch.cuda.empty_cache()  # Force deallocation
    torch.cuda.synchronize()
    cold_time = (time.perf_counter() - start) * 1000 / iterations
    
    # Test 2: Cached allocations (warm cache)
    torch.cuda.empty_cache()
    # Pre-warm the cache
    x = torch.randn(size, device='cuda')
    del x
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        x = torch.randn(size, device='cuda')
        del x
        # Don't empty cache - let PyTorch reuse memory
    torch.cuda.synchronize()
    warm_time = (time.perf_counter() - start) * 1000 / iterations
    
    # Test 3: Pre-allocated (no allocation in loop)
    x = torch.empty(size, device='cuda')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        x.normal_()  # In-place initialization
    torch.cuda.synchronize()
    preallocated_time = (time.perf_counter() - start) * 1000 / iterations
    
    print(f"\n Tensor size: {size * 4 / 1e6:.1f} MB")
    print(f"\n{'Allocation Pattern':<30} {'Time (ms)':<15} {'Relative'}")
    print("-" * 60)
    print(f"{'Cold (fresh allocation)':<30} {cold_time:<15.3f} {cold_time/preallocated_time:.1f}x")
    print(f"{'Warm (cached allocation)':<30} {warm_time:<15.3f} {warm_time/preallocated_time:.1f}x")
    print(f"{'Pre-allocated (in-place)':<30} {preallocated_time:<15.3f} 1.0x")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Fresh CUDA malloc is expensive (~{cold_time:.2f}ms)")
    print(f" - PyTorch's caching allocator helps (~{warm_time:.2f}ms)")
    print(f" - Pre-allocation is fastest (~{preallocated_time:.2f}ms)")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Pre-allocate KV cache for inference")
    print(f" - Use torch.empty() + in-place ops when possible")
    print(f" - Avoid creating tensors in tight loops")
    print(f" - Static shapes enable better memory planning")

# ============================================================================
# SUMMARY
# ============================================================================

def print_memory_hierarchy_summary():
    """Print comprehensive summary of GPU memory hierarchy."""
    print("\n" + "="*70)
    print(" GPU MEMORY HIERARCHY SUMMARY")
    print("="*70)
    
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                    GPU MEMORY HIERARCHY                            │
    ├────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  ┌─────────────┐ Registers                                        │
    │  │  ~256 KB    │ • Per-thread storage                             │
    │  │  ~0 cycles  │ • Compiler-managed                               │
    │  └──────┬──────┘ • Local variables, loop indices                  │
    │         │                                                          │
    │  ┌──────▼──────┐ Shared Memory / L1                               │
    │  │  48-164 KB  │ • Per-block, programmer-managed                  │
    │  │  ~20-30 cy  │ • Tiling, reductions, communication              │
    │  └──────┬──────┘ • CRITICAL for performance                       │
    │         │                                                          │
    │  ┌──────▼──────┐ L2 Cache                                         │
    │  │  40-50 MB   │ • Shared across all SMs                          │
    │  │  ~200 cy    │ • Automatic caching                              │
    │  └──────┬──────┘ • Read-mostly data benefits                      │
    │         │                                                          │
    │  ┌──────▼──────┐ Global Memory (HBM)                              │
    │  │  40-80 GB   │ • Main GPU memory                                │
    │  │  ~400-600cy │ • 2-8 TB/s bandwidth                             │
    │  └─────────────┘ • Weights, activations, KV cache                 │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
    
    OPTIMIZATION STRATEGIES:
    
    1. COALESCING: Access consecutive memory addresses
       Bad:  x[i * stride]  (strided)
       Good: x[i]           (consecutive)
    
    2. TILING: Use shared memory to reduce global memory traffic
       - Load tile to shared memory once
       - Reuse many times in computation
    
    3. OCCUPANCY: Keep many threads active to hide latency
       - More active warps = better latency hiding
       - Balance registers/shared memory usage
    
    4. ARITHMETIC INTENSITY: Maximize compute per byte moved
       - Fuse operations to avoid intermediate writes
       - Flash Attention: compute softmax without materializing full matrix
    
    MULTIMODAL TRAINING MEMORY PATTERNS:
    
    ┌─────────────────────────┬─────────────────────────────────────────┐
    │ Component               │ Memory Behavior                         │
    ├─────────────────────────┼─────────────────────────────────────────┤
    │ Embedding layers        │ Sparse access, low bandwidth util       │
    │ Linear layers (matmul)  │ High reuse, compute-bound               │
    │ Attention (QK^T)        │ Memory-bound, benefits from tiling      │
    │ Softmax                 │ Memory-bound, benefits from fusion      │
    │ Layer Norm              │ Memory-bound, benefits from fusion      │
    │ KV Cache                │ Huge memory footprint, sequential       │
    │ Activations             │ Recompute vs store tradeoff             │
    └─────────────────────────┴─────────────────────────────────────────┘
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " GPU MEMORY HIERARCHY: PROFILED EXPLORATION ".center(68) + "║")
    print("║" + " Understanding memory is KEY to GPU performance ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n GPU: {props.name}")
        print(f" Memory: {props.total_memory / 1e9:.1f} GB")
        print(f" SMs: {props.multi_processor_count}")
    else:
        print("\n WARNING: CUDA not available. GPU examples will be skipped.")
    
    experiment_access_patterns()
    experiment_bandwidth_vs_compute()
    experiment_shared_memory_concept()
    experiment_latency_hiding()
    experiment_hbm_bandwidth()
    experiment_memory_allocation()
    print_memory_hierarchy_summary()
    
    print("\n" + "="*70)
    print(" KEY TAKEAWAYS:")
    print(" 1. Memory bandwidth often limits performance (not compute)")
    print(" 2. Coalesced access patterns are essential")
    print(" 3. Shared memory enables data reuse (tiling)")
    print(" 4. High occupancy hides memory latency")
    print(" 5. Minimize memory traffic through fusion")
    print("="*70)
