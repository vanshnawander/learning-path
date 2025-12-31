"""
01_memory_deep_dive.py - GPU Memory Hierarchy Complete Guide

This module provides a comprehensive exploration of GPU memory hierarchy
with profiled experiments demonstrating each memory level.

Memory Levels (fastest to slowest):
1. Registers - Per-thread, ~0 cycles
2. Shared Memory - Per-block, ~20-30 cycles  
3. L1 Cache - Per-SM, ~30 cycles
4. L2 Cache - Device-wide, ~200 cycles
5. Global Memory (HBM) - ~400-600 cycles
6. Host Memory (CPU) - ~10000+ cycles

Understanding this hierarchy is essential for:
- Writing efficient CUDA/Triton kernels
- Understanding why Flash Attention works
- Optimizing data movement in training

Run: python 01_memory_deep_dive.py
"""

import torch
import time
import math

# ============================================================================
# PROFILING
# ============================================================================

def profile_op(func, warmup=10, iterations=100):
    """Profile GPU operation."""
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
# MEMORY HIERARCHY OVERVIEW
# ============================================================================

def print_memory_hierarchy():
    """Print comprehensive memory hierarchy overview."""
    print("\n" + "="*70)
    print(" GPU MEMORY HIERARCHY")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     GPU MEMORY HIERARCHY                                 │
    │                                                                         │
    │   FASTEST                                                               │
    │      │                                                                  │
    │      ▼                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐    │
    │   │  REGISTERS                                                     │    │
    │   │  • Per-thread storage                                         │    │
    │   │  • ~256 KB per SM total                                       │    │
    │   │  • Latency: ~0 cycles                                         │    │
    │   │  • Bandwidth: ~TB/s (internal)                               │    │
    │   │  • Used for: Local variables, loop counters                   │    │
    │   └───────────────────────────────────────────────────────────────┘    │
    │      │                                                                  │
    │      ▼                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐    │
    │   │  SHARED MEMORY / L1 CACHE                                      │    │
    │   │  • Per-SM, shared within block                                │    │
    │   │  • 48-228 KB per SM (configurable)                            │    │
    │   │  • Latency: ~20-30 cycles                                     │    │
    │   │  • Bandwidth: ~TB/s (per SM)                                  │    │
    │   │  • Used for: Tiling, reductions, inter-thread communication   │    │
    │   └───────────────────────────────────────────────────────────────┘    │
    │      │                                                                  │
    │      ▼                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐    │
    │   │  L2 CACHE                                                      │    │
    │   │  • Device-wide, automatic                                     │    │
    │   │  • 40-50 MB total                                             │    │
    │   │  • Latency: ~200 cycles                                       │    │
    │   │  • Bandwidth: ~TB/s                                           │    │
    │   │  • Used for: Read-mostly data, automatic caching              │    │
    │   └───────────────────────────────────────────────────────────────┘    │
    │      │                                                                  │
    │      ▼                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐    │
    │   │  GLOBAL MEMORY (HBM)                                           │    │
    │   │  • Main GPU memory                                            │    │
    │   │  • 40-80 GB capacity                                          │    │
    │   │  • Latency: ~400-600 cycles                                   │    │
    │   │  • Bandwidth: 2-8 TB/s                                        │    │
    │   │  • Used for: Weights, activations, inputs/outputs             │    │
    │   └───────────────────────────────────────────────────────────────┘    │
    │      │                                                                  │
    │      ▼                                                                  │
    │   ┌───────────────────────────────────────────────────────────────┐    │
    │   │  HOST MEMORY (CPU RAM)                                         │    │
    │   │  • System memory via PCIe                                     │    │
    │   │  • 64-256+ GB capacity                                        │    │
    │   │  • Latency: ~10000+ cycles                                    │    │
    │   │  • Bandwidth: 32-64 GB/s (PCIe)                              │    │
    │   │  • Used for: Data loading, CPU preprocessing                  │    │
    │   └───────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    │   SLOWEST                                                               │
    └─────────────────────────────────────────────────────────────────────────┘
    """)

# ============================================================================
# EXPERIMENT: CACHE EFFECTS
# ============================================================================

def experiment_cache_effects():
    """Demonstrate cache effects on GPU performance."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: CACHE EFFECTS")
    print(" Working set size affects performance")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Test different working set sizes
    sizes_kb = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    print(f"\n Working set size vs bandwidth:")
    print(f"{'Size':<12} {'Time (μs)':<15} {'Bandwidth (GB/s)':<18} {'Likely Level'}")
    print("-" * 65)
    
    for size_kb in sizes_kb:
        num_elements = (size_kb * 1024) // 4  # float32
        x = torch.randn(num_elements, device='cuda')
        
        # Multiple passes to stress cache
        def cache_test():
            total = x.sum()
            return total
        
        time_ms = profile_op(cache_test, iterations=500)
        time_us = time_ms * 1000
        bytes_read = num_elements * 4
        bandwidth = bytes_read / (time_ms / 1000) / 1e9
        
        # Estimate cache level based on size
        if size_kb <= 128:
            level = "L1/Shared"
        elif size_kb <= 4096:
            level = "L2 Cache"
        else:
            level = "HBM"
        
        print(f"{size_kb:>6} KB   {time_us:<15.2f} {bandwidth:<18.1f} {level}")
        
        del x
    
    torch.cuda.empty_cache()
    
    print(f"\n KEY INSIGHT:")
    print(f" - Small working sets fit in L1/L2 → higher effective bandwidth")
    print(f" - Large working sets spill to HBM → bandwidth limited")
    print(f" - This is why tiling (Flash Attention) is so effective!")

# ============================================================================
# EXPERIMENT: MEMORY ACCESS PATTERNS
# ============================================================================

def experiment_access_patterns():
    """Demonstrate impact of memory access patterns."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY ACCESS PATTERNS")
    print(" Coalesced vs strided vs random access")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    M = 4096
    x = torch.randn(N, M, device='cuda')
    
    print(f"\n Matrix access patterns ({N}×{M}):")
    print(f"{'Pattern':<30} {'Time (ms)':<15} {'BW (GB/s)':<15}")
    print("-" * 60)
    
    bytes_total = N * M * 4
    
    # Coalesced - access along last dimension
    def coalesced():
        return x.sum(dim=1)  # Sum rows
    
    time_coal = profile_op(coalesced)
    bw_coal = bytes_total / (time_coal / 1000) / 1e9
    print(f"{'Coalesced (row sum)':<30} {time_coal:<15.3f} {bw_coal:<15.1f}")
    
    # Strided - access along first dimension
    def strided():
        return x.sum(dim=0)  # Sum columns
    
    time_stride = profile_op(strided)
    bw_stride = bytes_total / (time_stride / 1000) / 1e9
    print(f"{'Strided (column sum)':<30} {time_stride:<15.3f} {bw_stride:<15.1f}")
    
    # Transpose then coalesced
    x_t = x.t().contiguous()
    def transpose_coal():
        return x_t.sum(dim=1)
    
    time_trans = profile_op(transpose_coal)
    bw_trans = bytes_total / (time_trans / 1000) / 1e9
    print(f"{'Transposed + coalesced':<30} {time_trans:<15.3f} {bw_trans:<15.1f}")
    
    print(f"\n Strided access is {time_stride/time_coal:.1f}x slower than coalesced!")
    
    print(f"\n WHY COALESCING MATTERS:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │ Coalesced: Threads 0-31 access addresses 0-31             │")
    print(f" │   → GPU combines into ONE memory transaction              │")
    print(f" │                                                           │")
    print(f" │ Strided: Thread 0 → addr 0, Thread 1 → addr 4096, ...    │")
    print(f" │   → GPU must issue MANY memory transactions              │")
    print(f" └────────────────────────────────────────────────────────────┘")

# ============================================================================
# EXPERIMENT: SHARED MEMORY BENEFIT
# ============================================================================

def experiment_shared_memory_benefit():
    """Demonstrate the benefit of shared memory (conceptually)."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SHARED MEMORY BENEFIT (CONCEPTUAL)")
    print(" Data reuse through on-chip memory")
    print("="*70)
    
    print("""
    SHARED MEMORY USE CASE: TILED MATRIX MULTIPLY
    
    Without tiling (naive):
    ┌─────────────────────────────────────────────────────────────────┐
    │ For each output C[i,j]:                                         │
    │   Load row i of A from HBM                                      │
    │   Load column j of B from HBM                                   │
    │   Compute dot product                                           │
    │                                                                 │
    │ Each element of A loaded N times (once per column of B)         │
    │ Each element of B loaded M times (once per row of A)            │
    │ Total HBM reads: 2 × M × N × K                                  │
    └─────────────────────────────────────────────────────────────────┘
    
    With tiling (using shared memory):
    ┌─────────────────────────────────────────────────────────────────┐
    │ For each tile of C:                                             │
    │   For each k-tile:                                              │
    │     Load A tile (BLOCK_M × BLOCK_K) to shared memory           │
    │     Load B tile (BLOCK_K × BLOCK_N) to shared memory           │
    │     __syncthreads()                                             │
    │     Compute using shared memory (fast!)                         │
    │     __syncthreads()                                             │
    │                                                                 │
    │ Each element loaded once per tile, reused BLOCK times           │
    │ Total HBM reads: M×K/BLOCK_N + K×N/BLOCK_M                     │
    │ Reduction: BLOCK_M × BLOCK_N (typically 64-256x less!)          │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    if not torch.cuda.is_available():
        print(" CUDA not available for benchmark")
        return
    
    # Show that PyTorch matmul uses tiling (achieves high efficiency)
    N = 4096
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    time_mm = profile_op(lambda: A @ B)
    
    flops = 2 * N * N * N
    tflops = flops / (time_mm / 1000) / 1e12
    
    # Theoretical reads without tiling
    naive_reads = 2 * N * N * N * 4  # bytes
    
    # Actual reads with tiling (approximately)
    block_size = 64
    tiled_reads = (N * N * 4 / block_size + N * N * 4 / block_size) * (N / block_size)
    
    print(f"\n Matrix multiply {N}×{N}:")
    print(f" Time: {time_mm:.2f} ms")
    print(f" Performance: {tflops:.1f} TFLOPS")
    print(f"\n Theoretical memory traffic:")
    print(f" Without tiling: ~{naive_reads/1e12:.1f} TB")
    print(f" With tiling (BLOCK=64): ~{tiled_reads/1e9:.1f} GB")
    print(f" Reduction: ~{naive_reads/tiled_reads:.0f}x")

# ============================================================================
# EXPERIMENT: BANDWIDTH VS COMPUTE
# ============================================================================

def experiment_bandwidth_vs_compute():
    """Show how to identify bandwidth vs compute bound operations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: IDENTIFYING THE BOTTLENECK")
    print(" Is it memory or compute?")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    
    print(f"\n Arithmetic Intensity = FLOPs / Bytes")
    print(f" Low AI → Memory bound | High AI → Compute bound")
    print(f"\n{'Operation':<25} {'AI':<12} {'Time (ms)':<12} {'Bottleneck'}")
    print("-" * 60)
    
    # Element-wise: 1 FLOP per 8 bytes (read + write)
    x = torch.randn(N * N, device='cuda')
    time_add = profile_op(lambda: x + 1.0)
    ai_add = 1 / 8
    print(f"{'x + 1.0':<25} {ai_add:<12.2f} {time_add:<12.3f} {'MEMORY'}")
    
    # More FLOPs, same bytes
    time_poly = profile_op(lambda: x**3 + 2*x**2 + x + 1)
    ai_poly = 6 / 8
    print(f"{'x³ + 2x² + x + 1':<25} {ai_poly:<12.2f} {time_poly:<12.3f} {'MEMORY'}")
    
    # Matrix multiply: 2N FLOPs per element, amortized
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    time_mm = profile_op(lambda: A @ B)
    ai_mm = (2 * N) / 8  # Simplified
    print(f"{'A @ B (4096×4096)':<25} {ai_mm:<12.0f} {time_mm:<12.3f} {'COMPUTE'}")
    
    # Softmax: ~10 FLOPs per element
    x_soft = torch.randn(N, N, device='cuda')
    time_soft = profile_op(lambda: torch.softmax(x_soft, dim=-1))
    ai_soft = 10 / 8
    print(f"{'softmax':<25} {ai_soft:<12.2f} {time_soft:<12.3f} {'MEMORY'}")
    
    print(f"\n RIDGE POINT (where memory BW = compute peak):")
    print(f" Typical GPU: ~100-200 FLOPs/byte")
    print(f" Below ridge → Memory bound (most element-wise ops)")
    print(f" Above ridge → Compute bound (matmul, convolution)")

# ============================================================================
# SUMMARY
# ============================================================================

def print_memory_summary():
    """Print memory hierarchy summary."""
    print("\n" + "="*70)
    print(" MEMORY HIERARCHY SUMMARY")
    print("="*70)
    
    print("""
    KEY OPTIMIZATION STRATEGIES:
    
    1. MAXIMIZE DATA REUSE
       • Use shared memory for repeated access
       • Tile computations to fit in cache
       • Example: Flash Attention tiles Q, K, V
    
    2. COALESCE MEMORY ACCESS
       • Adjacent threads → adjacent addresses
       • Access along contiguous dimension
       • Use .contiguous() when needed
    
    3. MINIMIZE MEMORY TRAFFIC
       • Fuse operations (keep data in registers)
       • Recompute cheap values vs storing
       • Example: Gradient checkpointing
    
    4. HIDE LATENCY
       • Keep many threads active (high occupancy)
       • Use prefetching when possible
       • Overlap transfer with compute (streams)
    
    MULTIMODAL TRAINING IMPLICATIONS:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Component          │ Memory Behavior         │ Optimization     │
    ├────────────────────┼─────────────────────────┼──────────────────┤
    │ Embeddings         │ Sparse, random access   │ Cache, batch     │
    │ Linear layers      │ High reuse (matmul)     │ Tensor Cores     │
    │ Attention QK^T     │ Memory-bound            │ Flash Attention  │
    │ Softmax            │ Memory-bound            │ Fused kernel     │
    │ Layer Norm         │ Memory-bound            │ Fused kernel     │
    │ Data loading       │ PCIe bottleneck         │ Prefetch, async  │
    └─────────────────────────────────────────────────────────────────┘
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " GPU MEMORY HIERARCHY DEEP DIVE ".center(68) + "║")
    print("║" + " Understanding memory is key to performance ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    print_memory_hierarchy()
    experiment_cache_effects()
    experiment_access_patterns()
    experiment_shared_memory_benefit()
    experiment_bandwidth_vs_compute()
    print_memory_summary()
    
    print("\n" + "="*70)
    print(" Memory optimization is often more important than compute optimization!")
    print("="*70)
