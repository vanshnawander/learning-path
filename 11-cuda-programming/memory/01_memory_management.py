"""
01_memory_management.py - CUDA Memory Management Deep Dive

Memory management is often the #1 source of performance issues in GPU code.
Understanding memory types, allocation patterns, and transfer costs
is essential for writing fast GPU programs.

Memory Types:
1. Global Memory (HBM) - Large, slow, accessible by all threads
2. Shared Memory - Small, fast, shared within a block
3. Registers - Fastest, per-thread, limited
4. Constant Memory - Read-only, cached, broadcast
5. Texture Memory - Optimized for 2D spatial locality
6. Unified Memory - Automatic CPU/GPU migration

For multimodal training:
- Large models stress global memory
- Attention benefits from shared memory tiling
- KV cache is memory-bound
- Data transfers can dominate if not overlapped

Run: python 01_memory_management.py
"""

import torch
import torch.nn as nn
import time
from contextlib import contextmanager
from typing import Tuple, List, Dict
import gc

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_cuda(func, warmup=10, iterations=100):
    """Profile CUDA function with proper timing."""
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

def get_memory_info() -> Dict:
    """Get current GPU memory state."""
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1e6,
        'reserved_mb': torch.cuda.memory_reserved() / 1e6,
        'peak_mb': torch.cuda.max_memory_allocated() / 1e6,
        'free_mb': (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_reserved()) / 1e6
    }

@contextmanager
def track_memory(name: str = ""):
    """Context manager to track memory changes."""
    if not torch.cuda.is_available():
        yield
        return
    
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    yield
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    delta = (after - before) / 1e6
    print(f" {name}: {delta:+.2f} MB (total: {after/1e6:.2f} MB)")

# ============================================================================
# EXPERIMENT 1: GLOBAL MEMORY (HBM)
# ============================================================================

def experiment_global_memory():
    """
    Global Memory is the main GPU memory (HBM on datacenter GPUs).
    
    Properties:
    - Large capacity (40-80GB on modern GPUs)
    - High bandwidth (2-8 TB/s)
    - High latency (~400-600 cycles)
    - Accessible by all threads
    
    This is where tensors, weights, and activations live.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: GLOBAL MEMORY (HBM)")
    print(" Main GPU memory - large but high latency")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory / 1e9
    
    print(f"\n GPU: {props.name}")
    print(f" Total Memory: {total_mem:.1f} GB")
    
    # Measure bandwidth at different sizes
    print(f"\n Global memory bandwidth (copy operation):")
    print(f"{'Size (MB)':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'Efficiency'}")
    print("-" * 70)
    
    sizes_mb = [1, 10, 100, 500, 1000, 2000]
    
    for size_mb in sizes_mb:
        if size_mb * 2 > total_mem * 1000 * 0.8:  # Leave 20% free
            continue
            
        num_elements = int(size_mb * 1e6 / 4)  # float32
        x = torch.randn(num_elements, device='cuda')
        
        def copy_op():
            return x.clone()
        
        time_ms = profile_cuda(copy_op, iterations=50)
        bytes_moved = num_elements * 4 * 2  # Read + write
        bandwidth = bytes_moved / (time_ms / 1000) / 1e9
        
        # Theoretical max varies by GPU
        theoretical = 900  # Approximate for RTX 3090
        efficiency = bandwidth / theoretical * 100
        
        print(f"{size_mb:<15} {time_ms:<15.3f} {bandwidth:<20.1f} ~{efficiency:.0f}%")
        
        del x
        torch.cuda.empty_cache()
    
    print(f"\n KEY INSIGHT:")
    print(f" - Large transfers achieve near-peak bandwidth")
    print(f" - Small transfers have overhead (not bandwidth-limited)")
    print(f" - HBM bandwidth: 900 GB/s (RTX 3090) to 3350 GB/s (H100)")

# ============================================================================
# EXPERIMENT 2: MEMORY ALLOCATION PATTERNS
# ============================================================================

def experiment_allocation_patterns():
    """
    GPU memory allocation is expensive!
    PyTorch uses a caching allocator to reduce overhead.
    
    Understanding allocation patterns helps avoid:
    - Unnecessary allocations in hot loops
    - Memory fragmentation
    - OOM errors from poor memory management
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY ALLOCATION PATTERNS")
    print(" Allocation is expensive - PyTorch caches to help")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n Memory tracking through allocations:")
    print(f"{'Operation':<45} {'Allocated':<15} {'Reserved':<15}")
    print("-" * 75)
    
    info = get_memory_info()
    print(f"{'Initial state':<45} {info['allocated_mb']:<15.2f} {info['reserved_mb']:<15.2f}")
    
    # Allocate tensor
    x = torch.randn(1024, 1024, 1024, device='cuda')  # 4 GB
    info = get_memory_info()
    print(f"{'After 4GB allocation':<45} {info['allocated_mb']:<15.2f} {info['reserved_mb']:<15.2f}")
    
    # Delete but don't empty cache
    del x
    info = get_memory_info()
    print(f"{'After del x (cached)':<45} {info['allocated_mb']:<15.2f} {info['reserved_mb']:<15.2f}")
    
    # Allocate same size - should reuse
    y = torch.randn(1024, 1024, 1024, device='cuda')
    info = get_memory_info()
    print(f"{'Reallocate 4GB (reused from cache)':<45} {info['allocated_mb']:<15.2f} {info['reserved_mb']:<15.2f}")
    
    del y
    torch.cuda.empty_cache()
    info = get_memory_info()
    print(f"{'After empty_cache()':<45} {info['allocated_mb']:<15.2f} {info['reserved_mb']:<15.2f}")
    
    # Allocation timing comparison
    print(f"\n Allocation timing comparison:")
    print(f"{'Method':<35} {'Time (μs)':<15}")
    print("-" * 50)
    
    # Cold allocation
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        x = torch.randn(1000000, device='cuda')
        del x
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cold_time = (time.perf_counter() - start) * 1e6 / 100
    
    # Warm allocation (cached)
    x = torch.randn(1000000, device='cuda')
    del x
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        x = torch.randn(1000000, device='cuda')
        del x
    torch.cuda.synchronize()
    warm_time = (time.perf_counter() - start) * 1e6 / 100
    
    # Pre-allocated (in-place)
    x = torch.empty(1000000, device='cuda')
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        x.normal_()
    torch.cuda.synchronize()
    inplace_time = (time.perf_counter() - start) * 1e6 / 100
    
    print(f"{'Cold allocation (fresh cudaMalloc)':<35} {cold_time:<15.1f}")
    print(f"{'Warm allocation (cached)':<35} {warm_time:<15.1f}")
    print(f"{'Pre-allocated (in-place fill)':<35} {inplace_time:<15.1f}")
    
    print(f"\n BEST PRACTICES:")
    print(f" 1. Pre-allocate buffers when size is known")
    print(f" 2. Use in-place operations when safe")
    print(f" 3. Don't call empty_cache() unless necessary")
    print(f" 4. Avoid allocations in tight loops")

# ============================================================================
# EXPERIMENT 3: CPU <-> GPU TRANSFERS
# ============================================================================

def experiment_cpu_gpu_transfers():
    """
    Data transfer between CPU and GPU is SLOW.
    
    PCIe 4.0 x16: ~32 GB/s (bidirectional)
    PCIe 5.0 x16: ~64 GB/s
    
    Compare to GPU memory: ~2000-3000 GB/s
    
    Transfers can easily become the bottleneck!
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: CPU <-> GPU TRANSFERS")
    print(" PCIe is 50-100x slower than GPU memory!")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    sizes_mb = [1, 10, 50, 100, 500]
    
    print(f"\n Transfer bandwidth comparison:")
    print(f"{'Size (MB)':<12} {'CPU→GPU (ms)':<15} {'GPU→CPU (ms)':<15} {'BW (GB/s)':<12} {'Memory Type'}")
    print("-" * 75)
    
    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1e6 / 4)
        
        # Pageable memory (default)
        x_pageable = torch.randn(num_elements)
        
        # CPU to GPU (pageable)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        x_gpu = x_pageable.to('cuda')
        end.record()
        torch.cuda.synchronize()
        
        time_h2d_pageable = start.elapsed_time(end)
        
        # GPU to CPU
        start.record()
        x_back = x_gpu.cpu()
        end.record()
        torch.cuda.synchronize()
        
        time_d2h = start.elapsed_time(end)
        
        bytes_moved = num_elements * 4
        bandwidth = bytes_moved / (time_h2d_pageable / 1000) / 1e9
        
        print(f"{size_mb:<12} {time_h2d_pageable:<15.2f} {time_d2h:<15.2f} {bandwidth:<12.1f} Pageable")
        
        del x_pageable, x_gpu, x_back
        torch.cuda.empty_cache()
    
    # Pinned memory comparison
    print(f"\n Pinned memory benefits:")
    print(f"{'Size (MB)':<12} {'Pageable (ms)':<18} {'Pinned (ms)':<18} {'Speedup'}")
    print("-" * 65)
    
    for size_mb in [10, 100, 500]:
        num_elements = int(size_mb * 1e6 / 4)
        
        # Pageable
        x_pageable = torch.randn(num_elements)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        x_gpu = x_pageable.to('cuda')
        end.record()
        torch.cuda.synchronize()
        time_pageable = start.elapsed_time(end)
        
        del x_pageable, x_gpu
        
        # Pinned
        x_pinned = torch.randn(num_elements).pin_memory()
        torch.cuda.synchronize()
        
        start.record()
        x_gpu = x_pinned.to('cuda', non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        time_pinned = start.elapsed_time(end)
        
        speedup = time_pageable / time_pinned
        print(f"{size_mb:<12} {time_pageable:<18.2f} {time_pinned:<18.2f} {speedup:.2f}x")
        
        del x_pinned, x_gpu
        torch.cuda.empty_cache()
    
    print(f"\n WHY PINNED MEMORY IS FASTER:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │ Pageable: CPU RAM → Staging Buffer → GPU (2 copies)       │")
    print(f" │ Pinned:   CPU RAM ──────────────────→ GPU (1 copy, DMA)   │")
    print(f" └────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - DataLoader: pin_memory=True for faster transfers")
    print(f" - Use non_blocking=True to overlap transfer with compute")
    print(f" - Prefetch data to GPU while processing current batch")

# ============================================================================
# EXPERIMENT 4: MEMORY-BOUND VS COMPUTE-BOUND
# ============================================================================

def experiment_memory_vs_compute_bound():
    """
    Understanding whether code is memory-bound or compute-bound
    is essential for optimization.
    
    Memory-bound: Waiting for data from memory
    Compute-bound: Waiting for arithmetic units
    
    Most ML operations are MEMORY-BOUND!
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: MEMORY-BOUND VS COMPUTE-BOUND")
    print(" Most ML ops are memory-bound!")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    
    print(f"\n Arithmetic Intensity Analysis:")
    print(f"{'Operation':<25} {'AI (FLOP/Byte)':<18} {'Type':<15} {'Time (ms)'}")
    print("-" * 75)
    
    operations = []
    
    # Element-wise add: 1 FLOP, 12 bytes (read A, B, write C)
    A = torch.randn(N * N, device='cuda')
    B = torch.randn(N * N, device='cuda')
    
    time_add = profile_cuda(lambda: A + B)
    ai_add = 1 / 12
    operations.append(("Vector add", ai_add, "MEMORY", time_add))
    
    # Element-wise multiply-add: 2 FLOPs, 12 bytes
    time_fma = profile_cuda(lambda: A * B + A)
    ai_fma = 2 / 12
    operations.append(("Vector FMA", ai_fma, "MEMORY", time_fma))
    
    # Softmax: ~5 FLOPs per element, 8 bytes
    x = torch.randn(N, N, device='cuda')
    time_soft = profile_cuda(lambda: torch.softmax(x, dim=-1))
    ai_soft = 5 / 8
    operations.append(("Softmax", ai_soft, "MEMORY", time_soft))
    
    # Layer Norm: ~10 FLOPs per element, 8 bytes
    ln = nn.LayerNorm(N).cuda()
    time_ln = profile_cuda(lambda: ln(x))
    ai_ln = 10 / 8
    operations.append(("Layer Norm", ai_ln, "MEMORY", time_ln))
    
    # Matrix multiply: 2*N FLOPs per element, 12 bytes (amortized)
    A_mat = torch.randn(N, N, device='cuda')
    B_mat = torch.randn(N, N, device='cuda')
    
    time_mm = profile_cuda(lambda: A_mat @ B_mat)
    ai_mm = 2 * N / 12  # N FLOPs per output, amortized over input reads
    operations.append(("Matrix multiply", ai_mm, "COMPUTE", time_mm))
    
    # Batched matmul
    A_batch = torch.randn(32, N, N, device='cuda')
    B_batch = torch.randn(32, N, N, device='cuda')
    
    time_bmm = profile_cuda(lambda: A_batch @ B_batch)
    ai_bmm = 2 * N / 12
    operations.append(("Batched matmul", ai_bmm, "COMPUTE", time_bmm))
    
    for name, ai, bound_type, time_ms in operations:
        print(f"{name:<25} {ai:<18.2f} {bound_type:<15} {time_ms:.3f}")
    
    print(f"\n ROOFLINE MODEL:")
    print(f" ┌────────────────────────────────────────────────────────────────┐")
    print(f" │ Performance                                                    │")
    print(f" │      ▲                    _______________Compute Peak          │")
    print(f" │      │                  /                                      │")
    print(f" │      │                /   ← Ridge point (AI = Peak/BW)        │")
    print(f" │      │              /                                          │")
    print(f" │      │            / Memory BW ceiling                          │")
    print(f" │      │          /                                              │")
    print(f" │      │        /  Element-wise ops here (low AI)               │")
    print(f" │      │      /                                                  │")
    print(f" │      └──────────────────────────────────────────────────────▶  │")
    print(f" │                 Arithmetic Intensity (FLOPs/Byte)              │")
    print(f" └────────────────────────────────────────────────────────────────┘")
    print(f"\n Ridge point for H100: ~250 FLOPs/byte")
    print(f" Ridge point for A100: ~150 FLOPs/byte")
    print(f" Matrix multiply (N=4096): ~680 FLOPs/byte → Compute-bound")

# ============================================================================
# EXPERIMENT 5: MEMORY FRAGMENTATION
# ============================================================================

def experiment_memory_fragmentation():
    """
    Memory fragmentation can cause OOM even when "enough" memory is free.
    
    Problem: Free memory is split into non-contiguous chunks.
    Large allocations need contiguous memory!
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: MEMORY FRAGMENTATION")
    print(" OOM can happen even with 'enough' free memory")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    torch.cuda.empty_cache()
    
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / 1e9
    
    # Use smaller allocations to avoid OOM on smaller GPUs
    alloc_size_gb = min(0.5, total_mem_gb / 20)
    alloc_size = int(alloc_size_gb * 1e9 / 4)  # float32 elements
    
    print(f"\n Demonstrating fragmentation:")
    print(f" Allocation size: {alloc_size_gb:.2f} GB each")
    
    # Allocate several tensors
    tensors = []
    for i in range(8):
        try:
            t = torch.randn(alloc_size, device='cuda')
            tensors.append(t)
        except RuntimeError:
            print(f" Could only allocate {i} tensors")
            break
    
    info = get_memory_info()
    print(f"\n After allocating {len(tensors)} tensors:")
    print(f" Allocated: {info['allocated_mb']:.0f} MB")
    print(f" Reserved:  {info['reserved_mb']:.0f} MB")
    
    # Delete every other tensor (creates fragmentation)
    for i in range(0, len(tensors), 2):
        del tensors[i]
        tensors[i] = None
    
    tensors = [t for t in tensors if t is not None]
    gc.collect()
    
    info = get_memory_info()
    print(f"\n After deleting every other tensor (fragmented):")
    print(f" Allocated: {info['allocated_mb']:.0f} MB")
    print(f" Reserved:  {info['reserved_mb']:.0f} MB")
    print(f" 'Free' in reserved: {info['reserved_mb'] - info['allocated_mb']:.0f} MB")
    
    # Try to allocate a larger tensor
    large_size = int(alloc_size * 1.5)
    print(f"\n Trying to allocate {alloc_size_gb * 1.5:.2f} GB (larger than gaps)...")
    
    try:
        large = torch.randn(large_size, device='cuda')
        print(" Success! (memory was defragmented)")
        del large
    except RuntimeError as e:
        print(f" Failed! Fragmentation prevented allocation")
        print(f" Error: {str(e)[:100]}...")
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache()
    
    print(f"\n HOW TO AVOID FRAGMENTATION:")
    print(f" 1. Allocate large tensors first")
    print(f" 2. Keep similar-sized tensors together")
    print(f" 3. Use memory pools for repeated allocations")
    print(f" 4. Call empty_cache() sparingly (can worsen fragmentation)")
    print(f" 5. Use gradient checkpointing to reduce peak memory")

# ============================================================================
# EXPERIMENT 6: PINNED MEMORY AND ASYNC TRANSFERS
# ============================================================================

def experiment_async_transfers():
    """
    Async transfers allow overlapping data movement with compute.
    
    This is CRITICAL for data loading pipelines!
    While GPU processes batch N, transfer batch N+1.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: ASYNC TRANSFERS AND OVERLAP")
    print(" Overlap data transfer with computation")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch_size = 64
    seq_len = 512
    hidden = 1024
    
    # Simulate a simple model
    model = nn.Sequential(
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden)
    ).cuda()
    
    # Create "batches" on CPU
    num_batches = 5
    cpu_batches = [torch.randn(batch_size, seq_len, hidden).pin_memory() 
                   for _ in range(num_batches)]
    
    print(f"\n Processing {num_batches} batches: ({batch_size}, {seq_len}, {hidden})")
    
    # Method 1: Sequential (no overlap)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for batch_cpu in cpu_batches:
        batch_gpu = batch_cpu.to('cuda')  # Blocking transfer
        output = model(batch_gpu)
    end.record()
    torch.cuda.synchronize()
    
    sequential_time = start.elapsed_time(end)
    
    # Method 2: Overlapped with streams
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start.record()
    
    # Pre-transfer first batch
    with torch.cuda.stream(transfer_stream):
        current_batch = cpu_batches[0].to('cuda', non_blocking=True)
    
    for i in range(num_batches):
        # Wait for current batch transfer
        compute_stream.wait_stream(transfer_stream)
        
        # Start transferring next batch (if any)
        if i + 1 < num_batches:
            with torch.cuda.stream(transfer_stream):
                next_batch = cpu_batches[i + 1].to('cuda', non_blocking=True)
        
        # Process current batch
        with torch.cuda.stream(compute_stream):
            output = model(current_batch)
        
        if i + 1 < num_batches:
            transfer_stream.wait_stream(compute_stream)
            current_batch = next_batch
    
    end.record()
    torch.cuda.synchronize()
    
    overlapped_time = start.elapsed_time(end)
    
    print(f"\n{'Method':<30} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 55)
    print(f"{'Sequential':<30} {sequential_time:<15.2f} 1.0x")
    print(f"{'Overlapped (streams)':<30} {overlapped_time:<15.2f} {sequential_time/overlapped_time:.2f}x")
    
    print(f"\n OVERLAP DIAGRAM:")
    print(f" Sequential:")
    print(f" Transfer: |████|    |████|    |████|")
    print(f" Compute:       |████|    |████|    |████|")
    print(f"")
    print(f" Overlapped:")
    print(f" Transfer: |████|████|████|████|")
    print(f" Compute:    |████|████|████|████|")
    print(f"          ↑ Overlap saves time!")
    
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - DataLoader uses this automatically with num_workers > 0")
    print(f" - prefetch_factor controls how many batches to pre-load")
    print(f" - Critical for image/video data with expensive decoding")

# ============================================================================
# SUMMARY
# ============================================================================

def print_memory_summary():
    """Print comprehensive memory management summary."""
    print("\n" + "="*70)
    print(" MEMORY MANAGEMENT SUMMARY")
    print("="*70)
    
    print("""
    MEMORY HIERARCHY RECAP:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Level        │ Size       │ Latency    │ Bandwidth  │ Scope    │
    ├──────────────┼────────────┼────────────┼────────────┼──────────┤
    │ Registers    │ ~256KB/SM  │ 0 cycles   │ ~TB/s      │ Thread   │
    │ Shared Mem   │ 48-228KB   │ ~20 cycles │ ~TB/s      │ Block    │
    │ L1 Cache     │ 128-256KB  │ ~30 cycles │ ~TB/s      │ SM       │
    │ L2 Cache     │ 40-50MB    │ ~200 cy    │ ~TB/s      │ Device   │
    │ Global (HBM) │ 40-80GB    │ ~400 cy    │ 2-8 TB/s   │ Device   │
    │ CPU RAM      │ 64-256GB   │ 1000s cy   │ 50-100GB/s │ Host     │
    │ PCIe         │ N/A        │ 1000s cy   │ 32-64 GB/s │ Transfer │
    └─────────────────────────────────────────────────────────────────┘
    
    KEY OPTIMIZATION STRATEGIES:
    
    1. MINIMIZE TRANSFERS
       - Keep data on GPU as long as possible
       - Batch small transfers into larger ones
       - Use pinned memory for faster transfers
    
    2. MAXIMIZE REUSE
       - Use shared memory for data accessed multiple times
       - Tile computations to fit in cache
       - Fuse operations to avoid intermediate writes
    
    3. OPTIMIZE ACCESS PATTERNS
       - Coalesced access: threads access consecutive addresses
       - Avoid bank conflicts in shared memory
       - Align data to cache line boundaries
    
    4. MANAGE ALLOCATION
       - Pre-allocate buffers when possible
       - Use in-place operations
       - Avoid allocations in hot loops
    
    MULTIMODAL TRAINING CHECKLIST:
    
    ✓ DataLoader: num_workers > 0, pin_memory=True
    ✓ Batch size: Large enough to saturate GPU
    ✓ Activation checkpointing for memory savings
    ✓ Mixed precision to halve memory usage
    ✓ Flash Attention for memory-efficient attention
    ✓ KV cache management for inference
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " CUDA MEMORY MANAGEMENT ".center(68) + "║")
    print("║" + " Understanding memory is key to GPU performance ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n GPU: {props.name}")
        print(f" Memory: {props.total_memory / 1e9:.1f} GB")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_global_memory()
    experiment_allocation_patterns()
    experiment_cpu_gpu_transfers()
    experiment_memory_vs_compute_bound()
    experiment_memory_fragmentation()
    experiment_async_transfers()
    print_memory_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Optimization techniques and advanced patterns")
    print("="*70)
