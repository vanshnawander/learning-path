"""
01_gpu_vs_cpu.py - Understanding GPU vs CPU Architecture Differences

This module demonstrates the fundamental architectural differences between
CPUs and GPUs through practical profiled examples. Every operation is timed
to build intuition about where GPUs excel and where they don't.

KEY INSIGHT: GPUs are throughput-oriented (many operations in parallel),
CPUs are latency-oriented (single operations fast).

For multimodal training:
- Image batches: GPU excels (parallel pixel processing)
- Text tokenization: CPU often better (sequential, branching)
- Audio FFT: GPU excels (parallel frequency computation)
- Data loading: CPU handles I/O, GPU handles compute

Run: python 01_gpu_vs_cpu.py
Requirements: torch, numpy, matplotlib
"""

import torch
import numpy as np
import time
from typing import Tuple, List, Dict
from dataclasses import dataclass
from contextlib import contextmanager

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

@dataclass
class TimingResult:
    """Store timing results with metadata for analysis."""
    operation: str
    device: str
    size: int
    time_ms: float
    throughput_gflops: float = 0.0
    bandwidth_gbps: float = 0.0

class Profiler:
    """Simple profiler for CPU/GPU operations."""
    
    def __init__(self):
        self.results: List[TimingResult] = []
    
    @contextmanager
    def time_operation(self, operation: str, device: str, size: int, 
                       flops: int = 0, bytes_moved: int = 0):
        """Context manager for timing operations."""
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            yield
            time_ms = (time.perf_counter() - start) * 1000
        
        throughput = (flops / (time_ms / 1000) / 1e9) if flops > 0 else 0
        bandwidth = (bytes_moved / (time_ms / 1000) / 1e9) if bytes_moved > 0 else 0
        
        result = TimingResult(operation, device, size, time_ms, throughput, bandwidth)
        self.results.append(result)
        
    def print_comparison(self, title: str):
        """Print formatted comparison of CPU vs GPU timings."""
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
        print(f"{'Operation':<25} {'Device':<8} {'Size':<12} {'Time (ms)':<12} {'Speedup':<10}")
        print(f"{'-'*70}")
        
        # Group by operation
        ops = {}
        for r in self.results:
            key = (r.operation, r.size)
            if key not in ops:
                ops[key] = {}
            ops[key][r.device] = r
        
        for (op, size), devices in ops.items():
            for device, r in devices.items():
                speedup = ""
                if "cpu" in devices and "cuda" in devices and device == "cuda":
                    speedup = f"{devices['cpu'].time_ms / r.time_ms:.1f}x"
                print(f"{op:<25} {device:<8} {size:<12} {r.time_ms:<12.3f} {speedup:<10}")
        
        self.results = []

profiler = Profiler()

# ============================================================================
# EXPERIMENT 1: LATENCY VS THROUGHPUT
# ============================================================================

def experiment_latency_vs_throughput():
    """
    Demonstrate the fundamental difference:
    - CPU: Optimized for low latency on single operations
    - GPU: Optimized for high throughput on parallel operations
    
    For multimodal training:
    - Small operations (single token embedding lookup): CPU wins
    - Large operations (batch of image convolutions): GPU wins
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: LATENCY VS THROUGHPUT")
    print(" CPU = Fast single operations | GPU = Fast parallel operations")
    print("="*70)
    
    # Test sizes from tiny to large
    sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    
    print(f"\n{'Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'GPU Speedup':<15} {'Winner'}")
    print("-" * 70)
    
    for size in sizes:
        # CPU timing
        x_cpu = torch.randn(size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.perf_counter()
        y_cpu = x_cpu * x_cpu + x_cpu  # Simple element-wise ops
        cpu_time = (time.perf_counter() - start) * 1000
        
        # GPU timing (including transfer for fair comparison)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            x_gpu = torch.randn(size, device='cuda')
            y_gpu = x_gpu * x_gpu + x_gpu
            end.record()
            torch.cuda.synchronize()
            
            gpu_time = start.elapsed_time(end)
        else:
            gpu_time = float('inf')
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        winner = "GPU" if speedup > 1 else "CPU"
        
        print(f"{size:<15} {cpu_time:<15.4f} {gpu_time:<15.4f} {speedup:<15.2f} {winner}")
    
    print("\n KEY INSIGHT:")
    print(" - Small sizes: GPU kernel launch overhead (~5-20μs) dominates")
    print(" - Large sizes: GPU's parallel compute dominates")
    print(" - Crossover point: ~10K-100K elements depending on operation")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Batch size matters! Small batches waste GPU potential")
    print(" - Token-by-token generation is CPU-bound (sequential)")
    print(" - Image processing in batches is GPU-optimal")

# ============================================================================
# EXPERIMENT 2: MEMORY BANDWIDTH COMPARISON
# ============================================================================

def experiment_memory_bandwidth():
    """
    Compare memory bandwidth between CPU and GPU.
    
    Typical bandwidths:
    - CPU DDR4/DDR5: 50-100 GB/s
    - GPU HBM2e (A100): 2039 GB/s
    - GPU HBM3 (H100): 3350 GB/s
    
    For multimodal training, memory bandwidth determines:
    - How fast we can load image batches
    - Attention mechanism speed (memory-bound)
    - Embedding table lookups
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY BANDWIDTH")
    print(" Bandwidth = Data moved / Time")
    print("="*70)
    
    sizes_mb = [1, 10, 100, 500, 1000]  # MB
    
    print(f"\n{'Size (MB)':<12} {'CPU BW (GB/s)':<18} {'GPU BW (GB/s)':<18} {'Ratio'}")
    print("-" * 65)
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4  # float32
        bytes_moved = num_elements * 4 * 2  # Read + Write
        
        # CPU bandwidth test (simple copy)
        x_cpu = torch.randn(num_elements)
        start = time.perf_counter()
        y_cpu = x_cpu.clone()
        cpu_time = time.perf_counter() - start
        cpu_bandwidth = bytes_moved / cpu_time / 1e9
        
        # GPU bandwidth test
        if torch.cuda.is_available():
            x_gpu = torch.randn(num_elements, device='cuda')
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            y_gpu = x_gpu.clone()
            end.record()
            torch.cuda.synchronize()
            
            gpu_time = start.elapsed_time(end) / 1000  # Convert to seconds
            gpu_bandwidth = bytes_moved / gpu_time / 1e9
        else:
            gpu_bandwidth = 0
        
        ratio = gpu_bandwidth / cpu_bandwidth if cpu_bandwidth > 0 else 0
        print(f"{size_mb:<12} {cpu_bandwidth:<18.2f} {gpu_bandwidth:<18.2f} {ratio:.1f}x")
    
    print("\n KEY INSIGHT:")
    print(" - GPU memory bandwidth is 20-40x higher than CPU")
    print(" - This is WHY GPUs are fast for ML - we move lots of data")
    print(" - Arithmetic intensity = FLOPs / Bytes determines GPU benefit")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Image tensors are large → GPU bandwidth helps")
    print(" - Attention is memory-bound → Flash Attention reduces memory traffic")
    print(" - Embedding lookups are random access → less bandwidth efficient")

# ============================================================================
# EXPERIMENT 3: PARALLEL COMPUTE - SIMT MODEL
# ============================================================================

def experiment_parallel_compute():
    """
    Demonstrate SIMT (Single Instruction Multiple Threads) parallelism.
    
    GPU executes the SAME operation on THOUSANDS of elements simultaneously.
    This is perfect for:
    - Matrix operations (each element computed independently)
    - Convolutions (each output pixel independent)
    - Attention scores (each query-key pair independent)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SIMT PARALLELISM")
    print(" GPU: Thousands of threads execute same instruction")
    print("="*70)
    
    # Matrix multiplication - the bread and butter of ML
    sizes = [128, 256, 512, 1024, 2048, 4096]
    
    print(f"\n Matrix Multiplication: (N x N) @ (N x N)")
    print(f"{'N':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'GPU TFLOPS':<15} {'Speedup'}")
    print("-" * 65)
    
    for N in sizes:
        flops = 2 * N * N * N  # matmul FLOPs
        
        # CPU timing
        A_cpu = torch.randn(N, N)
        B_cpu = torch.randn(N, N)
        
        start = time.perf_counter()
        C_cpu = A_cpu @ B_cpu
        cpu_time = (time.perf_counter() - start) * 1000
        
        # GPU timing
        if torch.cuda.is_available():
            A_gpu = torch.randn(N, N, device='cuda')
            B_gpu = torch.randn(N, N, device='cuda')
            torch.cuda.synchronize()
            
            # Warmup
            _ = A_gpu @ B_gpu
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            C_gpu = A_gpu @ B_gpu
            end.record()
            torch.cuda.synchronize()
            
            gpu_time = start.elapsed_time(end)
            gpu_tflops = flops / (gpu_time / 1000) / 1e12
        else:
            gpu_time = float('inf')
            gpu_tflops = 0
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"{N:<10} {cpu_time:<15.3f} {gpu_time:<15.3f} {gpu_tflops:<15.2f} {speedup:.0f}x")
    
    print("\n KEY INSIGHT:")
    print(" - Matmul has O(N³) compute with O(N²) data → high arithmetic intensity")
    print(" - This is IDEAL for GPU: lots of parallel, independent work")
    print(" - Modern GPUs achieve 100+ TFLOPS on large matmuls")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Transformers are mostly matmuls → GPU optimal")
    print(" - Batch matmul for attention: (B, H, S, D) @ (B, H, D, S)")
    print(" - Larger batch = better GPU utilization")

# ============================================================================
# EXPERIMENT 4: BRANCHING AND DIVERGENCE
# ============================================================================

def experiment_branching():
    """
    Demonstrate why GPUs struggle with branching/conditional code.
    
    GPU SIMT model: All threads in a warp (32 threads) must execute
    the SAME instruction. If threads diverge (different branches),
    both paths must be executed serially!
    
    This is critical for understanding:
    - Why tokenization is CPU-bound (lots of conditionals)
    - Why sparse operations are tricky on GPU
    - Why attention masking patterns matter
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: BRANCHING AND DIVERGENCE")
    print(" GPU threads in a warp (32) must execute same instruction")
    print("="*70)
    
    size = 10_000_000
    
    # No branching - all threads do same thing
    x = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def no_branch(x):
        return x * 2 + 1
    
    def with_branch(x):
        # This creates divergence - threads take different paths
        return torch.where(x > 0, x * 2, x * 0.5)
    
    def heavy_branch(x):
        # Even more divergence
        mask1 = x > 0.5
        mask2 = (x > -0.5) & (x <= 0.5)
        mask3 = x <= -0.5
        return torch.where(mask1, x * 2, 
               torch.where(mask2, x + 1, x * 0.1))
    
    # Profile each
    operations = [
        ("No branching", no_branch),
        ("Simple branch (where)", with_branch),
        ("Complex branching", heavy_branch)
    ]
    
    print(f"\n{'Operation':<25} {'Time (ms)':<15} {'Relative'}")
    print("-" * 55)
    
    base_time = None
    for name, func in operations:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            _ = func(x)
            torch.cuda.synchronize()
            
            start.record()
            for _ in range(100):
                _ = func(x)
            end.record()
            torch.cuda.synchronize()
            
            time_ms = start.elapsed_time(end) / 100
        else:
            start = time.perf_counter()
            for _ in range(100):
                _ = func(x)
            time_ms = (time.perf_counter() - start) * 1000 / 100
        
        if base_time is None:
            base_time = time_ms
        relative = time_ms / base_time
        
        print(f"{name:<25} {time_ms:<15.3f} {relative:.2f}x")
    
    print("\n KEY INSIGHT:")
    print(" - Branching causes 'warp divergence' - threads wait for each other")
    print(" - torch.where is optimized but still has overhead")
    print(" - Best practice: Avoid conditionals, use masking")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Text tokenization has many conditionals → CPU")
    print(" - Attention masking: Use additive masks, not conditionals")
    print(" - Sparse attention patterns need special kernels")

# ============================================================================
# EXPERIMENT 5: KERNEL LAUNCH OVERHEAD
# ============================================================================

def experiment_kernel_overhead():
    """
    Every GPU operation has launch overhead (~5-20 microseconds).
    
    This is why:
    - Fusing operations is important (fewer launches)
    - Small operations should stay on CPU
    - torch.compile helps by fusing kernels
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: KERNEL LAUNCH OVERHEAD")
    print(" Each GPU kernel launch costs ~5-20 microseconds")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available, skipping...")
        return
    
    x = torch.randn(1000, device='cuda')
    
    # Many separate operations (many kernel launches)
    def many_kernels(x):
        x = x + 1
        x = x * 2
        x = x - 1
        x = x / 2
        x = x + 0.5
        x = x * 1.5
        x = x - 0.25
        x = x / 1.5
        return x
    
    # Equivalent single operation
    def fused_equivalent(x):
        # Mathematically equivalent to many_kernels
        return (((((((x + 1) * 2) - 1) / 2) + 0.5) * 1.5) - 0.25) / 1.5
    
    # Using torch.compile (if available)
    try:
        compiled_many = torch.compile(many_kernels)
        has_compile = True
    except:
        has_compile = False
    
    # Profile
    print(f"\n Timing 1000 iterations on small tensor (1000 elements):")
    print(f"{'Method':<30} {'Time (ms)':<15} {'Per-call (μs)'}")
    print("-" * 60)
    
    # Many kernels
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):
        _ = many_kernels(x)
    end.record()
    torch.cuda.synchronize()
    many_time = start.elapsed_time(end)
    print(f"{'Separate operations (8 kernels)':<30} {many_time:<15.2f} {many_time:.1f}")
    
    # Try compiled version
    if has_compile:
        # Warmup
        _ = compiled_many(x)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(1000):
            _ = compiled_many(x)
        end.record()
        torch.cuda.synchronize()
        compiled_time = start.elapsed_time(end)
        print(f"{'torch.compile (fused)':<30} {compiled_time:<15.2f} {compiled_time:.1f}")
    
    print("\n KEY INSIGHT:")
    print(f" - 8 separate ops: ~{many_time/1000*1000:.0f}μs (kernel overhead dominates)")
    print(" - Kernel launch overhead: ~5-20μs PER kernel")
    print(" - For small tensors, launch overhead > compute time!")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Fuse operations when possible (Flash Attention fuses many ops)")
    print(" - Use torch.compile to auto-fuse")
    print(" - Batch multiple samples before GPU operations")

# ============================================================================
# EXPERIMENT 6: DATA TRANSFER OVERHEAD (CPU <-> GPU)
# ============================================================================

def experiment_data_transfer():
    """
    Moving data between CPU and GPU is EXPENSIVE!
    
    PCIe 4.0 x16: ~32 GB/s
    PCIe 5.0 x16: ~64 GB/s
    NVLink: ~600 GB/s
    
    Compare to:
    GPU memory: 2000-3000 GB/s
    
    This is why:
    - Keep data on GPU as long as possible
    - Overlap transfer with compute (streams)
    - Use pinned memory for faster transfers
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: CPU <-> GPU DATA TRANSFER")
    print(" PCIe is 50-100x slower than GPU memory!")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available, skipping...")
        return
    
    sizes_mb = [1, 10, 50, 100, 500]
    
    print(f"\n{'Size (MB)':<12} {'Pageable (ms)':<18} {'Pinned (ms)':<18} {'Speedup':<12} {'BW (GB/s)'}")
    print("-" * 75)
    
    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 4
        bytes_moved = num_elements * 4
        
        # Pageable memory transfer
        x_pageable = torch.randn(num_elements)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        x_gpu = x_pageable.to('cuda')
        end.record()
        torch.cuda.synchronize()
        pageable_time = start.elapsed_time(end)
        
        # Pinned memory transfer
        x_pinned = torch.randn(num_elements).pin_memory()
        torch.cuda.synchronize()
        
        start.record()
        x_gpu = x_pinned.to('cuda', non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        pinned_time = start.elapsed_time(end)
        
        speedup = pageable_time / pinned_time
        bandwidth = bytes_moved / (pinned_time / 1000) / 1e9
        
        print(f"{size_mb:<12} {pageable_time:<18.2f} {pinned_time:<18.2f} {speedup:<12.2f} {bandwidth:.1f}")
    
    print("\n KEY INSIGHT:")
    print(" - Pinned memory: 1.5-2x faster transfers (no staging copy)")
    print(" - Still only ~10-30 GB/s (100x slower than GPU memory)")
    print(" - Minimize CPU<->GPU transfers at all costs!")
    print("\n MULTIMODAL IMPLICATION:")
    print(" - Use DataLoader with pin_memory=True")
    print(" - Keep preprocessing on CPU, only send final tensors")
    print(" - Use non_blocking=True to overlap with compute")

# ============================================================================
# SUMMARY AND ARCHITECTURE COMPARISON
# ============================================================================

def print_architecture_summary():
    """Print summary of CPU vs GPU architecture differences."""
    print("\n" + "="*70)
    print(" CPU vs GPU ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("""
    ┌─────────────────┬────────────────────┬────────────────────┐
    │ Aspect          │ CPU                │ GPU                │
    ├─────────────────┼────────────────────┼────────────────────┤
    │ Cores           │ 8-64 complex       │ 1000s simple       │
    │ Clock Speed     │ 3-5 GHz            │ 1-2 GHz            │
    │ Cache           │ Large (32MB+)      │ Small per SM       │
    │ Memory BW       │ 50-100 GB/s        │ 2000-3000 GB/s     │
    │ Latency         │ Low (optimized)    │ High (hidden)      │
    │ Throughput      │ Low                │ Very High          │
    │ Branching       │ Excellent          │ Poor (divergence)  │
    │ Serial Code     │ Excellent          │ Poor               │
    │ Parallel Code   │ Good               │ Excellent          │
    └─────────────────┴────────────────────┴────────────────────┘
    
    MULTIMODAL TRAINING WORKLOAD MAPPING:
    
    ┌─────────────────────────────────┬─────────────────────────────┐
    │ CPU-Optimal                     │ GPU-Optimal                 │
    ├─────────────────────────────────┼─────────────────────────────┤
    │ • Text tokenization             │ • Matrix multiplications    │
    │ • Data loading/decoding         │ • Convolutions              │
    │ • Preprocessing with branches   │ • Attention computation     │
    │ • Small batch operations        │ • Batch normalization       │
    │ • File I/O                      │ • Large batch operations    │
    │ • Complex control flow          │ • Element-wise operations   │
    └─────────────────────────────────┴─────────────────────────────┘
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " GPU vs CPU ARCHITECTURE: PROFILED COMPARISON ".center(68) + "║")
    print("║" + " Understanding WHY GPUs are faster for ML ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" CUDA Version: {torch.version.cuda}")
        print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n WARNING: CUDA not available. GPU timings will be skipped.")
    
    experiment_latency_vs_throughput()
    experiment_memory_bandwidth()
    experiment_parallel_compute()
    experiment_branching()
    experiment_kernel_overhead()
    experiment_data_transfer()
    print_architecture_summary()
    
    print("\n" + "="*70)
    print(" NEXT STEPS:")
    print(" - Run with: python 01_gpu_vs_cpu.py")
    print(" - Study the timing differences at different sizes")
    print(" - Understand why crossover points exist")
    print(" - Apply to your multimodal training pipeline")
    print("="*70)
