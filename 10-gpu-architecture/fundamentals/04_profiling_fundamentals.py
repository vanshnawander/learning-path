"""
04_profiling_fundamentals.py - GPU Profiling: The Foundation of Optimization

PROFILING IS NOT OPTIONAL - It's the foundation of all GPU optimization.
Without profiling, you're optimizing blind.

This module covers:
1. PyTorch Profiler - High-level Python profiling
2. CUDA Events - Precise GPU timing
3. Nsight Systems - System-wide analysis
4. Nsight Compute - Kernel-level analysis
5. Memory profiling
6. Understanding profiler output

Key principle: MEASURE FIRST, OPTIMIZE SECOND
Never assume where the bottleneck is - always profile!

Run: python 04_profiling_fundamentals.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import os

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

@dataclass
class ProfileResult:
    """Store profiling results for analysis."""
    name: str
    cpu_time_ms: float
    cuda_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    flops: int = 0
    bytes_moved: int = 0

class GPUProfiler:
    """Comprehensive GPU profiler with multiple backends."""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.cuda_available = torch.cuda.is_available()
    
    @contextmanager
    def cuda_timer(self, name: str = "operation"):
        """
        Most accurate GPU timing using CUDA events.
        
        IMPORTANT: torch.cuda.synchronize() before and after!
        GPU operations are asynchronous - timing without sync is wrong.
        """
        if not self.cuda_available:
            start = time.perf_counter()
            yield
            elapsed = (time.perf_counter() - start) * 1000
            print(f" {name}: {elapsed:.3f} ms (CPU only)")
            return
        
        # Record memory before
        mem_before = torch.cuda.memory_allocated()
        
        # CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Sync before starting
        torch.cuda.synchronize()
        
        start_event.record()
        yield
        end_event.record()
        
        # Sync after ending
        torch.cuda.synchronize()
        
        # Calculate elapsed time
        elapsed_ms = start_event.elapsed_time(end_event)
        
        # Record memory after
        mem_after = torch.cuda.memory_allocated()
        mem_delta = (mem_after - mem_before) / 1e6
        
        result = ProfileResult(
            name=name,
            cpu_time_ms=0,  # Not measured here
            cuda_time_ms=elapsed_ms,
            memory_allocated_mb=mem_delta,
            memory_reserved_mb=torch.cuda.memory_reserved() / 1e6
        )
        self.results.append(result)
    
    def print_results(self, title: str = "Profiling Results"):
        """Print formatted profiling results."""
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
        print(f"{'Operation':<30} {'CUDA Time (ms)':<18} {'Memory (MB)':<15}")
        print(f"{'-'*70}")
        
        for r in self.results:
            print(f"{r.name:<30} {r.cuda_time_ms:<18.3f} {r.memory_allocated_mb:<15.2f}")
        
        self.results = []
    
    def warmup(self, func: Callable, iterations: int = 10):
        """Warmup function to ensure JIT compilation and cache warming."""
        for _ in range(iterations):
            func()
        if self.cuda_available:
            torch.cuda.synchronize()

profiler = GPUProfiler()

# ============================================================================
# EXPERIMENT 1: BASIC CUDA TIMING
# ============================================================================

def experiment_basic_timing():
    """
    Demonstrate correct vs incorrect GPU timing.
    
    CRITICAL: GPU operations are ASYNCHRONOUS!
    Python continues executing while GPU works.
    You MUST synchronize to get accurate timings.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: CORRECT VS INCORRECT GPU TIMING")
    print(" GPU ops are async - sync is required for accurate timing!")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = A @ B
    torch.cuda.synchronize()
    
    print(f"\n Matrix multiply: {N}x{N} @ {N}x{N}")
    print(f"\n{'Method':<35} {'Time (ms)':<15} {'Correct?'}")
    print("-" * 60)
    
    # WRONG: No synchronization
    start = time.perf_counter()
    C = A @ B
    wrong_time = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()  # Sync now to actually complete
    print(f"{'time.perf_counter() (NO sync)':<35} {wrong_time:<15.3f} NO!")
    
    # WRONG: Only sync after
    start = time.perf_counter()
    C = A @ B
    torch.cuda.synchronize()
    partial_time = (time.perf_counter() - start) * 1000
    print(f"{'Sync after only':<35} {partial_time:<15.3f} PARTIAL")
    
    # CORRECT: Sync before AND after
    torch.cuda.synchronize()
    start = time.perf_counter()
    C = A @ B
    torch.cuda.synchronize()
    correct_time = (time.perf_counter() - start) * 1000
    print(f"{'Sync before AND after':<35} {correct_time:<15.3f} YES")
    
    # BEST: CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    C = A @ B
    end_event.record()
    torch.cuda.synchronize()
    event_time = start_event.elapsed_time(end_event)
    print(f"{'CUDA Events (most accurate)':<35} {event_time:<15.3f} BEST")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Wrong timing can be 10-100x off!")
    print(f" - CUDA events have microsecond precision")
    print(f" - Always use events for benchmarking")
    print(f"\n COMMON MISTAKE:")
    print(f" - Timing a loop without sync: Only measures Python overhead")
    print(f" - Must sync inside loop OR use events properly")

# ============================================================================
# EXPERIMENT 2: PYTORCH PROFILER
# ============================================================================

def experiment_pytorch_profiler():
    """
    PyTorch's built-in profiler provides detailed operation breakdown.
    
    It traces:
    - CPU operations
    - CUDA kernels
    - Memory allocations
    - Call stacks
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: PYTORCH PROFILER")
    print(" Detailed breakdown of all operations")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, hidden=1024):
            super().__init__()
            self.fc1 = nn.Linear(hidden, hidden * 4)
            self.fc2 = nn.Linear(hidden * 4, hidden)
            self.ln = nn.LayerNorm(hidden)
        
        def forward(self, x):
            x = self.ln(x)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel().cuda()
    x = torch.randn(32, 128, 1024, device='cuda')  # (batch, seq, hidden)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    
    print(f"\n Profiling SimpleModel forward pass...")
    print(f" Input shape: {x.shape}")
    
    # Profile with activities
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_forward"):
            output = model(x)
    
    # Print summary table
    print(f"\n Top operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
    
    # Memory summary
    print(f"\n Top operations by CUDA memory:")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=5
    ))
    
    print(f"\n KEY INSIGHT:")
    print(f" - Profiler shows ACTUAL kernel names (e.g., 'aten::mm')")
    print(f" - Memory usage per operation visible")
    print(f" - Can identify which layers are slowest")
    print(f"\n USAGE TIP:")
    print(f" - Use record_function('name') to label your code sections")
    print(f" - Export to Chrome trace: prof.export_chrome_trace('trace.json')")
    print(f" - View in chrome://tracing")

# ============================================================================
# EXPERIMENT 3: MEMORY PROFILING
# ============================================================================

def experiment_memory_profiling():
    """
    Memory profiling is crucial for understanding:
    - Peak memory usage
    - Memory fragmentation
    - Where allocations happen
    - Memory leaks
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: MEMORY PROFILING")
    print(" Track allocations and find memory bottlenecks")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Clear memory state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    def get_memory_stats():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1e6,
            'reserved_mb': torch.cuda.memory_reserved() / 1e6,
            'peak_mb': torch.cuda.max_memory_allocated() / 1e6
        }
    
    print(f"\n Memory tracking through operations:")
    print(f"{'Operation':<40} {'Allocated (MB)':<18} {'Peak (MB)':<15}")
    print("-" * 75)
    
    # Initial state
    stats = get_memory_stats()
    print(f"{'Initial':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # Allocate input tensor
    x = torch.randn(32, 1024, 1024, device='cuda')  # ~128 MB
    stats = get_memory_stats()
    print(f"{'After input allocation (32,1024,1024)':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # Linear layer
    linear = nn.Linear(1024, 4096).cuda()  # ~16 MB weights
    stats = get_memory_stats()
    print(f"{'After Linear(1024,4096)':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # Forward pass (activations)
    y = linear(x)  # Output: (32, 1024, 4096) ~512 MB
    stats = get_memory_stats()
    print(f"{'After forward pass':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # Delete intermediate
    del y
    torch.cuda.empty_cache()
    stats = get_memory_stats()
    print(f"{'After del y + empty_cache()':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # In-place operation
    x.mul_(2.0)  # In-place, no new allocation
    stats = get_memory_stats()
    print(f"{'After in-place x.mul_(2.0)':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    # Non in-place
    z = x * 2.0  # New tensor allocated
    stats = get_memory_stats()
    print(f"{'After x * 2.0 (new tensor)':<40} {stats['allocated_mb']:<18.2f} {stats['peak_mb']:<15.2f}")
    
    print(f"\n MEMORY OPTIMIZATION TIPS:")
    print(f" 1. Use in-place operations when safe (add_, mul_, etc.)")
    print(f" 2. Delete large intermediates explicitly")
    print(f" 3. Use torch.no_grad() for inference")
    print(f" 4. Gradient checkpointing trades compute for memory")
    print(f" 5. Mixed precision halves activation memory")
    
    # Cleanup
    del x, z, linear
    torch.cuda.empty_cache()

# ============================================================================
# EXPERIMENT 4: PROFILING A TRAINING STEP
# ============================================================================

def experiment_training_profiling():
    """
    Profile a complete training step to understand:
    - Forward pass time
    - Backward pass time
    - Optimizer step time
    - Data loading (if applicable)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: PROFILING A TRAINING STEP")
    print(" Complete breakdown of forward, backward, optimizer")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Simple transformer-like block
    class TransformerBlock(nn.Module):
        def __init__(self, hidden=512, heads=8):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden)
            )
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)
        
        def forward(self, x):
            # Self-attention
            normed = self.ln1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            
            # Feed-forward
            normed = self.ln2(x)
            ff_out = self.ff(normed)
            x = x + ff_out
            return x
    
    model = TransformerBlock().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size, seq_len, hidden = 16, 256, 512
    x = torch.randn(batch_size, seq_len, hidden, device='cuda')
    target = torch.randn(batch_size, seq_len, hidden, device='cuda')
    
    # Warmup
    for _ in range(5):
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    print(f"\n Model: TransformerBlock")
    print(f" Input: ({batch_size}, {seq_len}, {hidden})")
    print(f"\n{'Phase':<25} {'Time (ms)':<15} {'% of Total'}")
    print("-" * 55)
    
    # Profile each phase
    times = {}
    
    # Forward
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    output = model(x)
    loss = F.mse_loss(output, target)
    end.record()
    torch.cuda.synchronize()
    times['forward'] = start.elapsed_time(end)
    
    # Backward
    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    times['backward'] = start.elapsed_time(end)
    
    # Optimizer
    start.record()
    optimizer.step()
    optimizer.zero_grad()
    end.record()
    torch.cuda.synchronize()
    times['optimizer'] = start.elapsed_time(end)
    
    total = sum(times.values())
    for phase, t in times.items():
        print(f"{phase.capitalize():<25} {t:<15.3f} {t/total*100:.1f}%")
    print(f"{'Total':<25} {total:<15.3f} 100%")
    
    print(f"\n TYPICAL BREAKDOWN:")
    print(f" - Forward: ~25-35% of step time")
    print(f" - Backward: ~50-65% (gradients + more memory)")
    print(f" - Optimizer: ~5-15% (Adam has state)")
    print(f"\n DATA LOADING (not shown):")
    print(f" - Can be 10-50% if not optimized!")
    print(f" - Use DataLoader with num_workers > 0")
    print(f" - Use pin_memory=True")

# ============================================================================
# EXPERIMENT 5: IDENTIFYING BOTTLENECKS
# ============================================================================

def experiment_bottleneck_analysis():
    """
    Learn to identify whether code is:
    - Compute-bound (GPU compute units busy)
    - Memory-bound (waiting for memory)
    - CPU-bound (Python/data loading)
    - Transfer-bound (PCIe bottleneck)
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: BOTTLENECK IDENTIFICATION")
    print(" Is it compute, memory, CPU, or transfer bound?")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 4096
    
    # Test cases
    tests = []
    
    # 1. Compute-bound: Large matrix multiply
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    def matmul_test():
        return A @ B
    
    # Time it
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(5):
        matmul_test()
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(100):
        matmul_test()
    end.record()
    torch.cuda.synchronize()
    
    mm_time = start.elapsed_time(end) / 100
    mm_flops = 2 * N * N * N
    mm_tflops = mm_flops / (mm_time / 1000) / 1e12
    tests.append(("Matrix Multiply", mm_time, "COMPUTE", mm_tflops))
    
    # 2. Memory-bound: Element-wise
    x = torch.randn(N * N, device='cuda')
    
    def elementwise_test():
        return x * 2.0 + 1.0
    
    for _ in range(5):
        elementwise_test()
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(100):
        elementwise_test()
    end.record()
    torch.cuda.synchronize()
    
    elem_time = start.elapsed_time(end) / 100
    elem_bw = N * N * 4 * 3 / (elem_time / 1000) / 1e9  # read x, write result
    tests.append(("Element-wise", elem_time, f"MEMORY ({elem_bw:.0f} GB/s)", 0))
    
    # 3. Transfer-bound: CPU->GPU copy
    x_cpu = torch.randn(N * N).pin_memory()
    
    def transfer_test():
        return x_cpu.to('cuda', non_blocking=False)
    
    for _ in range(3):
        transfer_test()
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(20):
        transfer_test()
    end.record()
    torch.cuda.synchronize()
    
    xfer_time = start.elapsed_time(end) / 20
    xfer_bw = N * N * 4 / (xfer_time / 1000) / 1e9
    tests.append(("CPU→GPU Transfer", xfer_time, f"TRANSFER ({xfer_bw:.0f} GB/s)", 0))
    
    # 4. CPU-bound: Python loop
    def cpu_bound_test():
        total = 0
        for i in range(10000):
            total += i
        return total
    
    start_cpu = time.perf_counter()
    for _ in range(100):
        cpu_bound_test()
    cpu_time = (time.perf_counter() - start_cpu) * 1000 / 100
    tests.append(("Python Loop", cpu_time, "CPU", 0))
    
    print(f"\n{'Operation':<25} {'Time (ms)':<15} {'Bottleneck':<25} {'Metric'}")
    print("-" * 80)
    
    for name, t, bound, metric in tests:
        metric_str = f"{metric:.1f} TFLOPS" if metric > 0 else ""
        print(f"{name:<25} {t:<15.3f} {bound:<25} {metric_str}")
    
    print(f"\n HOW TO IDENTIFY BOTTLENECKS:")
    print(f" ┌────────────────────────────────────────────────────────────────┐")
    print(f" │ Bottleneck    │ Symptom                │ Solution             │")
    print(f" ├───────────────┼────────────────────────┼──────────────────────┤")
    print(f" │ COMPUTE       │ High GPU util, low BW  │ Use Tensor Cores     │")
    print(f" │ MEMORY        │ Low GPU util, high BW  │ Fuse ops, use cache  │")
    print(f" │ CPU           │ GPU idle, CPU busy     │ Move to GPU/optimize │")
    print(f" │ TRANSFER      │ Low PCIe bandwidth     │ Overlap/batch/reduce │")
    print(f" └───────────────┴────────────────────────┴──────────────────────┘")

# ============================================================================
# EXPERIMENT 6: NSIGHT INTEGRATION
# ============================================================================

def experiment_nsight_info():
    """
    Information about NVIDIA Nsight tools for deep profiling.
    
    Nsight Systems: System-wide timeline view
    Nsight Compute: Kernel-level analysis
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: NVIDIA NSIGHT TOOLS")
    print(" Deep profiling for serious optimization")
    print("="*70)
    
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                    NSIGHT SYSTEMS                                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ Command: nsys profile python your_script.py                        │
    │                                                                    │
    │ Shows:                                                             │
    │ • Timeline of all CPU and GPU operations                          │
    │ • Kernel launches and duration                                    │
    │ • Memory transfers                                                │
    │ • CPU/GPU overlap                                                 │
    │ • CUDA API calls                                                  │
    │                                                                    │
    │ Key Metrics:                                                       │
    │ • GPU utilization over time                                       │
    │ • Kernel launch gaps (CPU bottlenecks)                            │
    │ • Memory transfer patterns                                         │
    │                                                                    │
    │ Usage:                                                             │
    │ $ nsys profile -o output python train.py                          │
    │ $ nsys-ui output.nsys-rep                                         │
    └────────────────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────────────────┐
    │                    NSIGHT COMPUTE                                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ Command: ncu python your_script.py                                 │
    │                                                                    │
    │ Shows (per kernel):                                                │
    │ • Achieved occupancy                                              │
    │ • Memory throughput                                               │
    │ • Compute throughput                                              │
    │ • Warp stall reasons                                              │
    │ • Register usage                                                  │
    │ • Shared memory usage                                             │
    │ • Roofline analysis                                               │
    │                                                                    │
    │ Key Questions Answered:                                            │
    │ • Is this kernel compute or memory bound?                         │
    │ • What's causing warp stalls?                                     │
    │ • Is memory access coalesced?                                     │
    │ • How close to theoretical peak?                                   │
    │                                                                    │
    │ Usage:                                                             │
    │ $ ncu --set full -o output python train.py                        │
    │ $ ncu-ui output.ncu-rep                                           │
    └────────────────────────────────────────────────────────────────────┘
    
    PROFILING WORKFLOW:
    
    1. Start with PyTorch Profiler for high-level view
       → Identify slow operations
    
    2. Use Nsight Systems for system-wide analysis
       → Find CPU/GPU gaps, transfer bottlenecks
    
    3. Use Nsight Compute for specific kernels
       → Deep dive into why a kernel is slow
    
    EXAMPLE NSIGHT COMMANDS:
    """)
    
    print(f" # Profile with Nsight Systems")
    print(f" nsys profile -w true -t cuda,nvtx -o profile_output \\")
    print(f"     --force-overwrite true python train.py")
    print()
    print(f" # Profile specific kernel with Nsight Compute")
    print(f" ncu --kernel-name \"volta_sgemm\" --launch-skip 5 --launch-count 1 \\")
    print(f"     python train.py")
    print()
    print(f" # Export PyTorch trace for Chrome")
    print(f" # In Python:")
    print(f" with torch.profiler.profile() as prof:")
    print(f"     model(x)")
    print(f" prof.export_chrome_trace('trace.json')")
    print(f" # Open chrome://tracing and load trace.json")

# ============================================================================
# PROFILING BEST PRACTICES SUMMARY
# ============================================================================

def print_profiling_summary():
    """Print comprehensive profiling best practices."""
    print("\n" + "="*70)
    print(" PROFILING BEST PRACTICES")
    print("="*70)
    
    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │                PROFILING IS A SKILL - PRACTICE IT!                 │
    └────────────────────────────────────────────────────────────────────┘
    
    RULE #1: ALWAYS PROFILE BEFORE OPTIMIZING
    ─────────────────────────────────────────
    • Assumptions about bottlenecks are often wrong
    • 5 minutes of profiling saves hours of wrong optimization
    • Profile representative workloads, not toy examples
    
    RULE #2: WARMUP IS ESSENTIAL
    ────────────────────────────
    • First runs include JIT compilation overhead
    • CUDA context initialization takes time
    • Cache warming affects results
    • Always skip first N iterations
    
    RULE #3: SYNCHRONIZE FOR ACCURATE TIMING
    ────────────────────────────────────────
    • GPU operations are asynchronous
    • torch.cuda.synchronize() before AND after
    • Use CUDA events for best precision
    
    RULE #4: PROFILE AT MULTIPLE LEVELS
    ───────────────────────────────────
    • High-level: Which operation is slow?
    • Mid-level: Why is the GPU underutilized?
    • Low-level: What's the kernel bottleneck?
    
    PROFILING CHECKLIST:
    
    □ Warmup iterations completed?
    □ Synchronization in place?
    □ Representative batch size?
    □ Multiple runs for statistics?
    □ Memory profiling if OOM issues?
    □ CPU vs GPU time breakdown?
    □ Data loading time measured?
    
    QUICK PROFILING TEMPLATE:
    
    ```python
    import torch
    from torch.profiler import profile, ProfilerActivity
    
    # Warmup
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()
    
    # Profile
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        output = model(x)
    end.record()
    torch.cuda.synchronize()
    
    print(f"Average time: {start.elapsed_time(end) / 100:.2f} ms")
    ```
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " GPU PROFILING FUNDAMENTALS ".center(68) + "║")
    print("║" + " Measure First, Optimize Second ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" CUDA: {torch.version.cuda}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_basic_timing()
    experiment_pytorch_profiler()
    experiment_memory_profiling()
    experiment_training_profiling()
    experiment_bottleneck_analysis()
    experiment_nsight_info()
    print_profiling_summary()
    
    print("\n" + "="*70)
    print(" KEY MESSAGE: Make profiling a HABIT, not an afterthought!")
    print(" Every optimization should start with: 'What does the profiler say?'")
    print("="*70)
