"""
01_optimization_techniques.py - CUDA Optimization Techniques

This module covers practical optimization techniques for GPU code.
Every technique is demonstrated with profiled examples showing
the before/after performance impact.

Optimization Categories:
1. Memory Optimization - Coalescing, caching, prefetching
2. Compute Optimization - Occupancy, ILP, Tensor Cores
3. Kernel Optimization - Fusion, launch configuration
4. Transfer Optimization - Overlap, pinned memory

Golden Rule: Profile first, optimize second!

Run: python 01_optimization_techniques.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import time
import math
from typing import Tuple, List, Callable
from functools import partial

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

def profile_cuda(func, warmup=10, iterations=100, name=""):
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

def compare_methods(methods: List[Tuple[str, Callable]], iterations=100):
    """Compare multiple methods and print results."""
    print(f"\n{'Method':<40} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 70)
    
    base_time = None
    for name, func in methods:
        time_ms = profile_cuda(func, iterations=iterations)
        if base_time is None:
            base_time = time_ms
        speedup = base_time / time_ms if time_ms > 0 else 0
        print(f"{name:<40} {time_ms:<15.4f} {speedup:.2f}x")
    
    return base_time

# ============================================================================
# OPTIMIZATION 1: BATCH SIZE OPTIMIZATION
# ============================================================================

def optimization_batch_size():
    """
    Batch size directly affects GPU utilization.
    
    Too small: GPU underutilized, kernel launch overhead dominates
    Too large: OOM, diminishing returns
    Optimal: Saturates GPU compute while fitting in memory
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 1: BATCH SIZE")
    print(" Larger batches = better GPU utilization (up to a point)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    hidden = 1024
    seq_len = 512
    
    # Simple transformer-like layer
    layer = nn.Sequential(
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden),
    ).cuda()
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"\n Model: Linear({hidden}, {hidden*4}) -> GELU -> Linear({hidden*4}, {hidden})")
    print(f" Sequence length: {seq_len}")
    print(f"\n{'Batch Size':<15} {'Time (ms)':<15} {'Throughput (samples/s)':<25} {'Efficiency'}")
    print("-" * 70)
    
    base_throughput = None
    for batch_size in batch_sizes:
        try:
            x = torch.randn(batch_size, seq_len, hidden, device='cuda')
            
            def forward():
                return layer(x)
            
            time_ms = profile_cuda(forward)
            throughput = batch_size / (time_ms / 1000)
            
            if base_throughput is None:
                base_throughput = throughput
            efficiency = throughput / (base_throughput * batch_size)
            
            print(f"{batch_size:<15} {time_ms:<15.3f} {throughput:<25.0f} {efficiency*100:.1f}%")
            
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"{batch_size:<15} OOM")
            break
    
    print(f"\n KEY INSIGHTS:")
    print(f" - Small batches: High overhead, low utilization")
    print(f" - Larger batches: Better throughput up to saturation")
    print(f" - Beyond saturation: Only memory increases, not speed")
    print(f"\n PRACTICAL GUIDANCE:")
    print(f" - Start with largest batch that fits in memory")
    print(f" - Use gradient accumulation for effective larger batches")
    print(f" - Profile to find the 'knee' of the curve")

# ============================================================================
# OPTIMIZATION 2: MIXED PRECISION (AMP)
# ============================================================================

def optimization_mixed_precision():
    """
    Mixed precision training uses FP16/BF16 for most operations,
    FP32 for sensitive operations (loss scaling).
    
    Benefits:
    - 2x memory reduction for activations
    - Faster compute (especially with Tensor Cores)
    - Same model quality (with proper implementation)
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 2: MIXED PRECISION (AMP)")
    print(" FP16/BF16 for speed, FP32 for accuracy")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch_size, seq_len, hidden = 32, 512, 1024
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
                nn.LayerNorm(hidden),
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel().cuda()
    x = torch.randn(batch_size, seq_len, hidden, device='cuda')
    
    # FP32 baseline
    def fp32_forward():
        return model(x)
    
    time_fp32 = profile_cuda(fp32_forward)
    
    # Measure FP32 memory
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    mem_fp32 = torch.cuda.max_memory_allocated() / 1e6
    
    # FP16 with autocast
    def fp16_forward():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model(x)
    
    time_fp16 = profile_cuda(fp16_forward)
    
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        _ = model(x)
    mem_fp16 = torch.cuda.max_memory_allocated() / 1e6
    
    # BF16 (if supported)
    try:
        def bf16_forward():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return model(x)
        
        time_bf16 = profile_cuda(bf16_forward)
        bf16_supported = True
    except:
        time_bf16 = 0
        bf16_supported = False
    
    print(f"\n Input shape: ({batch_size}, {seq_len}, {hidden})")
    print(f"\n{'Precision':<20} {'Time (ms)':<15} {'Speedup':<15} {'Memory (MB)'}")
    print("-" * 65)
    print(f"{'FP32':<20} {time_fp32:<15.3f} {'1.0x':<15} {mem_fp32:.0f}")
    print(f"{'FP16 (autocast)':<20} {time_fp16:<15.3f} {time_fp32/time_fp16:.2f}x{'':<10} {mem_fp16:.0f}")
    if bf16_supported:
        print(f"{'BF16 (autocast)':<20} {time_bf16:<15.3f} {time_fp32/time_bf16:.2f}x")
    
    # Matrix multiply comparison (where Tensor Cores shine)
    print(f"\n Matrix multiply (4096x4096) precision comparison:")
    
    N = 4096
    A_fp32 = torch.randn(N, N, device='cuda')
    B_fp32 = torch.randn(N, N, device='cuda')
    A_fp16 = A_fp32.half()
    B_fp16 = B_fp32.half()
    
    time_mm_fp32 = profile_cuda(lambda: A_fp32 @ B_fp32)
    time_mm_fp16 = profile_cuda(lambda: A_fp16 @ B_fp16)
    
    flops = 2 * N * N * N
    tflops_fp32 = flops / (time_mm_fp32 / 1000) / 1e12
    tflops_fp16 = flops / (time_mm_fp16 / 1000) / 1e12
    
    print(f"{'Precision':<15} {'Time (ms)':<15} {'TFLOPS':<15} {'Speedup'}")
    print("-" * 55)
    print(f"{'FP32':<15} {time_mm_fp32:<15.3f} {tflops_fp32:<15.1f} 1.0x")
    print(f"{'FP16':<15} {time_mm_fp16:<15.3f} {tflops_fp16:<15.1f} {time_mm_fp32/time_mm_fp16:.2f}x")
    
    print(f"\n WHY FP16 IS FASTER:")
    print(f" 1. Tensor Cores operate on FP16 (2-8x faster)")
    print(f" 2. Half the memory bandwidth needed")
    print(f" 3. More elements fit in cache")
    print(f"\n AMP BEST PRACTICES:")
    print(f" - Use torch.autocast() for forward pass")
    print(f" - Use GradScaler for backward pass (prevents underflow)")
    print(f" - BF16 is more robust (larger exponent range)")

# ============================================================================
# OPTIMIZATION 3: TORCH.COMPILE
# ============================================================================

def optimization_torch_compile():
    """
    torch.compile uses Triton to generate optimized kernels.
    
    Benefits:
    - Automatic kernel fusion
    - Optimized memory access
    - No code changes needed (mostly)
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 3: TORCH.COMPILE")
    print(" Automatic optimization through compilation")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch_size, seq_len, hidden = 32, 512, 1024
    
    class MLP(nn.Module):
        def __init__(self):
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
    
    model = MLP().cuda()
    x = torch.randn(batch_size, seq_len, hidden, device='cuda')
    
    # Baseline (eager mode)
    def eager_forward():
        return model(x)
    
    time_eager = profile_cuda(eager_forward)
    
    # Compiled
    try:
        compiled_model = torch.compile(model)
        
        # Warmup compilation
        for _ in range(5):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        
        def compiled_forward():
            return compiled_model(x)
        
        time_compiled = profile_cuda(compiled_forward)
        compile_available = True
    except Exception as e:
        print(f" torch.compile not available: {e}")
        time_compiled = time_eager
        compile_available = False
    
    # Different compile modes
    if compile_available:
        print(f"\n Comparing torch.compile modes:")
        print(f"{'Mode':<25} {'Time (ms)':<15} {'Speedup'}")
        print("-" * 55)
        print(f"{'Eager (no compile)':<25} {time_eager:<15.3f} 1.0x")
        print(f"{'compile (default)':<25} {time_compiled:<15.3f} {time_eager/time_compiled:.2f}x")
        
        # reduce-overhead mode
        try:
            compiled_reduce = torch.compile(model, mode="reduce-overhead")
            for _ in range(10):
                _ = compiled_reduce(x)
            torch.cuda.synchronize()
            
            time_reduce = profile_cuda(lambda: compiled_reduce(x))
            print(f"{'compile (reduce-overhead)':<25} {time_reduce:<15.3f} {time_eager/time_reduce:.2f}x")
        except:
            pass
        
        # max-autotune mode
        try:
            compiled_autotune = torch.compile(model, mode="max-autotune")
            for _ in range(10):
                _ = compiled_autotune(x)
            torch.cuda.synchronize()
            
            time_autotune = profile_cuda(lambda: compiled_autotune(x))
            print(f"{'compile (max-autotune)':<25} {time_autotune:<15.3f} {time_eager/time_autotune:.2f}x")
        except:
            pass
    
    print(f"\n TORCH.COMPILE MODES:")
    print(f" - default: Good balance of compile time and speedup")
    print(f" - reduce-overhead: Minimizes Python overhead (good for small models)")
    print(f" - max-autotune: Maximum optimization (longer compile time)")
    print(f"\n WHEN TO USE:")
    print(f" - Training: Usually worth the compile overhead")
    print(f" - Inference: Definitely use it")
    print(f" - Dynamic shapes: May cause recompilation (use dynamic=True)")

# ============================================================================
# OPTIMIZATION 4: MEMORY-EFFICIENT ATTENTION
# ============================================================================

def optimization_efficient_attention():
    """
    Standard attention: O(n²) memory for attention matrix
    Flash Attention: O(n) memory through tiling
    
    For long sequences, this is the difference between
    OOM and running successfully!
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 4: MEMORY-EFFICIENT ATTENTION")
    print(" Flash Attention: O(n) memory instead of O(n²)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    batch_size = 4
    num_heads = 8
    head_dim = 64
    
    print(f"\n Comparing attention implementations:")
    print(f" Batch: {batch_size}, Heads: {num_heads}, Head dim: {head_dim}")
    print(f"\n{'Seq Len':<12} {'Standard (ms)':<18} {'SDPA (ms)':<18} {'Speedup':<12} {'Mem Ratio'}")
    print("-" * 75)
    
    for seq_len in [256, 512, 1024, 2048, 4096]:
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Standard attention
        def standard_attention():
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, V)
        
        # Measure standard attention memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            _ = standard_attention()
            mem_standard = torch.cuda.max_memory_allocated() / 1e6
            time_standard = profile_cuda(standard_attention, iterations=50)
        except RuntimeError:
            mem_standard = float('inf')
            time_standard = float('inf')
        
        # SDPA (uses Flash Attention when available)
        def sdpa_attention():
            return F.scaled_dot_product_attention(Q, K, V)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        _ = sdpa_attention()
        mem_sdpa = torch.cuda.max_memory_allocated() / 1e6
        time_sdpa = profile_cuda(sdpa_attention, iterations=50)
        
        if time_standard != float('inf'):
            speedup = time_standard / time_sdpa
            mem_ratio = mem_standard / mem_sdpa
            print(f"{seq_len:<12} {time_standard:<18.3f} {time_sdpa:<18.3f} {speedup:<12.2f} {mem_ratio:.2f}x")
        else:
            print(f"{seq_len:<12} {'OOM':<18} {time_sdpa:<18.3f} {'-':<12} {'∞'}")
        
        del Q, K, V
        torch.cuda.empty_cache()
    
    print(f"\n WHY FLASH ATTENTION IS FASTER:")
    print(f" ┌────────────────────────────────────────────────────────────────┐")
    print(f" │ Standard Attention:                                           │")
    print(f" │ 1. Compute S = QK^T → Write to HBM (n² elements!)            │")
    print(f" │ 2. Compute P = softmax(S) → Write to HBM                     │")
    print(f" │ 3. Compute O = PV → Write to HBM                             │")
    print(f" │ Total: 3 round trips to HBM for n² data                      │")
    print(f" │                                                               │")
    print(f" │ Flash Attention:                                              │")
    print(f" │ 1. Load Q, K, V tiles to SRAM (on-chip)                      │")
    print(f" │ 2. Compute attention in SRAM (fused)                         │")
    print(f" │ 3. Write final O to HBM                                      │")
    print(f" │ Total: 1 round trip, n² stays in fast SRAM                   │")
    print(f" └────────────────────────────────────────────────────────────────┘")

# ============================================================================
# OPTIMIZATION 5: OPERATION FUSION
# ============================================================================

def optimization_fusion():
    """
    Fusing multiple operations into one kernel reduces:
    - Kernel launch overhead
    - Memory traffic (intermediates stay in registers)
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 5: OPERATION FUSION")
    print(" Fewer kernels = Less overhead + Less memory traffic")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 16_000_000  # 16M elements
    x = torch.randn(N, device='cuda')
    
    # Unfused operations
    def unfused_ops():
        a = x + 1
        b = a * 2
        c = b - 0.5
        d = torch.relu(c)
        return d
    
    # PyTorch's fused version (if available)
    def fused_relu():
        return torch.relu(x * 2 + 1.5)  # Mathematically equivalent
    
    time_unfused = profile_cuda(unfused_ops)
    time_fused = profile_cuda(fused_relu)
    
    print(f"\n Element-wise operations ({N/1e6:.0f}M elements):")
    print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 55)
    print(f"{'Unfused (4 kernels)':<30} {time_unfused:<15.4f} 1.0x")
    print(f"{'Partially fused':<30} {time_fused:<15.4f} {time_unfused/time_fused:.2f}x")
    
    # torch.compile fusion
    try:
        compiled_unfused = torch.compile(unfused_ops)
        # Warmup
        _ = compiled_unfused()
        torch.cuda.synchronize()
        
        time_compiled = profile_cuda(compiled_unfused)
        print(f"{'torch.compile (auto-fused)':<30} {time_compiled:<15.4f} {time_unfused/time_compiled:.2f}x")
    except:
        pass
    
    # More complex fusion example: GELU approximation
    print(f"\n GELU fusion comparison:")
    
    def unfused_gelu(x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608 * (x + 0.044715 * torch.pow(x, 3))
        ))
    
    x_gelu = torch.randn(N, device='cuda')
    
    time_unfused_gelu = profile_cuda(lambda: unfused_gelu(x_gelu))
    time_fused_gelu = profile_cuda(lambda: F.gelu(x_gelu))
    
    print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 55)
    print(f"{'Unfused GELU (manual)':<30} {time_unfused_gelu:<15.4f} 1.0x")
    print(f"{'Fused GELU (F.gelu)':<30} {time_fused_gelu:<15.4f} {time_unfused_gelu/time_fused_gelu:.2f}x")
    
    print(f"\n MEMORY TRAFFIC ANALYSIS:")
    print(f" Unfused (4 ops): Read x, write a, read a, write b, read b, write c, read c, write d")
    print(f"                  = 8 memory accesses × N elements")
    print(f" Fused (1 op):    Read x, write d")
    print(f"                  = 2 memory accesses × N elements")
    print(f" Theoretical speedup: 4x (memory-bound)")

# ============================================================================
# OPTIMIZATION 6: DATA LOADING
# ============================================================================

def optimization_data_loading():
    """
    Data loading is often the hidden bottleneck!
    
    Common issues:
    - Not enough workers
    - No pinned memory
    - No prefetching
    - CPU bottleneck in preprocessing
    """
    print("\n" + "="*70)
    print(" OPTIMIZATION 6: DATA LOADING")
    print(" Often the hidden bottleneck in training")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Simulate data loading scenarios
    batch_size = 64
    image_size = 224
    channels = 3
    
    # Simulate a batch of images
    def create_batch():
        return torch.randn(batch_size, channels, image_size, image_size)
    
    print(f"\n Simulating data transfer patterns:")
    print(f" Batch shape: ({batch_size}, {channels}, {image_size}, {image_size})")
    print(f" Batch size: {batch_size * channels * image_size * image_size * 4 / 1e6:.1f} MB")
    
    # Method 1: Pageable memory, blocking transfer
    batch_pageable = create_batch()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        batch_gpu = batch_pageable.to('cuda')
        del batch_gpu
    torch.cuda.synchronize()
    time_pageable = (time.perf_counter() - start) * 1000 / 10
    
    # Method 2: Pinned memory, blocking transfer
    batch_pinned = create_batch().pin_memory()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        batch_gpu = batch_pinned.to('cuda')
        del batch_gpu
    torch.cuda.synchronize()
    time_pinned = (time.perf_counter() - start) * 1000 / 10
    
    # Method 3: Pinned memory, non-blocking transfer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        batch_gpu = batch_pinned.to('cuda', non_blocking=True)
        torch.cuda.synchronize()  # Wait for this transfer
        del batch_gpu
    time_nonblocking = (time.perf_counter() - start) * 1000 / 10
    
    print(f"\n{'Method':<40} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 65)
    print(f"{'Pageable memory, blocking':<40} {time_pageable:<15.2f} 1.0x")
    print(f"{'Pinned memory, blocking':<40} {time_pinned:<15.2f} {time_pageable/time_pinned:.2f}x")
    print(f"{'Pinned memory, non-blocking':<40} {time_nonblocking:<15.2f} {time_pageable/time_nonblocking:.2f}x")
    
    print(f"\n DATALOADER BEST PRACTICES:")
    print(f" ┌────────────────────────────────────────────────────────────────┐")
    print(f" │ DataLoader(                                                   │")
    print(f" │     dataset,                                                  │")
    print(f" │     batch_size=64,                                            │")
    print(f" │     num_workers=4,      # Parallel data loading               │")
    print(f" │     pin_memory=True,    # Use pinned memory                   │")
    print(f" │     prefetch_factor=2,  # Prefetch 2 batches per worker       │")
    print(f" │     persistent_workers=True,  # Keep workers alive            │")
    print(f" │ )                                                             │")
    print(f" └────────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL DATA LOADING:")
    print(f" - Images: Use fast decoders (torchvision, DALI, ffcv)")
    print(f" - Text: Tokenize in advance or use fast tokenizers")
    print(f" - Audio: Use torchaudio with fast backends")
    print(f" - Video: Decode on GPU when possible")

# ============================================================================
# SUMMARY
# ============================================================================

def print_optimization_summary():
    """Print comprehensive optimization summary."""
    print("\n" + "="*70)
    print(" OPTIMIZATION SUMMARY")
    print("="*70)
    
    print("""
    OPTIMIZATION CHECKLIST (in order of impact):
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ 1. BATCH SIZE                                                      │
    │    □ Large enough to saturate GPU                                  │
    │    □ Not so large that overhead dominates                         │
    │    □ Use gradient accumulation for effective larger batches       │
    │                                                                    │
    │ 2. MIXED PRECISION (AMP)                                          │
    │    □ torch.autocast for forward pass                              │
    │    □ GradScaler for backward pass                                 │
    │    □ BF16 preferred if available (more robust)                    │
    │                                                                    │
    │ 3. TORCH.COMPILE                                                   │
    │    □ Compile model before training loop                           │
    │    □ Use mode='reduce-overhead' for small models                  │
    │    □ Use mode='max-autotune' for inference                        │
    │                                                                    │
    │ 4. EFFICIENT ATTENTION                                             │
    │    □ Use F.scaled_dot_product_attention (Flash Attention)         │
    │    □ Consider xformers for older PyTorch versions                 │
    │    □ Memory-efficient variants for long sequences                 │
    │                                                                    │
    │ 5. DATA LOADING                                                    │
    │    □ num_workers > 0 (typically 4-8)                              │
    │    □ pin_memory=True                                              │
    │    □ prefetch_factor=2 or more                                    │
    │    □ Profile to ensure GPU isn't waiting for data                 │
    │                                                                    │
    │ 6. MEMORY OPTIMIZATION                                             │
    │    □ Gradient checkpointing for large models                      │
    │    □ Optimizer state offloading (DeepSpeed ZeRO)                  │
    │    □ Activation compression                                       │
    └────────────────────────────────────────────────────────────────────┘
    
    PROFILING COMMANDS:
    
    # PyTorch Profiler
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    ) as prof:
        for step, data in enumerate(dataloader):
            train_step(data)
            prof.step()
    
    # Nsight Systems (command line)
    nsys profile -o output python train.py
    
    # Memory profiling
    torch.cuda.memory_summary()
    torch.cuda.max_memory_allocated()
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " CUDA OPTIMIZATION TECHNIQUES ".center(68) + "║")
    print("║" + " Practical techniques with profiled demonstrations ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    optimization_batch_size()
    optimization_mixed_precision()
    optimization_torch_compile()
    optimization_efficient_attention()
    optimization_fusion()
    optimization_data_loading()
    print_optimization_summary()
    
    print("\n" + "="*70)
    print(" Remember: Profile first, optimize second!")
    print("="*70)
