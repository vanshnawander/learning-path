"""
01_architecture_evolution.py - NVIDIA GPU Architecture Evolution

Understanding GPU architecture evolution helps you:
- Choose the right GPU for your workload
- Understand why certain optimizations work
- Anticipate future trends

Architecture Timeline:
- Tesla (2006): First CUDA GPUs
- Fermi (2010): First modern architecture
- Kepler (2012): Dynamic parallelism
- Maxwell (2014): Power efficiency
- Pascal (2016): Unified memory, NVLink
- Volta (2017): Tensor Cores!
- Turing (2018): RT Cores, improved Tensor Cores
- Ampere (2020): 3rd gen Tensor Cores, sparsity
- Hopper (2022): Transformer Engine, TMA
- Blackwell (2024): 5th gen Tensor Cores

Run: python 01_architecture_evolution.py
"""

import torch
import time

# ============================================================================
# PROFILING
# ============================================================================

def profile_operation(func, warmup=10, iterations=100):
    """Profile an operation."""
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
# ARCHITECTURE SPECIFICATIONS
# ============================================================================

def print_architecture_specs():
    """Print specifications of major NVIDIA architectures."""
    print("\n" + "="*70)
    print(" NVIDIA GPU ARCHITECTURE EVOLUTION")
    print("="*70)
    
    print("""
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                    NVIDIA ARCHITECTURE TIMELINE                             │
    ├──────────┬─────────┬──────────┬────────────┬───────────┬──────────────────┤
    │ Arch     │ Year    │ SM Count │ Tensor C.  │ Peak FP16 │ Memory BW        │
    │          │         │ (Flagship)│ per SM    │ (TFLOPS)  │ (GB/s)           │
    ├──────────┼─────────┼──────────┼────────────┼───────────┼──────────────────┤
    │ Pascal   │ 2016    │ 60 (P100)│ 0          │ 21.2      │ 732 (HBM2)       │
    │ Volta    │ 2017    │ 80 (V100)│ 8          │ 125       │ 900 (HBM2)       │
    │ Turing   │ 2018    │ 72 (2080)│ 8          │ 161       │ 616 (GDDR6)      │
    │ Ampere   │ 2020    │ 108(A100)│ 4          │ 312       │ 2039 (HBM2e)     │
    │ Hopper   │ 2022    │ 132(H100)│ 4          │ 989       │ 3350 (HBM3)      │
    │ Blackwell│ 2024    │ 192(B200)│ 4          │ 2250      │ 8000 (HBM3e)     │
    └──────────┴─────────┴──────────┴────────────┴───────────┴──────────────────┘
    
    KEY INNOVATIONS BY ARCHITECTURE:
    
    VOLTA (2017) - The Deep Learning Revolution
    ├── First Tensor Cores (8 per SM)
    ├── 4x4x4 FP16 matrix operations
    ├── Mixed precision training viable
    └── Independent thread scheduling
    
    AMPERE (2020) - Sparsity and Efficiency
    ├── 3rd gen Tensor Cores
    ├── TF32 precision (FP32 range, FP16 speed)
    ├── 2:4 structured sparsity (2x speedup)
    ├── BF16 support
    └── Multi-Instance GPU (MIG)
    
    HOPPER (2022) - Transformer Optimization
    ├── 4th gen Tensor Cores
    ├── Transformer Engine (FP8!)
    ├── TMA (Tensor Memory Accelerator)
    ├── DPX instructions (dynamic programming)
    └── NVLink 4.0 (900 GB/s)
    
    BLACKWELL (2024) - Next Generation
    ├── 5th gen Tensor Cores
    ├── 2nd gen Transformer Engine
    ├── FP4 support
    ├── 8 TB/s memory bandwidth
    └── NVLink 5.0 (1.8 TB/s)
    """)

# ============================================================================
# TENSOR CORE EVOLUTION
# ============================================================================

def experiment_tensor_cores():
    """Demonstrate Tensor Core evolution and impact."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: TENSOR CORE EVOLUTION")
    print(" Matrix acceleration across generations")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    TENSOR CORE OPERATION:
    
    D = A × B + C  (Matrix multiply-accumulate)
    
    Per Tensor Core operation (Volta):
    ┌─────────────────────────────────────────┐
    │ A[4×4] × B[4×4] + C[4×4] = D[4×4]      │
    │ 64 FMA operations in ONE cycle          │
    └─────────────────────────────────────────┘
    
    EVOLUTION:
    
    Volta:   FP16 × FP16 → FP16/FP32
    Turing:  + INT8 × INT8 → INT32
    Ampere:  + TF32, BF16, 2:4 sparsity
    Hopper:  + FP8 × FP8 → FP16/FP32
    Blackwell: + FP4, improved FP8
    """)
    
    # Benchmark different precisions
    N = 4096
    
    print(f"\n Matrix multiply {N}×{N} - Precision comparison:")
    print(f"{'Precision':<15} {'Time (ms)':<15} {'TFLOPS':<15} {'Tensor Cores?'}")
    print("-" * 60)
    
    flops = 2 * N * N * N
    
    # FP32
    A_fp32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B_fp32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
    
    # Disable TF32 for pure FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    time_fp32 = profile_operation(lambda: A_fp32 @ B_fp32)
    tflops_fp32 = flops / (time_fp32 / 1000) / 1e12
    print(f"{'FP32':<15} {time_fp32:<15.3f} {tflops_fp32:<15.1f} {'No'}")
    
    # TF32 (automatic on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    time_tf32 = profile_operation(lambda: A_fp32 @ B_fp32)
    tflops_tf32 = flops / (time_tf32 / 1000) / 1e12
    print(f"{'TF32 (auto)':<15} {time_tf32:<15.3f} {tflops_tf32:<15.1f} {'Yes'}")
    
    # FP16
    A_fp16 = A_fp32.half()
    B_fp16 = B_fp32.half()
    time_fp16 = profile_operation(lambda: A_fp16 @ B_fp16)
    tflops_fp16 = flops / (time_fp16 / 1000) / 1e12
    print(f"{'FP16':<15} {time_fp16:<15.3f} {tflops_fp16:<15.1f} {'Yes'}")
    
    # BF16
    try:
        A_bf16 = A_fp32.bfloat16()
        B_bf16 = B_fp32.bfloat16()
        time_bf16 = profile_operation(lambda: A_bf16 @ B_bf16)
        tflops_bf16 = flops / (time_bf16 / 1000) / 1e12
        print(f"{'BF16':<15} {time_bf16:<15.3f} {tflops_bf16:<15.1f} {'Yes'}")
    except:
        print(f"{'BF16':<15} {'Not supported'}")
    
    print(f"\n Tensor Core speedup: {time_fp32/time_fp16:.1f}x (FP32 → FP16)")

# ============================================================================
# MEMORY BANDWIDTH EVOLUTION
# ============================================================================

def experiment_memory_bandwidth():
    """Demonstrate memory bandwidth across GPU generations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: MEMORY BANDWIDTH EVOLUTION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    MEMORY TECHNOLOGY EVOLUTION:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Technology    │ GPUs                │ Bandwidth    │ Capacity  │
    ├───────────────┼─────────────────────┼──────────────┼───────────┤
    │ GDDR5         │ GTX 1080            │ 320 GB/s     │ 8 GB      │
    │ GDDR6         │ RTX 2080            │ 448 GB/s     │ 8 GB      │
    │ GDDR6X        │ RTX 3090, 4090      │ 936-1008 GB/s│ 24 GB     │
    │ HBM2          │ V100                │ 900 GB/s     │ 32 GB     │
    │ HBM2e         │ A100                │ 2039 GB/s    │ 80 GB     │
    │ HBM3          │ H100                │ 3350 GB/s    │ 80 GB     │
    │ HBM3e         │ B200                │ 8000 GB/s    │ 192 GB    │
    └───────────────┴─────────────────────┴──────────────┴───────────┘
    
    WHY BANDWIDTH MATTERS FOR ML:
    - Attention is memory-bound (QK^T, softmax, PV)
    - Embedding lookups are memory-bound
    - Large batch activations stress memory
    - KV cache for inference
    """)
    
    # Measure actual bandwidth
    sizes_gb = [0.5, 1.0, 2.0, 4.0]
    
    print(f"\n Actual bandwidth measurement (copy operation):")
    print(f"{'Size (GB)':<15} {'Time (ms)':<15} {'Bandwidth (GB/s)'}")
    print("-" * 45)
    
    for size_gb in sizes_gb:
        num_elements = int(size_gb * 1e9 / 4)
        
        try:
            x = torch.randn(num_elements, device='cuda')
            
            def copy_op():
                return x.clone()
            
            time_ms = profile_operation(copy_op, iterations=50)
            bytes_moved = num_elements * 4 * 2  # Read + write
            bandwidth = bytes_moved / (time_ms / 1000) / 1e9
            
            print(f"{size_gb:<15.1f} {time_ms:<15.2f} {bandwidth:.0f}")
            
            del x
            torch.cuda.empty_cache()
        except RuntimeError:
            print(f"{size_gb:<15.1f} OOM")
            break

# ============================================================================
# SM ARCHITECTURE EVOLUTION
# ============================================================================

def experiment_sm_evolution():
    """Explain SM architecture changes across generations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SM ARCHITECTURE EVOLUTION")
    print("="*70)
    
    print("""
    SM (STREAMING MULTIPROCESSOR) EVOLUTION:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        VOLTA SM (2017)                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ 4 Warp Schedulers                                               │   │
    │  │ 4 Dispatch Units                                                │   │
    │  │ 64 FP32 Cores    │ 64 INT32 Cores  │ 32 FP64 Cores             │   │
    │  │ 8 Tensor Cores   │ 16 SFUs         │ 4 Tex Units               │   │
    │  │ 128 KB L1/Shared │ 65536 Registers │                           │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        AMPERE SM (2020)                                 │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ 4 Warp Schedulers                                               │   │
    │  │ 64 FP32 Cores    │ 64 INT32 Cores  │ 32 FP64 Cores             │   │
    │  │ 4 Tensor Cores (3rd gen - more capable)                        │   │
    │  │ 192 KB L1/Shared (configurable)                                │   │
    │  │ + TF32, BF16, Structured Sparsity support                      │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        HOPPER SM (2022)                                 │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ 4 Warp Schedulers                                               │   │
    │  │ 128 FP32 Cores (2x Ampere!)                                     │   │
    │  │ 4 Tensor Cores (4th gen - FP8!)                                │   │
    │  │ 256 KB L1/Shared                                                │   │
    │  │ + Transformer Engine, TMA, Thread Block Clusters               │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    KEY CHANGES:
    
    Volta → Ampere:
    - Tensor Cores: 8→4 per SM but 2x faster each
    - Added TF32, BF16 support
    - 2:4 structured sparsity
    - Larger shared memory (192KB)
    
    Ampere → Hopper:
    - 2x FP32 cores (128 per SM)
    - FP8 Tensor Core support
    - Transformer Engine (automatic precision)
    - TMA for efficient memory access
    - Thread Block Clusters for SM cooperation
    """)

# ============================================================================
# PRACTICAL IMPLICATIONS
# ============================================================================

def experiment_practical_implications():
    """Show practical implications of architecture differences."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: PRACTICAL IMPLICATIONS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n Your GPU: {gpu_name}")
    
    # Detect architecture
    props = torch.cuda.get_device_properties(0)
    
    print(f"""
    OPTIMIZATION RECOMMENDATIONS BY ARCHITECTURE:
    
    VOLTA/TURING (V100, RTX 20xx):
    ├── Use FP16 for Tensor Cores
    ├── Avoid TF32 (not available)
    ├── Mixed precision training essential
    └── Batch size optimization critical
    
    AMPERE (A100, RTX 30xx):
    ├── TF32 enabled by default (good!)
    ├── BF16 available (more stable than FP16)
    ├── Consider 2:4 sparsity for inference
    └── Larger shared memory enables bigger tiles
    
    HOPPER (H100):
    ├── Use FP8 for maximum throughput
    ├── Transformer Engine handles precision automatically
    ├── TMA for efficient attention
    └── Thread Block Clusters for large problems
    
    ADA LOVELACE (RTX 40xx):
    ├── FP8 Tensor Cores available
    ├── Improved cache hierarchy
    ├── DLSS/AI features
    └── Good for inference workloads
    """)
    
    # Test what features are available
    print(f" Feature availability on {gpu_name}:")
    
    # Test BF16
    try:
        x = torch.randn(100, device='cuda', dtype=torch.bfloat16)
        _ = x @ x.T
        print(f" ✓ BF16 supported")
    except:
        print(f" ✗ BF16 not supported")
    
    # Test TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f" ✓ TF32 setting available (may not affect all GPUs)")
    
    # Check compute capability
    cc = props.major * 10 + props.minor
    print(f" Compute capability: {props.major}.{props.minor}")
    
    if cc >= 80:
        print(f" → Ampere+ features available")
    elif cc >= 70:
        print(f" → Volta/Turing features available")
    else:
        print(f" → Older architecture, limited Tensor Core support")

# ============================================================================
# SUMMARY
# ============================================================================

def print_evolution_summary():
    """Print architecture evolution summary."""
    print("\n" + "="*70)
    print(" ARCHITECTURE EVOLUTION SUMMARY")
    print("="*70)
    
    print("""
    KEY TAKEAWAYS:
    
    1. TENSOR CORES CHANGED EVERYTHING
       - 10x+ speedup for matrix operations
       - Enabled practical mixed precision training
       - Each generation more capable
    
    2. MEMORY BANDWIDTH KEEPS GROWING
       - HBM enables massive bandwidth
       - Critical for memory-bound operations
       - Often the real bottleneck
    
    3. PRECISION OPTIONS EXPANDING
       - FP32 → FP16 → BF16 → TF32 → FP8 → FP4
       - More precision choices = better optimization
       - Transformer Engine automates selection
    
    4. SOFTWARE/HARDWARE CO-EVOLUTION
       - cuDNN, cuBLAS optimized for each generation
       - Flash Attention designed for modern GPUs
       - PyTorch 2.0 leverages new features
    
    CHOOSING A GPU FOR ML:
    
    ┌─────────────────┬───────────────────────────────────────────────┐
    │ Use Case        │ Recommendation                                │
    ├─────────────────┼───────────────────────────────────────────────┤
    │ Learning/Dev    │ RTX 3090/4090 (24GB, good perf/$)            │
    │ Small Models    │ A10, L4 (cloud), RTX 4090 (local)            │
    │ Medium Models   │ A100 40GB, L40S                              │
    │ Large Models    │ A100 80GB, H100                              │
    │ LLM Training    │ H100, B200                                    │
    │ Inference       │ L4, L40S (cost-effective)                    │
    └─────────────────┴───────────────────────────────────────────────┘
    
    WHAT TO WATCH FOR:
    - Memory capacity (model + activations + optimizer)
    - Memory bandwidth (attention, embedding)
    - Tensor Core generation (training speed)
    - Multi-GPU connectivity (NVLink for training)
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " NVIDIA GPU ARCHITECTURE EVOLUTION ".center(68) + "║")
    print("║" + " From Tesla to Blackwell ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    print_architecture_specs()
    experiment_tensor_cores()
    experiment_memory_bandwidth()
    experiment_sm_evolution()
    experiment_practical_implications()
    print_evolution_summary()
    
    print("\n" + "="*70)
    print(" Understanding architecture helps choose the right optimizations!")
    print("="*70)
