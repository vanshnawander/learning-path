"""
03_streaming_multiprocessor.py - Understanding the SM (Streaming Multiprocessor)

The SM is the fundamental compute unit of NVIDIA GPUs.
Understanding SM architecture is essential for:
- Writing efficient CUDA kernels
- Understanding occupancy limits
- Optimizing for specific GPU generations

SM Components:
- CUDA Cores (FP32, INT32 units)
- Tensor Cores (matrix accelerators)
- Warp Schedulers
- Register File
- Shared Memory / L1 Cache
- Special Function Units (SFUs)
- Load/Store Units

Run: python 03_streaming_multiprocessor.py
"""

import torch
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

# ============================================================================
# GPU ARCHITECTURE INFORMATION
# ============================================================================

@dataclass
class SMArchitecture:
    """SM specifications for different GPU architectures."""
    name: str
    cuda_cores_per_sm: int
    tensor_cores_per_sm: int
    fp32_per_cycle_per_sm: int
    registers_per_sm: int
    shared_memory_per_sm_kb: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    warp_schedulers: int
    l1_cache_kb: int

# Architecture specifications
ARCHITECTURES = {
    "Pascal (P100)": SMArchitecture(
        name="Pascal", cuda_cores_per_sm=64, tensor_cores_per_sm=0,
        fp32_per_cycle_per_sm=64, registers_per_sm=65536,
        shared_memory_per_sm_kb=64, max_warps_per_sm=64, max_threads_per_sm=2048,
        max_blocks_per_sm=32, warp_schedulers=2, l1_cache_kb=24
    ),
    "Volta (V100)": SMArchitecture(
        name="Volta", cuda_cores_per_sm=64, tensor_cores_per_sm=8,
        fp32_per_cycle_per_sm=64, registers_per_sm=65536,
        shared_memory_per_sm_kb=96, max_warps_per_sm=64, max_threads_per_sm=2048,
        max_blocks_per_sm=32, warp_schedulers=4, l1_cache_kb=128
    ),
    "Ampere (A100)": SMArchitecture(
        name="Ampere", cuda_cores_per_sm=64, tensor_cores_per_sm=4,
        fp32_per_cycle_per_sm=64, registers_per_sm=65536,
        shared_memory_per_sm_kb=164, max_warps_per_sm=64, max_threads_per_sm=2048,
        max_blocks_per_sm=32, warp_schedulers=4, l1_cache_kb=192
    ),
    "Hopper (H100)": SMArchitecture(
        name="Hopper", cuda_cores_per_sm=128, tensor_cores_per_sm=4,
        fp32_per_cycle_per_sm=128, registers_per_sm=65536,
        shared_memory_per_sm_kb=228, max_warps_per_sm=64, max_threads_per_sm=2048,
        max_blocks_per_sm=32, warp_schedulers=4, l1_cache_kb=256
    ),
    "Ada (RTX 4090)": SMArchitecture(
        name="Ada", cuda_cores_per_sm=128, tensor_cores_per_sm=4,
        fp32_per_cycle_per_sm=128, registers_per_sm=65536,
        shared_memory_per_sm_kb=100, max_warps_per_sm=48, max_threads_per_sm=1536,
        max_blocks_per_sm=24, warp_schedulers=4, l1_cache_kb=128
    ),
}

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_operation(func, warmup=10, iterations=100):
    """Profile GPU operation with proper synchronization."""
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

def get_gpu_info() -> Dict:
    """Get current GPU information."""
    if not torch.cuda.is_available():
        return {}
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "total_memory_gb": props.total_memory / 1e9,
        "max_threads_per_sm": props.max_threads_per_multi_processor,
        "max_threads_per_block": props.max_threads_per_block,
        "max_shared_memory_per_block": props.max_shared_memory_per_block,
        "max_shared_memory_per_sm": props.max_shared_memory_per_multiprocessor,
        "warp_size": props.warp_size,
        "max_registers_per_block": props.regs_per_block,
    }

# ============================================================================
# EXPERIMENT 1: SM COUNT AND PARALLELISM
# ============================================================================

def experiment_sm_parallelism():
    """
    Demonstrate how work is distributed across SMs.
    
    Key insight: Each thread block runs on ONE SM.
    Multiple blocks can run on the same SM if resources allow.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 1: SM PARALLELISM")
    print(" Work is distributed across Streaming Multiprocessors")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    gpu_info = get_gpu_info()
    sm_count = gpu_info["sm_count"]
    
    print(f"\n Your GPU: {gpu_info['name']}")
    print(f" Number of SMs: {sm_count}")
    print(f" Max threads per SM: {gpu_info['max_threads_per_sm']}")
    
    # Test how performance scales with number of "blocks" of work
    N = 1024  # Elements per "block"
    
    print(f"\n Scaling test (matrix multiply, varying parallelism):")
    print(f"{'Work Units':<15} {'SMs Used':<15} {'Time (ms)':<15} {'Efficiency'}")
    print("-" * 60)
    
    block_counts = [1, sm_count // 4, sm_count // 2, sm_count, sm_count * 2, sm_count * 4]
    base_time = None
    
    for num_blocks in block_counts:
        if num_blocks < 1:
            continue
            
        # Create batch of matrices to simulate parallel work
        batch_size = num_blocks
        A = torch.randn(batch_size, N, N, device='cuda')
        B = torch.randn(batch_size, N, N, device='cuda')
        
        def batched_matmul():
            return torch.bmm(A, B)
        
        time_ms = profile_operation(batched_matmul)
        
        if base_time is None:
            base_time = time_ms
        
        # Ideal scaling: time should stay constant as we add more parallel work
        # up to the point where all SMs are utilized
        sms_used = min(num_blocks, sm_count)
        efficiency = (base_time * num_blocks) / time_ms / num_blocks * 100
        
        print(f"{num_blocks:<15} {sms_used:<15} {time_ms:<15.3f} {efficiency:.1f}%")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Below {sm_count} blocks: Not all SMs utilized")
    print(f" - At {sm_count} blocks: All SMs working (peak efficiency)")
    print(f" - Above {sm_count} blocks: Blocks queue on SMs")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Batch size affects SM utilization")
    print(f" - Small batch = idle SMs = wasted compute")
    print(f" - Batch dimension often maps to parallel blocks")

# ============================================================================
# EXPERIMENT 2: WARP EXECUTION
# ============================================================================

def experiment_warp_execution():
    """
    Warps are the fundamental execution unit - 32 threads.
    
    All threads in a warp execute the SAME instruction (SIMT).
    Warp divergence occurs when threads take different branches.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 2: WARP EXECUTION (32 THREADS)")
    print(" SIMT: Single Instruction, Multiple Threads")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    gpu_info = get_gpu_info()
    
    print(f"\n Warp size: {gpu_info['warp_size']} threads")
    print(f" Max warps per SM: {gpu_info['max_threads_per_sm'] // 32}")
    
    # Demonstrate warp-level efficiency
    N = 10_000_000
    x = torch.randn(N, device='cuda')
    
    print(f"\n Warp utilization scenarios:")
    print(f"{'Scenario':<35} {'Time (ms)':<15} {'Relative'}")
    print("-" * 65)
    
    # Full warp utilization - all threads do same thing
    def full_utilization():
        return x * 2.0
    
    time_full = profile_operation(full_utilization)
    
    # Partial utilization simulation - mask some elements
    mask = torch.rand(N, device='cuda') > 0.5
    def half_utilization():
        return torch.where(mask, x * 2.0, x)
    
    time_half = profile_operation(half_utilization)
    
    # Divergent branches - different operations
    def divergent():
        return torch.where(mask, x * 2.0, x + 1.0)
    
    time_div = profile_operation(divergent)
    
    # Multiple divergent paths
    mask2 = torch.rand(N, device='cuda')
    def multi_divergent():
        return torch.where(mask2 > 0.66, x * 2.0,
               torch.where(mask2 > 0.33, x + 1.0, x - 0.5))
    
    time_multi = profile_operation(multi_divergent)
    
    print(f"{'Full utilization (x * 2)':<35} {time_full:<15.3f} 1.0x")
    print(f"{'50% active (where, same op)':<35} {time_half:<15.3f} {time_half/time_full:.2f}x")
    print(f"{'Divergent (where, diff ops)':<35} {time_div:<15.3f} {time_div/time_full:.2f}x")
    print(f"{'Multi-path divergence':<35} {time_multi:<15.3f} {time_multi/time_full:.2f}x")
    
    print(f"\n WARP DIVERGENCE EXPLAINED:")
    print(f" ┌─────────────────────────────────────────────────────────────┐")
    print(f" │ Warp (32 threads)                                          │")
    print(f" │ ┌───┬───┬───┬───┬───┬───┬───┬───┐                         │")
    print(f" │ │ T0│ T1│ T2│...│T30│T31│   │   │                         │")
    print(f" │ └───┴───┴───┴───┴───┴───┴───┴───┘                         │")
    print(f" │                                                            │")
    print(f" │ No divergence: All 32 threads execute same path            │")
    print(f" │   → Full efficiency                                        │")
    print(f" │                                                            │")
    print(f" │ With divergence: if(cond) {{ A }} else {{ B }}               │")
    print(f" │   → Threads taking A wait while B executes                 │")
    print(f" │   → Then B threads wait while A executes                   │")
    print(f" │   → 2x time in worst case!                                 │")
    print(f" └─────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Attention masking: Use additive masks, not conditionals")
    print(f" - Sparse patterns: Need specialized kernels")
    print(f" - Dynamic shapes: Pad to warp boundaries (32)")

# ============================================================================
# EXPERIMENT 3: OCCUPANCY
# ============================================================================

def experiment_occupancy():
    """
    Occupancy = Active warps / Maximum warps per SM
    
    Higher occupancy generally means better latency hiding.
    But it's not always the bottleneck!
    
    Factors limiting occupancy:
    - Registers per thread
    - Shared memory per block
    - Threads per block
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 3: OCCUPANCY")
    print(" Occupancy = Active warps / Max warps per SM")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    gpu_info = get_gpu_info()
    
    print(f"\n GPU Occupancy Limits:")
    print(f" - Max threads per SM: {gpu_info['max_threads_per_sm']}")
    print(f" - Max warps per SM: {gpu_info['max_threads_per_sm'] // 32}")
    print(f" - Max threads per block: {gpu_info['max_threads_per_block']}")
    print(f" - Max shared memory per block: {gpu_info['max_shared_memory_per_block'] / 1024:.0f} KB")
    print(f" - Max registers per block: {gpu_info['max_registers_per_block']}")
    
    # Demonstrate occupancy impact with varying work sizes
    print(f"\n Occupancy impact demonstration:")
    print(f" (Simulated through varying tensor sizes)")
    
    sizes = [32, 128, 512, 2048, 8192, 32768, 131072, 524288]
    
    print(f"\n{'Size':<12} {'Threads':<12} {'Est. Warps':<12} {'Time (μs)':<12} {'Throughput'}")
    print("-" * 60)
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        
        def simple_op():
            return x * 2.0 + 1.0
        
        time_ms = profile_operation(simple_op, iterations=1000)
        time_us = time_ms * 1000
        
        # Estimate active threads (simplified)
        est_threads = min(size, gpu_info['max_threads_per_sm'] * gpu_info['sm_count'])
        est_warps = est_threads // 32
        throughput = size / time_us  # elements per microsecond
        
        print(f"{size:<12} {est_threads:<12} {est_warps:<12} {time_us:<12.2f} {throughput:.1f}")
    
    print(f"\n OCCUPANCY CALCULATOR:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │ Resource         │ Per Thread │ Per Block │ Limit        │")
    print(f" ├──────────────────┼────────────┼───────────┼──────────────┤")
    print(f" │ Registers        │ 32-255     │ ≤65536    │ 65536/SM     │")
    print(f" │ Shared Memory    │ -          │ ≤48KB*    │ 48-164KB/SM  │")
    print(f" │ Threads          │ 1          │ ≤1024     │ 2048/SM      │")
    print(f" └──────────────────┴────────────┴───────────┴──────────────┘")
    print(f" * Configurable, trades off with L1 cache")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - High register usage = lower occupancy")
    print(f" - Complex kernels may have low occupancy but still be fast")
    print(f" - Use NVIDIA's occupancy calculator for optimization")

# ============================================================================
# EXPERIMENT 4: REGISTER PRESSURE
# ============================================================================

def experiment_register_pressure():
    """
    Registers are the fastest storage but limited per thread.
    
    More registers per thread = fewer concurrent threads = lower occupancy.
    This is the classic occupancy vs. ILP tradeoff.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 4: REGISTER PRESSURE")
    print(" More registers per thread → Fewer concurrent threads")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    gpu_info = get_gpu_info()
    max_regs = gpu_info['max_registers_per_block']
    
    print(f"\n Max registers per block: {max_regs}")
    print(f" With 1024 threads per block: {max_regs // 1024} registers per thread")
    print(f" With 256 threads per block: {max_regs // 256} registers per thread")
    
    # Demonstrate with operations of varying complexity
    N = 1_000_000
    
    print(f"\n Operation complexity vs performance:")
    print(f"{'Operation':<30} {'Est. Regs':<12} {'Time (ms)':<12} {'Relative'}")
    print("-" * 65)
    
    x = torch.randn(N, device='cuda')
    
    # Simple operation - few registers
    def simple():
        return x + 1.0
    time_simple = profile_operation(simple)
    
    # Medium complexity - more intermediates
    def medium():
        a = x + 1.0
        b = a * 2.0
        c = b - 0.5
        return c * a
    time_medium = profile_operation(medium)
    
    # Complex - many intermediates
    def complex_op():
        a = x + 1.0
        b = x * 2.0
        c = a + b
        d = a * b
        e = c + d
        f = c * d
        g = e + f
        return g * x
    time_complex = profile_operation(complex_op)
    
    # Very complex - trigonometric functions use SFUs
    def trig():
        return torch.sin(x) + torch.cos(x) + torch.exp(x * 0.1)
    time_trig = profile_operation(trig)
    
    print(f"{'Simple (x + 1)':<30} {'~4':<12} {time_simple:<12.3f} 1.0x")
    print(f"{'Medium (4 ops)':<30} {'~8':<12} {time_medium:<12.3f} {time_medium/time_simple:.2f}x")
    print(f"{'Complex (8 ops)':<30} {'~16':<12} {time_complex:<12.3f} {time_complex/time_simple:.2f}x")
    print(f"{'Trig (sin+cos+exp)':<30} {'~32+':<12} {time_trig:<12.3f} {time_trig/time_simple:.2f}x")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Complex operations use more registers")
    print(f" - More registers → fewer warps → less latency hiding")
    print(f" - BUT more compute per thread can offset lower occupancy")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Fused kernels: Balance fusion depth with register pressure")
    print(f" - Flash Attention: Carefully manages registers for softmax")
    print(f" - Auto-tuning explores this tradeoff")

# ============================================================================
# EXPERIMENT 5: TENSOR CORES
# ============================================================================

def experiment_tensor_cores():
    """
    Tensor Cores are specialized matrix multiply-accumulate units.
    
    They perform D = A @ B + C in hardware for small tiles.
    Massive speedup for matrix operations at mixed precision.
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 5: TENSOR CORES")
    print(" Specialized hardware for matrix multiply")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Check if tensor cores are likely available
    gpu_info = get_gpu_info()
    has_tensor_cores = any(x in gpu_info['name'].lower() 
                          for x in ['v100', 'a100', 'h100', 'rtx', '2080', '3090', '4090'])
    
    print(f"\n GPU: {gpu_info['name']}")
    print(f" Tensor Cores likely available: {has_tensor_cores}")
    
    if not has_tensor_cores:
        print(" Skipping tensor core benchmarks...")
        return
    
    sizes = [512, 1024, 2048, 4096, 8192]
    
    print(f"\n Matrix multiply: FP32 vs FP16 (Tensor Core eligible)")
    print(f"{'Size':<10} {'FP32 (ms)':<15} {'FP16 (ms)':<15} {'TF32 (ms)':<15} {'Speedup'}")
    print("-" * 70)
    
    for N in sizes:
        # FP32 (no tensor cores)
        A_fp32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
        B_fp32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
        
        def mm_fp32():
            return A_fp32 @ B_fp32
        time_fp32 = profile_operation(mm_fp32)
        
        # FP16 (tensor cores)
        A_fp16 = A_fp32.half()
        B_fp16 = B_fp32.half()
        
        def mm_fp16():
            return A_fp16 @ B_fp16
        time_fp16 = profile_operation(mm_fp16)
        
        # TF32 (automatic on Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        def mm_tf32():
            return A_fp32 @ B_fp32
        time_tf32 = profile_operation(mm_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        
        speedup_fp16 = time_fp32 / time_fp16
        speedup_tf32 = time_fp32 / time_tf32
        
        print(f"{N:<10} {time_fp32:<15.3f} {time_fp16:<15.3f} {time_tf32:<15.3f} FP16:{speedup_fp16:.1f}x TF32:{speedup_tf32:.1f}x")
    
    # TFLOPS calculation
    N = 4096
    flops = 2 * N * N * N
    
    A_fp16 = torch.randn(N, N, device='cuda', dtype=torch.float16)
    B_fp16 = torch.randn(N, N, device='cuda', dtype=torch.float16)
    
    time_ms = profile_operation(lambda: A_fp16 @ B_fp16)
    tflops = flops / (time_ms / 1000) / 1e12
    
    print(f"\n Achieved FP16 TFLOPS at {N}x{N}: {tflops:.1f}")
    
    print(f"\n TENSOR CORE OPERATION:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │ D = A × B + C                                              │")
    print(f" │                                                            │")
    print(f" │ Per Tensor Core operation:                                 │")
    print(f" │ - 4×4×4 matrix multiply-add in ONE cycle                   │")
    print(f" │ - 256 FMA operations per cycle per core                    │")
    print(f" │                                                            │")
    print(f" │ Supported precisions:                                      │")
    print(f" │ - FP16 × FP16 → FP16/FP32 (Volta+)                        │")
    print(f" │ - BF16 × BF16 → FP32 (Ampere+)                            │")
    print(f" │ - TF32 × TF32 → FP32 (Ampere+, automatic)                 │")
    print(f" │ - FP8 × FP8 → FP16/FP32 (Hopper+)                         │")
    print(f" │ - INT8 × INT8 → INT32 (Turing+)                           │")
    print(f" └────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Use mixed precision training (AMP) to leverage Tensor Cores")
    print(f" - FP16/BF16 for forward/backward, FP32 for master weights")
    print(f" - cuBLAS/cuDNN automatically use Tensor Cores when applicable")
    print(f" - Batch size affects Tensor Core utilization")

# ============================================================================
# EXPERIMENT 6: WARP SCHEDULERS
# ============================================================================

def experiment_warp_schedulers():
    """
    Warp schedulers select which warps to execute each cycle.
    
    Modern GPUs have 4 warp schedulers per SM, each can issue
    1-2 instructions per cycle to different warps.
    
    This enables instruction-level parallelism (ILP).
    """
    print("\n" + "="*70)
    print(" EXPERIMENT 6: WARP SCHEDULERS AND ILP")
    print(" Multiple independent instructions can execute in parallel")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    N = 10_000_000
    
    # Demonstrate ILP benefit
    print(f"\n Instruction-Level Parallelism demonstration:")
    print(f"{'Pattern':<35} {'Time (ms)':<15} {'ILP Factor'}")
    print("-" * 60)
    
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    
    # Sequential dependent operations (no ILP)
    def sequential():
        a = x + 1
        b = a + 1  # Depends on a
        c = b + 1  # Depends on b
        d = c + 1  # Depends on c
        return d
    time_seq = profile_operation(sequential)
    
    # Independent operations (high ILP)
    def independent():
        a = x + 1
        b = y + 2
        c = x * 2
        d = y * 3
        return a + b + c + d
    time_ind = profile_operation(independent)
    
    # Mixed pattern
    def mixed():
        a = x + 1
        b = y + 2  # Independent of a
        c = a + b  # Depends on both
        d = c + x  # Depends on c
        return d
    time_mixed = profile_operation(mixed)
    
    print(f"{'Sequential (a→b→c→d)':<35} {time_seq:<15.3f} 1.0")
    print(f"{'Independent (a,b,c,d parallel)':<35} {time_ind:<15.3f} ~4.0")
    print(f"{'Mixed pattern':<35} {time_mixed:<15.3f} ~2.0")
    
    print(f"\n Observed speedup from ILP: {time_seq/time_ind:.2f}x")
    
    print(f"\n WARP SCHEDULER ARCHITECTURE:")
    print(f" ┌────────────────────────────────────────────────────────────┐")
    print(f" │              SM (Streaming Multiprocessor)                 │")
    print(f" │ ┌──────────┬──────────┬──────────┬──────────┐             │")
    print(f" │ │ Sched 0  │ Sched 1  │ Sched 2  │ Sched 3  │             │")
    print(f" │ └────┬─────┴────┬─────┴────┬─────┴────┬─────┘             │")
    print(f" │      │          │          │          │                   │")
    print(f" │      ▼          ▼          ▼          ▼                   │")
    print(f" │  [Warps]    [Warps]    [Warps]    [Warps]                 │")
    print(f" │                                                           │")
    print(f" │ Each scheduler can issue instructions to its warps        │")
    print(f" │ independently, enabling instruction-level parallelism     │")
    print(f" └────────────────────────────────────────────────────────────┘")
    print(f"\n MULTIMODAL IMPLICATION:")
    print(f" - Independent operations can overlap")
    print(f" - Memory loads can hide behind compute")
    print(f" - Loop unrolling creates more ILP opportunities")

# ============================================================================
# SM ARCHITECTURE SUMMARY
# ============================================================================

def print_sm_summary():
    """Print comprehensive SM architecture summary."""
    print("\n" + "="*70)
    print(" SM ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │              STREAMING MULTIPROCESSOR (SM) ANATOMY                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                    Warp Schedulers (4)                       │   │
    │  │    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                │   │
    │  │    │ WS0 │    │ WS1 │    │ WS2 │    │ WS3 │                │   │
    │  │    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                │   │
    │  └───────┼──────────┼──────────┼──────────┼────────────────────┘   │
    │          │          │          │          │                         │
    │  ┌───────▼──────────▼──────────▼──────────▼────────────────────┐   │
    │  │                    Execution Units                           │   │
    │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │   │
    │  │  │INT32   │ │FP32    │ │FP64    │ │Tensor  │ │SFU     │    │   │
    │  │  │Cores   │ │Cores   │ │Cores   │ │Cores   │ │Units   │    │   │
    │  │  │(64-128)│ │(64-128)│ │(32)    │ │(4-8)   │ │(16)    │    │   │
    │  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │   │
    │  └────────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                    Memory Subsystem                          │   │
    │  │  ┌─────────────────┐  ┌─────────────────┐                   │   │
    │  │  │  Register File  │  │ Shared Memory   │                   │   │
    │  │  │  (256 KB)       │  │ / L1 Cache      │                   │   │
    │  │  │                 │  │ (48-228 KB)     │                   │   │
    │  │  └─────────────────┘  └─────────────────┘                   │   │
    │  │  ┌─────────────────────────────────────────┐                │   │
    │  │  │        Load/Store Units (32)            │                │   │
    │  │  └─────────────────────────────────────────┘                │   │
    │  └────────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    KEY SM METRICS BY ARCHITECTURE:
    
    ┌────────────┬───────┬────────┬────────┬────────┬────────┐
    │ Component  │Pascal │ Volta  │ Ampere │ Hopper │ Ada    │
    ├────────────┼───────┼────────┼────────┼────────┼────────┤
    │ FP32 Cores │ 64    │ 64     │ 64     │ 128    │ 128    │
    │ Tensor C.  │ 0     │ 8      │ 4      │ 4      │ 4      │
    │ Registers  │ 64K   │ 64K    │ 64K    │ 64K    │ 64K    │
    │ Shared Mem │ 64KB  │ 96KB   │ 164KB  │ 228KB  │ 100KB  │
    │ Max Warps  │ 64    │ 64     │ 64     │ 64     │ 48     │
    │ Schedulers │ 2     │ 4      │ 4      │ 4      │ 4      │
    └────────────┴───────┴────────┴────────┴────────┴────────┘
    
    OPTIMIZATION CHECKLIST:
    
    ✓ Occupancy: Enough warps to hide memory latency?
    ✓ Registers: Not using too many per thread?
    ✓ Shared Memory: Using it for data reuse?
    ✓ Coalescing: Memory access patterns aligned?
    ✓ Divergence: Minimizing branch divergence?
    ✓ Tensor Cores: Using mixed precision when applicable?
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " STREAMING MULTIPROCESSOR (SM) DEEP DIVE ".center(68) + "║")
    print("║" + " The fundamental compute unit of NVIDIA GPUs ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"\n GPU: {gpu_info['name']}")
        print(f" SMs: {gpu_info['sm_count']}")
        print(f" Max threads/SM: {gpu_info['max_threads_per_sm']}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_sm_parallelism()
    experiment_warp_execution()
    experiment_occupancy()
    experiment_register_pressure()
    experiment_tensor_cores()
    experiment_warp_schedulers()
    print_sm_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Understanding CUDA execution model and memory patterns")
    print("="*70)
