"""
GPU Interconnects: NVLink, NVSwitch, PCIe, and InfiniBand
==========================================================

This module provides an in-depth exploration of GPU interconnect
technologies that enable distributed deep learning at scale.

Key Topics:
1. The Interconnect Bottleneck Problem
2. PCIe: The Baseline
3. NVLink: High-Speed GPU-to-GPU Communication
4. NVSwitch: All-to-All GPU Connectivity
5. InfiniBand vs Ethernet for Multi-Node
6. GPUDirect Technologies
7. Bandwidth vs Latency Trade-offs
"""

import torch
import time
from typing import Optional, Dict, Tuple
import subprocess
import os

# =============================================================================
# SECTION 1: THE INTERCONNECT BOTTLENECK
# =============================================================================
"""
WHY INTERCONNECTS MATTER:
═════════════════════════

Modern GPUs are incredibly fast at computation:
    - A100: 312 TFLOPS (FP16 Tensor Core)
    - H100: 989 TFLOPS (FP16 Tensor Core)

But they need DATA to compute on. Data must flow:
    1. From storage → CPU memory
    2. From CPU memory → GPU memory  
    3. Between GPUs (for multi-GPU training)
    4. Between nodes (for multi-node training)

BANDWIDTH HIERARCHY:
┌──────────────────────────────────────────────────────────────────────┐
│  Component                │ Bandwidth          │ Latency             │
├──────────────────────────────────────────────────────────────────────┤
│  GPU HBM (internal)       │ 3.35 TB/s (H100)   │ ~100 ns            │
│  NVLink 4.0 (per link)    │ 100 GB/s           │ ~1 μs              │
│  NVSwitch (full fabric)   │ 900 GB/s           │ ~2 μs              │
│  PCIe 5.0 x16             │ 64 GB/s            │ ~2-5 μs            │
│  InfiniBand NDR (400G)    │ 50 GB/s            │ ~1-2 μs            │
│  100GbE Ethernet          │ 12.5 GB/s          │ ~5-10 μs           │
│  NVMe SSD                 │ 7 GB/s             │ ~10-100 μs         │
│  HDD                      │ 0.2 GB/s           │ ~5-10 ms           │
└──────────────────────────────────────────────────────────────────────┘

THE PROBLEM:
    GPU compute: 989 TFLOPS
    GPU HBM bandwidth: 3.35 TB/s
    
    For training to be compute-bound (ideal):
        arithmetic_intensity = FLOPs / Bytes > 989 / 3350 ≈ 0.3
    
    But multi-GPU training adds:
        - Gradient synchronization (all-reduce)
        - Activation exchange (tensor parallel)
        - Layer transfers (pipeline parallel)
    
    These INCREASE byte requirements → LOWER arithmetic intensity
    → Training becomes communication-bound!

SOLUTION: Faster interconnects (NVLink, InfiniBand)
"""


def print_gpu_topology():
    """
    Print GPU topology and interconnect information.
    """
    print("\n" + "="*70)
    print("GPU TOPOLOGY AND INTERCONNECTS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  SM Count: {props.multi_processor_count}")
    
    if num_gpus > 1:
        print("\nPeer-to-Peer Access Matrix:")
        print("  " + "".join(f"GPU{j:2d} " for j in range(num_gpus)))
        for i in range(num_gpus):
            row = f"GPU{i} "
            for j in range(num_gpus):
                if i == j:
                    row += "  -   "
                else:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    row += "  ✓   " if can_access else "  ✗   "
            print(row)
        
        print("\nNote: ✓ indicates NVLink or PCIe P2P access available")


# =============================================================================
# SECTION 2: PCIe - THE BASELINE
# =============================================================================
"""
PCIe (Peripheral Component Interconnect Express):
═════════════════════════════════════════════════

The standard interface between CPU and peripherals (including GPUs).

PCIe GENERATIONS:
┌─────────────────────────────────────────────────────────────────────┐
│ Generation │ Per-Lane (GT/s) │ x16 Unidirectional │ x16 Bidirectional│
├─────────────────────────────────────────────────────────────────────┤
│ PCIe 3.0   │ 8 GT/s          │ 16 GB/s            │ 32 GB/s         │
│ PCIe 4.0   │ 16 GT/s         │ 32 GB/s            │ 64 GB/s         │
│ PCIe 5.0   │ 32 GT/s         │ 64 GB/s            │ 128 GB/s        │
│ PCIe 6.0   │ 64 GT/s         │ 128 GB/s           │ 256 GB/s        │
└─────────────────────────────────────────────────────────────────────┘

LIMITATIONS FOR MULTI-GPU:

1. Shared Bus Contention:
   - Multiple GPUs share lanes to CPU/chipset
   - Simultaneous transfers cause congestion
   - Unpredictable latency spikes

2. CPU-Centric Architecture:
   - GPU-to-GPU requires: GPU1 → CPU → GPU2
   - Doubles latency, halves effective bandwidth
   - CPU becomes bottleneck

3. Limited Bandwidth per GPU:
   - 64 GB/s (PCIe 5.0) vs 3.35 TB/s (HBM)
   - ~2% of GPU memory bandwidth
   - Severely limits multi-GPU scaling

PCIe P2P (Peer-to-Peer):
   - Direct GPU-to-GPU over PCIe switch
   - Avoids CPU hop
   - Only works if GPUs share PCIe switch
   - Still limited by PCIe bandwidth
"""


def measure_pcie_bandwidth():
    """
    Measure PCIe bandwidth between CPU and GPU.
    """
    print("\n" + "="*70)
    print("PCIe BANDWIDTH MEASUREMENT")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda:0")
    
    # Test different transfer sizes
    sizes_mb = [1, 10, 100, 500, 1000]
    
    print(f"\n{'Size (MB)':<12} {'H2D (GB/s)':<15} {'D2H (GB/s)':<15}")
    print("-" * 42)
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        
        # Create tensors
        cpu_tensor = torch.randn(size_bytes // 4, dtype=torch.float32, 
                                  pin_memory=True)
        gpu_tensor = torch.empty_like(cpu_tensor, device=device)
        
        # Warmup
        gpu_tensor.copy_(cpu_tensor)
        cpu_tensor.copy_(gpu_tensor)
        torch.cuda.synchronize()
        
        # Host to Device
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            gpu_tensor.copy_(cpu_tensor)
        torch.cuda.synchronize()
        h2d_time = (time.perf_counter() - start) / 10
        h2d_bw = size_bytes / h2d_time / 1e9
        
        # Device to Host
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            cpu_tensor.copy_(gpu_tensor)
        torch.cuda.synchronize()
        d2h_time = (time.perf_counter() - start) / 10
        d2h_bw = size_bytes / d2h_time / 1e9
        
        print(f"{size_mb:<12} {h2d_bw:<15.2f} {d2h_bw:<15.2f}")
    
    print("\nNote: Theoretical max depends on PCIe generation and lanes")


# =============================================================================
# SECTION 3: NVLink - HIGH-SPEED GPU-TO-GPU
# =============================================================================
"""
NVLink: NVIDIA's High-Speed Interconnect
═════════════════════════════════════════

NVLink is a point-to-point, high-speed serial link optimized for
GPU-to-GPU and GPU-to-CPU (POWER9) communication.

NVLINK GENERATIONS:
┌─────────────────────────────────────────────────────────────────────────┐
│ Generation │ Architecture │ Per-Link BW │ Links/GPU │ Total BW/GPU     │
├─────────────────────────────────────────────────────────────────────────┤
│ NVLink 1.0 │ Pascal P100  │ 40 GB/s     │ 4         │ 160 GB/s         │
│ NVLink 2.0 │ Volta V100   │ 50 GB/s     │ 6         │ 300 GB/s         │
│ NVLink 3.0 │ Ampere A100  │ 50 GB/s     │ 12        │ 600 GB/s         │
│ NVLink 4.0 │ Hopper H100  │ 100 GB/s    │ 18        │ 900 GB/s         │
│ NVLink 5.0 │ Blackwell   │ 200 GB/s    │ 18        │ 1.8 TB/s         │
└─────────────────────────────────────────────────────────────────────────┘

KEY FEATURES:

1. Point-to-Point Architecture:
   - Dedicated links between GPU pairs
   - No contention with other transfers
   - Predictable, low latency (~1 μs)

2. Higher Bandwidth than PCIe:
   - NVLink 4.0: 900 GB/s (total per GPU)
   - PCIe 5.0: 128 GB/s (bidirectional)
   - 7x bandwidth advantage!

3. Cache Coherence (NVLink 2.0+):
   - Hardware-enforced coherence
   - Unified address space GPU↔CPU (POWER9)
   - Eliminates explicit DMA programming

4. GPU Direct Memory Access:
   - GPU can directly read/write another GPU's memory
   - No CPU involvement
   - Enables efficient data sharing

TOPOLOGY EXAMPLES:

DGX-1 (P100): NVLink Hypercube
    Each GPU connected to 4 others
    2 hops maximum between any pair
    
    [GPU0]──[GPU1]
      │╲    ╱│
      │ ╲  ╱ │
      │  ╲╱  │
      │  ╱╲  │
      │ ╱  ╲ │
      │╱    ╲│
    [GPU2]──[GPU3]

DGX A100: Full NVSwitch Mesh
    All 8 GPUs fully connected via NVSwitch
    1 hop between any pair
    900 GB/s uniform bandwidth
"""


def measure_nvlink_bandwidth():
    """
    Measure NVLink bandwidth between GPUs (if available).
    """
    print("\n" + "="*70)
    print("GPU-TO-GPU BANDWIDTH MEASUREMENT")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need at least 2 GPUs for this measurement")
        return
    
    # Test GPU 0 to GPU 1
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    # Check P2P access
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"\nP2P access GPU0 ↔ GPU1: {'Yes (NVLink/P2P)' if can_p2p else 'No (via host)'}")
    
    # Enable P2P if available
    if can_p2p:
        torch.cuda.set_device(0)
    
    sizes_mb = [10, 100, 500, 1000, 2000]
    
    print(f"\n{'Size (MB)':<12} {'GPU0→GPU1 (GB/s)':<20} {'GPU1→GPU0 (GB/s)':<20}")
    print("-" * 52)
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        
        # Create tensors
        tensor0 = torch.randn(size_bytes // 4, dtype=torch.float32, device=device0)
        tensor1 = torch.empty_like(tensor0, device=device1)
        
        # Warmup
        tensor1.copy_(tensor0)
        torch.cuda.synchronize()
        
        # GPU0 to GPU1
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            tensor1.copy_(tensor0)
        torch.cuda.synchronize()
        time_01 = (time.perf_counter() - start) / 10
        bw_01 = size_bytes / time_01 / 1e9
        
        # GPU1 to GPU0
        tensor0_dst = torch.empty_like(tensor1, device=device0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            tensor0_dst.copy_(tensor1)
        torch.cuda.synchronize()
        time_10 = (time.perf_counter() - start) / 10
        bw_10 = size_bytes / time_10 / 1e9
        
        print(f"{size_mb:<12} {bw_01:<20.2f} {bw_10:<20.2f}")
    
    print("\nNote: High bandwidth (~200+ GB/s) indicates NVLink")
    print("      Lower bandwidth (~20-30 GB/s) indicates PCIe P2P or via host")


# =============================================================================
# SECTION 4: NVSWITCH - ALL-TO-ALL CONNECTIVITY
# =============================================================================
"""
NVSwitch: Scaling NVLink to Many GPUs
═════════════════════════════════════

Problem: Direct NVLink connects only pairs of GPUs
    - 8 GPUs fully meshed needs 28 links per GPU
    - Physical space constraints
    - Each GPU has limited NVLink ports

Solution: NVSwitch - A high-bandwidth crossbar switch

NVSWITCH GENERATIONS:
┌──────────────────────────────────────────────────────────────────────────┐
│ Generation  │ Ports │ Bandwidth/Port │ Total Switch BW │ GPUs Supported │
├──────────────────────────────────────────────────────────────────────────┤
│ NVSwitch 1  │ 18    │ 50 GB/s        │ 900 GB/s        │ 8 (DGX-2)      │
│ NVSwitch 2  │ 18    │ 50 GB/s        │ 900 GB/s        │ 8 (DGX A100)   │
│ NVSwitch 3  │ 64    │ 100 GB/s       │ 6.4 TB/s        │ 8+ (DGX H100)  │
│ NVSwitch 4  │ 72    │ 200 GB/s       │ 14.4 TB/s       │ 72+ (DGX GB200)│
└──────────────────────────────────────────────────────────────────────────┘

HOW NVSWITCH WORKS:

    Without NVSwitch (direct mesh):
    
    GPU0 ─── GPU1
      │╲   ╱│
      │ ╲ ╱ │         Bandwidth degrades with distance
      │  ╳  │         Multi-hop for distant pairs
      │ ╱ ╲ │
      │╱   ╲│
    GPU2 ─── GPU3
    
    With NVSwitch (switched fabric):
    
         [NVSwitch]
        ╱   │   │   ╲
    GPU0  GPU1  GPU2  GPU3
    
    Every GPU has SAME bandwidth to EVERY other GPU!
    - No multi-hop delays
    - 900 GB/s uniform (H100)
    - Ideal for collective operations

BANDWIDTH COMPARISON (8 GPUs):
┌─────────────────────────────────────────────────────────────────────────┐
│ Topology          │ Per-GPU BW    │ Bisection BW  │ Uniformity          │
├─────────────────────────────────────────────────────────────────────────┤
│ PCIe Only         │ 64 GB/s       │ 256 GB/s      │ Non-uniform         │
│ NVLink Ring       │ 600 GB/s      │ 1.2 TB/s      │ Varies by distance  │
│ NVLink Mesh       │ 600 GB/s      │ 2.4 TB/s      │ Mostly uniform      │
│ NVSwitch Fabric   │ 900 GB/s      │ 3.6 TB/s      │ Perfectly uniform   │
└─────────────────────────────────────────────────────────────────────────┘

DGXSUPERPOD AND MULTI-NODE:

For >8 GPUs (multi-node), NVSwitch connects to external network:
    
    Node 1 [8 GPUs ↔ NVSwitch] ←──InfiniBand──→ [8 GPUs ↔ NVSwitch] Node 2
    
    Intra-node: 900 GB/s (NVSwitch)
    Inter-node: 400 Gb/s = 50 GB/s (InfiniBand NDR)
    
    ~18x bandwidth difference!
    → Algorithms must minimize inter-node communication
"""


def nvswitch_topology_explanation():
    """Explain NVSwitch topology and its benefits."""
    print("\n" + "="*70)
    print("NVSWITCH TOPOLOGY BENEFITS")
    print("="*70)
    
    print("""
ALL-REDUCE OPERATION COMPARISON:
════════════════════════════════

All-reduce: Each GPU contributes gradients, all GPUs get sum

Ring All-Reduce (without NVSwitch):
    
    GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 → GPU7
      ↑                                               │
      └───────────────────────────────────────────────┘
    
    Steps: 2 × (N-1) = 14 for 8 GPUs
    Time: 14 × (data_size / link_bandwidth)
    Latency: O(N)

Tree All-Reduce (with NVSwitch):
    
              [NVSwitch]
          ╱   ╱    ╲    ╲
        GPU0 GPU1  GPU2 GPU3 ...
    
    Step 1: All GPUs → NVSwitch (parallel!)
    Step 2: NVSwitch → All GPUs (parallel!)
    
    Steps: 2 (reduce-scatter + all-gather)
    Time: 2 × (data_size / switch_bandwidth)
    Latency: O(1)

SPEEDUP EXAMPLE:
    Data: 1 GB gradients, 8 GPUs
    Link bandwidth: 600 GB/s (NVLink 3.0)
    
    Ring: 14 × (1 GB / 600 GB/s) = 23.3 ms
    Tree: 2 × (1 GB / 900 GB/s) = 2.2 ms
    
    Speedup: ~10x!
""")


# =============================================================================
# SECTION 5: INFINIBAND vs ETHERNET
# =============================================================================
"""
MULTI-NODE NETWORKING: InfiniBand vs Ethernet
══════════════════════════════════════════════

For training across multiple nodes, we need inter-node networking.
Two main options: InfiniBand and Ethernet.

INFINIBAND:
    - High-performance, low-latency switched fabric
    - Designed for HPC and data centers
    - RDMA (Remote Direct Memory Access) native
    - Lossless by design

ETHERNET:
    - Ubiquitous, well-understood
    - Lower cost, more vendors
    - RDMA via RoCE (RDMA over Converged Ethernet)
    - Lossy, needs congestion control

COMPARISON:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Feature              │ InfiniBand NDR     │ Ethernet 400GbE              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Raw Bandwidth        │ 400 Gb/s           │ 400 Gb/s                     │
│ Effective Bandwidth  │ ~50 GB/s           │ ~40-45 GB/s                  │
│ Latency              │ 0.5-1 μs           │ 2-5 μs (with RoCE)           │
│ RDMA                 │ Native             │ RoCE (needs configuration)   │
│ Lossless             │ Yes (credits)      │ No (needs PFC/ECN)           │
│ Cost                 │ Higher             │ Lower                        │
│ Ecosystem            │ Specialized        │ Ubiquitous                   │
│ Best for             │ HPC, large AI      │ General, inference           │
└─────────────────────────────────────────────────────────────────────────────┘

RDMA (Remote Direct Memory Access):
    - CPU not involved in data transfer
    - Network card reads/writes memory directly
    - Extremely low latency, high bandwidth
    - Essential for distributed training

INFINIBAND GENERATIONS:
┌──────────────────────────────────────────────────────────────────────────┐
│ Generation │ Speed per Lane │ 4x Aggregate │ Year                        │
├──────────────────────────────────────────────────────────────────────────┤
│ SDR        │ 2.5 Gb/s       │ 10 Gb/s      │ 2004                        │
│ DDR        │ 5 Gb/s         │ 20 Gb/s      │ 2005                        │
│ QDR        │ 10 Gb/s        │ 40 Gb/s      │ 2008                        │
│ FDR        │ 14 Gb/s        │ 56 Gb/s      │ 2011                        │
│ EDR        │ 25 Gb/s        │ 100 Gb/s     │ 2014                        │
│ HDR        │ 50 Gb/s        │ 200 Gb/s     │ 2018                        │
│ NDR        │ 100 Gb/s       │ 400 Gb/s     │ 2022                        │
│ XDR        │ 200 Gb/s       │ 800 Gb/s     │ 2024                        │
└──────────────────────────────────────────────────────────────────────────┘

WHEN TO USE EACH:
    
InfiniBand:
    ✓ Large-scale training (100s-1000s GPUs)
    ✓ HPC workloads requiring low latency
    ✓ When communication is bottleneck
    ✓ Budget allows premium networking
    
Ethernet:
    ✓ Smaller clusters (<100 GPUs)
    ✓ Mixed workloads (training + inference)
    ✓ Cost-sensitive environments
    ✓ Existing Ethernet infrastructure
"""


# =============================================================================
# SECTION 6: GPUDIRECT TECHNOLOGIES
# =============================================================================
"""
GPUDirect: Removing CPU from GPU Data Paths
════════════════════════════════════════════

GPUDirect is NVIDIA's umbrella for technologies that enable
direct GPU communication without CPU intervention.

GPUDIRECT VARIANTS:

1. GPUDirect Peer-to-Peer (P2P):
   - Direct GPU-to-GPU memory access
   - Over NVLink or PCIe
   - No CPU memory staging
   
   Without P2P:  GPU0 → CPU RAM → GPU1
   With P2P:     GPU0 → GPU1 (direct)

2. GPUDirect RDMA:
   - Network card reads/writes GPU memory directly
   - Bypasses CPU and system memory
   - Essential for multi-node training
   
   Without RDMA:  GPU → CPU → NIC → Network
   With RDMA:     GPU → NIC → Network (direct)

3. GPUDirect Storage:
   - NVMe SSD reads/writes GPU memory directly
   - Bypasses CPU for I/O
   - Useful for large datasets, checkpointing
   
   Without:  SSD → CPU → GPU
   With:     SSD → GPU (direct)

4. GPUDirect Async:
   - Asynchronous GPU-initiated operations
   - GPU kernel can trigger network sends
   - Overlap compute and communication

PERFORMANCE IMPACT:
┌──────────────────────────────────────────────────────────────────────┐
│ Operation              │ Without GPUDirect │ With GPUDirect         │
├──────────────────────────────────────────────────────────────────────┤
│ GPU-to-GPU (same node) │ ~15 GB/s (via CPU)│ ~600 GB/s (NVLink)     │
│ GPU-to-GPU (network)   │ ~10 GB/s          │ ~50 GB/s (IB RDMA)     │
│ GPU-to-NVMe            │ ~3 GB/s           │ ~7 GB/s                │
└──────────────────────────────────────────────────────────────────────┘

ENABLING GPUDIRECT:
    
    # Check GPUDirect RDMA support
    nvidia-smi -q | grep "GPUDirect"
    
    # In NCCL, GPUDirect is used automatically when available
    # Can disable for debugging:
    export NCCL_NET_GDR_LEVEL=0  # Disable GPUDirect RDMA
"""


# =============================================================================
# SECTION 7: BANDWIDTH BOTTLENECK ANALYSIS
# =============================================================================

def analyze_communication_bottleneck():
    """
    Analyze where communication bottlenecks occur.
    """
    print("\n" + "="*70)
    print("COMMUNICATION BOTTLENECK ANALYSIS")
    print("="*70)
    
    print("""
MODEL SIZE VS COMMUNICATION ANALYSIS:
═════════════════════════════════════

Consider training a 7B parameter model with:
    - 8 GPUs (single node with NVSwitch)
    - Batch size: 32 per GPU
    - BF16 gradients (2 bytes per param)

Gradient All-Reduce Volume:
    Total gradients = 7B × 2 bytes = 14 GB
    
    Ring all-reduce transfers: 2 × (N-1)/N × data = 1.75 × 14 GB = 24.5 GB
    
    Time on NVSwitch (900 GB/s): 24.5 GB / 900 GB/s = 27.2 ms
    Time on PCIe (64 GB/s): 24.5 GB / 64 GB/s = 383 ms
    
    Speedup from NVLink: 14x!

COMPUTE VS COMMUNICATION:

    Forward pass: ~50 ms
    Backward pass: ~100 ms
    All-reduce (NVSwitch): ~27 ms
    
    Total with NVSwitch: 177 ms (85% compute efficiency)
    Total with PCIe: 533 ms (28% compute efficiency)

SCALING TO MULTI-NODE:
    
    2 nodes (16 GPUs), InfiniBand 400G:
    
    Intra-node: 900 GB/s
    Inter-node: 50 GB/s
    
    Hierarchical all-reduce:
        1. Intra-node reduce: 14 GB / 900 GB/s = 15.5 ms
        2. Inter-node all-reduce: 14 GB / 50 GB/s = 280 ms
        3. Intra-node broadcast: 14 GB / 900 GB/s = 15.5 ms
        Total: 311 ms
    
    Efficiency drops to 48%!
    
    Solution: Gradient compression, overlap, hybrid parallelism
""")
    
    # Calculate real numbers
    model_params = 7e9  # 7B
    bytes_per_param = 2  # BF16
    gradient_size_gb = model_params * bytes_per_param / 1e9
    
    nvlink_bw = 900  # GB/s
    pcie_bw = 64  # GB/s
    ib_bw = 50  # GB/s
    
    num_gpus = 8
    ring_factor = 2 * (num_gpus - 1) / num_gpus
    
    total_transfer = ring_factor * gradient_size_gb
    
    time_nvlink = total_transfer / nvlink_bw * 1000  # ms
    time_pcie = total_transfer / pcie_bw * 1000  # ms
    
    print(f"\nNumerical Analysis (7B params, 8 GPUs):")
    print(f"  Gradient size: {gradient_size_gb:.1f} GB")
    print(f"  Ring all-reduce transfer: {total_transfer:.1f} GB")
    print(f"  Time on NVSwitch: {time_nvlink:.1f} ms")
    print(f"  Time on PCIe: {time_pcie:.1f} ms")
    print(f"  NVLink speedup: {time_pcie/time_nvlink:.1f}x")


# =============================================================================
# SECTION 8: PRACTICAL TOPOLOGY INSPECTION
# =============================================================================

def inspect_nvidia_topology():
    """
    Use nvidia-smi to inspect GPU topology.
    """
    print("\n" + "="*70)
    print("NVIDIA TOPOLOGY INSPECTION")
    print("="*70)
    
    print("""
COMMANDS TO INSPECT TOPOLOGY:

# Show GPU topology matrix
nvidia-smi topo -m

Output example (DGX A100):
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  CPU
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12  SYS   SYS
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12  SYS   SYS
...

Legend:
  X    = Self
  NV#  = Connected via NVLink (# = number of links)
  SYS  = Connected via system/PCIe
  NODE = Connected via NUMA node
  PHB  = Connected via PCIe Host Bridge
  PXB  = Connected via PCIe switch

# Show NVLink status
nvidia-smi nvlink -s

# Show detailed NVLink info
nvidia-smi nvlink -c 0  # For GPU 0

# Check P2P capabilities
nvidia-smi topo -p2p r  # Read access
nvidia-smi topo -p2p w  # Write access
nvidia-smi topo -p2p n  # Atomic access
""")
    
    # Try to run nvidia-smi topo if available
    try:
        if os.name != 'nt':  # Not Windows
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("\nActual topology on this system:")
                print(result.stdout)
    except Exception as e:
        print(f"\nCould not run nvidia-smi topo: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GPU INTERCONNECTS: NVLink, NVSwitch, PCIe, InfiniBand")
    print("="*70)
    
    # Print GPU topology
    print_gpu_topology()
    
    # Measure PCIe bandwidth
    measure_pcie_bandwidth()
    
    # Measure GPU-to-GPU bandwidth
    measure_nvlink_bandwidth()
    
    # Explain NVSwitch
    nvswitch_topology_explanation()
    
    # Analyze bottlenecks
    analyze_communication_bottleneck()
    
    # Inspect topology
    inspect_nvidia_topology()
    
    print("\n" + "="*70)
    print("GPU INTERCONNECTS MODULE COMPLETE")
    print("="*70)
