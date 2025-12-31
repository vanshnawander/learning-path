"""
NCCL and Collective Communication Operations
=============================================

Key Topics:
1. NCCL Overview
2. Collective Operations (All-Reduce, All-Gather, etc.)
3. Communication Algorithms (Ring, Tree)
4. NCCL Tuning and Configuration
"""

import torch
import torch.distributed as dist
import os
from typing import List

# =============================================================================
# SECTION 1: NCCL OVERVIEW
# =============================================================================
"""
NCCL (NVIDIA Collective Communications Library):
════════════════════════════════════════════════

NCCL is NVIDIA's optimized library for multi-GPU communication.

KEY FEATURES:
- Automatic topology detection (NVLink, PCIe, IB)
- Optimized algorithms for each topology
- GPUDirect RDMA support
- Integrated with PyTorch distributed

HIERARCHY:
    Application (PyTorch)
          ↓
    torch.distributed (Python API)
          ↓
    ProcessGroupNCCL (C++ wrapper)
          ↓
    NCCL Library (CUDA kernels)
          ↓
    Hardware (NVLink, IB, PCIe)
"""


# =============================================================================
# SECTION 2: COLLECTIVE OPERATIONS
# =============================================================================
"""
COLLECTIVE OPERATIONS:
══════════════════════

1. BROADCAST:
   One process sends to all others.
   
   Before: rank0=[A,B], rank1=[_,_], rank2=[_,_]
   After:  rank0=[A,B], rank1=[A,B], rank2=[A,B]
   
   Use: Initial model synchronization

2. REDUCE:
   All processes contribute, one receives result.
   
   Before: rank0=[1,2], rank1=[3,4], rank2=[5,6]
   After:  rank0=[9,12], rank1=[_,_], rank2=[_,_]
   
   Use: Final loss aggregation

3. ALL-REDUCE:
   All processes contribute, all receive result.
   
   Before: rank0=[1,2], rank1=[3,4], rank2=[5,6]
   After:  rank0=[9,12], rank1=[9,12], rank2=[9,12]
   
   Use: Gradient synchronization (DDP)

4. SCATTER:
   One process distributes different data to each.
   
   Before: rank0=[[A],[B],[C]], rank1=[_], rank2=[_]
   After:  rank0=[A], rank1=[B], rank2=[C]

5. GATHER:
   All processes send to one.
   
   Before: rank0=[A], rank1=[B], rank2=[C]
   After:  rank0=[[A],[B],[C]], rank1=[_], rank2=[_]

6. ALL-GATHER:
   All processes send to all.
   
   Before: rank0=[A], rank1=[B], rank2=[C]
   After:  rank0=[A,B,C], rank1=[A,B,C], rank2=[A,B,C]
   
   Use: FSDP parameter gathering

7. REDUCE-SCATTER:
   Reduce then scatter result.
   
   Before: rank0=[1,2,3], rank1=[4,5,6], rank2=[7,8,9]
   After:  rank0=[12], rank1=[15], rank2=[18]
   
   Use: FSDP gradient distribution
"""


def collective_bandwidth_formulas():
    """Print bandwidth formulas for collectives."""
    print("\n" + "="*60)
    print("COLLECTIVE BANDWIDTH FORMULAS")
    print("="*60)
    print("""
DATA TRANSFERRED (N ranks, S bytes):

┌──────────────────────────────────────────────────────────┐
│ Operation      │ Total Data      │ Algorithm Cost       │
├──────────────────────────────────────────────────────────┤
│ Broadcast      │ S               │ S × log(N)           │
│ Reduce         │ S               │ S × log(N)           │
│ All-Reduce     │ 2S(N-1)/N       │ 2S (ring)            │
│ All-Gather     │ S(N-1)          │ S(N-1)/N per link    │
│ Reduce-Scatter │ S(N-1)          │ S(N-1)/N per link    │
└──────────────────────────────────────────────────────────┘

RING ALL-REDUCE EXAMPLE (8 GPUs, 1GB data):
    Data per link: 2 × (8-1)/8 × 1GB = 1.75 GB
    Time on NVSwitch (900 GB/s): 1.94 ms
    Time on InfiniBand (50 GB/s): 35 ms
""")


# =============================================================================
# SECTION 3: COMMUNICATION ALGORITHMS
# =============================================================================
"""
RING ALL-REDUCE:
════════════════

GPUs arranged in a ring, data passed around.

Phase 1: Reduce-Scatter (N-1 steps)
    Each GPU ends up with 1/N of the final sum

Phase 2: All-Gather (N-1 steps)
    Each GPU gets the complete sum

Total: 2(N-1) steps, 2(N-1)/N × data transferred per link

    [GPU0]──►[GPU1]──►[GPU2]──►[GPU3]
      ▲                            │
      └────────────────────────────┘

Advantages: Bandwidth-optimal for large messages
Disadvantages: Latency O(N), sensitive to slow link


TREE ALL-REDUCE:
════════════════

Binary tree structure for reduce and broadcast.

Phase 1: Reduce up the tree (log N steps)
Phase 2: Broadcast down the tree (log N steps)

         [Root]
        /      \
    [GPU0]    [GPU1]
    /    \    /    \
 [G0] [G1] [G2] [G3]

Advantages: Latency O(log N)
Disadvantages: Root becomes bottleneck


NCCL ALGORITHM SELECTION:
═════════════════════════

NCCL automatically selects based on:
- Message size (small → tree, large → ring)
- Number of ranks
- Network topology
- Available bandwidth

Force algorithm:
    export NCCL_ALGO=Ring    # or Tree, CollNetDirect
"""


# =============================================================================
# SECTION 4: NCCL CONFIGURATION
# =============================================================================

def nccl_tuning_guide():
    """NCCL tuning guide."""
    print("\n" + "="*60)
    print("NCCL TUNING GUIDE")
    print("="*60)
    print("""
ESSENTIAL ENVIRONMENT VARIABLES:

# Debugging
NCCL_DEBUG=INFO              # Show initialization info
NCCL_DEBUG=WARN              # Warnings only
NCCL_DEBUG_SUBSYS=ALL        # All subsystems

# Network Selection
NCCL_SOCKET_IFNAME=eth0      # Use specific interface
NCCL_IB_DISABLE=0            # Enable InfiniBand
NCCL_NET_GDR_LEVEL=2         # GPUDirect RDMA level

# Algorithm Selection
NCCL_ALGO=Ring               # Force ring algorithm
NCCL_ALGO=Tree               # Force tree algorithm
NCCL_PROTO=Simple            # Protocol (Simple/LL/LL128)

# Performance
NCCL_BUFFSIZE=16777216       # Internal buffer size
NCCL_NTHREADS=512            # Threads per block
NCCL_NSOCKS_PERTHREAD=4      # Sockets per thread

# Multi-node
NCCL_CROSS_NIC=1             # Allow cross-NIC communication
NCCL_IB_HCA=mlx5             # InfiniBand HCA selection


COMMON ISSUES AND FIXES:

1. Slow all-reduce:
   - Check NVLink: nvidia-smi nvlink -s
   - Check IB: ibstat
   - Set NCCL_DEBUG=INFO to see chosen path

2. Hanging collectives:
   - NCCL_DEBUG=INFO for initialization
   - Check firewall on ports
   - Ensure same NCCL version on all nodes

3. Low bandwidth:
   - Enable GPUDirect: NCCL_NET_GDR_LEVEL=2
   - Check NIC binding: NCCL_IB_HCA
   - Profile with Nsight for kernel gaps
""")


# =============================================================================
# SECTION 5: PYTORCH DISTRIBUTED API
# =============================================================================

def pytorch_distributed_api():
    """PyTorch distributed collective API."""
    print("\n" + "="*60)
    print("PYTORCH DISTRIBUTED API")
    print("="*60)
    print('''
INITIALIZATION:
═══════════════

import torch.distributed as dist

# Environment variable method (torchrun sets these)
dist.init_process_group(backend='nccl')

# Explicit method
dist.init_process_group(
    backend='nccl',
    init_method='tcp://master_ip:port',
    rank=rank,
    world_size=world_size,
)


COLLECTIVE OPERATIONS:
══════════════════════

# All-reduce (in-place)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-gather
output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(output_tensors, tensor)

# Reduce-scatter
input_list = [chunk1, chunk2, chunk3, ...]  # one per rank
output = torch.zeros_like(chunk1)
dist.reduce_scatter(output, input_list)

# Broadcast
dist.broadcast(tensor, src=0)  # rank 0 sends

# Reduce
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)


ASYNC OPERATIONS:
═════════════════

# Start async all-reduce
handle = dist.all_reduce(tensor, async_op=True)

# Do other work...
other_computation()

# Wait for completion
handle.wait()


PROCESS GROUPS:
═══════════════

# Create subgroup
ranks = [0, 1, 2, 3]
group = dist.new_group(ranks)

# Use subgroup for collective
dist.all_reduce(tensor, group=group)
''')


if __name__ == "__main__":
    collective_bandwidth_formulas()
    nccl_tuning_guide()
    pytorch_distributed_api()
    print("\n" + "="*60)
    print("NCCL COLLECTIVES MODULE COMPLETE")
    print("="*60)
