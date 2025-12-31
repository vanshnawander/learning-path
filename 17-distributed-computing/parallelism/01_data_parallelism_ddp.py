"""
Data Parallelism and DistributedDataParallel (DDP)
===================================================

This module provides comprehensive coverage of data parallelism,
focusing on PyTorch's DistributedDataParallel (DDP) implementation.

Key Topics:
1. Data Parallelism Fundamentals
2. DDP Architecture and Implementation
3. Gradient Synchronization (All-Reduce)
4. Process Groups and Backends
5. Launch Methods (torchrun, mp.spawn)
6. Common Issues and Debugging
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os
from typing import Optional, Callable
import time

# =============================================================================
# SECTION 1: DATA PARALLELISM FUNDAMENTALS
# =============================================================================
"""
DATA PARALLELISM:
═════════════════

The simplest and most common distributed training strategy.

Core Idea:
    - REPLICATE the model on each GPU
    - SPLIT the data batch across GPUs
    - Each GPU computes gradients on its local data
    - SYNCHRONIZE gradients across all GPUs
    - Each GPU updates its local model copy

Diagram:
┌─────────────────────────────────────────────────────────────────────┐
│                        Global Batch (32)                            │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              │               │               │                      │
│              ▼               ▼               ▼                      │
│        Micro-batch 8   Micro-batch 8   Micro-batch 8   ...         │
│              │               │               │                      │
│              ▼               ▼               ▼                      │
│         [GPU 0]         [GPU 1]         [GPU 2]                    │
│         Model₀          Model₁          Model₂                     │
│         Grad₀           Grad₁           Grad₂                      │
│              │               │               │                      │
│              └───────────────┼───────────────┘                      │
│                              │                                      │
│                        All-Reduce                                   │
│                     (Average Gradients)                             │
│                              │                                      │
│              ┌───────────────┼───────────────┐                      │
│              ▼               ▼               ▼                      │
│         Grad_avg        Grad_avg        Grad_avg                   │
│              │               │               │                      │
│              ▼               ▼               ▼                      │
│        Optimizer        Optimizer        Optimizer                  │
│         Step             Step             Step                      │
│              │               │               │                      │
│              ▼               ▼               ▼                      │
│         Model₀'         Model₁'         Model₂'                    │
│         (identical models after update)                             │
└─────────────────────────────────────────────────────────────────────┘

MATHEMATICAL EQUIVALENCE:

Let B = global batch size, N = number of GPUs, b = B/N (local batch)

Full-batch gradient:
    ∇L = (1/B) Σᵢ₌₁ᴮ ∇ℓᵢ

Data-parallel gradient:
    ∇L = (1/N) Σⱼ₌₁ᴺ [(1/b) Σᵢ₌₁ᵇ ∇ℓᵢⱼ]
       = (1/B) Σᵢ₌₁ᴮ ∇ℓᵢ  ✓ Identical!

KEY INSIGHT: Data parallelism is MATHEMATICALLY EQUIVALENT
to single-GPU training with the same total batch size.
"""


# =============================================================================
# SECTION 2: DDP ARCHITECTURE
# =============================================================================
"""
DistributedDataParallel INTERNALS:
══════════════════════════════════

DDP is PyTorch's recommended way to do data parallel training.

WHY NOT DataParallel (DP)?
    - DP uses threading (GIL bottleneck)
    - DP replicates model every forward pass
    - DP gathers outputs to GPU 0 (memory imbalance)
    
    DDP uses processes (no GIL), replicates once, no gathering.

DDP INITIALIZATION:
    1. Constructor takes local model
    2. Broadcasts state_dict from rank 0 to all ranks
    3. Creates a Reducer for gradient synchronization
    4. Registers hooks on each parameter

DDP FORWARD PASS:
    1. Forward pass on local model (unchanged)
    2. No extra communication during forward

DDP BACKWARD PASS:
    1. Backward pass computes local gradients
    2. Gradient hooks trigger all-reduce
    3. Gradients are AVERAGED across ranks
    4. After backward, all ranks have identical gradients

BUCKET MECHANISM:
    - Parameters grouped into "buckets" (~25 MB default)
    - All-reduce happens per bucket, not per parameter
    - Buckets are processed in REVERSE order (as gradients computed)
    - Overlaps communication with backward computation

    Diagram:
    
    Backward computation: param_N → param_N-1 → ... → param_1
                               ↓         ↓              ↓
    Buckets:            [bucket_K] [bucket_K-1]... [bucket_1]
                               ↓         ↓              ↓
    All-reduce:              AR_K      AR_K-1    ...   AR_1
    
    Communication overlaps with gradient computation!
"""


class SimpleDDPModel(nn.Module):
    """Simple model for DDP demonstration."""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 2048, 
                 output_dim: int = 1000):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


# =============================================================================
# SECTION 3: COMPLETE DDP TRAINING EXAMPLE
# =============================================================================

def ddp_training_worker(rank: int, world_size: int, epochs: int = 2):
    """
    Worker function for DDP training.
    
    Each process runs this function with a different rank.
    """
    # Setup DDP
    setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    # Create model and move to device
    model = SimpleDDPModel().to(device)
    
    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    # Create dummy dataset
    dataset_size = 10000
    dataset = torch.utils.data.TensorDataset(
        torch.randn(dataset_size, 1024),
        torch.randint(0, 1000, (dataset_size,))
    )
    
    # Distributed sampler ensures each rank gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling!
        
        ddp_model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if rank == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    cleanup_ddp()


def run_ddp_training(world_size: int = 2, epochs: int = 2):
    """
    Launch DDP training across multiple GPUs.
    """
    mp.spawn(
        ddp_training_worker,
        args=(world_size, epochs),
        nprocs=world_size,
        join=True
    )


# =============================================================================
# SECTION 4: ALL-REDUCE OPERATIONS
# =============================================================================
"""
ALL-REDUCE: The Core of Data Parallelism
════════════════════════════════════════

All-reduce combines values from all processes and distributes
the result back to all processes.

OPERATION:
    Input: Each process has a tensor
    Output: All processes have SUM (or AVG) of all tensors

Example (4 processes):
    Before: rank0=[1,2], rank1=[3,4], rank2=[5,6], rank3=[7,8]
    After:  rank0=[16,20], rank1=[16,20], rank2=[16,20], rank3=[16,20]
    
    (16 = 1+3+5+7, 20 = 2+4+6+8)

ALGORITHMS:

1. Ring All-Reduce:
   - Pass data around a ring
   - O(N) latency, O(data) bandwidth per link
   - Best for large messages
   
   Step 1: Reduce-scatter (each rank gets 1/N of sum)
   Step 2: All-gather (distribute to all ranks)
   
   Total data transferred: 2 × (N-1)/N × data_size

2. Tree All-Reduce:
   - Binary tree structure
   - O(log N) latency, higher bandwidth per link
   - Better for small messages
   
   Step 1: Reduce up the tree
   Step 2: Broadcast down the tree

3. Recursive Halving/Doubling:
   - Divide and conquer
   - Good for power-of-2 process counts

NCCL (NVIDIA Collective Communications Library):
    - Highly optimized for GPU clusters
    - Automatically selects best algorithm
    - Uses NVLink, NVSwitch, InfiniBand
    - Default backend for PyTorch GPU training
"""


def demonstrate_all_reduce():
    """
    Demonstrate all-reduce operation.
    """
    print("\n" + "="*70)
    print("ALL-REDUCE DEMONSTRATION")
    print("="*70)
    
    print("""
ALL-REDUCE IN DDP:
══════════════════

During backward pass:
    1. Each GPU computes local gradients
    2. DDP triggers all-reduce on gradient buckets
    3. Gradients are averaged (sum then divide by world_size)
    4. All GPUs have identical averaged gradients

Code (what DDP does internally):

    # After local backward pass
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

Actually, DDP uses buckets and overlap:

    # Bucket callback (simplified)
    def bucket_hook(bucket):
        dist.all_reduce(bucket, op=dist.ReduceOp.AVG, async_op=True)

BANDWIDTH USAGE:

For model with P parameters (in bytes):
    Ring all-reduce: 2 × (N-1)/N × P bytes
    
    Example: 7B params × 2 bytes (BF16) × 2 × 7/8 = 24.5 GB
    
    On NVSwitch (900 GB/s): 27 ms
    On InfiniBand (50 GB/s): 490 ms per gradient sync!
""")


# =============================================================================
# SECTION 5: PROCESS GROUPS AND BACKENDS
# =============================================================================
"""
PROCESS GROUPS:
═══════════════

A process group defines a subset of processes that can communicate.

DEFAULT GROUP:
    - Created by init_process_group()
    - Contains all processes
    - Used for most collective operations

CUSTOM GROUPS:
    - Create subsets for hierarchical communication
    - Useful for hybrid parallelism
    
    Example: 2 nodes × 8 GPUs
    
    # Group for all 16 GPUs
    world_group = dist.group.WORLD
    
    # Group for 8 GPUs on each node
    node_group = dist.new_group([0,1,2,3,4,5,6,7])  # ranks on node 0
    
    # Group for same position across nodes
    cross_node_group = dist.new_group([0, 8])  # GPU 0 on each node

BACKENDS:

1. NCCL (Recommended for GPU):
   - NVIDIA's optimized library
   - Best performance for GPU-to-GPU
   - Supports all collective operations
   - Uses NVLink, NVSwitch, GPUDirect RDMA
   
2. Gloo:
   - CPU-based communication
   - Works without NVIDIA GPUs
   - Good for CPU training or heterogeneous
   
3. MPI:
   - Traditional HPC communication
   - Requires separate MPI installation
   - Useful if already using MPI
"""


def process_group_explanation():
    """Explain process groups and backends."""
    print("\n" + "="*70)
    print("PROCESS GROUPS AND BACKENDS")
    print("="*70)
    
    print("""
INITIALIZATION METHODS:
═══════════════════════

1. Environment Variables (recommended for multi-node):
   
   # On each node, set:
   export MASTER_ADDR=<master_ip>
   export MASTER_PORT=<port>
   export WORLD_SIZE=<total_gpus>
   export RANK=<global_rank>
   export LOCAL_RANK=<gpu_id_on_node>
   
   # In code:
   dist.init_process_group(backend='nccl')

2. TCP Store:
   
   dist.init_process_group(
       backend='nccl',
       init_method='tcp://master_ip:port',
       rank=rank,
       world_size=world_size
   )

3. File Store (shared filesystem):
   
   dist.init_process_group(
       backend='nccl',
       init_method='file:///shared/path/sync_file',
       rank=rank,
       world_size=world_size
   )


TORCHRUN LAUNCH (RECOMMENDED):
══════════════════════════════

Single node:
    torchrun --nproc_per_node=8 train.py

Multi-node:
    # On node 0:
    torchrun --nnodes=2 --nproc_per_node=8 \\
             --node_rank=0 --master_addr=node0_ip \\
             --master_port=12355 train.py
    
    # On node 1:
    torchrun --nnodes=2 --nproc_per_node=8 \\
             --node_rank=1 --master_addr=node0_ip \\
             --master_port=12355 train.py

In your code:
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
""")


# =============================================================================
# SECTION 6: DDP BEST PRACTICES
# =============================================================================

def ddp_best_practices():
    """DDP best practices and common issues."""
    print("\n" + "="*70)
    print("DDP BEST PRACTICES")
    print("="*70)
    
    print("""
1. USE DISTRIBUTEDSAMPLER:
══════════════════════════

WRONG (same data on all ranks):
    dataloader = DataLoader(dataset, shuffle=True)

CORRECT (different data per rank):
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler)
    
    # IMPORTANT: Set epoch for proper shuffling!
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            ...


2. SAVE/LOAD CHECKPOINTS PROPERLY:
══════════════════════════════════

SAVING (only on rank 0):
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), 'model.pt')
    
    # Wait for save to complete
    dist.barrier()

LOADING:
    # Load to CPU first, then move
    map_location = {'cuda:0': f'cuda:{rank}'}
    state_dict = torch.load('model.pt', map_location=map_location)
    model.load_state_dict(state_dict)
    
    # Then wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])


3. GRADIENT ACCUMULATION:
═════════════════════════

WRONG (syncs every micro-batch):
    for micro_batch in micro_batches:
        loss = model(micro_batch)
        loss.backward()

CORRECT (sync only on last micro-batch):
    for i, micro_batch in enumerate(micro_batches):
        # Don't sync until last micro-batch
        context = ddp_model.no_sync() if i < len(micro_batches)-1 else nullcontext()
        
        with context:
            loss = model(micro_batch)
            (loss / accumulation_steps).backward()
    
    optimizer.step()


4. AVOID UNUSED PARAMETERS:
═══════════════════════════

If some parameters don't receive gradients, DDP will hang!

Solution 1: find_unused_parameters=True (slower)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

Solution 2: Ensure all parameters are used or frozen
    for param in unused_params:
        param.requires_grad = False


5. PIN MEMORY FOR DATALOADER:
═════════════════════════════

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,  # Faster CPU→GPU transfer
    persistent_workers=True,  # Keep workers alive between epochs
)


6. BENCHMARK COMMUNICATION:
═══════════════════════════

# Set NCCL debug for performance analysis
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

# Profile with torch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Training step
    ...
print(prof.key_averages().table())
""")


# =============================================================================
# SECTION 7: DEBUGGING DDP ISSUES
# =============================================================================

def ddp_debugging_guide():
    """Guide for debugging DDP issues."""
    print("\n" + "="*70)
    print("DEBUGGING DDP ISSUES")
    print("="*70)
    
    print("""
COMMON ISSUES AND SOLUTIONS:
════════════════════════════

1. HANGS DURING TRAINING:
   
   Symptoms: Training freezes, no error message
   
   Causes:
   - Unused parameters (different gradients per rank)
   - Different control flow per rank
   - Deadlock in data loading
   
   Debug:
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   
   Solutions:
   - Use find_unused_parameters=True
   - Ensure same batch size on all ranks
   - Check sampler is giving data to all ranks


2. CUDA OUT OF MEMORY:
   
   Symptoms: OOM on some ranks
   
   Causes:
   - Uneven batch sizes
   - One rank doing extra work
   
   Debug:
   for rank in range(world_size):
       print(f"Rank {rank}: {torch.cuda.memory_allocated()}")
   
   Solutions:
   - Ensure even batch distribution
   - Use gradient checkpointing


3. GRADIENT NAN/INF:
   
   Symptoms: Loss becomes NaN
   
   Debug:
   torch.autograd.set_detect_anomaly(True)
   
   Solutions:
   - Gradient clipping
   - Lower learning rate
   - Check data for NaN


4. SLOW TRAINING:
   
   Symptoms: Multi-GPU slower than expected
   
   Debug:
   # Profile with nsys
   nsys profile -o profile python train.py
   
   Causes:
   - Data loading bottleneck (workers too few)
   - Small batch size (communication overhead)
   - Wrong NCCL settings
   
   Solutions:
   - Increase batch size
   - More dataloader workers
   - Check NVLink is being used


5. MODELS DIVERGE:
   
   Symptoms: Different losses on different ranks
   
   Causes:
   - Different random seeds
   - Dropout without sync
   
   Solutions:
   - Set same seed: torch.manual_seed(42)
   - Sync batch norm: nn.SyncBatchNorm.convert_sync_batchnorm(model)


USEFUL ENVIRONMENT VARIABLES:
═════════════════════════════

# NCCL debugging
export NCCL_DEBUG=INFO           # Basic info
export NCCL_DEBUG=WARN           # Warnings only
export NCCL_DEBUG_SUBSYS=ALL     # All subsystems

# PyTorch debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Detailed logs
export TORCH_SHOW_CPP_STACKTRACES=1    # C++ stack traces

# NCCL performance
export NCCL_IB_DISABLE=0         # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2      # GPUDirect RDMA level
export NCCL_SOCKET_IFNAME=eth0   # Network interface
""")


# =============================================================================
# SECTION 8: DDP WITH MIXED PRECISION
# =============================================================================

def ddp_with_amp_example():
    """DDP with automatic mixed precision."""
    print("\n" + "="*70)
    print("DDP WITH MIXED PRECISION")
    print("="*70)
    
    print("""
COMBINING DDP WITH AMP:
═══════════════════════

from torch.cuda.amp import autocast, GradScaler

def train_step(model, data, target, optimizer, scaler):
    optimizer.zero_grad()
    
    # Forward with autocast
    with autocast(dtype=torch.float16):
        output = model(data)
        loss = F.cross_entropy(output, target)
    
    # Scaled backward
    scaler.scale(loss).backward()
    
    # Unscale, clip, step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss

# Main training loop
model = SimpleDDPModel().to(device)
ddp_model = DDP(model, device_ids=[rank])
optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
scaler = GradScaler()

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for data, target in dataloader:
        loss = train_step(ddp_model, data, target, optimizer, scaler)


KEY POINTS:
═══════════

1. GradScaler is per-process (not shared)
2. Scaler state should be saved/loaded with checkpoint
3. DDP automatically handles AMP gradients
4. All-reduce happens on scaled gradients, unscaled after
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DATA PARALLELISM AND DDP")
    print("="*70)
    
    # Demonstrate all-reduce
    demonstrate_all_reduce()
    
    # Process groups explanation
    process_group_explanation()
    
    # Best practices
    ddp_best_practices()
    
    # Debugging guide
    ddp_debugging_guide()
    
    # DDP with AMP
    ddp_with_amp_example()
    
    # Note about running DDP
    print("\n" + "="*70)
    print("TO RUN DDP TRAINING:")
    print("="*70)
    print("""
# Single node, 4 GPUs:
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes, 8 GPUs each):
# On node 0:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
         --master_addr=<node0_ip> --master_port=12355 train.py

# On node 1:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \\
         --master_addr=<node0_ip> --master_port=12355 train.py
""")
    
    print("\n" + "="*70)
    print("DDP MODULE COMPLETE")
    print("="*70)
