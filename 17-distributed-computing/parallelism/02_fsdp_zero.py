"""
Fully Sharded Data Parallel (FSDP) and ZeRO
============================================

This module provides comprehensive coverage of FSDP and ZeRO,
the memory-efficient distributed training strategies.

Key Topics:
1. The Memory Wall Problem
2. ZeRO Stages (0, 1, 2, 3)
3. PyTorch FSDP Architecture
4. Sharding Strategies
5. FSDP vs DeepSpeed Comparison
6. Practical Configuration
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import os
from typing import Optional, Set, Type
from functools import partial

# =============================================================================
# SECTION 1: THE MEMORY WALL PROBLEM
# =============================================================================
"""
WHY WE NEED FSDP/ZeRO:
══════════════════════

Standard DDP replicates everything on each GPU:
    - Model parameters
    - Gradients  
    - Optimizer states

Memory per GPU for 7B model with Adam:
┌───────────────────────────────────────────────────────────────────────┐
│ Component              │ Bytes per Param │ 7B Params │ Total         │
├───────────────────────────────────────────────────────────────────────┤
│ Parameters (FP16)      │ 2               │ 7B        │ 14 GB         │
│ Gradients (FP16)       │ 2               │ 7B        │ 14 GB         │
│ Optimizer momentum     │ 4 (FP32)        │ 7B        │ 28 GB         │
│ Optimizer variance     │ 4 (FP32)        │ 7B        │ 28 GB         │
│ Master weights (FP32)  │ 4               │ 7B        │ 28 GB         │
├───────────────────────────────────────────────────────────────────────┤
│ TOTAL                  │ 16              │ 7B        │ 112 GB        │
└───────────────────────────────────────────────────────────────────────┘

Single A100 (80GB) can't even hold this!
And this is REPLICATED on every GPU in DDP!

INSIGHT: Most of this memory is REDUNDANT
    - Same model on every GPU
    - Same optimizer states on every GPU
    - Only gradients differ (temporarily)

SOLUTION: SHARD (partition) instead of REPLICATE
    - Each GPU holds only 1/N of the data
    - Gather when needed, discard after use
    - Trade compute for memory
"""


def memory_analysis():
    """Analyze memory requirements for different strategies."""
    print("\n" + "="*70)
    print("MEMORY ANALYSIS: DDP vs FSDP/ZeRO")
    print("="*70)
    
    model_params = 7e9  # 7B
    num_gpus = 8
    
    # DDP memory per GPU
    ddp_params = model_params * 2  # FP16
    ddp_grads = model_params * 2   # FP16
    ddp_opt_m = model_params * 4   # FP32 momentum
    ddp_opt_v = model_params * 4   # FP32 variance
    ddp_master = model_params * 4  # FP32 master weights
    ddp_total = ddp_params + ddp_grads + ddp_opt_m + ddp_opt_v + ddp_master
    
    # ZeRO Stage 1: Shard optimizer states
    z1_params = model_params * 2
    z1_grads = model_params * 2
    z1_opt = (ddp_opt_m + ddp_opt_v + ddp_master) / num_gpus
    z1_total = z1_params + z1_grads + z1_opt
    
    # ZeRO Stage 2: Shard optimizer + gradients
    z2_params = model_params * 2
    z2_grads = model_params * 2 / num_gpus
    z2_opt = (ddp_opt_m + ddp_opt_v + ddp_master) / num_gpus
    z2_total = z2_params + z2_grads + z2_opt
    
    # ZeRO Stage 3 / FSDP: Shard everything
    z3_params = model_params * 2 / num_gpus
    z3_grads = model_params * 2 / num_gpus
    z3_opt = (ddp_opt_m + ddp_opt_v + ddp_master) / num_gpus
    z3_total = z3_params + z3_grads + z3_opt
    
    print(f"\nModel: 7B parameters, {num_gpus} GPUs")
    print(f"\n{'Strategy':<20} {'Per-GPU Memory':<20} {'Reduction':<15}")
    print("-" * 55)
    print(f"{'DDP':<20} {ddp_total/1e9:.1f} GB{' '*10} 1x (baseline)")
    print(f"{'ZeRO Stage 1':<20} {z1_total/1e9:.1f} GB{' '*10} {ddp_total/z1_total:.1f}x")
    print(f"{'ZeRO Stage 2':<20} {z2_total/1e9:.1f} GB{' '*10} {ddp_total/z2_total:.1f}x")
    print(f"{'ZeRO Stage 3/FSDP':<20} {z3_total/1e9:.1f} GB{' '*10} {ddp_total/z3_total:.1f}x")


# =============================================================================
# SECTION 2: ZERO STAGES
# =============================================================================
"""
ZeRO (Zero Redundancy Optimizer) STAGES:
════════════════════════════════════════

ZeRO progressively shards more state to reduce memory.

STAGE 0 (DDP Baseline):
    - Replicate: Parameters, Gradients, Optimizer states
    - Memory: 16× model size
    - Communication: All-reduce gradients
    
STAGE 1 (Optimizer State Sharding):
    - Replicate: Parameters, Gradients
    - Shard: Optimizer states
    - Memory: 4× + 12×/N (e.g., 4× + 1.5× for 8 GPUs)
    - Communication: All-reduce gradients + reduce-scatter optimizer

STAGE 2 (+ Gradient Sharding):
    - Replicate: Parameters
    - Shard: Gradients, Optimizer states
    - Memory: 2× + 14×/N
    - Communication: Reduce-scatter gradients, gather parameters

STAGE 3 (+ Parameter Sharding):
    - Shard: Parameters, Gradients, Optimizer states
    - Memory: 16×/N (everything divided!)
    - Communication: All-gather parameters (forward+backward)

COMMUNICATION PATTERNS:
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage │ Forward          │ Backward              │ Optimizer Step       │
├──────────────────────────────────────────────────────────────────────────┤
│ 0     │ -                │ All-reduce gradients  │ Local update         │
│ 1     │ -                │ All-reduce gradients  │ Reduce-scatter→local │
│ 2     │ -                │ Reduce-scatter grads  │ Local update         │
│ 3     │ All-gather params│ All-gather + RS grads │ Local update         │
└──────────────────────────────────────────────────────────────────────────┘

STAGE 3 IN DETAIL:
    
    Forward Pass:
        1. Before layer computation, all-gather layer parameters
        2. Compute forward pass
        3. Free gathered parameters (only keep shard)
    
    Backward Pass:
        1. All-gather layer parameters (recompute)
        2. Compute gradients
        3. Reduce-scatter gradients (each GPU gets shard)
        4. Free gathered parameters
    
    Optimizer Step:
        1. Each GPU updates its shard of parameters
        2. No communication needed!
"""


def zero_stages_visualization():
    """Visualize ZeRO stages."""
    print("\n" + "="*70)
    print("ZeRO STAGES VISUALIZATION")
    print("="*70)
    
    print("""
MEMORY LAYOUT (4 GPUs, simplified):

DDP (Stage 0):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
├────────┤ ├────────┤ ├────────┤ ├────────┤
│ Params │ │ Params │ │ Params │ │ Params │ ← Same
│ Grads  │ │ Grads  │ │ Grads  │ │ Grads  │ ← Same (after AR)
│ Opt    │ │ Opt    │ │ Opt    │ │ Opt    │ ← Same
└────────┘ └────────┘ └────────┘ └────────┘

ZeRO Stage 1 (Optimizer Sharding):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
├────────┤ ├────────┤ ├────────┤ ├────────┤
│ Params │ │ Params │ │ Params │ │ Params │ ← Same
│ Grads  │ │ Grads  │ │ Grads  │ │ Grads  │ ← Same
│ Opt₀   │ │ Opt₁   │ │ Opt₂   │ │ Opt₃   │ ← SHARDED!
└────────┘ └────────┘ └────────┘ └────────┘

ZeRO Stage 2 (+ Gradient Sharding):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
├────────┤ ├────────┤ ├────────┤ ├────────┤
│ Params │ │ Params │ │ Params │ │ Params │ ← Same
│ Grad₀  │ │ Grad₁  │ │ Grad₂  │ │ Grad₃  │ ← SHARDED!
│ Opt₀   │ │ Opt₁   │ │ Opt₂   │ │ Opt₃   │ ← Sharded
└────────┘ └────────┘ └────────┘ └────────┘

ZeRO Stage 3 / FSDP (Full Sharding):
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │ │ GPU 3  │
├────────┤ ├────────┤ ├────────┤ ├────────┤
│ Param₀ │ │ Param₁ │ │ Param₂ │ │ Param₃ │ ← SHARDED!
│ Grad₀  │ │ Grad₁  │ │ Grad₂  │ │ Grad₃  │ ← Sharded
│ Opt₀   │ │ Opt₁   │ │ Opt₂   │ │ Opt₃   │ ← Sharded
└────────┘ └────────┘ └────────┘ └────────┘

All-Gather for Forward/Backward (Stage 3):
    Before computing layer L:
    [P₀_L] + [P₁_L] + [P₂_L] + [P₃_L] → [P_L, P_L, P_L, P_L]
    
    Each GPU now has full layer L parameters
    Compute forward/backward
    Discard non-local shards
""")


# =============================================================================
# SECTION 3: PYTORCH FSDP
# =============================================================================
"""
PyTorch FSDP (Fully Sharded Data Parallel):
═══════════════════════════════════════════

FSDP is PyTorch's native implementation of ZeRO Stage 3.

KEY CONCEPTS:

1. FSDP Units:
   - Model is divided into "FSDP units"
   - Each unit is sharded independently
   - Units are all-gathered as needed
   
2. Wrapping:
   - Decide which modules become FSDP units
   - More units = less peak memory, more communication
   - Typically wrap transformer layers

3. Sharding Strategy:
   - FULL_SHARD: ZeRO-3 (shard everything)
   - SHARD_GRAD_OP: ZeRO-2 (shard grads+optimizer)
   - NO_SHARD: DDP-like (for comparison)
   - HYBRID_SHARD: Full shard intra-node, replicate inter-node
"""


class TransformerBlock(nn.Module):
    """Transformer block for FSDP wrapping demonstration."""
    
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerModel(nn.Module):
    """Simple transformer model for FSDP demonstration."""
    
    def __init__(self, hidden_size: int = 1024, num_layers: int = 12):
        super().__init__()
        self.embed = nn.Embedding(50000, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, 50000)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def fsdp_wrapping_example():
    """Demonstrate FSDP wrapping strategies."""
    print("\n" + "="*70)
    print("FSDP WRAPPING STRATEGIES")
    print("="*70)
    
    print("""
WRAPPING APPROACHES:
════════════════════

1. AUTO WRAP (Recommended):
   
   # Wrap transformer blocks automatically
   auto_wrap_policy = partial(
       transformer_auto_wrap_policy,
       transformer_layer_cls={TransformerBlock}
   )
   
   model = FSDP(
       model,
       auto_wrap_policy=auto_wrap_policy,
       sharding_strategy=ShardingStrategy.FULL_SHARD,
   )

2. SIZE-BASED WRAP:
   
   # Wrap modules larger than min_num_params
   auto_wrap_policy = partial(
       size_based_auto_wrap_policy,
       min_num_params=100_000_000  # 100M params
   )

3. MANUAL WRAP:
   
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = FSDP(TransformerBlock())  # Wrapped
           self.layer2 = FSDP(TransformerBlock())  # Wrapped
           self.head = nn.Linear(...)  # Not wrapped


SHARDING STRATEGIES:
════════════════════

ShardingStrategy.FULL_SHARD (ZeRO-3):
    - Maximum memory efficiency
    - Highest communication overhead
    - Best for very large models

ShardingStrategy.SHARD_GRAD_OP (ZeRO-2):
    - Shard gradients and optimizer only
    - Parameters replicated
    - Less communication, more memory

ShardingStrategy.NO_SHARD:
    - Like DDP (no sharding)
    - Useful for comparison/debugging

ShardingStrategy.HYBRID_SHARD:
    - FULL_SHARD within node (fast NVLink)
    - Replicate across nodes (slow network)
    - Best for multi-node with slow inter-node
""")


# =============================================================================
# SECTION 4: FSDP CONFIGURATION
# =============================================================================

def fsdp_configuration_guide():
    """Comprehensive FSDP configuration guide."""
    print("\n" + "="*70)
    print("FSDP CONFIGURATION GUIDE")
    print("="*70)
    
    print("""
COMPLETE FSDP SETUP:
════════════════════

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 1. MIXED PRECISION
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # Parameters in BF16
    reduce_dtype=torch.bfloat16,     # Gradient reduction in BF16
    buffer_dtype=torch.bfloat16,     # Buffers in BF16
)

# 2. CPU OFFLOAD (optional, for extreme memory saving)
cpu_offload = CPUOffload(offload_params=True)

# 3. AUTO WRAP POLICY
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# 4. BACKWARD PREFETCH
# Prefetch next layer's params during backward
backward_prefetch = BackwardPrefetch.BACKWARD_PRE

# 5. CREATE FSDP MODEL
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=bf16_policy,
    # cpu_offload=cpu_offload,  # Enable for CPU offload
    backward_prefetch=backward_prefetch,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # Limit concurrent all-gathers
    use_orig_params=True,    # For compatibility with torch.compile
)


CONFIGURATION TRADE-OFFS:
═════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│ Option                 │ Memory ↓ │ Speed ↓ │ When to Use              │
├──────────────────────────────────────────────────────────────────────────┤
│ FULL_SHARD             │ Max      │ Some    │ Large models, limited GPU│
│ SHARD_GRAD_OP          │ Medium   │ Less    │ Moderate models          │
│ HYBRID_SHARD           │ Medium   │ Less    │ Multi-node, slow network │
│ cpu_offload=True       │ Extreme  │ High    │ Very large models        │
│ limit_all_gathers=True │ Some     │ Minimal │ Always recommended       │
│ backward_prefetch=PRE  │ Some ↑   │ Less    │ Balance memory/speed     │
└──────────────────────────────────────────────────────────────────────────┘


CHECKPOINTING WITH FSDP:
════════════════════════

# Option 1: Full state dict (for inference/single-GPU loading)
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

full_state_dict_config = FullStateDictConfig(
    offload_to_cpu=True,  # Save memory during gathering
    rank0_only=True,      # Only rank 0 saves
)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "checkpoint.pt")


# Option 2: Sharded state dict (faster, for resuming FSDP training)
from torch.distributed.fsdp import ShardedStateDictConfig

sharded_config = ShardedStateDictConfig(offload_to_cpu=True)

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_config):
    state_dict = model.state_dict()
    # Each rank saves its shard
    torch.save(state_dict, f"checkpoint_rank{rank}.pt")
""")


# =============================================================================
# SECTION 5: FSDP vs DEEPSPEED
# =============================================================================

def fsdp_vs_deepspeed():
    """Compare FSDP and DeepSpeed."""
    print("\n" + "="*70)
    print("FSDP vs DeepSpeed COMPARISON")
    print("="*70)
    
    print("""
FEATURE COMPARISON:
═══════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│ Feature                 │ FSDP (PyTorch)    │ DeepSpeed ZeRO            │
├──────────────────────────────────────────────────────────────────────────┤
│ ZeRO Stage 1            │ ✗                 │ ✓                         │
│ ZeRO Stage 2            │ SHARD_GRAD_OP     │ ✓                         │
│ ZeRO Stage 3            │ FULL_SHARD        │ ✓                         │
│ CPU Offload             │ ✓ (params only)   │ ✓ (params, grads, opt)    │
│ NVMe Offload            │ ✗                 │ ✓ (ZeRO-Infinity)         │
│ Activation Checkpointing│ Via PyTorch       │ Built-in                  │
│ Gradient Accumulation   │ Via PyTorch       │ Built-in (efficient)      │
│ torch.compile           │ ✓ (use_orig_params)│ Limited                  │
│ Mixed Precision         │ ✓                 │ ✓                         │
│ Activation Memory       │ Not sharded       │ Can be partitioned        │
│ Learning Curve          │ Native PyTorch    │ Config-driven             │
│ Maintenance             │ PyTorch team      │ Microsoft                 │
└──────────────────────────────────────────────────────────────────────────┘


WHEN TO USE FSDP:
═════════════════

✓ Pure PyTorch workflow
✓ Using torch.compile
✓ Single-node training
✓ Simpler debugging needs
✓ HuggingFace Trainer (native support)


WHEN TO USE DEEPSPEED:
══════════════════════

✓ Need ZeRO-Infinity (NVMe offload)
✓ Multi-node at scale (100s of GPUs)
✓ Complex parallelism (3D parallelism)
✓ Existing DeepSpeed infrastructure
✓ Need activation partitioning


PERFORMANCE NOTES:
══════════════════

- FSDP and DeepSpeed ZeRO-3 have similar throughput
- DeepSpeed has more mature multi-node optimizations
- FSDP integrates better with torch.compile
- Choice often depends on ecosystem fit
""")


# =============================================================================
# SECTION 6: HYBRID SHARDING (HSDP)
# =============================================================================

def hybrid_sharding_explanation():
    """Explain Hybrid Sharded Data Parallel."""
    print("\n" + "="*70)
    print("HYBRID SHARDING (HSDP)")
    print("="*70)
    
    print("""
THE MULTI-NODE PROBLEM:
═══════════════════════

Full sharding (ZeRO-3) requires all-gather for every layer.

Intra-node (NVSwitch): 900 GB/s → Fast all-gather
Inter-node (InfiniBand): 50 GB/s → Slow all-gather

With 2 nodes × 8 GPUs:
    Full shard across 16 GPUs:
        Every all-gather crosses the slow network!
        Severe communication bottleneck.


HYBRID SHARDING SOLUTION:
═════════════════════════

FULL_SHARD within each node (fast NVLink)
REPLICATE across nodes (avoid slow network)

    Node 0 (8 GPUs)              Node 1 (8 GPUs)
    ┌─────────────────┐          ┌─────────────────┐
    │ FSDP across 8   │          │ FSDP across 8   │
    │ (NVSwitch fast) │          │ (NVSwitch fast) │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             │      Gradient Sync         │
             │◄──────(InfiniBand)────────►│
             │    (like DDP between       │
             │      node replicas)        │

Memory: 16x/8 = 2x per GPU (not 16x/16 = 1x)
Communication: All-gathers stay intra-node


FSDP HYBRID SHARDING CONFIG:
════════════════════════════

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.device_mesh import init_device_mesh

# Create 2D device mesh: (nodes, gpus_per_node)
device_mesh = init_device_mesh("cuda", (2, 8))

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_mesh=device_mesh,
    auto_wrap_policy=auto_wrap_policy,
)

# OR with process groups
intra_node_group = dist.new_group([0,1,2,3,4,5,6,7])  # ranks on same node
inter_node_group = dist.new_group([0, 8])  # same local rank across nodes

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(intra_node_group, inter_node_group),
)
""")


# =============================================================================
# SECTION 7: PRACTICAL FSDP TRAINING
# =============================================================================

def fsdp_training_template():
    """Complete FSDP training template."""
    print("\n" + "="*70)
    print("FSDP TRAINING TEMPLATE")
    print("="*70)
    
    print('''
COMPLETE TRAINING SCRIPT:
═════════════════════════

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
import os

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def train():
    setup()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Create model
    model = TransformerModel().to(local_rank)
    
    # FSDP wrapping
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=local_rank,
        use_orig_params=True,
    )
    
    # Optimizer (after FSDP wrapping!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Data
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        
        for batch in dataloader:
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(**batch).loss
            
            loss.backward()
            
            # Gradient clipping (works with FSDP)
            model.clip_grad_norm_(1.0)
            
            optimizer.step()
        
        # Save checkpoint (rank 0 only with full state dict)
        if rank == 0:
            save_checkpoint(model, optimizer, epoch)
    
    cleanup()

if __name__ == "__main__":
    train()


LAUNCH:
═══════

# Single node, 8 GPUs
torchrun --nproc_per_node=8 train_fsdp.py

# Multi-node
torchrun --nnodes=2 --nproc_per_node=8 \\
         --node_rank=$NODE_RANK \\
         --master_addr=$MASTER_ADDR \\
         --master_port=12355 \\
         train_fsdp.py
''')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FSDP AND ZeRO DEEP DIVE")
    print("="*70)
    
    # Memory analysis
    memory_analysis()
    
    # ZeRO stages visualization
    zero_stages_visualization()
    
    # FSDP wrapping
    fsdp_wrapping_example()
    
    # Configuration guide
    fsdp_configuration_guide()
    
    # FSDP vs DeepSpeed
    fsdp_vs_deepspeed()
    
    # Hybrid sharding
    hybrid_sharding_explanation()
    
    # Training template
    fsdp_training_template()
    
    print("\n" + "="*70)
    print("FSDP/ZeRO MODULE COMPLETE")
    print("="*70)
