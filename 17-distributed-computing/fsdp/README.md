# FSDP: Fully Sharded Data Parallel

PyTorch's native ZeRO-style distributed training.

## The Problem

Data Parallel (DDP):
- Each GPU has full model copy
- Memory: O(model_size × num_gpus)
- Doesn't scale for large models

## FSDP Solution

Shard everything across GPUs:
- Parameters
- Gradients  
- Optimizer states

Memory per GPU: O(model_size / num_gpus)

## How It Works

### Forward Pass
1. All-gather: Collect full parameter
2. Compute forward
3. Discard non-owned parameters

### Backward Pass
1. All-gather: Collect full parameter
2. Compute backward
3. Reduce-scatter: Sum and shard gradients
4. Discard non-owned parameters

## Basic Usage

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
)

# Training loop is same as DDP
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

## Sharding Strategies

| Strategy | Memory | Speed |
|----------|--------|-------|
| FULL_SHARD | Lowest | Slowest |
| SHARD_GRAD_OP | Medium | Medium |
| NO_SHARD | Highest (DDP) | Fastest |

## Comparison with DeepSpeed ZeRO

| Feature | FSDP | DeepSpeed |
|---------|------|-----------|
| Native PyTorch | ✅ | ❌ |
| CPU Offload | Limited | Full |
| NVMe Offload | ❌ | ✅ |
| Ease of use | Higher | Lower |

## Best Practices
1. Use BF16 mixed precision
2. Enable activation checkpointing
3. Tune sharding strategy for your model size
