# LoRA: Low-Rank Adaptation

Efficient fine-tuning by adding small trainable matrices.

## The Problem

Full fine-tuning of LLMs:
- Requires storing full gradients
- Updates all parameters
- Memory-intensive

## LoRA Solution

Instead of updating W directly:
```
W_new = W_old + ΔW
```

Decompose ΔW as low-rank:
```
W_new = W_old + BA
```
Where:
- B: d × r matrix (r << d)
- A: r × k matrix
- r: rank (typically 8-64)

## Memory Savings

For a 7B parameter model:
- Full fine-tuning: ~28GB (FP32 gradients)
- LoRA (r=8): ~0.1GB of trainable params

## Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x, original_output):
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return original_output + lora_output
```

## Where to Apply LoRA

Typically applied to:
- Query/Key/Value projections in attention
- Sometimes MLP layers
- Not layer norms

## QLoRA

Combines LoRA with quantization:
1. Quantize base model to 4-bit
2. Add LoRA adapters in FP16/BF16
3. Train only LoRA adapters

## Reference
- `unsloth/` - Optimized LoRA training
- Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
