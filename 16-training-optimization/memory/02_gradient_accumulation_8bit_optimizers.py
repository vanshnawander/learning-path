"""
Gradient Accumulation and 8-bit Optimizers
============================================

This module covers two critical memory optimization techniques:
1. Gradient Accumulation - Simulate larger batches
2. 8-bit Optimizers - Reduce optimizer state memory

Key Topics:
1. Gradient Accumulation Theory and Implementation
2. Effective Batch Size Calculation
3. 8-bit Optimizer Theory (bitsandbytes)
4. Dynamic Quantization of Optimizer States
5. Combining Techniques for Maximum Memory Efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math
from typing import Optional, Dict, Any

# =============================================================================
# SECTION 1: GRADIENT ACCUMULATION FUNDAMENTALS
# =============================================================================
"""
GRADIENT ACCUMULATION:
══════════════════════

Problem: Limited GPU memory prevents large batch sizes

Solution: Accumulate gradients over multiple mini-batches

How it works:
    Instead of: 
        loss = model(batch_32)
        loss.backward()
        optimizer.step()
        
    Do:
        for i in range(4):
            loss = model(batch_8) / 4  # Scale loss!
            loss.backward()            # Accumulates gradients
        optimizer.step()              # Update with accumulated gradients
        optimizer.zero_grad()

Mathematical Equivalence:
    
    Batch 32: ∇L = (1/32) Σᵢ ∇Lᵢ
    
    4 × Batch 8: ∇L = (1/4) Σⱼ [(1/8) Σᵢ ∇Lᵢⱼ]
                    = (1/32) Σᵢ ∇Lᵢ  ✓ Same!

Key Points:
    - Scale loss by 1/accumulation_steps
    - Only step optimizer after all accumulations
    - Only zero gradients after optimizer step
"""


class GradientAccumulationTrainer:
    """
    Training loop with gradient accumulation support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        
        # For mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Track current accumulation step
        self.current_step = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step with gradient accumulation.
        
        Returns loss value (unscaled).
        """
        self.model.train()
        
        # Forward pass
        if self.use_amp:
            with autocast():
                outputs = self.model(batch['input_ids'])
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch['labels'].view(-1)
                )
        else:
            outputs = self.model(batch['input_ids'])
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch['labels'].view(-1)
            )
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.current_step += 1
        
        # Check if we should update
        if self.current_step >= self.accumulation_steps:
            self._optimizer_step()
            self.current_step = 0
        
        return loss.item()
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscale for gradient clipping
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()


def demonstrate_gradient_accumulation():
    """Demonstrate gradient accumulation equivalence."""
    print("\n" + "="*70)
    print("GRADIENT ACCUMULATION DEMONSTRATION")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Simple model
    model = nn.Linear(100, 10)
    
    # Create full batch
    x_full = torch.randn(32, 100)
    y_full = torch.randint(0, 10, (32,))
    
    # Method 1: Full batch gradient
    model_full = nn.Linear(100, 10)
    model_full.load_state_dict(model.state_dict())
    
    loss_full = F.cross_entropy(model_full(x_full), y_full)
    loss_full.backward()
    grad_full = model_full.weight.grad.clone()
    
    print(f"Full batch (32):")
    print(f"  Loss: {loss_full.item():.4f}")
    print(f"  Gradient norm: {grad_full.norm():.4f}")
    
    # Method 2: Accumulated gradients (4 × batch 8)
    model_accum = nn.Linear(100, 10)
    model_accum.load_state_dict(model.state_dict())
    
    accumulation_steps = 4
    batch_size = 8
    total_loss = 0
    
    for i in range(accumulation_steps):
        start = i * batch_size
        end = start + batch_size
        x_mini = x_full[start:end]
        y_mini = y_full[start:end]
        
        loss_mini = F.cross_entropy(model_accum(x_mini), y_mini)
        scaled_loss = loss_mini / accumulation_steps
        scaled_loss.backward()
        total_loss += loss_mini.item()
    
    grad_accum = model_accum.weight.grad.clone()
    
    print(f"\nAccumulated (4 × 8):")
    print(f"  Average loss: {total_loss / accumulation_steps:.4f}")
    print(f"  Gradient norm: {grad_accum.norm():.4f}")
    
    # Compare
    grad_diff = (grad_full - grad_accum).abs().max()
    print(f"\nGradient difference: {grad_diff:.2e}")
    print(f"Gradients equivalent: {grad_diff < 1e-5}")


# =============================================================================
# SECTION 2: EFFECTIVE BATCH SIZE
# =============================================================================

def effective_batch_size_guide():
    """Guide for calculating effective batch size."""
    print("\n" + "="*70)
    print("EFFECTIVE BATCH SIZE CALCULATION")
    print("="*70)
    
    print("""
EFFECTIVE BATCH SIZE FORMULA:
═════════════════════════════

effective_batch_size = micro_batch_size × accumulation_steps × num_gpus

Examples:
┌─────────────────────────────────────────────────────────────────────┐
│ Micro Batch │ Accum Steps │ GPUs │ Effective Batch                  │
├─────────────────────────────────────────────────────────────────────┤
│     8       │      4      │   1  │    32                            │
│     4       │      8      │   1  │    32                            │
│     8       │      4      │   4  │   128                            │
│    16       │      2      │   8  │   256                            │
└─────────────────────────────────────────────────────────────────────┘


LEARNING RATE SCALING:
══════════════════════

When changing batch size, adjust learning rate:

Linear Scaling (common):
    lr_new = lr_base × (effective_batch_new / effective_batch_base)
    
Square Root Scaling (more conservative):
    lr_new = lr_base × sqrt(effective_batch_new / effective_batch_base)

Example:
    Base: batch=32, lr=1e-4
    New:  batch=256 (8× larger)
    
    Linear:    lr = 1e-4 × 8 = 8e-4
    Sqrt:      lr = 1e-4 × √8 ≈ 2.8e-4


WARMUP ADJUSTMENT:
══════════════════

Warmup steps should be adjusted for effective batch size:

    warmup_steps_new = warmup_steps_base × (batch_base / effective_batch_new)

Example:
    Base: batch=32, warmup=1000 steps
    New:  batch=256
    
    warmup_new = 1000 × (32/256) = 125 steps


CHOOSING ACCUMULATION STEPS:
════════════════════════════

Factors to consider:
1. Target effective batch size (determined by training recipe)
2. Available GPU memory (determines micro batch)
3. Training speed (more accumulation = more memory reads)

Rule of thumb:
- Start with accumulation_steps = target_batch / max_micro_batch
- If training is unstable, reduce accumulation and use gradient clipping
- For very long sequences, use smaller micro batch with more accumulation
""")


# =============================================================================
# SECTION 3: 8-BIT OPTIMIZERS
# =============================================================================
"""
8-BIT OPTIMIZERS (bitsandbytes):
════════════════════════════════

Problem: Adam optimizer stores 2 states per parameter
    - First moment (m): FP32
    - Second moment (v): FP32
    - Total: 8 bytes per parameter!

For 7B model:
    - Model: 14 GB (FP16)
    - Optimizer states: 56 GB (!)
    
Solution: Quantize optimizer states to 8-bit

8-bit Adam:
    - m, v stored in INT8 (1 byte each)
    - Per-block scaling factors
    - Dynamic quantization during training
    
Memory savings: 4x for optimizer states!
    - 56 GB → 14 GB
"""


def eight_bit_optimizer_explanation():
    """Explain 8-bit optimizer implementation."""
    print("\n" + "="*70)
    print("8-BIT OPTIMIZERS (BITSANDBYTES)")
    print("="*70)
    
    print("""
ADAM OPTIMIZER MEMORY:
══════════════════════

Standard Adam stores per parameter:
    - m (first moment): FP32 (4 bytes)
    - v (second moment): FP32 (4 bytes)
    - Total: 8 bytes per parameter

7B Parameter Model:
    - Parameters: 7 billion
    - Optimizer states: 7B × 8 = 56 GB
    - This often exceeds GPU memory!


8-BIT QUANTIZATION:
═══════════════════

Insight: Optimizer states don't need full precision

Quantization scheme:
1. Divide tensor into blocks (e.g., 2048 elements)
2. For each block:
   - Find max absolute value
   - Compute scale = max_abs / 127
   - Quantize: q = round(value / scale)
   - Store: INT8 values + FP32 scale

Memory per block:
    - Standard: 2048 × 4 = 8192 bytes
    - 8-bit: 2048 × 1 + 4 = 2052 bytes
    - Savings: ~4x


DYNAMIC QUANTIZATION:
═════════════════════

During training:
1. Dequantize state: m = m_q × scale
2. Perform Adam update in FP32
3. Requantize state: m_q = round(m / new_scale)

This happens every step, but overhead is minimal.


USAGE:
══════

import bitsandbytes as bnb

# Replace standard Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

# Or AdamW (more common for LLMs)
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# With different settings
optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
    min_8bit_size=4096,      # Only 8-bit for params > this size
    percentile_clipping=5,    # Gradient clipping percentile
)


STABLE EMBEDDINGS:
══════════════════

Embeddings can be unstable with 8-bit optimizers.
Use StableEmbedding for better results:

from bitsandbytes.nn import StableEmbedding

# Replace nn.Embedding
# self.embed = nn.Embedding(vocab_size, hidden_size)
self.embed = StableEmbedding(vocab_size, hidden_size)


MEMORY COMPARISON:
══════════════════

7B Model Training Memory:
┌────────────────────────────────────────────────────────────────┐
│ Component              │ Standard Adam │ 8-bit Adam           │
├────────────────────────────────────────────────────────────────┤
│ Model (FP16)           │    14 GB      │    14 GB             │
│ Gradients (FP16)       │    14 GB      │    14 GB             │
│ Optimizer states       │    56 GB      │    14 GB             │
│ Activations            │   ~30 GB      │   ~30 GB             │
├────────────────────────────────────────────────────────────────┤
│ TOTAL                  │  ~114 GB      │   ~72 GB             │
│ Savings                │      -        │    42 GB (37%)       │
└────────────────────────────────────────────────────────────────┘
""")


def demonstrate_8bit_optimizer():
    """Demonstrate 8-bit optimizer usage."""
    print("\n" + "="*70)
    print("8-BIT OPTIMIZER DEMONSTRATION")
    print("="*70)
    
    try:
        import bitsandbytes as bnb
        bnb_available = True
    except ImportError:
        bnb_available = False
        print("bitsandbytes not installed. Install with: pip install bitsandbytes")
    
    if not bnb_available:
        print("\nShowing usage patterns (bitsandbytes not installed):")
        print("""
# Installation
pip install bitsandbytes

# Basic usage
import bitsandbytes as bnb

model = YourModel()
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# Training loop remains the same
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
""")
        return
    
    # If bitsandbytes is available
    print("\nbitsandbytes available!")
    
    # Compare memory usage
    hidden_size = 4096
    model = nn.Linear(hidden_size, hidden_size)
    
    # Standard optimizer
    opt_standard = torch.optim.AdamW(model.parameters())
    
    # 8-bit optimizer
    opt_8bit = bnb.optim.AdamW8bit(model.parameters())
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Standard Adam state size: ~{sum(p.numel() for p in model.parameters()) * 8 / 1e6:.1f} MB")
    print(f"8-bit Adam state size: ~{sum(p.numel() for p in model.parameters()) * 2 / 1e6:.1f} MB")


# =============================================================================
# SECTION 4: PAGED OPTIMIZERS
# =============================================================================

def paged_optimizers_explanation():
    """Explain paged optimizers for memory management."""
    print("\n" + "="*70)
    print("PAGED OPTIMIZERS")
    print("="*70)
    
    print("""
PAGED OPTIMIZERS:
═════════════════

Problem: Even with 8-bit states, optimizer memory can exceed GPU

Solution: Paged memory - offload to CPU when needed

How it works:
1. Optimizer states allocated in "paged" memory
2. When GPU memory is full, pages transferred to CPU
3. When needed, pages transferred back to GPU
4. Automatic, transparent to user


USAGE:
══════

import bitsandbytes as bnb

# Paged 8-bit AdamW
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,
)

# Paged 32-bit AdamW (less compression but still paged)
optimizer = bnb.optim.PagedAdamW32bit(
    model.parameters(),
    lr=2e-4,
)


WHEN TO USE:
════════════

┌──────────────────────────────────────────────────────────────────┐
│ Scenario                      │ Recommendation                   │
├──────────────────────────────────────────────────────────────────┤
│ Plenty of GPU memory          │ Standard AdamW or Adam8bit       │
│ Tight on memory               │ Adam8bit                         │
│ Very tight / OOM during peaks │ PagedAdam8bit                    │
│ CPU memory also limited       │ Reduce model/batch size          │
└──────────────────────────────────────────────────────────────────┘


PERFORMANCE IMPACT:
═══════════════════

Paging adds overhead from CPU↔GPU transfers:
- Minimal impact if pages rarely swapped
- Noticeable slowdown if constantly swapping
- Use only when necessary

Best practice:
- Start with Adam8bit
- Switch to PagedAdam8bit only if OOM
- Consider gradient checkpointing first (often better tradeoff)
""")


# =============================================================================
# SECTION 5: COMBINING TECHNIQUES
# =============================================================================

def combined_optimization_example():
    """Show how to combine all memory optimization techniques."""
    print("\n" + "="*70)
    print("COMBINING MEMORY OPTIMIZATION TECHNIQUES")
    print("="*70)
    
    print("""
MAXIMUM MEMORY EFFICIENCY SETUP:
════════════════════════════════

Combine all techniques for training large models on limited hardware:

```python
import torch
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes as bnb

# 1. Load model in 4-bit (QLoRA)
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Apply LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# 4. Use 8-bit optimizer
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-4,
)

# 5. Gradient accumulation + mixed precision
accumulation_steps = 16
scaler = GradScaler()

for step, batch in enumerate(dataloader):
    with autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```


MEMORY BREAKDOWN (7B Model):
════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│ Optimization             │ Memory Saved    │ Cumulative          │
├──────────────────────────────────────────────────────────────────┤
│ Full FP32 training       │ (baseline)      │ ~120 GB             │
│ + BF16 mixed precision   │ -30 GB          │ ~90 GB              │
│ + 4-bit base weights     │ -10 GB          │ ~80 GB → ~6 GB      │
│ + LoRA (0.1% trainable)  │ -50 GB optim    │ ~15 GB              │
│ + Gradient checkpointing │ -5 GB activ     │ ~10 GB              │
│ + 8-bit optimizer        │ -2 GB           │ ~8 GB               │
│ + Gradient accumulation  │ smaller batches │ ~6 GB               │
└──────────────────────────────────────────────────────────────────┘

Result: Train 7B model on single 8GB GPU!


RECOMMENDED COMBINATIONS:
═════════════════════════

Consumer GPU (8-16 GB):
    4-bit QLoRA + Checkpointing + 8-bit Adam + Accum 8-16

Mid-range GPU (24-48 GB):
    BF16 LoRA + Checkpointing + 8-bit Adam + Accum 4-8
    
High-end GPU (80 GB):
    BF16 Full/LoRA + Optional Checkpointing + Standard Adam
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GRADIENT ACCUMULATION AND 8-BIT OPTIMIZERS")
    print("="*70)
    
    # Gradient accumulation
    demonstrate_gradient_accumulation()
    
    # Effective batch size
    effective_batch_size_guide()
    
    # 8-bit optimizers
    eight_bit_optimizer_explanation()
    demonstrate_8bit_optimizer()
    
    # Paged optimizers
    paged_optimizers_explanation()
    
    # Combined techniques
    combined_optimization_example()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
