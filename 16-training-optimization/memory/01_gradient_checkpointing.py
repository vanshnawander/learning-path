"""
Gradient Checkpointing (Activation Recomputation)
==================================================

This module provides a deep dive into gradient checkpointing,
the technique that trades compute for memory during training.

Key Topics:
1. The Activation Memory Problem
2. Gradient Checkpointing Theory
3. PyTorch Implementation
4. Selective Checkpointing Strategies
5. Memory vs Compute Trade-offs
6. Integration with Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import math
from typing import List, Optional, Callable

# =============================================================================
# SECTION 1: THE ACTIVATION MEMORY PROBLEM
# =============================================================================
"""
WHY ACTIVATIONS USE SO MUCH MEMORY:
═══════════════════════════════════

During forward pass, we compute:
    layer1_out = layer1(input)
    layer2_out = layer2(layer1_out)
    layer3_out = layer3(layer2_out)
    ...

For backward pass, we need the INPUTS to each layer to compute gradients.
This means we must STORE all intermediate activations!

Memory Usage Example (7B LLM, batch=1, seq=2048):
┌─────────────────────────────────────────────────────────────┐
│ Component              │ Memory                             │
├─────────────────────────────────────────────────────────────┤
│ Model weights (BF16)   │ ~14 GB                             │
│ Optimizer states       │ ~28 GB (Adam: 2x weights)          │
│ Gradients             │ ~14 GB                              │
│ Activations           │ ~30-50 GB (!!)                      │
└─────────────────────────────────────────────────────────────┘

Activations often dominate memory, especially for:
- Long sequences (attention: O(N²))
- Large batch sizes
- Deep networks

GRADIENT CHECKPOINTING SOLUTION:
Don't save all activations - recompute them during backward!

Memory: O(√N) instead of O(N) for N layers
Compute: ~33% more FLOPs (recompute forward during backward)
"""


class ActivationMemoryDemo(nn.Module):
    """Demonstrate activation memory growth."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x


def measure_activation_memory():
    """
    Measure activation memory with and without checkpointing.
    """
    print("\n" + "="*70)
    print("ACTIVATION MEMORY MEASUREMENT")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required for memory measurement")
        return
    
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    
    # Model parameters
    hidden_dim = 2048
    num_layers = 24
    batch_size = 32
    seq_len = 512
    
    model = ActivationMemoryDemo(hidden_dim, num_layers).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Measure without checkpointing
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    peak_mem_no_ckpt = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Batch × Seq: {batch_size} × {seq_len}")
    print(f"\nPeak memory (no checkpointing): {peak_mem_no_ckpt:.2f} GB")
    
    # Clear memory
    del output, loss
    torch.cuda.empty_cache()


# =============================================================================
# SECTION 2: HOW GRADIENT CHECKPOINTING WORKS
# =============================================================================
"""
GRADIENT CHECKPOINTING ALGORITHM:
═════════════════════════════════

Standard Backward Pass:
    Forward: Save all activations a₁, a₂, ..., aₙ
    Backward: Use saved activations to compute gradients
    Memory: O(N) for N layers

Checkpointed Backward Pass:
    Forward: Only save "checkpoint" activations at intervals
    Backward: Recompute activations from nearest checkpoint
    Memory: O(√N) with optimal checkpoint placement

Visualization:
    Standard:
    Layer:    [1] → [2] → [3] → [4] → [5] → [6] → [7] → [8]
    Saved:     ●     ●     ●     ●     ●     ●     ●     ●
    Memory: 8 activations
    
    Checkpointed (every 2 layers):
    Layer:    [1] → [2] → [3] → [4] → [5] → [6] → [7] → [8]
    Saved:     ●           ●           ●           ●
    Memory: 4 activations + recompute intermediate

TRADE-OFF:
    Memory saved: ~(1 - 1/k) where k = checkpoint interval
    Extra compute: ~(k-1)/k of forward pass per checkpointed segment
    
    Typical: k=1 (every layer) → ~2x forward compute, ~50% memory
"""


def visualize_checkpointing():
    """Visualize how checkpointing saves memory."""
    print("\n" + "="*70)
    print("CHECKPOINTING VISUALIZATION")
    print("="*70)
    
    print("""
STANDARD FORWARD/BACKWARD:
══════════════════════════

Forward Pass (save all activations):
    Input → [Layer 1] → a₁ → [Layer 2] → a₂ → [Layer 3] → a₃ → Output
              save ●          save ●          save ●
              
    Memory: [input, a₁, a₂, a₃, output] = 5 tensors

Backward Pass (use saved activations):
    ∂L/∂a₃ ← [Layer 3] ← ∂L/∂a₂ ← [Layer 2] ← ∂L/∂a₁ ← [Layer 1]
               uses a₂           uses a₁           uses input


CHECKPOINTED FORWARD/BACKWARD:
══════════════════════════════

Forward Pass (save only checkpoints):
    Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
              (no save)   checkpoint ●   (no save)
              
    Memory: [input, a₂, output] = 3 tensors (40% saved!)

Backward Pass (recompute when needed):
    
    To compute ∂L/∂a₃:
        Need a₂ ✓ (saved as checkpoint)
        
    To compute ∂L/∂a₂:
        Need a₁ ✗ (not saved)
        Recompute: a₁ = Layer1(input)  ← Extra forward!
        
    To compute ∂L/∂a₁:
        Need input ✓ (always saved)
""")


# =============================================================================
# SECTION 3: PYTORCH CHECKPOINT API
# =============================================================================
"""
PyTorch provides two main checkpointing functions:

1. checkpoint(function, *args):
   - Wraps a single function/module
   - Doesn't save intermediate activations within the function
   - Recomputes forward during backward
   
2. checkpoint_sequential(functions, segments, input):
   - For sequential models
   - Divides into segments, checkpoints each segment
   - More convenient for nn.Sequential
"""


class CheckpointedBlock(nn.Module):
    """A transformer-like block with optional checkpointing."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.use_checkpoint = False
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Use checkpointing during training
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class CheckpointedTransformer(nn.Module):
    """Transformer with gradient checkpointing support."""
    
    def __init__(self, hidden_dim: int = 512, num_layers: int = 12,
                 checkpoint_every: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CheckpointedBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.checkpoint_every = checkpoint_every
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing for all blocks."""
        for i, block in enumerate(self.blocks):
            # Checkpoint based on interval
            block.use_checkpoint = (i % self.checkpoint_every == 0)
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        for block in self.blocks:
            block.use_checkpoint = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def demonstrate_checkpoint_api():
    """
    Demonstrate PyTorch checkpoint API usage.
    """
    print("\n" + "="*70)
    print("PYTORCH CHECKPOINT API")
    print("="*70)
    
    print("""
BASIC USAGE:
────────────

from torch.utils.checkpoint import checkpoint

def expensive_function(x):
    # This function's intermediate activations won't be saved
    y = layer1(x)
    z = layer2(y)
    return layer3(z)

# Wrap with checkpoint
output = checkpoint(expensive_function, input_tensor, use_reentrant=False)


IMPORTANT PARAMETERS:
─────────────────────

use_reentrant (bool):
    - False (recommended): New implementation, supports complex autograd
    - True (legacy): Old implementation, faster but limited
    
    Always use use_reentrant=False for new code!

preserve_rng_state (bool):
    - True: Save and restore RNG state (for dropout consistency)
    - False: Don't save RNG (faster but dropout may differ)


SEQUENTIAL MODELS:
──────────────────

from torch.utils.checkpoint import checkpoint_sequential

# Divide sequential model into segments
model = nn.Sequential(layer1, layer2, layer3, layer4, layer5, layer6)
segments = 2  # Checkpoint every 3 layers

output = checkpoint_sequential(model, segments, input_tensor)
""")
    
    if not torch.cuda.is_available():
        print("\nCUDA required for memory comparison")
        return
    
    device = torch.device("cuda")
    
    # Compare memory usage
    model = CheckpointedTransformer(hidden_dim=512, num_layers=12).to(device)
    x = torch.randn(8, 256, 512, device=device)
    
    # Without checkpointing
    model.disable_checkpointing()
    torch.cuda.reset_peak_memory_stats()
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    mem_no_ckpt = torch.cuda.max_memory_allocated() / 1e9
    
    # Clear
    del output, loss
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    torch.cuda.empty_cache()
    
    # With checkpointing
    model.enable_checkpointing()
    torch.cuda.reset_peak_memory_stats()
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    mem_with_ckpt = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nMemory Comparison:")
    print(f"  Without checkpointing: {mem_no_ckpt:.2f} GB")
    print(f"  With checkpointing:    {mem_with_ckpt:.2f} GB")
    print(f"  Memory saved:          {(1 - mem_with_ckpt/mem_no_ckpt)*100:.1f}%")


# =============================================================================
# SECTION 4: SELECTIVE CHECKPOINTING STRATEGIES
# =============================================================================
"""
SELECTIVE CHECKPOINTING:
════════════════════════

Not all layers benefit equally from checkpointing.
Strategic placement can optimize memory/compute trade-off.

STRATEGIES:

1. Every-N-Layers:
   - Checkpoint every N transformer blocks
   - Simple, predictable memory
   
2. Memory-Heavy Layers Only:
   - Only checkpoint attention layers (O(N²) memory)
   - Skip MLP (less memory benefit)
   
3. Offloading-Aware:
   - Combine with CPU offloading
   - Checkpoint when offloading is too slow
   
4. Adaptive/Dynamic:
   - Adjust based on sequence length
   - More checkpointing for longer sequences
"""


class SelectiveCheckpointTransformer(nn.Module):
    """Transformer with selective checkpointing strategies."""
    
    def __init__(self, hidden_dim: int = 512, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList([
            CheckpointedBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def set_checkpoint_strategy(self, strategy: str, **kwargs):
        """
        Set checkpointing strategy.
        
        Args:
            strategy: 'none', 'all', 'every_n', 'first_half', 'gradient'
            **kwargs: Strategy-specific parameters
        """
        if strategy == 'none':
            for layer in self.layers:
                layer.use_checkpoint = False
                
        elif strategy == 'all':
            for layer in self.layers:
                layer.use_checkpoint = True
                
        elif strategy == 'every_n':
            n = kwargs.get('n', 2)
            for i, layer in enumerate(self.layers):
                layer.use_checkpoint = (i % n == 0)
                
        elif strategy == 'first_half':
            # Checkpoint first half (earlier layers have more downstream deps)
            for i, layer in enumerate(self.layers):
                layer.use_checkpoint = (i < self.num_layers // 2)
                
        elif strategy == 'gradient':
            # More checkpointing for earlier layers (gradient flow longer)
            for i, layer in enumerate(self.layers):
                # Probability decreases with layer depth
                layer.use_checkpoint = (i < self.num_layers * 0.7)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def compare_checkpoint_strategies():
    """
    Compare different checkpointing strategies.
    """
    print("\n" + "="*70)
    print("CHECKPOINTING STRATEGIES COMPARISON")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    device = torch.device("cuda")
    
    model = SelectiveCheckpointTransformer(hidden_dim=512, num_layers=24).to(device)
    x = torch.randn(8, 256, 512, device=device)
    
    strategies = ['none', 'all', 'every_n', 'first_half']
    results = {}
    
    for strategy in strategies:
        # Set strategy
        if strategy == 'every_n':
            model.set_checkpoint_strategy(strategy, n=3)
        else:
            model.set_checkpoint_strategy(strategy)
        
        # Count checkpointed layers
        num_ckpt = sum(1 for l in model.layers if l.use_checkpoint)
        
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward
        model.train()
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        mem = torch.cuda.max_memory_allocated() / 1e9
        results[strategy] = (mem, num_ckpt)
        
        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        del output, loss
    
    print(f"\n{'Strategy':<15} {'Checkpointed':<15} {'Peak Memory':<15}")
    print("-" * 45)
    for strategy, (mem, num_ckpt) in results.items():
        print(f"{strategy:<15} {num_ckpt}/{model.num_layers} layers{' '*5} {mem:.2f} GB")


# =============================================================================
# SECTION 5: HUGGING FACE INTEGRATION
# =============================================================================

def huggingface_checkpointing():
    """
    Show how to enable checkpointing in HuggingFace models.
    """
    print("\n" + "="*70)
    print("HUGGING FACE GRADIENT CHECKPOINTING")
    print("="*70)
    
    print("""
ENABLING CHECKPOINTING IN TRANSFORMERS:
═══════════════════════════════════════

Method 1: Model Configuration
─────────────────────────────
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()


Method 2: Training Arguments
────────────────────────────
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,  # Enable here
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)


Method 3: PEFT/LoRA with Checkpointing
──────────────────────────────────────
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained(...)
model.gradient_checkpointing_enable()

# Then apply LoRA
peft_config = LoraConfig(...)
model = get_peft_model(model, peft_config)


IMPORTANT NOTES:
────────────────

1. Use with use_reentrant=False:
   model.gradient_checkpointing_enable(
       gradient_checkpointing_kwargs={"use_reentrant": False}
   )

2. Required for inputs_embeds if using gradient checkpointing:
   model.config.use_cache = False  # Disable KV cache

3. Incompatible with:
   - KV caching (disable for training)
   - Some flash attention implementations (check compatibility)
""")


# =============================================================================
# SECTION 6: COMPUTE vs MEMORY TRADE-OFF ANALYSIS
# =============================================================================

def tradeoff_analysis():
    """
    Analyze the compute vs memory trade-off.
    """
    print("\n" + "="*70)
    print("COMPUTE vs MEMORY TRADE-OFF ANALYSIS")
    print("="*70)
    
    print("""
THEORETICAL ANALYSIS:
═════════════════════

For a network with N layers:

Standard Training:
    Memory: O(N) - store all N activations
    Compute: F (forward) + B (backward) = 1F + 1B

Full Checkpointing (every layer):
    Memory: O(1) - only store input
    Compute: Recompute all N layers during backward
            = 1F + 1B + 1F = 2F + 1B (~33% more)

Optimal Checkpointing (√N segments):
    Memory: O(√N) - store √N checkpoints
    Compute: Recompute √N layers per segment
            ≈ 1F + 1B + F/√N ≈ 1.X × (F + B)


PRACTICAL MEASUREMENTS:
═══════════════════════

Typical overhead with gradient checkpointing:
┌─────────────────────────────────────────────────────────────┐
│ Scenario              │ Memory Saved  │ Compute Overhead    │
├─────────────────────────────────────────────────────────────┤
│ Every layer           │ ~60-70%       │ ~30-40%            │
│ Every 2 layers        │ ~40-50%       │ ~20-25%            │
│ Every 4 layers        │ ~25-35%       │ ~10-15%            │
│ Attention only        │ ~30-50%       │ ~15-25%            │
└─────────────────────────────────────────────────────────────┘


WHEN TO USE:
════════════

✓ Use checkpointing when:
  - Training large models (>1B parameters)
  - Long sequences (>2K tokens)
  - Memory-limited (single GPU training)
  - Want larger batch sizes

✗ Avoid checkpointing when:
  - Inference (no backward pass)
  - Abundant memory available
  - Latency-critical training
  - Very shallow networks


COMBINING WITH OTHER TECHNIQUES:
════════════════════════════════

1. Checkpointing + Mixed Precision:
   - Both reduce memory
   - Multiplicative benefit
   - BF16 + checkpointing = ~4x memory reduction

2. Checkpointing + Gradient Accumulation:
   - Accumulation: reduce batch dimension
   - Checkpointing: reduce activation storage
   - Together: train with very large effective batch

3. Checkpointing + DeepSpeed ZeRO:
   - ZeRO: shards optimizer/gradients/params
   - Checkpointing: reduces activations
   - Orthogonal optimizations
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GRADIENT CHECKPOINTING DEEP DIVE")
    print("="*70)
    
    # Measure activation memory
    measure_activation_memory()
    
    # Visualize checkpointing
    visualize_checkpointing()
    
    # PyTorch API demonstration
    demonstrate_checkpoint_api()
    
    # Compare strategies
    compare_checkpoint_strategies()
    
    # HuggingFace integration
    huggingface_checkpointing()
    
    # Trade-off analysis
    tradeoff_analysis()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
