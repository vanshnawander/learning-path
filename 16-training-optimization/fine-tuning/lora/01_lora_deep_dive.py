"""
LoRA (Low-Rank Adaptation) Deep Dive
=====================================

This module provides comprehensive coverage of LoRA, the parameter-efficient
fine-tuning technique that revolutionized LLM adaptation.

Key Topics:
1. The Full Fine-tuning Problem
2. Low-Rank Matrix Decomposition Theory
3. LoRA Mathematics and Implementation
4. QLoRA: Quantized LoRA
5. Hyperparameter Selection (rank, alpha, target modules)
6. Advanced Variants (DoRA, LoRA+, rsLoRA)
7. Merging and Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# =============================================================================
# SECTION 1: THE FULL FINE-TUNING PROBLEM
# =============================================================================
"""
FULL FINE-TUNING CHALLENGES:
════════════════════════════

For a 7B parameter model with FP16:
┌────────────────────────────────────────────────────────────────┐
│ Component              │ Memory (FP16/FP32)                    │
├────────────────────────────────────────────────────────────────┤
│ Model weights          │ 14 GB (FP16)                          │
│ Gradients              │ 14 GB (FP16)                          │
│ Optimizer states       │ 56 GB (Adam: 2 FP32 per param)        │
│ Activations            │ 20-50 GB (depends on batch/seq)       │
├────────────────────────────────────────────────────────────────┤
│ TOTAL                  │ 100+ GB                                │
└────────────────────────────────────────────────────────────────┘

Problems:
1. GPU Memory: Requires 4-8 A100 GPUs
2. Storage: Each fine-tuned model = 14 GB
3. Serving: Need separate model per task
4. Catastrophic Forgetting: Risk of losing pretrained knowledge

LoRA SOLUTION:
Instead of updating all 7B parameters, update only ~0.1% (7M parameters)
- Memory: Trainable params + their states only
- Storage: Only save LoRA weights (~50 MB vs 14 GB)
- Serving: Share base model, swap adapters
- Forgetting: Base weights frozen, preserved
"""


def compare_finetuning_memory():
    """Compare memory requirements of full vs LoRA fine-tuning."""
    print("\n" + "="*70)
    print("FULL FINE-TUNING vs LoRA MEMORY COMPARISON")
    print("="*70)
    
    # 7B model parameters
    total_params = 7e9
    
    # Full fine-tuning
    full_model = total_params * 2  # FP16 weights
    full_grad = total_params * 2   # FP16 gradients
    full_opt = total_params * 8    # Adam: 2 * FP32 = 8 bytes per param
    
    print(f"\nFULL FINE-TUNING (7B model):")
    print(f"  Model weights (FP16):    {full_model / 1e9:.1f} GB")
    print(f"  Gradients (FP16):        {full_grad / 1e9:.1f} GB")
    print(f"  Optimizer states (Adam): {full_opt / 1e9:.1f} GB")
    print(f"  TOTAL (excl. activations): {(full_model + full_grad + full_opt) / 1e9:.1f} GB")
    
    # LoRA fine-tuning (rank=16, typical target modules)
    # Typical: q, k, v, o projections = 4 * 2 * (hidden * rank) per layer
    hidden = 4096
    layers = 32
    rank = 16
    lora_params = 4 * 2 * hidden * rank * layers  # ~67M params
    
    lora_weights = lora_params * 2  # FP16
    lora_grad = lora_params * 2
    lora_opt = lora_params * 8
    frozen_model = total_params * 2  # Still need frozen model in memory
    
    print(f"\nLoRA FINE-TUNING (rank={rank}):")
    print(f"  Frozen model (FP16):     {frozen_model / 1e9:.1f} GB")
    print(f"  LoRA weights:            {lora_weights / 1e6:.1f} MB")
    print(f"  LoRA gradients:          {lora_grad / 1e6:.1f} MB")
    print(f"  LoRA optimizer states:   {lora_opt / 1e6:.1f} MB")
    print(f"  Trainable params:        {lora_params / 1e6:.1f}M ({100*lora_params/total_params:.2f}%)")
    print(f"  TOTAL:                   {(frozen_model + lora_weights + lora_grad + lora_opt) / 1e9:.1f} GB")
    print(f"\n  Optimizer memory saved:  {(full_opt - lora_opt) / 1e9:.1f} GB ({100*(1-lora_opt/full_opt):.1f}%)")


# =============================================================================
# SECTION 2: LOW-RANK MATRIX DECOMPOSITION
# =============================================================================
"""
LOW-RANK DECOMPOSITION THEORY:
══════════════════════════════

Key Insight: Weight changes during fine-tuning have LOW INTRINSIC RANK

For a weight matrix W ∈ R^{d×k}:
    Full rank: W has d×k = dk parameters
    
    Low-rank r: W ≈ BA where B ∈ R^{d×r}, A ∈ R^{r×k}
                Parameters: dr + rk = r(d+k) << dk when r << min(d,k)

WHY LOW RANK WORKS FOR FINE-TUNING:

1. Pretrained models already capture general knowledge
2. Task-specific adaptation lies in low-dimensional subspace
3. Empirically, ΔW during fine-tuning is low-rank

MATHEMATICAL FOUNDATION:
    Original: y = Wx
    Fine-tuned: y = (W + ΔW)x
    
    With LoRA: ΔW = BA (low-rank)
    So: y = Wx + BAx = Wx + B(Ax)
    
    Forward pass:
    1. Compute Ax (r×k @ k = r)
    2. Compute B(Ax) (d×r @ r = d)
    3. Add to Wx
    
    Total extra compute: O(r(d+k)) vs O(dk) for full update
"""


def demonstrate_low_rank():
    """Demonstrate low-rank approximation of weight updates."""
    print("\n" + "="*70)
    print("LOW-RANK DECOMPOSITION DEMONSTRATION")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Simulate weight change during fine-tuning
    d, k = 4096, 4096  # Typical LLM dimensions
    
    # Create a synthetic "low-rank" weight change
    # (Real fine-tuning changes are empirically low-rank)
    true_rank = 16
    delta_W = torch.randn(d, true_rank) @ torch.randn(true_rank, k)
    delta_W = delta_W / delta_W.norm() * 10  # Scale
    
    print(f"\nSimulated weight change ΔW: {d}×{k} = {d*k:,} parameters")
    print(f"True intrinsic rank: {true_rank}")
    
    # SVD to verify rank
    U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
    
    # Show singular value decay
    print(f"\nTop 20 singular values:")
    for i in range(min(20, len(S))):
        bar = "█" * int(S[i] / S[0] * 30)
        print(f"  σ_{i+1:2d}: {S[i]:8.2f} {bar}")
    
    # Reconstruction with different ranks
    print(f"\nReconstruction error with different ranks:")
    for r in [4, 8, 16, 32, 64]:
        # Low-rank approximation
        approx = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]
        error = (delta_W - approx).norm() / delta_W.norm()
        
        params_saved = (1 - r*(d+k)/(d*k)) * 100
        print(f"  Rank {r:3d}: error={error:.4f}, params={r*(d+k):,} ({params_saved:.1f}% saved)")


# =============================================================================
# SECTION 3: LORA IMPLEMENTATION
# =============================================================================
"""
LoRA FORMULATION:
═════════════════

For a pretrained weight W₀ ∈ R^{d×k}:

    h = W₀x + ΔWx = W₀x + BAx

Where:
    B ∈ R^{d×r} initialized to zeros
    A ∈ R^{r×k} initialized with random Gaussian
    
Scaling factor:
    h = W₀x + (α/r) · BAx
    
    α (alpha): scaling hyperparameter
    α/r: keeps magnitude stable as r changes
    
Typical: α = 2r (so scaling = 2)

INITIALIZATION:
    A ~ N(0, σ²)  (Kaiming/He initialization)
    B = 0         (ΔW starts at 0, no initial change)
    
This ensures fine-tuning starts from pretrained behavior!
"""


class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer.
    
    Implements: y = Wx + (α/r) * B @ A @ x
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False
        
        # Pretrained weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize base weight (would normally be from pretrained)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # LoRA initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Start with ΔW = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Use merged weights (for inference)
            return F.linear(x, self.weight, self.bias)
        
        # Base forward
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA forward: (α/r) * x @ A^T @ B^T
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output
    
    def merge(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            # W_merged = W + (α/r) * B @ A
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True
    
    def unmerge(self):
        """Separate LoRA weights from base weights."""
        if self.merged:
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False
    
    def get_lora_parameters(self) -> int:
        """Count LoRA parameters."""
        return self.lora_A.numel() + self.lora_B.numel()


def demonstrate_lora_forward():
    """Demonstrate LoRA forward pass."""
    print("\n" + "="*70)
    print("LoRA FORWARD PASS DEMONSTRATION")
    print("="*70)
    
    # Create LoRA layer
    in_feat, out_feat = 1024, 1024
    rank = 8
    alpha = 16.0
    
    layer = LoRALinear(in_feat, out_feat, rank=rank, alpha=alpha)
    
    print(f"\nLoRA Configuration:")
    print(f"  Input dim: {in_feat}")
    print(f"  Output dim: {out_feat}")
    print(f"  Rank (r): {rank}")
    print(f"  Alpha (α): {alpha}")
    print(f"  Scaling (α/r): {alpha/rank}")
    
    # Parameter count
    base_params = in_feat * out_feat + out_feat
    lora_params = layer.get_lora_parameters()
    
    print(f"\nParameter Count:")
    print(f"  Base layer: {base_params:,}")
    print(f"  LoRA params: {lora_params:,}")
    print(f"  Reduction: {100*(1-lora_params/base_params):.1f}%")
    
    # Forward pass
    x = torch.randn(32, 1024)
    y = layer(x)
    
    print(f"\nForward Pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    
    # Verify initialization starts at pretrained behavior
    layer_no_lora = nn.Linear(in_feat, out_feat)
    layer_no_lora.weight.data = layer.weight.data.clone()
    layer_no_lora.bias.data = layer.bias.data.clone()
    
    with torch.no_grad():
        y_base = layer_no_lora(x)
        y_lora = layer(x)
        diff = (y_base - y_lora).abs().max()
    
    print(f"\nInitialization Check (B=0):")
    print(f"  Max diff from base: {diff:.2e}")
    print(f"  Starts at pretrained behavior: {diff < 1e-5}")


# =============================================================================
# SECTION 4: QLoRA - QUANTIZED LoRA
# =============================================================================
"""
QLoRA: Memory-Efficient LoRA with Quantization
═══════════════════════════════════════════════

Key Innovations:

1. 4-bit NormalFloat (NF4):
   - Quantize base weights to 4-bit
   - Information-theoretically optimal for normal distributions
   - Memory: 14 GB → 3.5 GB for 7B model

2. Double Quantization:
   - Quantize the quantization constants too!
   - Additional ~0.4 GB savings

3. Paged Optimizers:
   - Offload optimizer states to CPU when OOM
   - Automatic GPU↔CPU transfer

Memory Comparison (7B model):
┌─────────────────────────────────────────────────┐
│ Method         │ GPU Memory                     │
├─────────────────────────────────────────────────┤
│ Full FP16      │ 14 GB                          │
│ LoRA FP16      │ 14 GB (frozen) + 0.1 GB        │
│ QLoRA 4-bit    │ 3.5 GB (quantized) + 0.1 GB    │
│ QLoRA + DQ     │ 3.1 GB + 0.1 GB                │
└─────────────────────────────────────────────────┘

FORWARD PASS:
    y = dequantize(W_4bit) @ x + (α/r) * B @ A @ x
    
    - W_4bit: 4-bit quantized weights
    - dequantize: On-the-fly to FP16/BF16
    - LoRA computation still in full precision
"""


def qlora_explanation():
    """Explain QLoRA in detail."""
    print("\n" + "="*70)
    print("QLoRA: QUANTIZED LoRA")
    print("="*70)
    
    print("""
NF4 (4-bit NormalFloat) QUANTIZATION:
═════════════════════════════════════

Problem: Standard quantization wastes bits on outliers
Solution: Quantization levels optimized for normal distribution

NF4 quantization bins:
    [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
     
These 16 values (4 bits) are optimal for N(0,1) distribution!

DOUBLE QUANTIZATION:
════════════════════

Problem: Quantization needs scaling factors per block
        Block size 64 → 1 FP32 scale per 64 params
        7B params → 110M FP32 scales = 440 MB overhead!

Solution: Quantize the scales too (to FP8)
        440 MB → 55 MB

Total savings: ~0.4 GB


BITSANDBYTES IMPLEMENTATION:
════════════════════════════

import bitsandbytes as bnb

# Create 4-bit linear layer
linear_4bit = bnb.nn.Linear4bit(
    input_features=4096,
    output_features=4096,
    bias=False,
    compute_dtype=torch.bfloat16,
    compress_statistics=True,  # Double quantization
    quant_type='nf4',          # NormalFloat4
)

# Load pretrained model in 4-bit
from transformers import BitsAndBytesConfig

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

# Then apply LoRA as usual
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
""")


# =============================================================================
# SECTION 5: HYPERPARAMETER SELECTION
# =============================================================================
"""
LoRA HYPERPARAMETERS:
═════════════════════

1. RANK (r):
   - Lower r = fewer parameters, more compression
   - Higher r = more expressiveness
   - Typical: 8-64 for most tasks
   - Rule of thumb: Start with r=8, increase if underfitting

2. ALPHA (α):
   - Scaling factor: scaling = α/r
   - Controls magnitude of LoRA updates
   - Common: α = 2r (scaling = 2)
   - Higher α = larger updates, may need lower LR

3. TARGET MODULES:
   - Which layers to apply LoRA
   - Attention: q_proj, k_proj, v_proj, o_proj
   - MLP: gate_proj, up_proj, down_proj
   - More modules = more parameters but better adaptation

4. DROPOUT:
   - Regularization for LoRA path
   - Typical: 0.05-0.1
   - Higher for small datasets

5. LEARNING RATE:
   - Usually higher than full fine-tuning
   - Typical: 1e-4 to 3e-4
   - Scale with rank (lower rank → higher LR)
"""


@dataclass
class LoRAConfig:
    """LoRA configuration dataclass."""
    r: int = 8
    alpha: float = 16.0
    target_modules: List[str] = None
    dropout: float = 0.05
    bias: str = "none"  # "none", "all", "lora_only"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


def hyperparameter_guide():
    """Guide for LoRA hyperparameter selection."""
    print("\n" + "="*70)
    print("LoRA HYPERPARAMETER SELECTION GUIDE")
    print("="*70)
    
    print("""
RANK SELECTION:
═══════════════

┌─────────────────────────────────────────────────────────────────┐
│ Task Type              │ Recommended Rank    │ Notes            │
├─────────────────────────────────────────────────────────────────┤
│ Simple classification  │ 4-8                 │ Low complexity   │
│ Text generation        │ 8-32                │ Medium           │
│ Complex reasoning      │ 32-64               │ High complexity  │
│ Domain adaptation      │ 16-64               │ Depends on shift │
│ Instruction tuning     │ 16-32               │ Standard         │
└─────────────────────────────────────────────────────────────────┘


TARGET MODULES SELECTION:
═════════════════════════

Minimal (fastest, least params):
    target_modules = ["q_proj", "v_proj"]
    
Standard (good balance):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
Full (best quality, most params):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]


ALPHA/RANK RATIO:
═════════════════

Scaling = α/r controls update magnitude

Conservative (α = r):    scaling = 1
Standard (α = 2r):       scaling = 2 (recommended)
Aggressive (α = 4r):     scaling = 4 (may need lower LR)


LEARNING RATE GUIDELINES:
═════════════════════════

LoRA typically uses HIGHER learning rates than full fine-tuning:

Full fine-tuning:  1e-5 to 5e-5
LoRA:              1e-4 to 3e-4

Reasoning: LoRA params start at 0 and need larger updates


PRACTICAL RECOMMENDATIONS:
══════════════════════════

Starting Configuration:
    r = 16
    alpha = 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout = 0.05
    lr = 2e-4
    
Memory Limited (QLoRA):
    load_in_4bit = True
    r = 64 (can afford higher rank)
    alpha = 128
    target_modules = all linear layers
    
Quality Critical:
    r = 64
    alpha = 128
    target_modules = all linear layers
    lr = 1e-4 (lower for stability)
""")


# =============================================================================
# SECTION 6: ADVANCED VARIANTS
# =============================================================================
"""
LORA VARIANTS:
══════════════

1. DoRA (Weight-Decomposed Low-Rank Adaptation):
   - Decomposes weight into magnitude and direction
   - LoRA only updates direction, separate magnitude scalar
   - Better performance, minimal overhead
   
   W = m * (W₀ + BA) / ||W₀ + BA||
   
2. LoRA+ (Differential Learning Rates):
   - Different LR for A and B matrices
   - B typically needs 2-4x higher LR
   - Improves convergence

3. rsLoRA (Rank-Stabilized LoRA):
   - Scaling: α/√r instead of α/r
   - More stable for higher ranks
   - Used by default in some implementations

4. AdaLoRA (Adaptive Budget Allocation):
   - Learns optimal rank per layer
   - Prunes unimportant singular values
   - Better parameter efficiency

5. QA-LoRA (Quantization-Aware LoRA):
   - Considers quantization during LoRA training
   - Better quality after PTQ
"""


class DoRALinear(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.
    
    Decomposes W into magnitude m and direction d:
    W' = m * (W + BA) / ||W + BA||
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        # DoRA magnitude (learnable per output dimension)
        self.magnitude = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # Initialize magnitude from weight norms
        with torch.no_grad():
            self.magnitude.copy_(self.weight.norm(dim=1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adapted weight
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)
        adapted_weight = self.weight + lora_weight
        
        # Normalize to unit direction
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)
        
        # Apply learned magnitude
        final_weight = self.magnitude.unsqueeze(1) * direction
        
        return F.linear(x, final_weight)


# =============================================================================
# SECTION 7: MERGING AND INFERENCE
# =============================================================================

def merging_guide():
    """Guide for merging LoRA weights and inference."""
    print("\n" + "="*70)
    print("LoRA MERGING AND INFERENCE")
    print("="*70)
    
    print("""
MERGING LoRA WEIGHTS:
═════════════════════

After training, LoRA weights can be merged into base model:

    W_merged = W_base + (α/r) * B @ A

Benefits:
✓ No inference overhead (same as base model)
✓ Single model file for deployment
✓ Works with any inference engine

Code:
    from peft import PeftModel
    
    # Load base + LoRA
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, lora_adapter)
    
    # Merge and unload
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained("merged_model")


ADAPTER SWITCHING:
══════════════════

Keep multiple LoRA adapters for different tasks:

    # Load base model once
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, "adapter_task1")
    
    # Switch adapters
    model.load_adapter("adapter_task2", adapter_name="task2")
    model.set_adapter("task2")  # Activate task2 adapter
    
    # Switch back
    model.set_adapter("default")  # Back to task1


MULTIPLE ADAPTERS:
══════════════════

Combine multiple LoRA adapters:

    # Method 1: Linear combination
    from peft import add_weighted_adapter
    
    model.add_weighted_adapter(
        adapters=["task1", "task2"],
        weights=[0.7, 0.3],
        adapter_name="combined"
    )
    
    # Method 2: Stack adapters (experimental)
    # Each adapter applied sequentially


INFERENCE OPTIMIZATION:
═══════════════════════

1. Merge for production (no overhead)
2. Keep separate for multi-task (swap adapters)
3. Batch different adapters with specialized kernels

Memory at inference (unmerged):
    Base model + LoRA overhead
    Overhead: ~0.1-1% of model size
    
Memory at inference (merged):
    Same as base model
    No additional overhead
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LoRA (LOW-RANK ADAPTATION) DEEP DIVE")
    print("="*70)
    
    # Compare fine-tuning approaches
    compare_finetuning_memory()
    
    # Demonstrate low-rank decomposition
    demonstrate_low_rank()
    
    # Show LoRA forward pass
    demonstrate_lora_forward()
    
    # QLoRA explanation
    qlora_explanation()
    
    # Hyperparameter guide
    hyperparameter_guide()
    
    # Merging guide
    merging_guide()
    
    print("\n" + "="*70)
    print("LoRA MODULE COMPLETE")
    print("="*70)
