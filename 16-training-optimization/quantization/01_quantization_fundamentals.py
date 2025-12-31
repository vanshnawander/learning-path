"""
Quantization Fundamentals for LLMs
===================================

This module provides comprehensive coverage of quantization techniques
for large language models, from theory to practical implementation.

Key Topics:
1. Quantization Theory and Mathematics
2. Post-Training Quantization (PTQ)
3. GPTQ: Weight Quantization
4. AWQ: Activation-Aware Quantization
5. Quantization-Aware Training (QAT)
6. Inference Formats (GGUF, GGML)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

# =============================================================================
# SECTION 1: QUANTIZATION THEORY
# =============================================================================
"""
QUANTIZATION FUNDAMENTALS:
══════════════════════════

Quantization maps continuous values to discrete levels:
    float32 → int8/int4/int2

Why Quantize?
┌──────────────────────────────────────────────────────────────────┐
│ Benefit              │ Impact                                   │
├──────────────────────────────────────────────────────────────────┤
│ Memory Reduction     │ 4x (FP32→INT8) or 8x (FP32→INT4)        │
│ Bandwidth Savings    │ Faster memory transfers                  │
│ Compute Speedup      │ INT8 ops are faster on many hardware    │
│ Deployment           │ Fit larger models on edge devices        │
└──────────────────────────────────────────────────────────────────┘

QUANTIZATION FORMULA:
    
    Affine Quantization:
        x_q = round(x / scale + zero_point)
        x_dequant = (x_q - zero_point) * scale
    
    Symmetric Quantization (zero_point = 0):
        x_q = round(x / scale)
        x_dequant = x_q * scale
    
    Scale calculation:
        scale = (max_val - min_val) / (2^bits - 1)

GRANULARITY:
    - Per-tensor: One scale for entire tensor
    - Per-channel: One scale per output channel (better accuracy)
    - Per-group: One scale per group of weights (best accuracy)
"""


def quantize_tensor(x: torch.Tensor, bits: int = 8, 
                    symmetric: bool = True) -> Tuple[torch.Tensor, float, int]:
    """
    Basic quantization of a tensor.
    
    Args:
        x: Input tensor
        bits: Number of bits for quantization
        symmetric: Use symmetric quantization
    
    Returns:
        Quantized tensor, scale, zero_point
    """
    if symmetric:
        # Symmetric: range is [-max_abs, max_abs]
        max_abs = x.abs().max()
        qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1
        scale = max_abs / qmax
        zero_point = 0
    else:
        # Asymmetric: range is [min, max]
        min_val, max_val = x.min(), x.max()
        qmin, qmax = 0, 2**bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))
    
    # Quantize
    x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax).to(torch.int8)
    
    return x_q, scale.item(), zero_point


def dequantize_tensor(x_q: torch.Tensor, scale: float, 
                      zero_point: int = 0) -> torch.Tensor:
    """Dequantize a tensor."""
    return (x_q.float() - zero_point) * scale


def demonstrate_quantization():
    """Demonstrate basic quantization."""
    print("\n" + "="*70)
    print("QUANTIZATION DEMONSTRATION")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Create sample weights
    weights = torch.randn(4, 4) * 2
    print(f"Original weights (FP32):\n{weights}")
    
    # 8-bit quantization
    w_q8, scale8, zp8 = quantize_tensor(weights, bits=8)
    w_dq8 = dequantize_tensor(w_q8, scale8, zp8)
    error8 = (weights - w_dq8).abs().mean()
    
    print(f"\nINT8 quantized:\n{w_q8}")
    print(f"Scale: {scale8:.6f}, Zero point: {zp8}")
    print(f"Dequantized:\n{w_dq8}")
    print(f"Mean absolute error: {error8:.6f}")
    
    # 4-bit quantization
    w_q4, scale4, zp4 = quantize_tensor(weights, bits=4)
    w_dq4 = dequantize_tensor(w_q4, scale4, zp4)
    error4 = (weights - w_dq4).abs().mean()
    
    print(f"\nINT4 quantized:\n{w_q4}")
    print(f"Scale: {scale4:.6f}")
    print(f"Mean absolute error: {error4:.6f}")
    print(f"Error increase (4-bit vs 8-bit): {error4/error8:.1f}x")


# =============================================================================
# SECTION 2: POST-TRAINING QUANTIZATION (PTQ)
# =============================================================================
"""
POST-TRAINING QUANTIZATION:
═══════════════════════════

PTQ quantizes a trained model without additional training.

Types:
1. Weight-Only Quantization:
   - Only quantize weights
   - Activations stay in FP16/BF16
   - Simpler, good for LLMs

2. Weight + Activation Quantization:
   - Quantize both weights and activations
   - More compression but harder
   - Requires calibration data

CALIBRATION:
    For activation quantization, we need to know the typical range.
    Run forward passes on representative data to collect statistics.

CHALLENGES:
    - Outliers in activations/weights
    - Accumulation of quantization errors
    - Different sensitivity per layer
"""


class PTQLinear(nn.Module):
    """Linear layer with post-training quantization support."""
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 bits: int = 8, per_channel: bool = True):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        
        # Quantize weights
        if per_channel:
            # Per-output-channel quantization
            self.scales = []
            self.zero_points = []
            quantized_rows = []
            
            for i in range(weight.shape[0]):
                row = weight[i]
                max_abs = row.abs().max()
                scale = max_abs / (2**(bits-1) - 1)
                self.scales.append(scale.item())
                self.zero_points.append(0)
                
                q_row = torch.clamp(
                    torch.round(row / scale), 
                    -(2**(bits-1)), 2**(bits-1) - 1
                )
                quantized_rows.append(q_row)
            
            self.weight_q = nn.Parameter(
                torch.stack(quantized_rows).to(torch.int8),
                requires_grad=False
            )
            self.scales = torch.tensor(self.scales)
        else:
            # Per-tensor quantization
            self.weight_q, self.scale, self.zero_point = quantize_tensor(weight, bits)
            self.weight_q = nn.Parameter(self.weight_q, requires_grad=False)
        
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly
        if self.per_channel:
            weight = self.weight_q.float() * self.scales.unsqueeze(1).to(x.device)
        else:
            weight = dequantize_tensor(self.weight_q, self.scale, self.zero_point)
        
        return F.linear(x, weight, self.bias)


def ptq_example():
    """Demonstrate post-training quantization."""
    print("\n" + "="*70)
    print("POST-TRAINING QUANTIZATION EXAMPLE")
    print("="*70)
    
    torch.manual_seed(42)
    
    # Original FP32 layer
    in_features, out_features = 1024, 1024
    weight = torch.randn(out_features, in_features) * 0.02
    bias = torch.zeros(out_features)
    
    original = nn.Linear(in_features, out_features)
    original.weight.data = weight
    original.bias.data = bias
    
    # PTQ version
    ptq_layer = PTQLinear(weight, bias, bits=8, per_channel=True)
    
    # Compare outputs
    x = torch.randn(32, in_features)
    
    with torch.no_grad():
        y_orig = original(x)
        y_ptq = ptq_layer(x)
    
    error = (y_orig - y_ptq).abs()
    
    print(f"\nLayer: {in_features} → {out_features}")
    print(f"Original memory: {weight.numel() * 4 / 1e6:.2f} MB (FP32)")
    print(f"PTQ memory: {weight.numel() * 1 / 1e6:.2f} MB (INT8)")
    print(f"\nOutput comparison:")
    print(f"  Mean absolute error: {error.mean():.6f}")
    print(f"  Max absolute error: {error.max():.6f}")
    print(f"  Relative error: {(error / y_orig.abs()).mean() * 100:.2f}%")


# =============================================================================
# SECTION 3: GPTQ - OPTIMAL WEIGHT QUANTIZATION
# =============================================================================
"""
GPTQ (Generative Pre-trained Transformer Quantization):
═══════════════════════════════════════════════════════

GPTQ is a one-shot weight quantization method based on Optimal Brain Quantization.

Key Insight: Quantize weights to MINIMIZE OUTPUT ERROR, not weight error!

Algorithm:
1. For each layer, collect input activations (calibration)
2. Quantize weights column by column
3. After quantizing each column, UPDATE remaining columns to compensate

Mathematical Foundation:
    Minimize: ||WX - Q(W)X||²
    
    Where:
    - W = original weights
    - Q(W) = quantized weights
    - X = input activations
    
    This is different from minimizing ||W - Q(W)||²!

Hessian-based Update:
    After quantizing column i, update remaining columns:
    W[:, j] -= (w_i - q_i) * H_ij / H_ii
    
    Where H = X @ X.T is the Hessian (correlation of inputs)
"""


def gptq_explanation():
    """Explain GPTQ algorithm."""
    print("\n" + "="*70)
    print("GPTQ: OPTIMAL WEIGHT QUANTIZATION")
    print("="*70)
    
    print("""
GPTQ ALGORITHM OVERVIEW:
════════════════════════

Step 1: Collect Calibration Data
    - Run forward pass on 128-1024 samples
    - Collect input activations X for each layer
    - Compute Hessian H = X @ X.T

Step 2: Quantize Column by Column
    For each column i:
        1. Quantize w_i to q_i
        2. Compute error: δ = (w_i - q_i) / H_ii
        3. Update remaining columns: W[:, i+1:] -= δ * H[i, i+1:]
        
Step 3: Handle Ordering
    - Process columns in order of increasing H_ii (error sensitivity)
    - Use block-wise processing for efficiency (block size ~128)


WHY GPTQ WORKS WELL:
════════════════════

1. Minimizes OUTPUT error, not weight error
   - What matters is y = Wx, not W itself
   
2. Error compensation
   - Quantization error in one column is compensated by others
   
3. Considers input distribution
   - Weights that process correlated inputs can compensate each other


USAGE WITH TRANSFORMERS:
════════════════════════

from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,                    # 4-bit quantization
    dataset="c4",              # Calibration dataset
    tokenizer=tokenizer,
    group_size=128,            # Per-group quantization
    desc_act=True,             # Activation ordering
    sym=False,                 # Asymmetric quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto",
)


PERFORMANCE:
════════════

7B Model Performance:
┌────────────────────────────────────────────────────────────┐
│ Format      │ Memory    │ Perplexity (vs FP16)            │
├────────────────────────────────────────────────────────────┤
│ FP16        │ 14 GB     │ baseline                        │
│ GPTQ 8-bit  │ 7 GB      │ +0.01 (negligible)              │
│ GPTQ 4-bit  │ 3.5 GB    │ +0.1-0.3 (small degradation)    │
│ GPTQ 3-bit  │ 2.6 GB    │ +0.5-1.0 (noticeable)           │
└────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# SECTION 4: AWQ - ACTIVATION-AWARE QUANTIZATION
# =============================================================================
"""
AWQ (Activation-Aware Weight Quantization):
═══════════════════════════════════════════

Key Insight: Not all weights are equally important!

Observation:
    1% of weights (those processing outlier activations) 
    contribute disproportionately to model quality.
    
    Protecting these "salient" weights is critical.

AWQ Strategy:
1. Identify salient weight channels (based on activation magnitude)
2. Scale these channels UP before quantization
3. Scale activations DOWN to compensate
4. Quantize the scaled weights

Mathematical Formulation:
    Original: y = W @ x
    Scaled:   y = (W * s) @ (x / s) = W' @ x'
    
    Where s is per-channel scaling factor
    
    Larger s for important channels → less quantization error
    
Scaling Factor Selection:
    s_i = (mean(|x_i|))^α
    α is typically 0.5 (square root)
"""


def awq_explanation():
    """Explain AWQ algorithm."""
    print("\n" + "="*70)
    print("AWQ: ACTIVATION-AWARE QUANTIZATION")
    print("="*70)
    
    print("""
AWQ KEY INSIGHT:
════════════════

Observation from analyzing LLMs:
    
    ┌─────────────────────────────────────────────────────────┐
    │ Weight Channel   │ Activation Magnitude │ Importance    │
    ├─────────────────────────────────────────────────────────┤
    │ 99% of channels  │ Normal (~1.0)        │ Standard      │
    │ 1% of channels   │ Outliers (10-100x)   │ CRITICAL      │
    └─────────────────────────────────────────────────────────┘
    
    These 1% of channels carry most of the model's capability!


AWQ ALGORITHM:
══════════════

Step 1: Collect Activation Statistics
    - Run calibration data through model
    - Track mean absolute activation per channel
    
Step 2: Compute Scaling Factors
    s_i = (mean(|X[:, i]|))^0.5
    
    Channels with larger activations get larger scales
    
Step 3: Apply Scaling (per-channel)
    W_scaled = W * s        # Scale up weights
    X_scaled = X / s        # Scale down activations
    
    Output unchanged: W_scaled @ X_scaled = W @ X
    
Step 4: Quantize Scaled Weights
    - Important channels are scaled up
    - Quantization error is relatively smaller
    - Quality preserved!


COMPARISON: GPTQ vs AWQ
═══════════════════════

┌───────────────────────────────────────────────────────────────────┐
│ Aspect            │ GPTQ                  │ AWQ                   │
├───────────────────────────────────────────────────────────────────┤
│ Approach          │ Error compensation    │ Importance weighting  │
│ Calibration       │ Required (slow)       │ Minimal (fast)        │
│ Quality (4-bit)   │ Very good             │ Very good             │
│ Speed             │ Slower quantization   │ Faster quantization   │
│ Kernel support    │ ExLlama, AutoGPTQ     │ AutoAWQ, vLLM         │
└───────────────────────────────────────────────────────────────────┘


USAGE:
══════

from awq import AutoAWQForCausalLM

# Load and quantize
model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)

model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
    }
)

# Save quantized model
model.save_quantized("llama-2-7b-awq")
""")


# =============================================================================
# SECTION 5: QUANTIZATION-AWARE TRAINING (QAT)
# =============================================================================
"""
QUANTIZATION-AWARE TRAINING:
════════════════════════════

QAT simulates quantization during training, allowing the model
to learn to be robust to quantization errors.

Forward Pass:
    Use fake quantization: quantize → dequantize
    This simulates quantization error while keeping gradients

Backward Pass:
    Use Straight-Through Estimator (STE)
    Gradient flows through as if no quantization happened

Benefits:
    - Better quality than PTQ for low bit-widths
    - Model learns to avoid quantization-sensitive weights
    
Drawbacks:
    - Requires training (expensive for LLMs)
    - Complex implementation
"""


class FakeQuantize(torch.autograd.Function):
    """Fake quantization with straight-through estimator."""
    
    @staticmethod
    def forward(ctx, x, bits=8, symmetric=True):
        if symmetric:
            max_abs = x.abs().max()
            qmax = 2**(bits-1) - 1
            scale = max_abs / qmax
        else:
            min_val, max_val = x.min(), x.max()
            qmax = 2**bits - 1
            scale = (max_val - min_val) / qmax
        
        # Quantize and dequantize
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -qmax if symmetric else 0, qmax)
        x_dq = x_q * scale
        
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient unchanged
        return grad_output, None, None


def fake_quantize(x, bits=8):
    """Apply fake quantization."""
    return FakeQuantize.apply(x, bits)


class QATLinear(nn.Module):
    """Linear layer with quantization-aware training."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Fake quantize weights during training
            weight_q = fake_quantize(self.linear.weight, self.bits)
            return F.linear(x, weight_q, self.linear.bias)
        else:
            # Use real quantized weights for inference
            return self.linear(x)


# =============================================================================
# SECTION 6: INFERENCE FORMATS
# =============================================================================

def inference_formats():
    """Explain different inference formats."""
    print("\n" + "="*70)
    print("INFERENCE FORMATS: GGUF, GGML, AND MORE")
    print("="*70)
    
    print("""
GGML/GGUF (llama.cpp):
══════════════════════

GGML: Original format by Georgi Gerganov
GGUF: Updated format with better metadata support

Quantization Types:
┌──────────────────────────────────────────────────────────────────┐
│ Type      │ Bits │ Description                                  │
├──────────────────────────────────────────────────────────────────┤
│ F32       │ 32   │ Full precision (baseline)                    │
│ F16       │ 16   │ Half precision                               │
│ Q8_0      │ 8    │ 8-bit, symmetric, block size 32              │
│ Q5_K_M    │ 5.5  │ 5-bit with importance matrix (good balance)  │
│ Q4_K_M    │ 4.5  │ 4-bit with importance matrix (recommended)   │
│ Q4_0      │ 4    │ 4-bit, simple                                │
│ Q3_K_M    │ 3.5  │ 3-bit with importance (aggressive)           │
│ Q2_K      │ 2.5  │ 2-bit (very aggressive, quality loss)        │
└──────────────────────────────────────────────────────────────────┘

K-Quants (Q4_K_M, Q5_K_M, etc.):
    - 'K' = mixture of quantization levels
    - Important layers get more bits
    - '_M' = medium quality/size trade-off
    - '_S' = small (more compression)


EXLLAMA/EXLLAMAV2:
══════════════════

Optimized inference for GPTQ models:
    - Custom CUDA kernels
    - Flash attention integration
    - Continuous batching
    - 2-3x faster than base GPTQ


VLLM QUANTIZATION:
══════════════════

vLLM supports multiple formats:
    - GPTQ (via AutoGPTQ)
    - AWQ (via AutoAWQ)
    - SqueezeLLM
    - FP8 (native)
    
Best for: High-throughput serving


CHOOSING A FORMAT:
══════════════════

┌─────────────────────────────────────────────────────────────────┐
│ Use Case               │ Recommended Format                     │
├─────────────────────────────────────────────────────────────────┤
│ Local/CPU inference    │ GGUF Q4_K_M or Q5_K_M                  │
│ GPU inference (single) │ AWQ or GPTQ 4-bit                      │
│ High-throughput server │ vLLM with AWQ/GPTQ                     │
│ Edge deployment        │ GGUF Q2_K or Q3_K_M                    │
│ Quality critical       │ GPTQ 8-bit or AWQ                      │
└─────────────────────────────────────────────────────────────────┘


CONVERTING BETWEEN FORMATS:
═══════════════════════════

# HuggingFace → GGUF
python convert.py model_path --outtype q4_k_m

# GPTQ → GGUF
python convert.py model_path --outtype q4_k_m

# AWQ model can be used directly with vLLM
from vllm import LLM
model = LLM("TheBloke/Llama-2-7B-AWQ", quantization="awq")
""")


# =============================================================================
# SECTION 7: PRACTICAL QUANTIZATION GUIDE
# =============================================================================

def quantization_guide():
    """Practical guide for choosing quantization."""
    print("\n" + "="*70)
    print("PRACTICAL QUANTIZATION GUIDE")
    print("="*70)
    
    print("""
DECISION FLOWCHART:
═══════════════════

                    ┌─────────────────────┐
                    │  What's your goal?  │
                    └──────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Min memory  │    │ Best quality│    │ Fast quant  │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ GGUF Q2/Q3  │    │ GPTQ 8-bit  │    │    AWQ      │
    │ or INT4     │    │ or QAT      │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘


QUALITY vs SIZE TRADE-OFF:
══════════════════════════

For Llama-2 7B (14 GB FP16):

┌────────────────────────────────────────────────────────────────┐
│ Format        │ Size    │ Perplexity │ Quality Assessment      │
├────────────────────────────────────────────────────────────────┤
│ FP16          │ 14 GB   │ 5.47       │ Baseline                │
│ GPTQ 8-bit    │ 7 GB    │ 5.48       │ Essentially identical   │
│ AWQ 4-bit     │ 3.9 GB  │ 5.60       │ Minimal degradation     │
│ GPTQ 4-bit    │ 3.5 GB  │ 5.65       │ Small degradation       │
│ GGUF Q4_K_M   │ 4.1 GB  │ 5.68       │ Good for CPU            │
│ GPTQ 3-bit    │ 2.6 GB  │ 6.10       │ Noticeable degradation  │
│ GGUF Q2_K     │ 2.3 GB  │ 7.20       │ Significant degradation │
└────────────────────────────────────────────────────────────────┘


RECOMMENDATIONS BY SCENARIO:
════════════════════════════

1. Production Serving (GPU):
   - AWQ 4-bit with vLLM
   - Best throughput/quality balance
   
2. Local Development (GPU):
   - GPTQ 4-bit with ExLlamaV2
   - Good speed, reasonable quality
   
3. CPU Inference:
   - GGUF Q4_K_M with llama.cpp
   - Optimized for CPU
   
4. Embedded/Mobile:
   - GGUF Q3_K_M or Q2_K
   - Maximum compression
   
5. Quality Critical:
   - GPTQ 8-bit or keep FP16
   - Minimal quality loss


COMMON PITFALLS:
════════════════

1. Wrong calibration data:
   - Use data similar to deployment use case
   - 128-1024 samples usually sufficient
   
2. Ignoring perplexity:
   - Always measure perplexity before deployment
   - Task-specific evaluation is even better
   
3. Over-quantizing:
   - Start with 4-bit, go lower only if needed
   - Test on your actual use case
   
4. Mixing quantization types:
   - Stick to one format for consistency
   - Conversion between formats can add error
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("QUANTIZATION FUNDAMENTALS FOR LLMs")
    print("="*70)
    
    # Basic quantization
    demonstrate_quantization()
    
    # PTQ example
    ptq_example()
    
    # GPTQ explanation
    gptq_explanation()
    
    # AWQ explanation
    awq_explanation()
    
    # Inference formats
    inference_formats()
    
    # Practical guide
    quantization_guide()
    
    print("\n" + "="*70)
    print("QUANTIZATION MODULE COMPLETE")
    print("="*70)
