"""
Automatic Mixed Precision (AMP) Training
=========================================

This module provides comprehensive coverage of AMP training in PyTorch,
including loss scaling, GradScaler, autocast, and best practices.

Key Topics:
1. Why Mixed Precision Training
2. PyTorch AMP API (autocast + GradScaler)
3. Loss Scaling Deep Dive
4. Dynamic vs Static Loss Scaling
5. Common Pitfalls and Solutions
6. BF16 AMP (no scaling needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math
from typing import Optional

# =============================================================================
# SECTION 1: THE CORE AMP COMPONENTS
# =============================================================================
"""
PyTorch AMP has two main components:

1. AUTOCAST: Automatically chooses precision for operations
   - Wraps forward pass
   - Matmuls/convs → FP16
   - Reductions/softmax → FP32 (for stability)

2. GRADSCALER: Handles loss scaling for FP16
   - Scales loss before backward (prevents underflow)
   - Unscales gradients before optimizer step
   - Dynamically adjusts scale factor
"""


class SimpleModel(nn.Module):
    """Simple model for AMP demonstrations."""
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1000):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm(x)
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# SECTION 2: BASIC AMP TRAINING LOOP
# =============================================================================

def basic_amp_training():
    """
    Basic AMP training loop with autocast and GradScaler.
    """
    print("\n" + "="*70)
    print("BASIC AMP TRAINING LOOP")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required for AMP demonstration")
        return
    
    device = torch.device("cuda")
    
    # Model and optimizer
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create GradScaler for FP16
    scaler = GradScaler()
    
    # Dummy data
    batch_size = 32
    x = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)
    
    print("\nTraining step with AMP:")
    print("-" * 50)
    
    # Standard training loop with AMP
    optimizer.zero_grad()
    
    # Step 1: Forward pass with autocast
    with autocast():
        output = model(x)
        loss = F.cross_entropy(output, target)
    
    print(f"Output dtype: {output.dtype}")  # float16
    print(f"Loss dtype: {loss.dtype}")  # float32 (cross_entropy promotes)
    print(f"Loss value: {loss.item():.4f}")
    
    # Step 2: Scaled backward pass
    scaler.scale(loss).backward()
    
    # Step 3: Unscale and step
    scaler.step(optimizer)
    
    # Step 4: Update scaler
    scaler.update()
    
    print(f"Current scale: {scaler.get_scale()}")
    print("Training step complete!")


# =============================================================================
# SECTION 3: LOSS SCALING DEEP DIVE
# =============================================================================
"""
LOSS SCALING: Preventing Gradient Underflow

Problem:
- Gradients can be very small (1e-5 to 1e-8)
- FP16 min normal value: ~6e-5
- Small gradients underflow to 0 → no learning!

Solution: Scale up the loss before backward pass

    scaled_loss = loss * scale_factor
    scaled_loss.backward()
    # Gradients are now: grad * scale_factor
    
    # Before optimizer step, unscale:
    grad = grad / scale_factor

The GradScaler automates this:
1. Tracks a scale factor (starts at 2^16 = 65536)
2. Checks for inf/nan in gradients
3. If inf/nan: skip step, reduce scale by 0.5
4. If OK for N steps: increase scale by 2
"""


def demonstrate_loss_scaling():
    """
    Demonstrate how loss scaling prevents gradient underflow.
    """
    print("\n" + "="*70)
    print("LOSS SCALING DEMONSTRATION")
    print("="*70)
    
    # Simulate small gradients
    small_grad_fp32 = torch.tensor([1e-6, 1e-7, 1e-8])
    print(f"Small gradients (FP32): {small_grad_fp32.tolist()}")
    
    # Convert to FP16 - underflow!
    small_grad_fp16 = small_grad_fp32.half()
    print(f"Converted to FP16: {small_grad_fp16.tolist()}")
    print("Problem: Small values become 0!")
    
    # With loss scaling
    scale = 65536.0  # 2^16
    scaled_grads = small_grad_fp32 * scale
    scaled_fp16 = scaled_grads.half()
    unscaled = scaled_fp16.float() / scale
    
    print(f"\nWith loss scaling (scale={scale}):")
    print(f"Scaled gradients: {scaled_grads.tolist()}")
    print(f"Scaled in FP16: {scaled_fp16.tolist()}")
    print(f"After unscaling: {unscaled.tolist()}")
    print("Solution: Values preserved!")
    
    print("\n" + "-"*50)
    print("DYNAMIC SCALING BEHAVIOR:")
    print("-"*50)
    print("""
GradScaler adjusts scale dynamically:

Initial scale: 65536 (2^16)

If gradients have inf/nan:
    scale = scale * backoff_factor (default 0.5)
    Skip optimizer step
    
If N consecutive steps OK (default N=2000):
    scale = scale * growth_factor (default 2.0)

This finds the optimal scale automatically!
""")


# =============================================================================
# SECTION 4: AUTOCAST OPERATION CATEGORIES
# =============================================================================
"""
AUTOCAST categorizes operations:

1. FP16-SAFE (run in FP16):
   - Linear layers (matmul)
   - Convolutions
   - Batch/Layer normalization
   - RNN cells
   
2. FP32-REQUIRED (always FP32):
   - Softmax, log_softmax
   - Loss functions
   - Exponentials, log
   - Large reductions (sum over many elements)
   - Layer norm (inputs converted to FP32)
   
3. PROMOTE (highest input precision):
   - Pointwise operations
   - Concatenation
   - Element-wise comparisons
"""


def demonstrate_autocast_dtypes():
    """
    Show how autocast chooses dtypes for different operations.
    """
    print("\n" + "="*70)
    print("AUTOCAST DTYPE SELECTION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    device = torch.device("cuda")
    x = torch.randn(32, 1024, device=device)
    
    # Linear layer
    linear = nn.Linear(1024, 1024).to(device)
    
    print("\nOperation dtypes with autocast(dtype=torch.float16):")
    print("-" * 50)
    
    with autocast(dtype=torch.float16):
        # FP16 operations
        y_linear = linear(x)
        print(f"Linear output: {y_linear.dtype}")
        
        y_matmul = torch.matmul(x, x.T)
        print(f"Matmul output: {y_matmul.dtype}")
        
        # FP32 operations (promoted for stability)
        y_softmax = F.softmax(y_linear, dim=-1)
        print(f"Softmax output: {y_softmax.dtype}")
        
        y_layernorm = F.layer_norm(y_linear, [1024])
        print(f"LayerNorm output: {y_layernorm.dtype}")
        
        # Reductions
        y_sum = y_linear.sum()
        print(f"Sum output: {y_sum.dtype}")
        
        y_mean = y_linear.mean()
        print(f"Mean output: {y_mean.dtype}")


# =============================================================================
# SECTION 5: COMPLETE AMP TRAINING EXAMPLE
# =============================================================================

def complete_amp_training_loop():
    """
    Complete training loop with all AMP best practices.
    """
    print("\n" + "="*70)
    print("COMPLETE AMP TRAINING LOOP")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    device = torch.device("cuda")
    
    # Model setup
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # Training settings
    num_steps = 10
    accumulation_steps = 4
    
    print(f"\nTraining for {num_steps} steps with gradient accumulation={accumulation_steps}")
    print("-" * 50)
    
    for step in range(num_steps):
        # Accumulate gradients
        for micro_step in range(accumulation_steps):
            x = torch.randn(16, 1024, device=device)
            target = torch.randint(0, 1000, (16,), device=device)
            
            with autocast():
                output = model(x)
                loss = F.cross_entropy(output, target)
                loss = loss / accumulation_steps  # Scale for accumulation
            
            # Scaled backward
            scaler.scale(loss).backward()
        
        # Gradient clipping (with unscaling)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for inf/nan and step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if step % 2 == 0:
            print(f"Step {step}: loss={loss.item()*accumulation_steps:.4f}, scale={scaler.get_scale():.0f}")
    
    print("\nTraining complete!")


# =============================================================================
# SECTION 6: BF16 TRAINING (NO SCALING NEEDED)
# =============================================================================
"""
BF16 TRAINING: Simpler than FP16

BF16 has the same range as FP32, so:
- No gradient underflow (no loss scaling needed!)
- No loss overflow
- Simpler code

When to use BF16:
- NVIDIA Ampere (A100) or newer
- Google TPUs
- AMD MI200+
"""


def bf16_training_example():
    """
    BF16 training - simpler than FP16.
    """
    print("\n" + "="*70)
    print("BF16 TRAINING (NO LOSS SCALING)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    # Check BF16 support
    if not torch.cuda.is_bf16_supported():
        print("BF16 not supported on this GPU")
        return
    
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\nBF16 training is simpler - no GradScaler needed!")
    print("-" * 50)
    
    for step in range(5):
        x = torch.randn(32, 1024, device=device)
        target = torch.randint(0, 1000, (32,), device=device)
        
        optimizer.zero_grad()
        
        # Just use autocast with bfloat16
        with autocast(dtype=torch.bfloat16):
            output = model(x)
            loss = F.cross_entropy(output, target)
        
        # No scaling needed!
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: loss={loss.item():.4f}, output dtype={output.dtype}")
    
    print("\nNo overflow/underflow issues with BF16!")


# =============================================================================
# SECTION 7: COMMON PITFALLS AND SOLUTIONS
# =============================================================================

def common_pitfalls():
    """
    Document common AMP pitfalls and solutions.
    """
    print("\n" + "="*70)
    print("COMMON AMP PITFALLS AND SOLUTIONS")
    print("="*70)
    
    print("""
PITFALL 1: Forgetting to unscale before gradient clipping
──────────────────────────────────────────────────────────
WRONG:
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clips scaled grads!
    scaler.step(optimizer)

CORRECT:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Unscale first!
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)


PITFALL 2: Using autocast for the entire training step
───────────────────────────────────────────────────────
WRONG:
    with autocast():
        output = model(x)
        loss = criterion(output, target)
        loss.backward()  # Don't do backward in autocast!
        optimizer.step()

CORRECT:
    with autocast():
        output = model(x)
        loss = criterion(output, target)
    # Outside autocast for backward
    scaler.scale(loss).backward()


PITFALL 3: Not handling inf/nan in custom operations
────────────────────────────────────────────────────
Custom ops may need explicit dtype handling:

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        # Forces FP32 inputs
        return custom_op(x)
    
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return custom_grad(grad)


PITFALL 4: Calling scaler.step() after skipped steps
─────────────────────────────────────────────────────
scaler.step() returns None if step was skipped.
Learning rate schedulers should check:

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    
    if scaler.get_scale() >= scale_before:
        # Step was not skipped
        scheduler.step()


PITFALL 5: Not using autocast for inference
───────────────────────────────────────────
Autocast also helps inference speed:

    model.eval()
    with torch.no_grad():
        with autocast():
            output = model(x)


PITFALL 6: Mixing autocast with torch.compile incorrectly
─────────────────────────────────────────────────────────
Put autocast OUTSIDE the compiled function:

    @torch.compile
    def forward_fn(model, x):
        return model(x)
    
    with autocast():
        output = forward_fn(model, x)
""")


# =============================================================================
# SECTION 8: PERFORMANCE COMPARISON
# =============================================================================

def benchmark_precision():
    """
    Benchmark different precision modes.
    """
    print("\n" + "="*70)
    print("PRECISION PERFORMANCE BENCHMARK")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA required for benchmark")
        return
    
    device = torch.device("cuda")
    
    # Larger model for meaningful benchmark
    model_fp32 = SimpleModel(2048, 4096, 2048).to(device)
    model_fp16 = SimpleModel(2048, 4096, 2048).to(device).half()
    
    x = torch.randn(64, 2048, device=device)
    x_half = x.half()
    
    # Warmup
    for _ in range(10):
        _ = model_fp32(x)
        _ = model_fp16(x_half)
    torch.cuda.synchronize()
    
    num_iters = 100
    
    # FP32 benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = model_fp32(x)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / num_iters * 1000
    
    # FP16 with autocast
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        with autocast():
            _ = model_fp32(x)
    torch.cuda.synchronize()
    amp_time = (time.time() - start) / num_iters * 1000
    
    # Pure FP16
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = model_fp16(x_half)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / num_iters * 1000
    
    print(f"\nForward pass time (batch=64, hidden=4096):")
    print(f"  FP32:       {fp32_time:.2f} ms")
    print(f"  AMP (FP16): {amp_time:.2f} ms (speedup: {fp32_time/amp_time:.2f}x)")
    print(f"  Pure FP16:  {fp16_time:.2f} ms (speedup: {fp32_time/fp16_time:.2f}x)")
    
    # Memory comparison
    print(f"\nModel memory:")
    print(f"  FP32: {sum(p.numel() * 4 for p in model_fp32.parameters()) / 1e6:.1f} MB")
    print(f"  FP16: {sum(p.numel() * 2 for p in model_fp16.parameters()) / 1e6:.1f} MB")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("="*70)
    print("AUTOMATIC MIXED PRECISION (AMP) TRAINING")
    print("="*70)
    
    # Basic AMP
    basic_amp_training()
    
    # Loss scaling explanation
    demonstrate_loss_scaling()
    
    # Autocast dtypes
    demonstrate_autocast_dtypes()
    
    # Complete training loop
    complete_amp_training_loop()
    
    # BF16 training
    bf16_training_example()
    
    # Common pitfalls
    common_pitfalls()
    
    # Benchmark
    benchmark_precision()
    
    print("\n" + "="*70)
    print("AMP MODULE COMPLETE")
    print("="*70)
