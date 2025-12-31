"""
05_attention_pitfalls.py - Common Issues and Solutions

This module covers common pitfalls, debugging techniques, and
best practices for working with attention mechanisms.

Topics:
1. Numerical Stability Issues
2. Memory Problems
3. Training Instabilities
4. Attention Pattern Pathologies
5. Positional Encoding Issues
6. Debugging Techniques

Run: python 05_attention_pitfalls.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional

# ============================================================================
# NUMERICAL STABILITY ISSUES
# ============================================================================

def explain_numerical_stability():
    """Explain numerical stability issues in attention."""
    print("\n" + "="*70)
    print(" NUMERICAL STABILITY ISSUES")
    print(" When math goes wrong")
    print("="*70)
    
    print("""
    ISSUE 1: SOFTMAX OVERFLOW/UNDERFLOW
    ─────────────────────────────────────────────────────────────────
    
    Problem: exp(x) overflows for large x, underflows for small x
    
    Example:
    scores = [1000, 1, 2]
    exp(scores) = [inf, 2.7, 7.4]  # Overflow!
    
    Solution: Subtract max before softmax (numerically stable)
    
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    PyTorch's F.softmax does this automatically.
    """)
    
    # Demonstrate
    print(" DEMONSTRATION:")
    print("-" * 50)
    
    # Unstable (manual, wrong)
    scores = torch.tensor([1000.0, 1.0, 2.0])
    
    print(f" Scores: {scores}")
    
    # Wrong way
    try:
        exp_scores = torch.exp(scores)
        wrong_softmax = exp_scores / exp_scores.sum()
        print(f" Naive softmax: {wrong_softmax} (contains NaN/inf!)")
    except:
        print(f" Naive softmax: OVERFLOW")
    
    # Right way (PyTorch)
    correct_softmax = F.softmax(scores, dim=-1)
    print(f" Stable softmax: {correct_softmax}")
    
    print("""
    
    ISSUE 2: SCALE FACTOR IMPORTANCE
    ─────────────────────────────────────────────────────────────────
    
    Problem: Without scaling, dot products grow with dimension
    
    For d-dimensional vectors with unit variance:
    E[q·k] = 0, Var[q·k] = d
    
    Large d → large variance → softmax saturates
    
    Solution: Scale by 1/√d
    
    Attention = softmax(QK^T / √d) V
    """)
    
    # Demonstrate scaling issue
    print(" SCALING DEMONSTRATION:")
    print("-" * 50)
    
    for d in [64, 512, 4096]:
        q = torch.randn(1000, d)
        k = torch.randn(1000, d)
        
        dots = (q * k).sum(dim=-1)
        scaled_dots = dots / math.sqrt(d)
        
        # Softmax entropy (lower = more peaked)
        probs = F.softmax(dots.unsqueeze(0), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        scaled_probs = F.softmax(scaled_dots.unsqueeze(0), dim=-1)
        scaled_entropy = -(scaled_probs * torch.log(scaled_probs + 1e-10)).sum()
        
        print(f" d={d:4d}: unscaled_std={dots.std():.1f}, "
              f"scaled_std={scaled_dots.std():.2f}, "
              f"entropy_gain={scaled_entropy - entropy:.1f}")
    
    print("""
    
    ISSUE 3: ATTENTION SINK / FIRST TOKEN BIAS
    ─────────────────────────────────────────────────────────────────
    
    Problem: Models often over-attend to first token (especially [BOS])
    
    Why: First token accumulates "junk" attention when nothing relevant
    
    Solutions:
    1. Attention sinks: Keep first few tokens in KV cache always
    2. Sink tokens: Add learnable sink tokens
    3. Masking: Don't include special tokens in attention stats
    """)

# ============================================================================
# MEMORY ISSUES
# ============================================================================

def explain_memory_issues():
    """Explain memory issues and solutions."""
    print("\n" + "="*70)
    print(" MEMORY ISSUES")
    print(" When GPU runs out of memory")
    print("="*70)
    
    print("""
    ISSUE 1: QUADRATIC MEMORY FROM ATTENTION MATRIX
    ─────────────────────────────────────────────────────────────────
    
    Memory for N×N attention matrix (FP16):
    - N=1K:   2 MB
    - N=4K:   32 MB
    - N=16K:  512 MB
    - N=64K:  8 GB per layer!
    
    With batch size B and H heads:
    Memory = B × H × N² × 2 bytes
    
    SOLUTIONS:
    
    1. Flash Attention: O(N) memory via tiling
       - Use torch.nn.functional.scaled_dot_product_attention
       - Automatic in PyTorch 2.0+
    
    2. Gradient Checkpointing: Recompute instead of store
       - torch.utils.checkpoint.checkpoint()
       - Trades 30% more compute for massive memory savings
    
    3. Reduce precision:
       - FP16/BF16 instead of FP32
       - FP8 for inference (emerging)
    """)
    
    if torch.cuda.is_available():
        print("\n MEMORY DEMONSTRATION:")
        print("-" * 50)
        
        for seq_len in [1024, 2048, 4096, 8192]:
            batch, heads, head_dim = 4, 8, 64
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            q = torch.randn(batch, heads, seq_len, head_dim, 
                          device='cuda', dtype=torch.float16)
            k = torch.randn(batch, heads, seq_len, head_dim,
                          device='cuda', dtype=torch.float16)
            v = torch.randn(batch, heads, seq_len, head_dim,
                          device='cuda', dtype=torch.float16)
            
            # Standard attention
            torch.cuda.reset_peak_memory_stats()
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)
            torch.cuda.synchronize()
            standard_mem = torch.cuda.max_memory_allocated() / 1e6
            
            # Flash attention
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            q2 = q.clone()
            k2 = k.clone()
            v2 = v.clone()
            _ = F.scaled_dot_product_attention(q2, k2, v2)
            torch.cuda.synchronize()
            flash_mem = torch.cuda.max_memory_allocated() / 1e6
            
            print(f" seq_len={seq_len:4d}: Standard={standard_mem:6.1f}MB, "
                  f"Flash={flash_mem:6.1f}MB, Savings={1-flash_mem/standard_mem:.0%}")
    
    print("""
    
    ISSUE 2: KV CACHE EXPLOSION
    ─────────────────────────────────────────────────────────────────
    
    During autoregressive generation, must cache K and V:
    
    KV cache size = 2 × layers × heads × seq_len × head_dim × batch
    
    For LLaMA 70B at 4K context:
    = 2 × 80 × 64 × 4096 × 128 × 2 bytes = 10.7 GB!
    
    SOLUTIONS:
    
    1. Grouped Query Attention (GQA):
       - Fewer KV heads than Q heads
       - LLaMA 2 70B: 64 Q heads, 8 KV heads → 8x smaller cache
    
    2. Sliding Window:
       - Only cache last W tokens
       - Mistral: W=4096
    
    3. Quantized KV Cache:
       - INT8 or INT4 KV cache
       - 2-4x memory reduction
    
    4. Paged Attention (vLLM):
       - Dynamic memory allocation
       - Virtual memory for KV cache
    """)

# ============================================================================
# TRAINING INSTABILITIES
# ============================================================================

def explain_training_instabilities():
    """Explain training instabilities in attention models."""
    print("\n" + "="*70)
    print(" TRAINING INSTABILITIES")
    print(" When training goes wrong")
    print("="*70)
    
    print("""
    ISSUE 1: LOSS SPIKES / NaN LOSSES
    ─────────────────────────────────────────────────────────────────
    
    Symptoms:
    - Sudden loss increase
    - Loss becomes NaN
    - Gradients explode
    
    Common causes:
    
    1. Learning rate too high
       → Solution: Warmup, lower LR, use LR scheduler
    
    2. Attention logits explode
       → Solution: QK layer norm (used in LLaMA 2)
       
       # Before attention
       q = layer_norm(q)
       k = layer_norm(k)
    
    3. FP16 overflow
       → Solution: Use BF16, or gradient scaling (AMP)
    
    4. Bad initialization
       → Solution: Proper init (Xavier/He), smaller init scale for deep models
    
    ISSUE 2: TRAINING DOESN'T CONVERGE
    ─────────────────────────────────────────────────────────────────
    
    Symptoms:
    - Loss plateaus early
    - Poor validation performance
    
    Common causes:
    
    1. Pre-norm vs Post-norm confusion
       
       Post-norm (original transformer, harder to train):
       x = x + Attention(LayerNorm(x))  # WRONG order
       x = LayerNorm(x + Attention(x))  # Correct post-norm
       
       Pre-norm (easier to train, standard now):
       x = x + Attention(LayerNorm(x))  # Correct pre-norm
    
    2. Missing residual connections
       → Always add skip connections!
    
    3. Wrong attention mask
       → Causal mask for autoregressive, padding mask for batching
    
    ISSUE 3: ATTENTION COLLAPSE
    ─────────────────────────────────────────────────────────────────
    
    Symptoms:
    - All heads learn similar patterns
    - Attention becomes uniform
    - Model ignores input structure
    
    Solutions:
    
    1. Head dropout: Randomly drop entire heads during training
    2. Attention regularization: Encourage diversity between heads
    3. Multi-query → Grouped Query: Force different groups
    """)
    
    # Demonstrate attention mask importance
    print("\n CAUSAL MASK DEMONSTRATION:")
    print("-" * 50)
    
    seq_len = 5
    
    # Correct causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f" Correct causal mask:")
    print(causal_mask)
    
    # Common mistake: wrong mask direction
    wrong_mask = torch.triu(torch.ones(seq_len, seq_len))
    print(f"\n WRONG mask (future only!):")
    print(wrong_mask)

# ============================================================================
# ATTENTION PATTERN PATHOLOGIES
# ============================================================================

def explain_attention_pathologies():
    """Explain pathological attention patterns."""
    print("\n" + "="*70)
    print(" ATTENTION PATTERN PATHOLOGIES")
    print(" When attention behaves unexpectedly")
    print("="*70)
    
    print("""
    PATHOLOGY 1: UNIFORM ATTENTION
    ─────────────────────────────────────────────────────────────────
    
    All positions get equal attention weight:
    
    [0.2, 0.2, 0.2, 0.2, 0.2]
    
    Causes:
    - Scores too similar (needs more expressiveness)
    - Temperature too high (softmax too soft)
    - Poorly initialized Q, K projections
    
    Detection:
    entropy = -sum(p * log(p))
    High entropy = more uniform
    
    Solutions:
    - Lower temperature (larger scale)
    - More expressive projections
    - Check initialization
    
    PATHOLOGY 2: SPARSE/PEAKED ATTENTION
    ─────────────────────────────────────────────────────────────────
    
    One position gets all attention:
    
    [0.0, 0.0, 1.0, 0.0, 0.0]
    
    Causes:
    - Temperature too low
    - One key dominates
    - Potential embedding collapse
    
    Not always bad! But check if it's learned or a bug.
    
    PATHOLOGY 3: DIAGONAL DOMINANCE
    ─────────────────────────────────────────────────────────────────
    
    Every position only attends to itself:
    
    ┌─────────────────┐
    │ 1  0  0  0  0   │
    │ 0  1  0  0  0   │
    │ 0  0  1  0  0   │
    │ 0  0  0  1  0   │
    │ 0  0  0  0  1   │
    └─────────────────┘
    
    This is a no-op! Attention isn't mixing information.
    
    Causes:
    - Q and K too similar (identity-like)
    - Residual connection dominates (attention contribution ignored)
    
    PATHOLOGY 4: FIRST/LAST TOKEN SINKS
    ─────────────────────────────────────────────────────────────────
    
    All positions attend to first (or last) token:
    
    ┌─────────────────┐
    │ 1  0  0  0  0   │
    │ 1  0  0  0  0   │
    │ 1  0  0  0  0   │
    │ 1  0  0  0  0   │
    │ 1  0  0  0  0   │
    └─────────────────┘
    
    The [BOS] token becomes an "attention sink".
    
    Actually learned behavior! Models dump "junk" attention there.
    Can cause issues when removing first token.
    
    PATHOLOGY 5: POSITIONAL OVER-RELIANCE
    ─────────────────────────────────────────────────────────────────
    
    Attention based purely on position, not content:
    
    All sequences show same attention pattern regardless of content.
    
    Causes:
    - Position embeddings too strong
    - Content projections too weak
    - Relative position bias dominates
    
    Check by:
    - Visualizing attention across different inputs
    - Should see input-dependent patterns
    """)

# ============================================================================
# POSITIONAL ENCODING ISSUES
# ============================================================================

def explain_positional_issues():
    """Explain positional encoding issues."""
    print("\n" + "="*70)
    print(" POSITIONAL ENCODING ISSUES")
    print(" When position information goes wrong")
    print("="*70)
    
    print("""
    ISSUE 1: LENGTH EXTRAPOLATION
    ─────────────────────────────────────────────────────────────────
    
    Problem: Trained on length N, fails on length > N
    
    Learned positions:
    - Fixed embedding table
    - Position N+1 has no embedding!
    - Model collapses on longer sequences
    
    Sinusoidal:
    - Can extrapolate mathematically
    - But not trained on those positions
    - Performance degrades smoothly
    
    RoPE:
    - Better extrapolation than learned
    - Still degrades beyond training length
    - Solutions: NTK-aware scaling, YaRN
    
    ALiBi:
    - Best extrapolation
    - Simple linear bias
    - Works at 2-8x training length
    
    ISSUE 2: POSITION EMBEDDING SCALE
    ─────────────────────────────────────────────────────────────────
    
    Problem: Position embeddings dominate content
    
    If ||position|| >> ||content||:
    - Attention based on position, not content
    - Model ignores actual tokens
    
    Solutions:
    - Normalize position embeddings
    - Scale down position embeddings
    - Use relative positions (content-agnostic)
    
    ISSUE 3: ROPE FREQUENCY SELECTION
    ─────────────────────────────────────────────────────────────────
    
    RoPE uses different frequencies per dimension:
    
    freq_i = base^(-2i/d)
    
    Default base=10000 works for ~4K context.
    
    For longer context:
    - Increase base (NTK-aware): base' = base * (L/L_train)
    - Interpolation: Scale positions down
    - YaRN: Combination of techniques
    
    ISSUE 4: ALIBI HEAD ASSIGNMENT
    ─────────────────────────────────────────────────────────────────
    
    ALiBi uses different slopes per head:
    
    m_i = 2^(-8/H * i)  for head i
    
    Head 0: Steep slope (very local attention)
    Head H: Shallow slope (more global attention)
    
    If you change number of heads, must recalculate slopes!
    """)

# ============================================================================
# DEBUGGING TECHNIQUES
# ============================================================================

def explain_debugging():
    """Explain debugging techniques for attention."""
    print("\n" + "="*70)
    print(" DEBUGGING TECHNIQUES")
    print(" How to find and fix attention issues")
    print("="*70)
    
    print("""
    TECHNIQUE 1: ATTENTION VISUALIZATION
    ─────────────────────────────────────────────────────────────────
    
    Visualize attention weights to spot issues:
    
    ```python
    import matplotlib.pyplot as plt
    
    # Get attention weights
    with torch.no_grad():
        output, attn_weights = model(x, return_attention=True)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_weights[0, 0].cpu(), cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.title('Attention weights (Head 0)')
    plt.show()
    ```
    
    What to look for:
    - Diagonal patterns (position-based)
    - Vertical lines (attention sinks)
    - Uniform patterns (no differentiation)
    - Block patterns (segment attention)
    
    TECHNIQUE 2: ATTENTION STATISTICS
    ─────────────────────────────────────────────────────────────────
    
    Compute statistics to detect issues:
    
    ```python
    # Entropy (higher = more uniform)
    entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
    
    # Max attention (higher = more peaked)
    max_attn = attn.max(dim=-1).values.mean()
    
    # Effective context (how many positions contribute)
    # Using entropy: exp(entropy) approximates effective context
    effective_context = torch.exp(entropy)
    
    print(f"Entropy: {entropy:.3f}")
    print(f"Max attention: {max_attn:.3f}")
    print(f"Effective context: {effective_context:.1f}")
    ```
    
    TECHNIQUE 3: GRADIENT ANALYSIS
    ─────────────────────────────────────────────────────────────────
    
    Check gradient flow through attention:
    
    ```python
    # Gradient of output w.r.t. input
    x.requires_grad_(True)
    output = model(x)
    output.sum().backward()
    
    # Check gradient magnitude
    grad_norm = x.grad.norm()
    print(f"Input gradient norm: {grad_norm:.4f}")
    
    # Zero gradient = no gradient flow (problem!)
    ```
    
    TECHNIQUE 4: ABLATION STUDIES
    ─────────────────────────────────────────────────────────────────
    
    Test components in isolation:
    
    1. Remove attention (identity attention):
       attn_weights = torch.eye(seq_len)
       → How much does removing attention hurt?
    
    2. Remove specific heads:
       Mask out individual heads
       → Which heads are important?
    
    3. Remove position encodings:
       pos_emb = 0
       → Is model relying too much on position?
    
    TECHNIQUE 5: SANITY CHECKS
    ─────────────────────────────────────────────────────────────────
    
    Always verify:
    
    1. Attention weights sum to 1:
       assert torch.allclose(attn.sum(dim=-1), torch.ones(...))
    
    2. Causal mask is correct:
       # No future information leaks
       assert (attn.triu(diagonal=1) == 0).all()
    
    3. Shapes are correct:
       # Q @ K^T should be (batch, heads, seq, seq)
       assert scores.shape == (batch, heads, seq_len, seq_len)
    
    4. No NaN/Inf:
       assert torch.isfinite(output).all()
    """)
    
    # Demonstrate attention statistics
    print("\n ATTENTION STATISTICS DEMO:")
    print("-" * 50)
    
    seq_len = 10
    
    # Different attention patterns
    patterns = {
        "Uniform": torch.ones(seq_len, seq_len) / seq_len,
        "Peaked": F.softmax(torch.randn(seq_len, seq_len) * 10, dim=-1),
        "Diagonal": torch.eye(seq_len),
        "Causal uniform": torch.tril(torch.ones(seq_len, seq_len)),
    }
    
    # Normalize causal
    patterns["Causal uniform"] = patterns["Causal uniform"] / patterns["Causal uniform"].sum(dim=-1, keepdim=True)
    
    print(f" {'Pattern':<15} {'Entropy':<10} {'Max':<10} {'Eff. Context'}")
    print("-" * 50)
    
    for name, attn in patterns.items():
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
        max_attn = attn.max(dim=-1).values.mean()
        eff_ctx = torch.exp(entropy)
        print(f" {name:<15} {entropy.item():<10.3f} {max_attn.item():<10.3f} {eff_ctx.item():.1f}")

# ============================================================================
# BEST PRACTICES
# ============================================================================

def print_best_practices():
    """Print best practices for attention."""
    print("\n" + "="*70)
    print(" BEST PRACTICES")
    print("="*70)
    
    print("""
    IMPLEMENTATION:
    ─────────────────────────────────────────────────────────────────
    
    1. Always use Flash Attention (via SDPA)
       output = F.scaled_dot_product_attention(q, k, v)
       # Faster, less memory, fused kernel
    
    2. Use proper dtypes
       # BF16 for training (avoids FP16 overflow)
       # FP16 for inference
       # Never FP32 for attention (too slow/memory)
    
    3. Pre-allocate buffers for inference
       # Avoid memory fragmentation
    
    4. Use GQA for inference efficiency
       # Smaller KV cache = faster
    
    TRAINING:
    ─────────────────────────────────────────────────────────────────
    
    1. Use Pre-LN (LayerNorm before attention)
       # More stable training
    
    2. Initialize properly
       # Xavier/He init
       # Scale down for deeper models
    
    3. Warmup learning rate
       # Attention is sensitive to LR
    
    4. Gradient clipping
       # Prevents explosion
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    5. Use QK LayerNorm for large models
       # Prevents attention logit explosion
    
    DEBUGGING:
    ─────────────────────────────────────────────────────────────────
    
    1. Visualize attention patterns
    2. Check for NaN/Inf regularly
    3. Monitor attention entropy
    4. Verify masks are correct
    5. Test on small scale first
    
    PRODUCTION:
    ─────────────────────────────────────────────────────────────────
    
    1. Use optimized inference libraries
       # vLLM, TensorRT-LLM, etc.
    
    2. Quantize KV cache
       # INT8 or lower
    
    3. Use speculative decoding
       # For latency-sensitive applications
    
    4. Implement proper batching
       # Continuous batching for throughput
    
    5. Monitor attention memory
       # KV cache can grow unexpectedly
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " ATTENTION PITFALLS AND DEBUGGING ".center(68) + "║")
    print("║" + " Common issues and how to fix them ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    explain_numerical_stability()
    explain_memory_issues()
    explain_training_instabilities()
    explain_attention_pathologies()
    explain_positional_issues()
    explain_debugging()
    print_best_practices()
    
    print("\n" + "="*70)
    print(" Knowing the pitfalls helps you build better models!")
    print("="*70)
