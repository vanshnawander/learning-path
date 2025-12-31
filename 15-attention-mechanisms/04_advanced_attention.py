"""
04_advanced_attention.py - Advanced Attention Mechanisms

This module covers cutting-edge alternatives to standard attention:
- RWKV (Receptance Weighted Key Value)
- Mamba / State Space Models
- Gated Attention
- Retention Networks
- Hyena / Long Convolutions

The Quest: O(N) complexity with attention-like quality

Timeline:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2020: Linear Attention attempts (Performer, Linformer)                      │
│ 2021: S4 - Structured State Spaces for Sequences                           │
│ 2022: RWKV - RNN-Transformer hybrid                                        │
│ 2023: Mamba - Selective State Spaces (breakthrough!)                       │
│ 2023: RetNet - Retention Networks                                          │
│ 2024: Mamba-2, RWKV-6, Hybrid architectures                               │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 04_advanced_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple

# ============================================================================
# PROFILING
# ============================================================================

def profile_fn(func, warmup=5, iterations=20):
    """Profile with CUDA timing."""
    if torch.cuda.is_available():
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations
    else:
        for _ in range(warmup):
            func()
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        return (time.perf_counter() - start) * 1000 / iterations

# ============================================================================
# RWKV - RECEPTANCE WEIGHTED KEY VALUE
# ============================================================================

class RWKVTimeMixing(nn.Module):
    """
    RWKV Time Mixing Layer - Linear attention with learned decay.
    
    RWKV combines ideas from:
    - Linear attention (no softmax)
    - RNNs (sequential state updates)
    - Transformers (parallelizable training)
    
    Key innovation: Token shift + exponential decay
    
    wkv_t = Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} v_i + e^{u+k_t} v_t
            ─────────────────────────────────────────────────────
            Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}
    
    Where:
    - w: learned decay (how fast to forget)
    - u: learned bonus for current token
    - k, v: key and value from input
    """
    
    def __init__(self, d_model: int, layer_id: int):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id
        
        # Time mixing coefficients (for token shift)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Decay and bonus parameters
        self.time_decay = nn.Parameter(torch.ones(d_model))  # w
        self.time_first = nn.Parameter(torch.ones(d_model))  # u (bonus for current)
        
        # Linear projections
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)  # Gate
        self.output = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: (batch, seq_len, d_model)
            state: (num, denom, prev_x) for RNN-mode inference
        """
        batch_size, seq_len, _ = x.shape
        
        # Token shift: mix current with previous token
        if state is not None:
            prev_x = state[2]
        else:
            prev_x = torch.zeros_like(x[:, :1, :])
        
        # Shift by concatenating prev with current (excluding last)
        shifted = torch.cat([prev_x, x[:, :-1, :]], dim=1)
        
        # Time mixing
        xk = x * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + shifted * (1 - self.time_mix_r)
        
        # Project to K, V, R
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))  # Gating
        
        # WKV computation (simplified - real impl uses CUDA kernel)
        # This is the "attention" part with exponential decay
        wkv = self._compute_wkv(k, v, self.time_decay, self.time_first)
        
        # Apply receptance gate and output projection
        output = self.output(r * wkv)
        
        # Update state for next step
        new_state = (None, None, x[:, -1:, :])  # Simplified
        
        return output, new_state
    
    def _compute_wkv(self, k, v, w, u):
        """
        Simplified WKV computation.
        Real RWKV uses a custom CUDA kernel for efficiency.
        """
        batch, seq_len, d = k.shape
        
        # Initialize accumulators
        output = torch.zeros_like(v)
        
        # Exponential decay
        w = -torch.exp(w)  # Make decay negative for exp
        
        # Sequential computation (parallel version exists but complex)
        aa = torch.zeros(batch, d, device=k.device)  # Numerator accumulator
        bb = torch.zeros(batch, d, device=k.device)  # Denominator accumulator
        pp = torch.full((batch, d), -1e30, device=k.device)  # Previous max for stability
        
        for t in range(seq_len):
            kk = k[:, t]
            vv = v[:, t]
            
            ww = u + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            
            output[:, t] = (e1 * aa + e2 * vv) / (e1 * bb + e2)
            
            ww = w + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        
        return output


def explain_rwkv():
    """Explain RWKV architecture in detail."""
    print("\n" + "="*70)
    print(" RWKV: RECEPTANCE WEIGHTED KEY VALUE")
    print(" RNN efficiency + Transformer quality")
    print("="*70)
    
    print("""
    WHAT IS RWKV?
    ─────────────────────────────────────────────────────────────────
    
    RWKV = "Receptance Weighted Key Value"
    
    A hybrid architecture that combines:
    - RNN: O(1) inference per token, maintains state
    - Transformer: Parallelizable training, good quality
    - Linear Attention: No softmax, O(N) training
    
    THE RWKV FORMULA:
    ─────────────────────────────────────────────────────────────────
    
    For each position t:
    
              Σ_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t
    wkv_t = ───────────────────────────────────────────────────────
              Σ_{i<t} e^{-(t-1-i)w + k_i} + e^{u + k_t}
    
    Key components:
    - w: Decay rate (learned, per-channel)
         Higher w → forget faster
         Lower w → remember longer
    
    - u: Bonus for current token
         Allows special handling of present vs past
    
    - k, v: Keys and values (like attention)
    
    - Receptance (r): Gating mechanism
         r = sigmoid(W_r · x)
         output = r ⊙ wkv
    
    TOKEN SHIFTING - THE KEY INSIGHT:
    ─────────────────────────────────────────────────────────────────
    
    RWKV uses "token shift" - mixing current and previous token:
    
    x_k = μ_k · x_t + (1 - μ_k) · x_{t-1}
    
    This creates implicit positional information without
    explicit position embeddings!
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Why token shift works:                                          │
    │                                                                 │
    │ Token:      The    cat    sat    on     the    mat             │
    │               │      │      │      │      │      │             │
    │ x_t:        [The] [cat]  [sat]  [on]  [the]  [mat]            │
    │ x_{t-1}:    [pad] [The]  [cat]  [sat]  [on]  [the]            │
    │               │      │      │      │      │      │             │
    │ mixed:     Creates local context! Each position                │
    │            "sees" its neighbor implicitly.                     │
    └─────────────────────────────────────────────────────────────────┘
    
    RWKV LAYER STRUCTURE:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ RWKV Block                                                      │
    │                                                                 │
    │ Input x                                                        │
    │    │                                                           │
    │    ├──→ LayerNorm ──→ Time Mixing (attention-like) ──→ + ──→  │
    │    │                                                    ↑      │
    │    └────────────────────────────────────────────────────┘      │
    │                              │                                  │
    │    ├──→ LayerNorm ──→ Channel Mixing (FFN-like) ──→ + ──→     │
    │    │                                                ↑          │
    │    └────────────────────────────────────────────────┘          │
    │                                                                 │
    │ Output                                                         │
    └─────────────────────────────────────────────────────────────────┘
    
    RWKV VERSIONS:
    ─────────────────────────────────────────────────────────────────
    
    RWKV-4: Original version, proven to work at scale
    RWKV-5: Improved architecture, better performance
    RWKV-6: Latest (2024), multi-head, improved training
    
    PROS AND CONS:
    ─────────────────────────────────────────────────────────────────
    
    ✓ O(N) training complexity
    ✓ O(1) inference per token (like RNN)
    ✓ Constant memory during generation
    ✓ No KV cache needed!
    ✓ Works on CPU/edge devices
    ✓ Open source, community-driven
    
    ✗ Slightly lower quality than transformers (closing gap)
    ✗ Requires custom CUDA kernels for speed
    ✗ Less mature ecosystem
    ✗ Cannot "re-read" - purely sequential
    """)

# ============================================================================
# MAMBA / STATE SPACE MODELS
# ============================================================================

class SimplifiedS4Layer(nn.Module):
    """
    Simplified State Space Model layer for educational purposes.
    
    State Space Model (continuous):
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)
    
    Discretized (for sequences):
        h_t = Ā h_{t-1} + B̄ x_t
        y_t = C h_t + D x_t
    
    Where Ā, B̄ are discretized versions of A, B.
    """
    
    def __init__(self, d_model: int, state_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        
        # SSM parameters (simplified)
        self.A = nn.Parameter(torch.randn(d_model, state_size))
        self.B = nn.Parameter(torch.randn(d_model, state_size))
        self.C = nn.Parameter(torch.randn(d_model, state_size))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Discretization step
        self.delta = nn.Parameter(torch.ones(d_model) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape
        
        # Discretize A, B using delta (simplified ZOH discretization)
        delta = F.softplus(self.delta)
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A)  # Simplified
        B_bar = delta.unsqueeze(-1) * self.B
        
        # Run SSM (sequential - can be parallelized with convolution)
        h = torch.zeros(batch, d, self.state_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d)
            
            # State update: h = A_bar * h + B_bar * x
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(-1)
            
            # Output: y = C * h + D * x
            y = (self.C.unsqueeze(0) * h).sum(-1) + self.D * x_t
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style).
    
    Key innovation: Make B, C, delta INPUT-DEPENDENT!
    
    Standard SSM: Fixed dynamics (same for all inputs)
    Selective SSM: Input-dependent dynamics (can "select" what to remember)
    
    This is the breakthrough of Mamba!
    """
    
    def __init__(self, d_model: int, state_size: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.state_size = state_size
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution (like in Mamba)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=4, padding=3, groups=self.d_inner
        )
        
        # SSM parameters
        # A is static, but B, C, delta are input-dependent!
        self.A = nn.Parameter(torch.randn(self.d_inner, state_size))
        
        # Input-dependent projections
        self.x_proj = nn.Linear(self.d_inner, state_size * 2 + 1, bias=False)  # B, C, delta
        
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        x = F.silu(x)
        
        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x)  # (batch, seq_len, state_size*2 + 1)
        delta, B, C = torch.split(
            x_dbl, [1, self.state_size, self.state_size], dim=-1
        )
        
        # Discretize delta
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)
        
        # Run selective SSM
        y = self._selective_scan(x, delta, self.A, B, C)
        
        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def _selective_scan(self, x, delta, A, B, C):
        """Selective scan (simplified sequential version)."""
        batch, seq_len, d = x.shape
        
        h = torch.zeros(batch, d, self.state_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]  # (batch, d)
            delta_t = delta[:, t]  # (batch, d)
            B_t = B[:, t]  # (batch, state_size)
            C_t = C[:, t]  # (batch, state_size)
            
            # Discretize
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # State update
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output
            y = (C_t.unsqueeze(1) * h).sum(-1)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


def explain_mamba():
    """Explain Mamba and State Space Models."""
    print("\n" + "="*70)
    print(" MAMBA: SELECTIVE STATE SPACE MODELS")
    print(" The 2023 breakthrough in sequence modeling")
    print("="*70)
    
    print("""
    STATE SPACE MODELS (SSMs) - THE FOUNDATION:
    ─────────────────────────────────────────────────────────────────
    
    From control theory - continuous dynamical system:
    
        h'(t) = A h(t) + B x(t)    (state update)
        y(t) = C h(t) + D x(t)     (output)
    
    Discretized for sequences:
    
        h_t = Ā h_{t-1} + B̄ x_t
        y_t = C h_t + D x_t
    
    KEY INSIGHT: This is a LINEAR RNN!
    
    RNN:  h_t = σ(W_h h_{t-1} + W_x x_t)  ← Non-linear, hard to train
    SSM:  h_t = A h_{t-1} + B x_t          ← Linear, can parallelize!
    
    S4: STRUCTURED STATE SPACES (2021):
    ─────────────────────────────────────────────────────────────────
    
    Problem: Random A matrix doesn't work well
    Solution: Use STRUCTURED A (HiPPO - for long-range dependencies)
    
    HiPPO A matrix encodes "polynomial projection" - optimal for
    compressing history into fixed-size state.
    
    S4 showed SSMs can match transformers on Long Range Arena!
    
    THE MAMBA BREAKTHROUGH (2023):
    ─────────────────────────────────────────────────────────────────
    
    Key insight: SSMs have FIXED dynamics (same A, B, C for all inputs)
    
    This is like attention with fixed attention pattern - not adaptive!
    
    Mamba solution: Make B, C, Δ INPUT-DEPENDENT
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Standard SSM (S4):                                              │
    │                                                                 │
    │ x_t ──→ [ Fixed A, B, C ] ──→ y_t                              │
    │         (same for all inputs)                                   │
    │                                                                 │
    │ Selective SSM (Mamba):                                          │
    │                                                                 │
    │ x_t ──→ [Project to B, C, Δ] ──→ [ A, B(x), C(x), Δ(x) ] ──→ y│
    │         (input-dependent!)                                      │
    │                                                                 │
    │ The model can now SELECT what information to keep!             │
    └─────────────────────────────────────────────────────────────────┘
    
    WHY "SELECTIVE" MATTERS:
    ─────────────────────────────────────────────────────────────────
    
    Selective SSM can implement "attention-like" behavior:
    
    Input: "The capital of France is [MASK]"
    
    - When processing "France": Large B → store in state
    - When processing "of": Small B → mostly ignore
    - At [MASK]: Large C for "France" dimension → retrieve
    
    This is like attention but with O(N) complexity!
    
    MAMBA ARCHITECTURE:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Mamba Block                                                     │
    │                                                                 │
    │    Input x ──→ Linear (expand) ──┬──→ Conv1D ──→ SiLU ──→      │
    │                                  │                              │
    │                                  └──→ (skip) ──→ SiLU ──→ Gate  │
    │                                                      ↓          │
    │              ┌──────────────────────────────────────┘          │
    │              ↓                                                  │
    │    ──→ Selective SSM ──→ × Gate ──→ Linear (project) ──→ Out   │
    └─────────────────────────────────────────────────────────────────┘
    
    MAMBA vs TRANSFORMERS:
    ─────────────────────────────────────────────────────────────────
    
    ┌───────────────────┬────────────────┬───────────────────────────┐
    │ Aspect            │ Transformer    │ Mamba                     │
    ├───────────────────┼────────────────┼───────────────────────────┤
    │ Training          │ O(N²)          │ O(N)                      │
    │ Inference/token   │ O(N) + KV cache│ O(1) + state              │
    │ Memory            │ O(N²) or O(N)  │ O(N)                      │
    │ Long sequences    │ Challenging    │ Natural                   │
    │ Associative recall│ Excellent      │ Good (improving)          │
    │ In-context learning│ Excellent     │ Good                      │
    └───────────────────┴────────────────┴───────────────────────────┘
    
    MAMBA-2 (2024):
    ─────────────────────────────────────────────────────────────────
    
    Key insight: Selective SSMs ARE a form of linear attention!
    
    Shows connection between:
    - State Space Models
    - Linear Attention
    - Structured Matrices
    
    Mamba-2 is faster (2-8x) and better quality than Mamba-1.
    """)

# ============================================================================
# GATED ATTENTION
# ============================================================================

class GatedAttention(nn.Module):
    """
    Gated Attention - Add gates to control attention flow.
    
    Used in various forms:
    - Gated Linear Attention (GLA)
    - MEGA (Moving Average Equipped Gated Attention)
    - Gated State Spaces
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_g = nn.Linear(d_model, d_model)  # Gate
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.W_qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        
        # Gating
        gate = torch.sigmoid(self.W_g(x))
        output = gate * self.W_o(context)
        
        return output


def explain_gated_attention():
    """Explain gated attention variants."""
    print("\n" + "="*70)
    print(" GATED ATTENTION VARIANTS")
    print(" Adding learnable gates for better control")
    print("="*70)
    
    print("""
    WHY GATES?
    ─────────────────────────────────────────────────────────────────
    
    Gates allow the model to:
    - Control information flow
    - Learn when to use attention vs skip
    - Stabilize training
    - Improve gradient flow
    
    GATED LINEAR ATTENTION (GLA):
    ─────────────────────────────────────────────────────────────────
    
    Combines linear attention with gating:
    
    y = gate ⊙ (Q @ (K^T @ V)) + (1 - gate) ⊙ x
    
    - Linear attention for long-range
    - Gate controls how much attention to use
    - Skip connection preserves local info
    
    MEGA (Moving Average Equipped Gated Attention):
    ─────────────────────────────────────────────────────────────────
    
    Combines:
    1. Exponential Moving Average (EMA) - local smoothing
    2. Single-head gated attention - global mixing
    3. Gating between them
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ MEGA Block                                                      │
    │                                                                 │
    │ Input ──→ EMA ──┬──→ Gated Attention ──→ Gate ──→ Output       │
    │                 │                          ↑                    │
    │                 └──────────────────────────┘                    │
    │                      (controls mixing)                          │
    └─────────────────────────────────────────────────────────────────┘
    
    EMA provides:
    - Local inductive bias (like convolution)
    - O(N) computation
    - Complements global attention
    
    RETENTION NETWORKS (RetNet):
    ─────────────────────────────────────────────────────────────────
    
    Combines:
    - Multi-head retention (like attention with decay)
    - Group normalization
    - Swish activation
    
    Retention formula:
    
    Retention(X) = (Q @ K^T ⊙ D) @ V
    
    Where D is a decay matrix:
    D_{ij} = γ^{i-j} if i ≥ j else 0
    
    - Parallel mode: Full matrix (like attention)
    - Recurrent mode: O(1) per token
    - Chunk mode: Best of both worlds
    
    BENEFITS OF GATING:
    ─────────────────────────────────────────────────────────────────
    
    ✓ Better gradient flow (like skip connections)
    ✓ Learnable balance between components
    ✓ Often improves training stability
    ✓ Can implement "mixture of experts" behavior
    """)

# ============================================================================
# BOUNDED ATTENTION
# ============================================================================

def explain_bounded_attention():
    """Explain bounded/constrained attention mechanisms."""
    print("\n" + "="*70)
    print(" BOUNDED ATTENTION")
    print(" Constraining attention for efficiency and interpretability")
    print("="*70)
    
    print("""
    BOUNDED ATTENTION TYPES:
    ─────────────────────────────────────────────────────────────────
    
    1. SPARSE ATTENTION BOUNDS
       - Only attend to k neighbors
       - Fixed sparsity pattern
       - Examples: Local, strided, block-sparse
    
    2. TOP-K ATTENTION
       - Only keep top-k attention weights
       - Sparse but adaptive
       - Can be implemented efficiently with custom kernels
    
    3. THRESHOLD ATTENTION
       - Zero out weights below threshold
       - Adaptive sparsity
       - Interpretability benefit
    
    4. ENTROPY-CONSTRAINED
       - Encourage concentrated attention
       - Add entropy penalty to loss
       - Leads to more interpretable patterns
    
    5. HARD ATTENTION
       - Use argmax instead of softmax
       - Non-differentiable (use REINFORCE/Gumbel-Softmax)
       - Maximum sparsity (only attend to one position)
    
    LOCALITY BIAS:
    ─────────────────────────────────────────────────────────────────
    
    Many tasks have local structure - bias attention towards it:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Relative position bias:                                         │
    │                                                                 │
    │ Attention(Q, K, V) = softmax(QK^T / √d + B) V                  │
    │                                                                 │
    │ Where B_{ij} = bias[i - j]                                     │
    │                                                                 │
    │ Example bias (encourages local attention):                     │
    │   distance:  -3   -2   -1    0   +1   +2   +3                 │
    │   bias:     -0.5 -0.2  0.0  0.5  0.0 -0.2 -0.5                │
    │                          ↑                                      │
    │                    current position (bonus)                    │
    └─────────────────────────────────────────────────────────────────┘
    
    ALiBi (Attention with Linear Biases):
    
    bias[i - j] = -m × |i - j|
    
    Where m is different per head (log-spaced).
    Simple but effective for length extrapolation!
    """)

# ============================================================================
# HYBRID ARCHITECTURES
# ============================================================================

def explain_hybrid_architectures():
    """Explain hybrid attention architectures."""
    print("\n" + "="*70)
    print(" HYBRID ARCHITECTURES (2024)")
    print(" Combining the best of different approaches")
    print("="*70)
    
    print("""
    WHY HYBRIDS?
    ─────────────────────────────────────────────────────────────────
    
    Different mechanisms excel at different things:
    
    - Attention: Associative recall, in-context learning
    - SSMs/RNNs: Long sequences, efficient inference
    - Convolutions: Local patterns, inductive bias
    
    Hybrid = Best of all worlds!
    
    MAMBA-ATTENTION HYBRIDS:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Jamba (AI21, 2024):                                             │
    │                                                                 │
    │ Layer 1: Mamba                                                 │
    │ Layer 2: Mamba                                                 │
    │ Layer 3: Attention  ← Full attention every few layers         │
    │ Layer 4: Mamba                                                 │
    │ Layer 5: Mamba                                                 │
    │ Layer 6: Attention                                             │
    │ ...                                                            │
    │                                                                 │
    │ Benefits:                                                       │
    │ - Mostly O(N) from Mamba layers                                │
    │ - Attention for complex reasoning when needed                  │
    │ - 256K context length achieved                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    STRIPED HYBRIDS:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ StripedHyena, Zamba:                                            │
    │                                                                 │
    │ Interleave different mechanisms:                               │
    │ [Mamba] [Attention] [Mamba] [Attention] ...                   │
    │                                                                 │
    │ Or per-head:                                                    │
    │ Head 1-4: SSM                                                  │
    │ Head 5-6: Attention                                            │
    │ Head 7-8: Sliding Window                                       │
    └─────────────────────────────────────────────────────────────────┘
    
    GRIFFIN (DeepMind, 2024):
    ─────────────────────────────────────────────────────────────────
    
    Combines:
    - Gated Linear Recurrent (like Mamba)
    - Local Attention (sliding window)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Griffin Block:                                                  │
    │                                                                 │
    │ Input ──┬──→ Gated Linear Recurrent ──┬──→ Output              │
    │         │                              │                        │
    │         └──→ Local Attention ──────────┘                       │
    │              (window size ~1024)                               │
    │                                                                 │
    │ Recurrent: Global context (O(1) per token)                    │
    │ Local Attn: Precise local modeling                            │
    └─────────────────────────────────────────────────────────────────┘
    
    2024 TRENDS:
    ─────────────────────────────────────────────────────────────────
    
    1. Pure attention still dominates at large scale (GPT-4, Claude)
    2. Hybrids promising for efficiency
    3. Mamba/SSMs closing quality gap
    4. Sliding window attention in most new LLMs
    5. GQA standard for inference efficiency
    
    THE FUTURE:
    ─────────────────────────────────────────────────────────────────
    
    - More sophisticated hybrids
    - Hardware-aware architecture design
    - Automatic architecture search
    - Task-specific attention patterns
    """)

# ============================================================================
# SUMMARY
# ============================================================================

def print_advanced_summary():
    """Print advanced attention summary."""
    print("\n" + "="*70)
    print(" ADVANCED ATTENTION SUMMARY")
    print("="*70)
    
    print("""
    ARCHITECTURE COMPARISON:
    
    ┌────────────────┬──────────┬───────────┬─────────────┬────────────┐
    │ Architecture   │ Training │ Inference │ Long Context│ Quality    │
    ├────────────────┼──────────┼───────────┼─────────────┼────────────┤
    │ Transformer    │ O(N²)    │ O(N)+KV   │ Limited     │ Excellent  │
    │ + Flash Attn   │ O(N²)    │ O(N)+KV   │ Better      │ Excellent  │
    │ + Sliding Win  │ O(Nw)    │ O(w)+KV   │ Good        │ Very Good  │
    │ RWKV           │ O(N)     │ O(1)      │ Excellent   │ Good       │
    │ Mamba          │ O(N)     │ O(1)      │ Excellent   │ Very Good  │
    │ Hybrid         │ O(N)~    │ Mixed     │ Excellent   │ Excellent  │
    └────────────────┴──────────┴───────────┴─────────────┴────────────┘
    
    WHEN TO USE WHAT:
    ─────────────────────────────────────────────────────────────────
    
    Standard Transformer (+ Flash):
    - Best quality needed
    - Context < 32K
    - Compute budget available
    
    Sliding Window:
    - Very long context (100K+)
    - Mostly local dependencies
    - Inference efficiency important
    
    RWKV:
    - Edge/mobile deployment
    - Very limited compute
    - Streaming applications
    
    Mamba:
    - Long sequences (millions)
    - Continuous signals
    - When attention quality gap acceptable
    
    Hybrid:
    - Best of both worlds
    - Research/new applications
    - When optimizing for specific task
    
    KEY TAKEAWAYS:
    ─────────────────────────────────────────────────────────────────
    
    1. Attention is NOT all you need - alternatives are viable
    2. O(N²) → O(N) is possible with acceptable quality loss
    3. Selection/gating is key to making linear models work
    4. Hybrids may be the future
    5. Hardware co-design matters as much as algorithm
    
    NEXT: 05_attention_pitfalls.py - Common issues and solutions
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " ADVANCED ATTENTION MECHANISMS ".center(68) + "║")
    print("║" + " RWKV, Mamba, and beyond ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    explain_rwkv()
    explain_mamba()
    explain_gated_attention()
    explain_bounded_attention()
    explain_hybrid_architectures()
    print_advanced_summary()
    
    print("\n" + "="*70)
    print(" The landscape of sequence modeling is rapidly evolving!")
    print("="*70)
