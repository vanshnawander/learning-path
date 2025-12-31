"""
01_attention_history_fundamentals.py - History and Fundamentals of Attention

This module traces the evolution of attention from its origins to transformers.
Understanding history helps you appreciate WHY attention works the way it does.

Timeline:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1990s: Neural Attention (early cognitive science inspirations)              │
│ 2014:  Bahdanau Attention - Seq2Seq NMT breakthrough                       │
│ 2015:  Luong Attention - Simplified attention variants                      │
│ 2017:  Transformer - "Attention Is All You Need"                           │
│ 2018:  BERT, GPT - Pre-training revolution                                 │
│ 2020:  Linear Attention attempts (Performer, Linformer)                    │
│ 2021:  Flash Attention - IO-aware exact attention                          │
│ 2023:  Mamba, RWKV - State Space Models as alternatives                    │
│ 2024:  Hybrid architectures, Mamba-2, RWKV-6                               │
└─────────────────────────────────────────────────────────────────────────────┘

Key Papers:
- Bahdanau et al. 2014: "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani et al. 2017: "Attention Is All You Need"
- Gu et al. 2023: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Run: python 01_attention_history_fundamentals.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_attention(func, warmup=5, iterations=20):
    """Profile attention operation with CUDA timing."""
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
# THE PROBLEM THAT LED TO ATTENTION
# ============================================================================

def explain_seq2seq_problem():
    """Explain the bottleneck problem that motivated attention."""
    print("\n" + "="*70)
    print(" THE PROBLEM: SEQ2SEQ BOTTLENECK")
    print(" Why attention was invented")
    print("="*70)
    
    print("""
    BEFORE ATTENTION: Encoder-Decoder with Fixed Context
    ─────────────────────────────────────────────────────────────────
    
    Seq2Seq for Machine Translation (Sutskever 2014):
    
    Input: "The cat sat on the mat"
         ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ ENCODER (LSTM)                                                  │
    │ The → h1 → cat → h2 → sat → h3 → on → h4 → the → h5 → mat → h6 │
    │                                                              ↓   │
    │                                              CONTEXT VECTOR: c  │
    │                                              (fixed-size!)      │
    └─────────────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ DECODER (LSTM)                                                  │
    │ c → Le → chat → est → assis → sur → le → tapis                 │
    │     ↑     ↑       ↑      ↑      ↑    ↑     ↑                    │
    │   All use the SAME context vector c!                           │
    └─────────────────────────────────────────────────────────────────┘
    
    THE BOTTLENECK PROBLEM:
    ─────────────────────────────────────────────────────────────────
    
    1. ALL information must compress into ONE fixed-size vector
    2. Long sequences → information loss
    3. Decoder has no way to "look back" at specific source words
    4. Performance degrades on long sequences
    
    EVIDENCE (Cho et al. 2014):
    ┌─────────────────────────────────────────────────────────────────┐
    │ BLEU Score vs Sentence Length                                   │
    │                                                                 │
    │ 35 │ ****                                                       │
    │    │     ****                                                   │
    │ 30 │         ****                                               │
    │    │             ****                                           │
    │ 25 │                 ****                                       │
    │    │                     ****                                   │
    │ 20 │                         ****                               │
    │    └──────────────────────────────────────────────────────────  │
    │      10   20   30   40   50   60  (sentence length)             │
    │                                                                 │
    │ Performance DROPS as sentences get longer!                      │
    └─────────────────────────────────────────────────────────────────┘
    
    THE INTUITION FOR ATTENTION:
    ─────────────────────────────────────────────────────────────────
    
    When translating "The cat sat on the mat" → "Le chat est assis..."
    
    - When generating "chat" (cat), focus on "cat"
    - When generating "assis" (sat), focus on "sat"
    - Don't compress everything - let decoder ATTEND to relevant parts!
    
    Human analogy: When reading a long document and answering questions,
    you don't memorize everything - you look back at relevant sections.
    """)

# ============================================================================
# BAHDANAU ATTENTION (2014)
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention - The original attention mechanism.
    
    Paper: "Neural Machine Translation by Jointly Learning to Align and Translate"
    Authors: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)
    
    Key idea: Learn alignment between decoder state and encoder outputs.
    
    Alignment score: e_ij = v^T * tanh(W_s * s_{i-1} + W_h * h_j)
    
    Where:
    - s_{i-1}: Previous decoder hidden state
    - h_j: Encoder hidden state at position j
    - W_s, W_h, v: Learnable parameters
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Alignment model parameters
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(
        self,
        decoder_state: torch.Tensor,  # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
        mask: Optional[torch.Tensor] = None  # (batch, src_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention context and weights.
        
        Returns:
            context: (batch, hidden) - Weighted sum of encoder outputs
            weights: (batch, src_len) - Attention distribution
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Expand decoder state: (batch, hidden) -> (batch, src_len, hidden)
        decoder_expanded = decoder_state.unsqueeze(1).expand(-1, src_len, -1)
        
        # Compute alignment scores
        # e_ij = v^T * tanh(W_s * s + W_h * h)
        energy = torch.tanh(
            self.W_s(decoder_expanded) + self.W_h(encoder_outputs)
        )  # (batch, src_len, hidden)
        
        scores = self.v(energy).squeeze(-1)  # (batch, src_len)
        
        # Apply mask (for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize with softmax
        weights = F.softmax(scores, dim=-1)  # (batch, src_len)
        
        # Compute context vector
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, hidden)
        
        return context, weights


def demonstrate_bahdanau_attention():
    """Demonstrate Bahdanau attention with visualization."""
    print("\n" + "="*70)
    print(" BAHDANAU ATTENTION (2014)")
    print(" The breakthrough that started it all")
    print("="*70)
    
    print("""
    BAHDANAU ATTENTION MECHANISM:
    ─────────────────────────────────────────────────────────────────
    
    Instead of one context vector, compute DYNAMIC context at each step:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ ENCODER                                                         │
    │ The → h1 → cat → h2 → sat → h3 → on → h4 → the → h5 → mat → h6 │
    │       ↓         ↓         ↓        ↓        ↓         ↓        │
    │      [h1]     [h2]      [h3]     [h4]     [h5]      [h6]       │
    │       ↓         ↓         ↓        ↓        ↓         ↓        │
    └───────┼─────────┼─────────┼────────┼────────┼─────────┼────────┘
            │         │         │        │        │         │
            ├─────────┼─────────┼────────┼────────┼─────────┤
            │    ATTENTION WEIGHTS (example for "chat")     │
            │   0.05     0.70     0.05    0.05    0.05    0.10       │
            └─────────────────────┬───────────────────────────────────┘
                                  ↓
                          context = Σ α_j * h_j
                                  ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ DECODER                                                         │
    │ s_{i-1} ──→ Attention ──→ context ──→ [s_{i-1}; context] ──→ s_i│
    └─────────────────────────────────────────────────────────────────┘
    
    ALIGNMENT SCORE COMPUTATION:
    ─────────────────────────────────────────────────────────────────
    
    e_ij = v^T * tanh(W_s * s_{i-1} + W_h * h_j)
    
    α_ij = softmax(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
    
    context_i = Σ_j α_ij * h_j
    """)
    
    # Live demonstration
    print("\n LIVE DEMONSTRATION:")
    print("-" * 50)
    
    hidden_size = 256
    batch_size = 2
    src_len = 6  # "The cat sat on the mat"
    
    attention = BahdanauAttention(hidden_size)
    
    # Simulated encoder outputs and decoder state
    encoder_outputs = torch.randn(batch_size, src_len, hidden_size)
    decoder_state = torch.randn(batch_size, hidden_size)
    
    context, weights = attention(decoder_state, encoder_outputs)
    
    print(f" Encoder outputs shape: {encoder_outputs.shape}")
    print(f" Decoder state shape: {decoder_state.shape}")
    print(f" Context shape: {context.shape}")
    print(f" Attention weights shape: {weights.shape}")
    print(f"\n Attention weights (batch 0):")
    print(f" {weights[0].detach().numpy().round(3)}")
    print(f" Sum of weights: {weights[0].sum().item():.4f} (should be 1.0)")

# ============================================================================
# LUONG ATTENTION (2015)
# ============================================================================

class LuongAttention(nn.Module):
    """
    Luong (Multiplicative) Attention variants.
    
    Paper: "Effective Approaches to Attention-based Neural Machine Translation"
    Authors: Minh-Thang Luong, Hieu Pham, Christopher D. Manning (2015)
    
    Three scoring functions:
    - dot: score = s^T * h
    - general: score = s^T * W * h
    - concat: score = v^T * tanh(W * [s; h])
    """
    
    def __init__(self, hidden_size: int, score_type: str = 'general'):
        super().__init__()
        self.hidden_size = hidden_size
        self.score_type = score_type
        
        if score_type == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif score_type == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(
        self,
        decoder_state: torch.Tensor,  # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, src_len, _ = encoder_outputs.shape
        
        if self.score_type == 'dot':
            # score = s^T * h
            scores = torch.bmm(
                encoder_outputs, 
                decoder_state.unsqueeze(-1)
            ).squeeze(-1)
            
        elif self.score_type == 'general':
            # score = s^T * W * h
            transformed = self.W(encoder_outputs)
            scores = torch.bmm(
                transformed,
                decoder_state.unsqueeze(-1)
            ).squeeze(-1)
            
        elif self.score_type == 'concat':
            # score = v^T * tanh(W * [s; h])
            decoder_expanded = decoder_state.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat([decoder_expanded, encoder_outputs], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, weights


def explain_luong_attention():
    """Explain Luong attention variants."""
    print("\n" + "="*70)
    print(" LUONG ATTENTION (2015)")
    print(" Simplified and efficient variants")
    print("="*70)
    
    print("""
    LUONG ATTENTION VARIANTS:
    ─────────────────────────────────────────────────────────────────
    
    1. DOT PRODUCT (simplest, fastest)
       score(s, h) = s^T · h
       
       Pros: No parameters, fast
       Cons: Requires same dimensionality
    
    2. GENERAL (learnable transformation)
       score(s, h) = s^T · W · h
       
       Pros: More expressive
       Cons: O(d²) parameters
    
    3. CONCAT (like Bahdanau but different)
       score(s, h) = v^T · tanh(W · [s; h])
       
       Pros: Most expressive
       Cons: Slower
    
    KEY DIFFERENCE FROM BAHDANAU:
    ─────────────────────────────────────────────────────────────────
    
    Bahdanau: Uses s_{i-1} (PREVIOUS decoder state) for attention
    Luong:    Uses s_i (CURRENT decoder state) for attention
    
    Luong also introduced:
    - Global attention (attend to all source positions)
    - Local attention (attend to a window around predicted position)
    
    COMPUTATIONAL COMPARISON:
    ─────────────────────────────────────────────────────────────────
    
    Let d = hidden_size, n = sequence_length
    
    ┌─────────────────┬───────────────┬───────────────────────────────┐
    │ Method          │ Parameters    │ Compute per step              │
    ├─────────────────┼───────────────┼───────────────────────────────┤
    │ Dot             │ 0             │ O(n·d)                        │
    │ General         │ d²            │ O(n·d²)                       │
    │ Concat          │ 2d² + d       │ O(n·d²)                       │
    │ Bahdanau        │ 2d² + d       │ O(n·d²)                       │
    └─────────────────┴───────────────┴───────────────────────────────┘
    """)
    
    # Profile different attention types
    print("\n PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    hidden_size = 256
    batch_size = 32
    src_len = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    encoder_outputs = torch.randn(batch_size, src_len, hidden_size, device=device)
    decoder_state = torch.randn(batch_size, hidden_size, device=device)
    
    for score_type in ['dot', 'general', 'concat']:
        attention = LuongAttention(hidden_size, score_type).to(device)
        
        time_ms = profile_attention(
            lambda: attention(decoder_state, encoder_outputs)
        )
        
        print(f" {score_type:10s}: {time_ms:.4f} ms")

# ============================================================================
# FROM RNN ATTENTION TO SELF-ATTENTION
# ============================================================================

def explain_transition_to_self_attention():
    """Explain the conceptual transition to self-attention."""
    print("\n" + "="*70)
    print(" FROM RNN ATTENTION TO SELF-ATTENTION")
    print(" The path to transformers")
    print("="*70)
    
    print("""
    LIMITATIONS OF RNN + ATTENTION:
    ─────────────────────────────────────────────────────────────────
    
    1. SEQUENTIAL COMPUTATION
       - RNN must process tokens one-by-one
       - Can't parallelize across sequence
       - Training is slow
    
    2. LONG-RANGE DEPENDENCIES
       - Information must flow through many RNN steps
       - Gradient vanishing/exploding
       - Still struggles with very long sequences
    
    3. ATTENTION IS AN "ADD-ON"
       - Core representation still from RNN
       - Attention just helps decoder
    
    THE KEY INSIGHT:
    ─────────────────────────────────────────────────────────────────
    
    What if we ONLY use attention?
    
    - Remove the RNN entirely
    - Let each position attend to ALL other positions
    - This is "self-attention" or "intra-attention"
    
    SELF-ATTENTION vs CROSS-ATTENTION:
    ─────────────────────────────────────────────────────────────────
    
    Cross-attention (Bahdanau/Luong):
    - Query: decoder state
    - Keys/Values: encoder outputs
    - Decoder attends to encoder
    
    Self-attention:
    - Query, Keys, Values: ALL from same sequence
    - Each position attends to ALL positions
    - Captures relationships WITHIN a sequence
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Self-Attention Example: "The cat sat on the mat"                │
    │                                                                 │
    │ For "sat":                                                      │
    │   - Attends to "cat" (who sat?)                                │
    │   - Attends to "mat" (where sat?)                              │
    │   - Attends to "on" (how sat?)                                 │
    │   - Learns relationships without sequential processing!        │
    └─────────────────────────────────────────────────────────────────┘
    
    BENEFITS OF SELF-ATTENTION:
    ─────────────────────────────────────────────────────────────────
    
    1. PARALLELIZATION
       - All positions computed simultaneously
       - Massive GPU speedup
    
    2. CONSTANT PATH LENGTH
       - Any two positions: O(1) operations apart
       - No vanishing gradients through long sequences
    
    3. INTERPRETABILITY
       - Attention weights show what model focuses on
       - Can visualize relationships
    
    THE COST: QUADRATIC COMPLEXITY
    ─────────────────────────────────────────────────────────────────
    
    - Each position attends to ALL others
    - N positions × N positions = O(N²) computation
    - O(N²) memory for attention matrix
    
    This is THE fundamental challenge that later work addresses!
    """)

# ============================================================================
# ATTENTION AS INFORMATION RETRIEVAL
# ============================================================================

def explain_attention_as_retrieval():
    """Explain attention from information retrieval perspective."""
    print("\n" + "="*70)
    print(" ATTENTION AS INFORMATION RETRIEVAL")
    print(" The Query-Key-Value framework")
    print("="*70)
    
    print("""
    INTUITION: ATTENTION = SOFT DATABASE LOOKUP
    ─────────────────────────────────────────────────────────────────
    
    Think of attention like searching a database:
    
    Hard lookup (traditional database):
    - Query: "What is the capital of France?"
    - Exact match: Return "Paris"
    
    Soft lookup (attention):
    - Query: Representation of what you're looking for
    - Keys: Representations of what's available
    - Values: The actual information to retrieve
    - Return: Weighted combination based on query-key similarity
    
    THE QKV FORMULATION:
    ─────────────────────────────────────────────────────────────────
    
    Given input X ∈ ℝ^(n×d):
    
    Q = X · W_Q    (Queries: what am I looking for?)
    K = X · W_K    (Keys: what do I have?)
    V = X · W_V    (Values: what information to return?)
    
    Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
    
    Step by step:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Q · K^T                                                      │
    │    Compute similarity between each query and all keys           │
    │    Shape: (n × d) · (d × n) = (n × n)                          │
    │                                                                 │
    │ 2. / √d_k                                                       │
    │    Scale to prevent softmax saturation                          │
    │    (Dot products grow with dimension)                           │
    │                                                                 │
    │ 3. softmax(...)                                                 │
    │    Convert to probability distribution over keys                │
    │    Each row sums to 1                                          │
    │                                                                 │
    │ 4. · V                                                          │
    │    Weighted sum of values                                       │
    │    Shape: (n × n) · (n × d) = (n × d)                          │
    └─────────────────────────────────────────────────────────────────┘
    
    WHY SCALE BY √d_k?
    ─────────────────────────────────────────────────────────────────
    
    For random unit vectors q, k ∈ ℝ^d:
    
    E[q · k] = 0
    Var[q · k] = d
    
    As d grows, dot products get larger → softmax saturates
    
    Scaling by √d keeps variance ≈ 1, softmax stays well-behaved
    """)
    
    # Demonstrate the scaling issue
    print("\n DEMONSTRATION: Why scaling matters")
    print("-" * 50)
    
    for d in [64, 256, 1024]:
        q = torch.randn(1000, d) / math.sqrt(d)  # Unit variance
        k = torch.randn(1000, d) / math.sqrt(d)
        
        # Unscaled dot product
        dots_unscaled = (q * k).sum(dim=-1)
        
        # Scaled dot product
        dots_scaled = dots_unscaled / math.sqrt(d)
        
        print(f" d={d:4d}: unscaled std={dots_unscaled.std():.2f}, "
              f"scaled std={dots_scaled.std():.2f}")

# ============================================================================
# SIMPLE SELF-ATTENTION IMPLEMENTATION
# ============================================================================

class SimpleSelfAttention(nn.Module):
    """
    Basic self-attention (single head) for educational purposes.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = math.sqrt(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or (seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            weights: (batch, seq_len, seq_len)
        """
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale
        # (batch, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        
        output = torch.bmm(weights, V)
        
        return output, weights


def demonstrate_self_attention():
    """Demonstrate self-attention with profiling."""
    print("\n" + "="*70)
    print(" SELF-ATTENTION DEMONSTRATION")
    print(" The core operation of transformers")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    d_model = 256
    batch_size = 4
    seq_len = 32
    
    attention = SimpleSelfAttention(d_model).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output, weights = attention(x)
    
    print(f"\n Input shape: {x.shape}")
    print(f" Output shape: {output.shape}")
    print(f" Weights shape: {weights.shape}")
    
    print(f"\n Attention weights (first query, first batch):")
    print(f" {weights[0, 0, :8].detach().cpu().numpy().round(3)}...")
    
    # Profile at different sequence lengths
    print(f"\n QUADRATIC SCALING DEMONSTRATION:")
    print("-" * 50)
    print(f" {'Seq Len':<10} {'Time (ms)':<15} {'Ratio'}")
    print("-" * 50)
    
    times = []
    for seq_len in [64, 128, 256, 512]:
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        time_ms = profile_attention(lambda: attention(x))
        times.append(time_ms)
        
        ratio = times[-1] / times[0] if len(times) > 1 else 1.0
        expected_ratio = (seq_len / 64) ** 2
        
        print(f" {seq_len:<10} {time_ms:<15.4f} {ratio:.2f}x (expected: {expected_ratio:.2f}x)")

# ============================================================================
# SUMMARY
# ============================================================================

def print_history_summary():
    """Print summary of attention history."""
    print("\n" + "="*70)
    print(" ATTENTION HISTORY SUMMARY")
    print("="*70)
    
    print("""
    KEY MILESTONES:
    
    2014 - BAHDANAU ATTENTION
    └── Solved seq2seq bottleneck
    └── Dynamic context per decoder step
    └── Alignment learned jointly with translation
    
    2015 - LUONG ATTENTION
    └── Simplified scoring functions
    └── Dot product attention (efficient!)
    └── Local vs global attention
    
    2017 - TRANSFORMER (Vaswani et al.)
    └── "Attention Is All You Need"
    └── Remove RNNs entirely
    └── Self-attention + Multi-head attention
    └── Position encodings for sequence order
    
    2018+ - THE ATTENTION ERA
    └── BERT, GPT, T5, etc.
    └── Pre-training + fine-tuning paradigm
    └── Attention dominates NLP, vision, audio
    
    KEY EQUATIONS TO REMEMBER:
    ─────────────────────────────────────────────────────────────────
    
    Bahdanau: e_ij = v^T · tanh(W_s·s + W_h·h)
    
    Luong:    score = q^T · k  (dot product)
    
    Scaled:   Attention(Q,K,V) = softmax(QK^T / √d) · V
    
    THE BIG TRADE-OFF:
    ─────────────────────────────────────────────────────────────────
    
    Attention gives us:
    ✓ Parallelization
    ✓ Direct long-range dependencies
    ✓ Interpretable weights
    
    But costs us:
    ✗ O(N²) time and memory
    ✗ No inherent sequence order (need positional encoding)
    
    NEXT: 02_transformer_attention.py - Multi-head attention & transformers
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " ATTENTION MECHANISMS: HISTORY & FUNDAMENTALS ".center(68) + "║")
    print("║" + " From Bahdanau to Self-Attention ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    explain_seq2seq_problem()
    demonstrate_bahdanau_attention()
    explain_luong_attention()
    explain_transition_to_self_attention()
    explain_attention_as_retrieval()
    demonstrate_self_attention()
    print_history_summary()
    
    print("\n" + "="*70)
    print(" Understanding history helps you understand the future!")
    print("="*70)
