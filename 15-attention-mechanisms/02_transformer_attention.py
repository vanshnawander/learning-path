"""
02_transformer_attention.py - Transformer Architecture Deep Dive

The Transformer ("Attention Is All You Need", Vaswani et al. 2017)
revolutionized deep learning by using ONLY attention mechanisms.

Key Innovations:
1. Multi-Head Attention - Multiple attention "perspectives"
2. Self-Attention - Each position attends to all positions
3. Position Encodings - Inject sequence order information
4. Layer Normalization + Residual connections

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORMER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ ENCODER (x N)              │ DECODER (x N)                                  │
│ ┌─────────────────────┐    │ ┌─────────────────────┐                       │
│ │ Multi-Head          │    │ │ Masked Multi-Head   │ ← Self-attention      │
│ │ Self-Attention      │    │ │ Self-Attention      │   (causal)            │
│ ├─────────────────────┤    │ ├─────────────────────┤                       │
│ │ Add & Norm          │    │ │ Add & Norm          │                       │
│ ├─────────────────────┤    │ ├─────────────────────┤                       │
│ │ Feed-Forward        │    │ │ Multi-Head          │ ← Cross-attention     │
│ │ Network             │    │ │ Cross-Attention     │   (encoder-decoder)   │
│ ├─────────────────────┤    │ ├─────────────────────┤                       │
│ │ Add & Norm          │    │ │ Add & Norm          │                       │
│ └─────────────────────┘    │ ├─────────────────────┤                       │
│                            │ │ Feed-Forward        │                       │
│                            │ ├─────────────────────┤                       │
│                            │ │ Add & Norm          │                       │
│                            │ └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 02_transformer_attention.py
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
    """Profile with proper synchronization."""
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
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from "Attention Is All You Need".
    
    Key insight: Instead of one attention function, run h parallel
    attention "heads" and concatenate results.
    
    Why multiple heads?
    - Different heads can focus on different aspects
    - Head 1: syntactic relationships
    - Head 2: semantic relationships  
    - Head 3: positional patterns
    - etc.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
    where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Combined projections for efficiency
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Self-attention forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        # (batch, heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention for encoder-decoder models.
    
    Query comes from decoder, Keys/Values from encoder.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,  # Decoder states (query source)
        encoder_output: torch.Tensor,  # Encoder output (key/value source)
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention: decoder attends to encoder.
        
        Args:
            x: (batch, tgt_len, d_model) - decoder input
            encoder_output: (batch, src_len, d_model) - encoder output
            mask: (batch, 1, tgt_len, src_len) - padding mask
        """
        batch_size, tgt_len, _ = x.shape
        src_len = encoder_output.shape[1]
        
        # Query from decoder
        Q = self.W_q(x).reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, heads, tgt_len, head_dim)
        
        # Key, Value from encoder
        kv = self.W_kv(encoder_output)
        kv = kv.reshape(batch_size, src_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        K, V = kv[0], kv[1]  # (batch, heads, src_len, head_dim)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(batch_size, tgt_len, self.d_model)
        
        return self.W_o(context)


def explain_multi_head_attention():
    """Explain multi-head attention in detail."""
    print("\n" + "="*70)
    print(" MULTI-HEAD ATTENTION")
    print(" Multiple attention 'perspectives'")
    print("="*70)
    
    print("""
    WHY MULTIPLE HEADS?
    ─────────────────────────────────────────────────────────────────
    
    Single-head attention:
    - One attention pattern per layer
    - Limited expressiveness
    
    Multi-head attention:
    - h different attention patterns
    - Each head can specialize
    - Richer representations
    
    INTUITION:
    ─────────────────────────────────────────────────────────────────
    
    For sentence: "The animal didn't cross the street because it was tired"
    
    Head 1 might focus on: "it" → "animal" (coreference)
    Head 2 might focus on: "tired" → "animal" (who is tired?)
    Head 3 might focus on: "cross" → "street" (what's being crossed?)
    
    MATHEMATICAL FORMULATION:
    ─────────────────────────────────────────────────────────────────
    
    Input: X ∈ ℝ^(n×d_model)
    
    For each head i ∈ {1, ..., h}:
        Q_i = X · W_Q^i    where W_Q^i ∈ ℝ^(d_model × d_k)
        K_i = X · W_K^i    where W_K^i ∈ ℝ^(d_model × d_k)  
        V_i = X · W_V^i    where W_V^i ∈ ℝ^(d_model × d_v)
        
        head_i = Attention(Q_i, K_i, V_i)
    
    MultiHead(X) = Concat(head_1, ..., head_h) · W_O
    
    Typical dimensions:
    - d_model = 512
    - h = 8 heads
    - d_k = d_v = d_model / h = 64
    
    COMPUTATIONAL COST:
    ─────────────────────────────────────────────────────────────────
    
    Single head (d_model):
    - QK^T: O(n² · d_model)
    
    Multi-head (h heads, d_k = d_model/h):
    - h × QK^T: h × O(n² · d_k) = O(n² · d_model)
    
    Same asymptotic cost! But more expressive.
    """)
    
    # Demonstration
    print("\n DEMONSTRATION:")
    print("-" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 32
    
    mha = MultiHeadAttention(d_model, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    output, attn = mha(x, return_attention=True)
    
    print(f" Input: {x.shape}")
    print(f" Output: {output.shape}")
    print(f" Attention weights: {attn.shape}")
    print(f" (batch, num_heads, seq_len, seq_len)")
    
    # Show different heads have different patterns
    print(f"\n Each head attends differently:")
    for h in range(min(4, num_heads)):
        entropy = -(attn[0, h] * torch.log(attn[0, h] + 1e-9)).sum(-1).mean()
        print(f"  Head {h}: entropy = {entropy.item():.3f}")

# ============================================================================
# POSITION ENCODINGS
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal position encodings from original Transformer.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why sinusoids?
    - Can extrapolate to longer sequences
    - Relative positions can be computed as linear function
    - PE(pos+k) can be represented as linear transform of PE(pos)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer.
    
    Used in LLaMA, GPT-NeoX, and modern LLMs.
    
    Key idea: Encode position by ROTATING the query and key vectors.
    
    Benefits:
    - Relative position naturally encoded
    - Better extrapolation
    - Works well with KV caching
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        """Apply rotary embeddings to queries and keys."""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


def explain_position_encodings():
    """Explain different position encoding schemes."""
    print("\n" + "="*70)
    print(" POSITION ENCODINGS")
    print(" How transformers know sequence order")
    print("="*70)
    
    print("""
    THE PROBLEM:
    ─────────────────────────────────────────────────────────────────
    
    Self-attention is PERMUTATION INVARIANT!
    
    Attention("cat sat mat") = Attention("mat cat sat")
    
    Without position info, transformer can't distinguish order.
    
    SOLUTIONS:
    ─────────────────────────────────────────────────────────────────
    
    1. SINUSOIDAL (Original Transformer, 2017)
       PE(pos, 2i) = sin(pos / 10000^(2i/d))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
       
       ✓ Can extrapolate to longer sequences
       ✓ No learned parameters
       ✗ Absolute positions only
    
    2. LEARNED (BERT, GPT-2)
       PE = nn.Embedding(max_len, d_model)
       
       ✓ Can learn complex patterns
       ✗ Fixed maximum length
       ✗ More parameters
    
    3. ROTARY (RoPE) - LLaMA, GPT-NeoX
       Rotate Q and K vectors based on position
       
       ✓ Encodes RELATIVE positions naturally
       ✓ Better extrapolation than learned
       ✓ Works with KV cache
       → Now standard for modern LLMs
    
    4. ALiBi (Attention with Linear Biases)
       Add linear bias based on distance to attention scores
       
       ✓ No position embeddings needed
       ✓ Excellent extrapolation
       ✓ Very simple
    
    COMPARISON:
    ─────────────────────────────────────────────────────────────────
    
    ┌──────────────┬──────────────┬──────────────┬──────────────────┐
    │ Method       │ Extrapolation│ Relative Pos │ Used In          │
    ├──────────────┼──────────────┼──────────────┼──────────────────┤
    │ Sinusoidal   │ Moderate     │ Implicit     │ Original Trans.  │
    │ Learned      │ Poor         │ No           │ BERT, GPT-2      │
    │ RoPE         │ Good         │ Yes          │ LLaMA, Mistral   │
    │ ALiBi        │ Excellent    │ Yes          │ MPT, BLOOM       │
    └──────────────┴──────────────┴──────────────┴──────────────────┘
    """)

# ============================================================================
# ENCODER-DECODER ARCHITECTURE
# ============================================================================

def explain_encoder_decoder():
    """Explain encoder-decoder architecture deeply."""
    print("\n" + "="*70)
    print(" ENCODER-DECODER ARCHITECTURE")
    print(" Understanding what encoding and decoding DO")
    print("="*70)
    
    print("""
    WHAT DOES THE ENCODER DO?
    ─────────────────────────────────────────────────────────────────
    
    The encoder COMPRESSES the input into a rich representation.
    
    Input: "The cat sat on the mat"
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ ENCODER                                                         │
    │                                                                 │
    │ Token Embedding:  [The] [cat] [sat] [on] [the] [mat]           │
    │                     ↓     ↓     ↓    ↓    ↓     ↓              │
    │                   [e1]  [e2]  [e3] [e4] [e5]  [e6]             │
    │                     ↓     ↓     ↓    ↓    ↓     ↓              │
    │ Self-Attention:   Each token attends to ALL others             │
    │                     ↓     ↓     ↓    ↓    ↓     ↓              │
    │                   [h1]  [h2]  [h3] [h4] [h5]  [h6]             │
    │                                                                 │
    │ After N layers:   Contextualized representations               │
    │                   h2 now "knows" about cat, sat, mat, etc.     │
    └─────────────────────────────────────────────────────────────────┘
    
    ENCODING = COMPRESSION + CONTEXTUALIZATION
    ─────────────────────────────────────────────────────────────────
    
    1. COMPRESSION: Variable-length input → fixed-dimensional vectors
       - Each token gets a d_model dimensional representation
       - Information about entire sequence distributed across all vectors
    
    2. CONTEXTUALIZATION: Each position contains global context
       - "bank" in "river bank" vs "bank account" has DIFFERENT encodings
       - Context resolved through self-attention
    
    WHAT DOES THE DECODER DO?
    ─────────────────────────────────────────────────────────────────
    
    The decoder DECOMPRESSES/GENERATES from the encoder representation.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ DECODER                                                         │
    │                                                                 │
    │ Step 1: Generate "Le"                                          │
    │   ├── Masked self-attention on previous outputs: [<start>]     │
    │   ├── Cross-attention to encoder: focus on "The"              │
    │   └── Output: "Le"                                             │
    │                                                                 │
    │ Step 2: Generate "chat"                                        │
    │   ├── Masked self-attention: [<start>, Le]                     │
    │   ├── Cross-attention: focus on "cat" in encoder              │
    │   └── Output: "chat"                                           │
    │                                                                 │
    │ Step 3: Generate "est"                                         │
    │   ├── Masked self-attention: [<start>, Le, chat]               │
    │   ├── Cross-attention: focus on "sat" in encoder              │
    │   └── Output: "est"                                            │
    │                                                                 │
    │ ... continues until <end> token                                │
    └─────────────────────────────────────────────────────────────────┘
    
    WHY MASKED SELF-ATTENTION?
    ─────────────────────────────────────────────────────────────────
    
    During training, we process all positions in parallel.
    But position i should NOT see future positions i+1, i+2, ...
    
    Mask ensures CAUSAL attention:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ Attention mask for sequence length 4:                          │
    │                                                                 │
    │         To:  1    2    3    4                                  │
    │ From:                                                          │
    │   1       [  1    0    0    0  ]  ← Position 1 only sees 1    │
    │   2       [  1    1    0    0  ]  ← Position 2 sees 1,2       │
    │   3       [  1    1    1    0  ]  ← Position 3 sees 1,2,3     │
    │   4       [  1    1    1    1  ]  ← Position 4 sees all       │
    │                                                                 │
    │ 1 = can attend, 0 = cannot attend (masked to -inf)             │
    └─────────────────────────────────────────────────────────────────┘
    
    INFORMATION FLOW SUMMARY:
    ─────────────────────────────────────────────────────────────────
    
    1. Encoder: Bidirectional (sees full input)
       - Self-attention: ALL positions attend to ALL positions
       - Captures full context of input
    
    2. Decoder: Unidirectional + Cross-attention
       - Masked self-attention: Only see PAST positions
       - Cross-attention: Can see ALL encoder positions
       - Generates one token at a time (autoregressively)
    
    ENCODER-ONLY vs DECODER-ONLY vs ENCODER-DECODER:
    ─────────────────────────────────────────────────────────────────
    
    ┌──────────────────┬───────────────────┬──────────────────────────┐
    │ Architecture     │ Examples          │ Best For                 │
    ├──────────────────┼───────────────────┼──────────────────────────┤
    │ Encoder-only     │ BERT, RoBERTa     │ Understanding, class.    │
    │ Decoder-only     │ GPT, LLaMA        │ Generation               │
    │ Encoder-Decoder  │ T5, BART          │ Seq2seq (translation)    │
    └──────────────────┴───────────────────┴──────────────────────────┘
    """)

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm style (more stable training)
        attn_out, _ = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        x = x + self.ffn(self.norm2(x))
        
        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with cross-attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ):
        # Masked self-attention
        attn_out, _ = self.self_attn(self.norm1(x), self_attn_mask)
        x = x + self.dropout(attn_out)
        
        # Cross-attention to encoder
        cross_out = self.cross_attn(self.norm2(x), encoder_output, cross_attn_mask)
        x = x + self.dropout(cross_out)
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        
        return x


def profile_transformer_components():
    """Profile different transformer components."""
    print("\n" + "="*70)
    print(" TRANSFORMER COMPONENT PROFILING")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    d_model = 512
    num_heads = 8
    d_ff = 2048
    batch_size = 8
    seq_len = 512
    
    # Create components
    mha = MultiHeadAttention(d_model, num_heads).to(device)
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Profile
    print(f"\n Configuration: d_model={d_model}, heads={num_heads}, seq_len={seq_len}")
    print("-" * 50)
    
    time_mha = profile_fn(lambda: mha(x))
    time_block = profile_fn(lambda: encoder_block(x))
    
    print(f" Multi-Head Attention: {time_mha:.3f} ms")
    print(f" Full Encoder Block:   {time_block:.3f} ms")
    print(f" FFN overhead:         {time_block - time_mha:.3f} ms")
    
    # Quadratic scaling
    print(f"\n QUADRATIC SCALING:")
    print("-" * 50)
    
    times = []
    for seq_len in [128, 256, 512, 1024]:
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        t = profile_fn(lambda: mha(x))
        times.append(t)
        ratio = t / times[0] if times[0] > 0 else 1
        print(f" seq_len={seq_len:4d}: {t:.3f} ms ({ratio:.1f}x)")

# ============================================================================
# SUMMARY
# ============================================================================

def print_transformer_summary():
    """Print transformer architecture summary."""
    print("\n" + "="*70)
    print(" TRANSFORMER ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("""
    KEY COMPONENTS:
    
    1. MULTI-HEAD ATTENTION
       - h parallel attention heads
       - Each head has d_k = d_model/h dimensions
       - Concatenate and project outputs
    
    2. POSITION ENCODINGS
       - Sinusoidal (original)
       - Learned (BERT, GPT-2)
       - RoPE (LLaMA, modern LLMs)
       - ALiBi (MPT)
    
    3. FEED-FORWARD NETWORK
       - Two linear layers with activation
       - Expands then contracts: d → 4d → d
       - Applied position-wise
    
    4. RESIDUAL + LAYER NORM
       - Skip connections for gradient flow
       - Layer norm for training stability
       - Pre-norm vs Post-norm variants
    
    ENCODER vs DECODER:
    ─────────────────────────────────────────────────────────────────
    
    Encoder:
    - Bidirectional self-attention
    - Sees full input
    - Good for understanding
    
    Decoder:
    - Causal (masked) self-attention
    - Cross-attention to encoder
    - Good for generation
    
    THE QUADRATIC PROBLEM:
    ─────────────────────────────────────────────────────────────────
    
    Self-attention: O(N²) time and memory
    
    This limits:
    - Maximum sequence length
    - Batch size
    - Training/inference efficiency
    
    → Motivates efficient attention variants (next section!)
    
    NEXT: 03_efficient_attention.py - Linear attention, Flash Attention
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " TRANSFORMER ATTENTION DEEP DIVE ".center(68) + "║")
    print("║" + " Multi-head attention and architecture ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    explain_multi_head_attention()
    explain_position_encodings()
    explain_encoder_decoder()
    profile_transformer_components()
    print_transformer_summary()
    
    print("\n" + "="*70)
    print(" The transformer changed everything!")
    print("="*70)
