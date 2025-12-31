"""
Flex Attention: PyTorch 2.5+ Flexible Attention API
=====================================================

Flex Attention is PyTorch's new API for custom attention patterns.
It provides the flexibility to implement ANY attention variant while
maintaining Flash Attention-level performance through torch.compile.

Key Concepts:
1. Score Modification (score_mod) - Modify attention scores
2. Block Masks (block_mask) - Sparsity patterns for efficiency
3. Compilation - torch.compile integration for performance

This module covers:
- score_mod function patterns
- Block mask creation and usage
- Common attention variants (ALiBi, causal, sliding window, etc.)
- Performance considerations
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable
from functools import partial

# Check PyTorch version for Flex Attention availability
FLEX_ATTENTION_AVAILABLE = hasattr(torch.nn.attention, 'flex_attention') if hasattr(torch.nn, 'attention') else False

if FLEX_ATTENTION_AVAILABLE:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        create_mask,
        and_masks,
        or_masks,
    )
    print("✓ Flex Attention is available (PyTorch 2.5+)")
else:
    print("✗ Flex Attention not available. Requires PyTorch 2.5+")
    print("  This module will demonstrate concepts with mock implementations.")


# =============================================================================
# SECTION 1: UNDERSTANDING SCORE_MOD
# =============================================================================
"""
SCORE_MOD: A function that modifies attention scores BEFORE softmax.

Standard attention:
    Attention(Q, K, V) = softmax(QK^T / √d) V

With score_mod:
    Attention(Q, K, V) = softmax(score_mod(QK^T / √d)) V

The score_mod function signature:
    def score_mod(score, b, h, q_idx, kv_idx) -> modified_score
    
    Args:
        score: The attention score (scalar) for position (q_idx, kv_idx)
        b: Batch index
        h: Head index  
        q_idx: Query position index
        kv_idx: Key/Value position index
    
    Returns:
        Modified score (scalar)

Key Properties:
- Applied element-wise to the attention score matrix
- Must be a pure function (no side effects)
- Gets compiled by torch.compile for efficiency
- Can implement: causal, ALiBi, relative position, document masks, etc.
"""


def identity_score_mod(score, b, h, q_idx, kv_idx):
    """Identity score modification - no change."""
    return score


def causal_score_mod(score, b, h, q_idx, kv_idx):
    """
    Causal attention: mask future positions.
    
    Sets score to -inf when query position < key position.
    After softmax, this becomes 0 attention weight.
    """
    return torch.where(q_idx >= kv_idx, score, float('-inf'))


def sliding_window_score_mod(score, b, h, q_idx, kv_idx, window_size: int = 256):
    """
    Sliding window attention: only attend to nearby positions.
    
    Each position can only attend to positions within window_size.
    Commonly used in Mistral, Longformer, etc.
    """
    in_window = (q_idx - kv_idx).abs() <= window_size
    return torch.where(in_window, score, float('-inf'))


def alibi_score_mod(score, b, h, q_idx, kv_idx, num_heads: int = 8):
    """
    ALiBi (Attention with Linear Biases).
    
    Instead of positional embeddings, ALiBi adds a linear bias
    based on the distance between positions:
        score = score - m * |q_idx - kv_idx|
    
    where m is head-specific and follows a geometric sequence.
    
    Benefits:
    - No positional embeddings needed
    - Better length extrapolation
    - Used in BLOOM, MPT models
    """
    # Compute head-specific slope (geometric sequence)
    # m = 2^(-8/num_heads) for head 0, 2^(-16/num_heads) for head 1, etc.
    # This creates slopes: [1/4, 1/8, 1/16, 1/32, ...] for 8 heads
    ratio = 2 ** (-8.0 / num_heads)
    slope = ratio ** (h + 1)
    
    # Apply linear bias based on distance
    distance = (q_idx - kv_idx).abs()
    bias = -slope * distance
    
    return score + bias


def relative_position_score_mod(score, b, h, q_idx, kv_idx, max_distance: int = 128):
    """
    Relative position bias.
    
    Adds learnable bias based on relative position.
    Used in T5, DeBERTa, etc.
    
    Note: In practice, the bias table would be a learned parameter.
    Here we use a simple distance-based formula for demonstration.
    """
    # Compute relative position
    rel_pos = kv_idx - q_idx
    
    # Clip to max distance
    rel_pos = torch.clamp(rel_pos, -max_distance, max_distance)
    
    # Simple sinusoidal bias (in practice, this would be learned)
    bias = torch.sin(rel_pos.float() * math.pi / max_distance) * 0.5
    
    return score + bias


def prefix_lm_score_mod(score, b, h, q_idx, kv_idx, prefix_length: int = 32):
    """
    Prefix LM attention pattern.
    
    - Prefix tokens (0 to prefix_length-1): Bidirectional attention
    - Generation tokens (prefix_length onwards): Causal attention
    
    Used in encoder-decoder models and prefix tuning.
    """
    # In prefix region: allow bidirectional
    in_prefix = q_idx < prefix_length
    
    # In generation region: causal only
    causal_ok = q_idx >= kv_idx
    
    # Combine: prefix is always OK, generation needs causal
    valid = in_prefix | causal_ok
    
    return torch.where(valid, score, float('-inf'))


def document_mask_score_mod(score, b, h, q_idx, kv_idx, 
                            document_ids: torch.Tensor):
    """
    Document masking for multi-document batches.
    
    Prevents attention across document boundaries.
    Essential for training on concatenated documents.
    
    Args:
        document_ids: Tensor of shape (batch, seq_len) containing
                     document ID for each position.
    """
    # Get document IDs for query and key positions
    q_doc = document_ids[b, q_idx]
    k_doc = document_ids[b, kv_idx]
    
    # Only allow attention within same document
    same_doc = q_doc == k_doc
    
    return torch.where(same_doc, score, float('-inf'))


# =============================================================================
# SECTION 2: BLOCK MASKS FOR EFFICIENCY
# =============================================================================
"""
BLOCK MASKS: Compile-time sparsity optimization.

Problem with score_mod alone:
- Still computes ALL scores, then masks
- O(N²) computation even for sparse patterns

Block Masks solve this:
- Define which BLOCKS of the attention matrix to compute
- Skip entire blocks that would be masked
- Reduces actual FLOPs, not just masks

Block Structure:
    Attention matrix is divided into blocks of size BLOCK_SIZE × BLOCK_SIZE
    Block mask indicates which blocks contain ANY valid attention
    
Example for causal with BLOCK_SIZE=128:
    [■ □ □ □]   ■ = Compute (has valid scores)
    [■ ■ □ □]   □ = Skip (all masked anyway)
    [■ ■ ■ □]
    [■ ■ ■ ■]
"""


def causal_mask_fn(b, h, q_idx, kv_idx):
    """Mask function for causal attention."""
    return q_idx >= kv_idx


def sliding_window_mask_fn(b, h, q_idx, kv_idx, window_size: int = 256):
    """Mask function for sliding window attention."""
    return (q_idx - kv_idx).abs() <= window_size


def causal_sliding_window_mask_fn(b, h, q_idx, kv_idx, window_size: int = 256):
    """Combined causal + sliding window mask."""
    causal = q_idx >= kv_idx
    in_window = (q_idx - kv_idx) <= window_size
    return causal & in_window


def create_document_mask_fn(document_ids: torch.Tensor):
    """Create document mask function with captured document IDs."""
    def mask_fn(b, h, q_idx, kv_idx):
        return document_ids[b, q_idx] == document_ids[b, kv_idx]
    return mask_fn


# =============================================================================
# SECTION 3: CREATING AND USING BLOCK MASKS
# =============================================================================
"""
Block Mask Creation:

create_block_mask(mask_fn, B, H, Q_LEN, KV_LEN, device, ...)

Args:
    mask_fn: Function that returns True for valid attention positions
    B: Batch size (or None for batch-independent masks)
    H: Number of heads (or None for head-independent masks)
    Q_LEN: Query sequence length
    KV_LEN: Key/Value sequence length
    device: Target device
    BLOCK_SIZE: Size of blocks (default 128)

The function:
1. Evaluates mask_fn at block granularity
2. Creates sparse representation of valid blocks
3. Returns BlockMask object for flex_attention

Performance Tip:
- Create block_mask ONCE at the start of your model
- Reuse for all attention layers if pattern is same
- Only recreate when sequence length changes
"""


def demonstrate_block_mask_creation():
    """
    Demonstrate block mask creation and visualization.
    """
    print("\n" + "="*70)
    print("BLOCK MASK CREATION DEMONSTRATION")
    print("="*70)
    
    if not FLEX_ATTENTION_AVAILABLE:
        print("\nFlex Attention not available. Showing conceptual example.")
        
        # Conceptual demonstration
        seq_len = 512
        block_size = 128
        num_blocks = seq_len // block_size
        
        print(f"\nSequence length: {seq_len}")
        print(f"Block size: {block_size}")
        print(f"Number of blocks: {num_blocks} × {num_blocks} = {num_blocks**2}")
        
        print("\nCausal Block Mask Pattern:")
        for i in range(num_blocks):
            row = ""
            for j in range(num_blocks):
                # Block (i, j) is valid if any position in block i can attend to block j
                # For causal: block is valid if i*block_size >= j*block_size (i >= j)
                if i >= j:
                    row += "■ "  # Compute this block
                else:
                    row += "□ "  # Skip this block
            print(f"  Q[{i}]: {row}")
        
        valid_blocks = sum(1 for i in range(num_blocks) for j in range(num_blocks) if i >= j)
        total_blocks = num_blocks ** 2
        sparsity = (total_blocks - valid_blocks) / total_blocks * 100
        
        print(f"\nValid blocks: {valid_blocks}/{total_blocks}")
        print(f"Sparsity: {sparsity:.1f}%")
        print(f"FLOPs saved: {sparsity:.1f}%")
        
        return
    
    # Real Flex Attention demonstration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, H = 1, 8
    Q_LEN = KV_LEN = 1024
    BLOCK_SIZE = 128
    
    print(f"\nCreating block mask for:")
    print(f"  Batch: {B}, Heads: {H}")
    print(f"  Sequence length: {Q_LEN}")
    print(f"  Block size: {BLOCK_SIZE}")
    
    # Create causal block mask
    causal_block_mask = create_block_mask(
        causal_mask_fn,
        B=B, H=H,
        Q_LEN=Q_LEN, KV_LEN=KV_LEN,
        device=device,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    print(f"\nCausal block mask created!")
    print(f"  Type: {type(causal_block_mask)}")
    
    # Create sliding window block mask
    window_size = 256
    sliding_mask_fn = partial(sliding_window_mask_fn, window_size=window_size)
    
    sliding_block_mask = create_block_mask(
        sliding_mask_fn,
        B=B, H=H,
        Q_LEN=Q_LEN, KV_LEN=KV_LEN,
        device=device,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    print(f"\nSliding window block mask created (window={window_size})!")


# =============================================================================
# SECTION 4: FLEX ATTENTION USAGE PATTERNS
# =============================================================================

def flex_attention_examples():
    """
    Show common flex_attention usage patterns.
    """
    print("\n" + "="*70)
    print("FLEX ATTENTION USAGE EXAMPLES")
    print("="*70)
    
    if not FLEX_ATTENTION_AVAILABLE:
        print("\nFlex Attention not available. Showing code patterns.")
        
        print("""
# Example 1: Basic causal attention
# ---------------------------------
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, float('-inf'))

# Create block mask (do this once, reuse for all layers)
block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=2048, KV_LEN=2048)

# In forward pass:
output = flex_attention(Q, K, V, score_mod=causal_score_mod, block_mask=block_mask)


# Example 2: ALiBi attention
# --------------------------
def alibi_score_mod(score, b, h, q_idx, kv_idx):
    slope = 2 ** (-(h + 1) * 8.0 / num_heads)
    bias = -slope * (q_idx - kv_idx).abs()
    return score + bias

# ALiBi is dense, so no block_mask optimization
output = flex_attention(Q, K, V, score_mod=alibi_score_mod)


# Example 3: Sliding window + Causal
# ----------------------------------
WINDOW = 256

def sliding_causal_mask(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= WINDOW)

def sliding_causal_score(score, b, h, q_idx, kv_idx):
    valid = (q_idx >= kv_idx) & ((q_idx - kv_idx) <= WINDOW)
    return torch.where(valid, score, float('-inf'))

block_mask = create_block_mask(sliding_causal_mask, ...)
output = flex_attention(Q, K, V, score_mod=sliding_causal_score, block_mask=block_mask)


# Example 4: Document masking with torch.compile
# ---------------------------------------------
@torch.compile
def document_attention(Q, K, V, doc_ids, block_mask):
    def doc_score_mod(score, b, h, q_idx, kv_idx):
        same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
        return torch.where(same_doc, score, float('-inf'))
    
    return flex_attention(Q, K, V, score_mod=doc_score_mod, block_mask=block_mask)


# Example 5: Combining multiple patterns
# -------------------------------------
# You can combine masks using and_masks, or_masks

from torch.nn.attention.flex_attention import and_masks, or_masks

# Causal AND sliding window
combined_mask = and_masks(causal_mask, sliding_window_mask)

# Causal OR some special tokens can attend globally  
combined_mask = or_masks(causal_mask, global_attention_mask)
""")
        return
    
    # Real implementation when Flex Attention is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    batch, heads, seq_len, head_dim = 2, 8, 512, 64
    
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device)
    
    # Example 1: Causal attention
    print("\n1. Causal Attention:")
    
    causal_block_mask = create_block_mask(
        causal_mask_fn, B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len,
        device=device
    )
    
    output_causal = flex_attention(Q, K, V, 
                                   score_mod=causal_score_mod,
                                   block_mask=causal_block_mask)
    print(f"   Output shape: {output_causal.shape}")
    
    # Example 2: ALiBi
    print("\n2. ALiBi Attention:")
    alibi_mod = partial(alibi_score_mod, num_heads=heads)
    output_alibi = flex_attention(Q, K, V, score_mod=alibi_mod)
    print(f"   Output shape: {output_alibi.shape}")
    
    # Example 3: Sliding window
    print("\n3. Sliding Window Attention:")
    window_size = 128
    
    sliding_mask = partial(sliding_window_mask_fn, window_size=window_size)
    sliding_block_mask = create_block_mask(
        sliding_mask, B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len,
        device=device
    )
    
    sliding_score = partial(sliding_window_score_mod, window_size=window_size)
    output_sliding = flex_attention(Q, K, V,
                                    score_mod=sliding_score,
                                    block_mask=sliding_block_mask)
    print(f"   Output shape: {output_sliding.shape}")


# =============================================================================
# SECTION 5: PERFORMANCE CONSIDERATIONS
# =============================================================================
"""
PERFORMANCE BEST PRACTICES:

1. Always use torch.compile
   - Flex Attention is designed for compilation
   - Without compile, it falls back to slower path
   
   @torch.compile
   def attention_layer(Q, K, V, block_mask):
       return flex_attention(Q, K, V, score_mod=my_mod, block_mask=block_mask)

2. Reuse Block Masks
   - create_block_mask is expensive
   - Create once, reuse for all layers with same pattern
   - Only recreate when sequence length changes
   
   class Transformer:
       def forward(self, x):
           if self.block_mask is None or self.block_mask.shape != (seq_len, seq_len):
               self.block_mask = create_block_mask(...)
           for layer in self.layers:
               x = layer(x, block_mask=self.block_mask)

3. Use Block Masks for Sparse Patterns
   - Causal: ~50% blocks skipped
   - Sliding window: Even more savings
   - Without block_mask, all N² scores computed

4. Head-Independent vs Head-Dependent
   - If mask is same for all heads, use H=None
   - Reduces block_mask memory and creation time
   - Same for batch dimension

5. score_mod Should Be Simple
   - Complex logic slows down compilation
   - Pre-compute what you can outside score_mod
   - Use closures to capture constants

6. Watch Compilation Time
   - First call triggers compilation
   - Subsequent calls use cached kernel
   - Different sequence lengths may trigger recompilation
"""


def performance_comparison():
    """
    Compare performance of different attention implementations.
    """
    print("\n" + "="*70)
    print("ATTENTION PERFORMANCE COMPARISON")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping benchmark.")
        return
    
    import time
    
    device = torch.device("cuda")
    
    # Test configurations
    configs = [
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (2, 8, 4096, 64),
    ]
    
    print("\nBenchmark: Causal Attention")
    print("-" * 60)
    print(f"{'Config':<25} {'Standard':<15} {'SDPA':<15}")
    print("-" * 60)
    
    for batch, heads, seq_len, head_dim in configs:
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Warmup
        for _ in range(3):
            _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        torch.cuda.synchronize()
        
        # Standard attention
        try:
            scale = 1.0 / math.sqrt(head_dim)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                S = torch.matmul(Q, K.transpose(-2, -1)) * scale
                S = S.masked_fill(causal_mask, float('-inf'))
                P = F.softmax(S, dim=-1)
                O = torch.matmul(P, V)
            torch.cuda.synchronize()
            standard_time = (time.perf_counter() - start) / 10 * 1000
        except RuntimeError:
            standard_time = float('inf')
        
        # PyTorch SDPA (uses Flash Attention internally)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / 10 * 1000
        
        config_str = f"B={batch}, H={heads}, S={seq_len}"
        print(f"{config_str:<25} {standard_time:.2f} ms{' '*5} {sdpa_time:.2f} ms")
    
    print("-" * 60)
    print("\nNote: SDPA uses Flash Attention kernel when available")
    print("Flex Attention adds flexibility while maintaining similar performance")


# =============================================================================
# SECTION 6: ADVANCED PATTERNS
# =============================================================================

def advanced_attention_patterns():
    """
    Demonstrate advanced attention pattern implementations.
    """
    print("\n" + "="*70)
    print("ADVANCED ATTENTION PATTERNS")
    print("="*70)
    
    print("""
1. SPARSE ATTENTION (BigBird/Longformer style)
   ─────────────────────────────────────────
   Combines: Global + Local (sliding window) + Random
   
   def sparse_attention_mask(b, h, q_idx, kv_idx, 
                             global_tokens, window_size, random_indices):
       # Global tokens attend to and are attended by all
       is_global = (q_idx < global_tokens) | (kv_idx < global_tokens)
       
       # Local sliding window
       is_local = (q_idx - kv_idx).abs() <= window_size
       
       # Random attention pattern (precomputed)
       is_random = random_indices[q_idx, kv_idx]
       
       return is_global | is_local | is_random


2. DILATED ATTENTION
   ──────────────────
   Attend to every k-th position (different k per head for coverage)
   
   def dilated_attention_mask(b, h, q_idx, kv_idx, dilation_rates):
       rate = dilation_rates[h]
       return (q_idx - kv_idx) % rate == 0


3. CHUNKED/BLOCKWISE ATTENTION
   ────────────────────────────
   Divide sequence into chunks, attend within chunks
   
   def chunked_attention_mask(b, h, q_idx, kv_idx, chunk_size):
       q_chunk = q_idx // chunk_size
       kv_chunk = kv_idx // chunk_size
       return q_chunk == kv_chunk


4. MEMORY/SINK TOKENS
   ───────────────────
   First N tokens always attended (memory/anchor)
   Rest is causal with sliding window
   
   def sink_attention_mask(b, h, q_idx, kv_idx, num_sink, window_size):
       # Sink tokens are always visible
       is_sink = kv_idx < num_sink
       
       # Normal causal sliding window for rest
       in_window = (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)
       
       return is_sink | in_window


5. CROSS-ATTENTION WITH MASKS
   ───────────────────────────
   For encoder-decoder with variable length inputs
   
   def cross_attention_mask(b, h, q_idx, kv_idx, encoder_lengths):
       # Only attend to valid encoder positions
       return kv_idx < encoder_lengths[b]


6. GROUPED QUERY ATTENTION (GQA) PATTERN
   ──────────────────────────────────────
   Multiple query heads share fewer KV heads
   
   # GQA is handled at the tensor level, not mask level
   # Q: (batch, num_q_heads, seq, head_dim)
   # K, V: (batch, num_kv_heads, seq, head_dim)
   # 
   # Repeat K, V to match Q heads:
   # K = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
   
   
7. MULTI-DOCUMENT WITH CAUSAL
   ───────────────────────────
   Each document is causal, but no cross-document attention
   
   def multi_doc_causal_mask(b, h, q_idx, kv_idx, doc_ids):
       same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
       causal = q_idx >= kv_idx
       return same_doc & causal
""")


# =============================================================================
# SECTION 7: MIGRATION FROM STANDARD ATTENTION
# =============================================================================

def migration_guide():
    """
    Guide for migrating from standard attention to Flex Attention.
    """
    print("\n" + "="*70)
    print("MIGRATION GUIDE: Standard → Flex Attention")
    print("="*70)
    
    print("""
STEP 1: IDENTIFY YOUR ATTENTION PATTERN
────────────────────────────────────────
Current code might look like:

    # Standard attention with manual masking
    scores = Q @ K.T / sqrt(d)
    
    # Manual mask creation
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, -inf)
    
    # Maybe add positional bias
    scores = scores + position_bias
    
    attn = softmax(scores)
    output = attn @ V


STEP 2: CONVERT MASK TO MASK FUNCTION
────────────────────────────────────────
Replace mask tensor with mask function:

    # Old: mask = torch.triu(torch.ones(...), diagonal=1).bool()
    
    # New:
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx


STEP 3: CONVERT BIAS TO SCORE_MOD
────────────────────────────────────────
Replace additive bias with score_mod:

    # Old: scores = scores + position_bias[q_idx, kv_idx]
    
    # New:
    def position_score_mod(score, b, h, q_idx, kv_idx):
        return score + position_bias[q_idx, kv_idx]


STEP 4: COMBINE MASK AND MODIFICATION
────────────────────────────────────────
If you have both mask and bias:

    def combined_score_mod(score, b, h, q_idx, kv_idx):
        # Add position bias
        score = score + position_bias[q_idx, kv_idx]
        # Apply mask
        score = torch.where(q_idx >= kv_idx, score, float('-inf'))
        return score


STEP 5: CREATE BLOCK MASK FOR EFFICIENCY
────────────────────────────────────────
Add block_mask for sparse patterns:

    block_mask = create_block_mask(
        causal_mask,
        B=None, H=None,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=device
    )


STEP 6: CALL FLEX_ATTENTION
────────────────────────────────────────
Replace your attention code:

    # Old:
    output = manual_attention(Q, K, V, mask, bias)
    
    # New:
    output = flex_attention(Q, K, V, 
                           score_mod=combined_score_mod,
                           block_mask=block_mask)


STEP 7: ADD TORCH.COMPILE
────────────────────────────────────────
Wrap in compile for best performance:

    @torch.compile
    def attention_forward(Q, K, V, block_mask):
        return flex_attention(Q, K, V,
                            score_mod=my_score_mod,
                            block_mask=block_mask)


COMMON GOTCHAS:
───────────────
1. score_mod must work element-wise (no batch operations)
2. Mask function should return bool tensor
3. Block mask creation is expensive - cache it
4. First compile takes time, subsequent calls are fast
5. Different seq_lens may trigger recompilation
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FLEX ATTENTION COMPREHENSIVE GUIDE")
    print("="*70)
    
    # Demonstrate block mask creation
    demonstrate_block_mask_creation()
    
    # Show usage examples
    flex_attention_examples()
    
    # Performance comparison
    performance_comparison()
    
    # Advanced patterns
    advanced_attention_patterns()
    
    # Migration guide
    migration_guide()
    
    print("\n" + "="*70)
    print("GUIDE COMPLETE")
    print("="*70)
