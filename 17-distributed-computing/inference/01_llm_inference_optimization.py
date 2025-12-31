"""
01_llm_inference_optimization.py - LLM Inference Optimization Deep Dive

Comprehensive coverage of modern LLM inference optimization techniques.

Key Topics:
1. KV Cache fundamentals and memory analysis
2. PagedAttention (vLLM) - Virtual memory for KV cache
3. Continuous Batching - Dynamic request handling
4. Speculative Decoding - Parallel token generation
5. Flash Decoding - Optimized decode phase
6. Quantization for inference (INT8, INT4, GGUF)

These techniques enable:
- 10-100x throughput improvement
- 2-4x latency reduction
- Efficient GPU memory utilization

Run: python 01_llm_inference_optimization.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# SECTION 1: LLM INFERENCE BASICS
# ============================================================================
"""
LLM INFERENCE PHASES:
=====================

1. PREFILL PHASE (Prompt Processing):
   - Process all input tokens in parallel
   - Compute-bound (matrix multiplications)
   - Generate KV cache for all input positions
   - Single forward pass

2. DECODE PHASE (Token Generation):
   - Generate one token at a time
   - Memory-bound (loading KV cache)
   - Autoregressive: each token depends on previous
   - Many sequential forward passes

KEY BOTTLENECKS:
================

Prefill: Compute-bound
    Time ∝ batch_size × seq_len × model_size

Decode: Memory-bound  
    Time ∝ batch_size × model_size × num_layers
    (KV cache must be loaded for each token)

MEMORY REQUIREMENTS:
====================

For Llama-7B with batch=1, seq=2048:
    Model weights: 14 GB (FP16)
    KV Cache: 2 × 32 layers × 2048 × 4096 × 2 bytes = 1 GB
    
For batch=32, seq=2048:
    KV Cache: 32 GB (!)
    
KV cache grows linearly with batch size and sequence length!
"""

@dataclass
class LLMConfig:
    """Configuration for LLM inference analysis."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # For GQA
    head_dim: int = 128
    intermediate_size: int = 11008
    max_seq_len: int = 4096
    
    def kv_cache_size_per_token(self, dtype_bytes: int = 2) -> int:
        """Memory for KV cache per token per layer."""
        # K and V each: num_kv_heads × head_dim
        return 2 * self.num_kv_heads * self.head_dim * dtype_bytes
    
    def total_kv_cache_size(self, batch_size: int, seq_len: int, 
                            dtype_bytes: int = 2) -> int:
        """Total KV cache memory in bytes."""
        per_token = self.kv_cache_size_per_token(dtype_bytes)
        return batch_size * seq_len * self.num_layers * per_token


def analyze_memory_requirements():
    """Analyze memory requirements for LLM inference."""
    print("\n" + "="*70)
    print(" LLM INFERENCE MEMORY ANALYSIS")
    print("="*70)
    
    config = LLMConfig()  # Llama-7B like config
    
    print(f"\n Model Configuration (Llama-7B like):")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Num heads: {config.num_heads}")
    print(f"   Head dim: {config.head_dim}")
    
    # Model weights
    # Rough estimate: 4 * hidden² per layer (QKV + O + MLP)
    params_per_layer = 4 * config.hidden_size * config.hidden_size + \
                       3 * config.hidden_size * config.intermediate_size
    total_params = config.num_layers * params_per_layer + \
                   config.vocab_size * config.hidden_size * 2
    
    print(f"\n Model Weights:")
    print(f"   Parameters: {total_params / 1e9:.1f}B")
    print(f"   FP16: {total_params * 2 / 1e9:.1f} GB")
    print(f"   INT8: {total_params * 1 / 1e9:.1f} GB")
    print(f"   INT4: {total_params * 0.5 / 1e9:.1f} GB")
    
    # KV cache analysis
    print(f"\n KV Cache Analysis:")
    
    for batch_size in [1, 8, 32, 128]:
        for seq_len in [512, 2048, 8192]:
            kv_size = config.total_kv_cache_size(batch_size, seq_len)
            print(f"   Batch={batch_size:3d}, Seq={seq_len:5d}: "
                  f"{kv_size / 1e9:.2f} GB")
    
    # Memory bandwidth analysis
    print(f"\n Memory Bandwidth Analysis (A100 @ 2TB/s):")
    
    a100_bandwidth = 2e12  # 2 TB/s
    
    for batch_size in [1, 8, 32]:
        # Per decode step: load KV cache + model weights
        kv_per_step = config.total_kv_cache_size(batch_size, 2048)
        model_size = total_params * 2
        
        # Time = data / bandwidth
        time_ms = (kv_per_step + model_size) / a100_bandwidth * 1000
        tokens_per_sec = batch_size / time_ms * 1000
        
        print(f"   Batch={batch_size}: {time_ms:.2f} ms/token, "
              f"{tokens_per_sec:.0f} tok/s")


# ============================================================================
# SECTION 2: KV CACHE IMPLEMENTATION
# ============================================================================
"""
KV CACHE FUNDAMENTALS:
======================

Without KV cache:
    For each new token, recompute attention for ALL previous tokens
    Complexity: O(n²) per token, O(n³) total
    
With KV cache:
    Store K and V for all previous positions
    Only compute attention for new token against cached KV
    Complexity: O(n) per token, O(n²) total

IMPLEMENTATION:
===============

During prefill:
    K_cache[0:n] = K_proj(hidden[0:n])
    V_cache[0:n] = V_proj(hidden[0:n])
    
During decode:
    K_new = K_proj(hidden[n])
    V_new = V_proj(hidden[n])
    K_cache[n] = K_new
    V_cache[n] = V_new
    
    # Attention with full cache
    scores = Q_new @ K_cache[0:n+1].T
    output = softmax(scores) @ V_cache[0:n+1]
"""

class KVCache:
    """
    Basic KV Cache implementation for understanding.
    
    Real implementations use:
    - Pre-allocated buffers
    - Efficient memory layout
    - Optional quantization
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        
        # Track current position
        self.seq_len = 0
    
    def update(
        self,
        k: torch.Tensor,  # (batch, heads, new_len, head_dim)
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full K, V for attention."""
        new_len = k.shape[2]
        
        # Store new values
        self.k_cache[:, :, self.seq_len:self.seq_len + new_len] = k
        self.v_cache[:, :, self.seq_len:self.seq_len + new_len] = v
        
        self.seq_len += new_len
        
        # Return full cache up to current position
        return (
            self.k_cache[:, :, :self.seq_len],
            self.v_cache[:, :, :self.seq_len],
        )
    
    def get_memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return (self.k_cache.numel() + self.v_cache.numel()) * \
               self.k_cache.element_size()


def demonstrate_kv_cache():
    """Demonstrate KV cache usage."""
    print("\n" + "="*70)
    print(" KV CACHE DEMONSTRATION")
    print("="*70)
    
    batch_size = 2
    num_heads = 8
    head_dim = 64
    max_seq_len = 1024
    
    cache = KVCache(batch_size, max_seq_len, num_heads, head_dim, 
                    device='cpu', dtype=torch.float32)
    
    print(f"\n Cache Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Max seq len: {max_seq_len}")
    print(f"   Num heads: {num_heads}")
    print(f"   Head dim: {head_dim}")
    print(f"   Memory allocated: {cache.get_memory_usage() / 1e6:.2f} MB")
    
    # Simulate prefill (10 tokens)
    prefill_len = 10
    k_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    v_prefill = torch.randn(batch_size, num_heads, prefill_len, head_dim)
    
    k_full, v_full = cache.update(k_prefill, v_prefill)
    print(f"\n After prefill ({prefill_len} tokens):")
    print(f"   Cache seq_len: {cache.seq_len}")
    print(f"   K shape: {k_full.shape}")
    
    # Simulate decode (3 tokens)
    for i in range(3):
        k_new = torch.randn(batch_size, num_heads, 1, head_dim)
        v_new = torch.randn(batch_size, num_heads, 1, head_dim)
        
        k_full, v_full = cache.update(k_new, v_new)
        print(f"   After decode step {i+1}: seq_len={cache.seq_len}")


# ============================================================================
# SECTION 3: PAGEDATTENTION (vLLM)
# ============================================================================
"""
PAGEDATTENTION - Virtual Memory for KV Cache:
==============================================

PROBLEM:
- KV cache size is unpredictable (variable output lengths)
- Static allocation wastes memory
- Fragmentation prevents batching more requests

SOLUTION (inspired by OS paging):
- Divide KV cache into fixed-size BLOCKS (e.g., 16 tokens)
- Allocate blocks on-demand from a block pool
- Map logical positions to physical blocks via block table

EXAMPLE:
========

Request 1: "Hello world" (10 tokens generated)
    Logical blocks: [0, 1] (needs 10/16 = 1 block + partial)
    Physical blocks: [5, 12] (wherever free)
    Block table: {0: 5, 1: 12}

Request 2: "What is AI" (5 tokens generated)
    Logical blocks: [0]
    Physical blocks: [3]
    Block table: {0: 3}

BENEFITS:
=========
1. Near-zero memory waste
2. Easy memory sharing (copy-on-write for beam search)
3. Better batching (no fragmentation)
4. Dynamic memory allocation

IMPLEMENTATION INSIGHT:
=======================
Attention computation must be modified to:
1. Look up physical block from block table
2. Gather KV from non-contiguous physical blocks
3. Compute attention as usual
"""

@dataclass
class PagedAttentionConfig:
    """Configuration for PagedAttention."""
    block_size: int = 16  # Tokens per block
    num_blocks: int = 1000  # Total blocks in pool
    num_heads: int = 32
    head_dim: int = 128


class BlockManager:
    """
    Simplified block manager for PagedAttention.
    
    Manages allocation of physical blocks to sequences.
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.block_size = config.block_size
        self.num_blocks = config.num_blocks
        
        # Track which blocks are free
        self.free_blocks = list(range(config.num_blocks))
        
        # Map sequence_id -> list of physical block indices
        self.block_tables: Dict[int, List[int]] = {}
    
    def allocate_block(self, seq_id: int) -> int:
        """Allocate a new block for a sequence."""
        if not self.free_blocks:
            raise RuntimeError("Out of blocks!")
        
        block_idx = self.free_blocks.pop()
        
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
        
        self.block_tables[seq_id].append(block_idx)
        return block_idx
    
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
    
    def get_block_table(self, seq_id: int) -> List[int]:
        """Get block table for a sequence."""
        return self.block_tables.get(seq_id, [])
    
    def num_free_blocks(self) -> int:
        """Return number of free blocks."""
        return len(self.free_blocks)


def demonstrate_paged_attention():
    """Demonstrate PagedAttention concepts."""
    print("\n" + "="*70)
    print(" PAGEDATTENTION (vLLM) DEMONSTRATION")
    print("="*70)
    
    config = PagedAttentionConfig(
        block_size=16,
        num_blocks=100,
        num_heads=32,
        head_dim=128,
    )
    
    manager = BlockManager(config)
    
    print(f"\n Configuration:")
    print(f"   Block size: {config.block_size} tokens")
    print(f"   Total blocks: {config.num_blocks}")
    print(f"   Memory per block: {config.block_size * config.num_heads * config.head_dim * 2 * 2 / 1024:.1f} KB")
    
    # Simulate multiple requests
    print(f"\n Simulating requests:")
    print(f"   Free blocks: {manager.num_free_blocks()}")
    
    # Request 1: Long generation
    seq1_tokens = 50
    seq1_blocks_needed = (seq1_tokens + config.block_size - 1) // config.block_size
    
    for _ in range(seq1_blocks_needed):
        manager.allocate_block(seq_id=1)
    
    print(f"\n   Request 1 ({seq1_tokens} tokens):")
    print(f"      Blocks allocated: {seq1_blocks_needed}")
    print(f"      Block table: {manager.get_block_table(1)}")
    print(f"      Free blocks: {manager.num_free_blocks()}")
    
    # Request 2: Short generation
    seq2_tokens = 20
    seq2_blocks_needed = (seq2_tokens + config.block_size - 1) // config.block_size
    
    for _ in range(seq2_blocks_needed):
        manager.allocate_block(seq_id=2)
    
    print(f"\n   Request 2 ({seq2_tokens} tokens):")
    print(f"      Blocks allocated: {seq2_blocks_needed}")
    print(f"      Block table: {manager.get_block_table(2)}")
    print(f"      Free blocks: {manager.num_free_blocks()}")
    
    # Free request 1
    manager.free_sequence(seq_id=1)
    print(f"\n   After freeing Request 1:")
    print(f"      Free blocks: {manager.num_free_blocks()}")
    
    # Memory efficiency comparison
    print(f"\n Memory Efficiency Comparison:")
    
    max_seq_len = 2048
    batch_size = 32
    
    # Static allocation
    static_memory = batch_size * max_seq_len * config.num_heads * \
                    config.head_dim * 2 * 2  # K and V, FP16
    
    # PagedAttention with 50% average utilization
    avg_seq_len = max_seq_len // 2
    paged_memory = batch_size * avg_seq_len * config.num_heads * \
                   config.head_dim * 2 * 2
    
    print(f"   Static allocation (worst case): {static_memory / 1e9:.2f} GB")
    print(f"   PagedAttention (50% util): {paged_memory / 1e9:.2f} GB")
    print(f"   Memory saved: {(1 - paged_memory/static_memory) * 100:.0f}%")


# ============================================================================
# SECTION 4: CONTINUOUS BATCHING
# ============================================================================
"""
CONTINUOUS BATCHING (In-flight Batching):
==========================================

PROBLEM with Static Batching:
- Wait for ALL requests in batch to complete
- Short requests wait for long ones
- GPU underutilized

SOLUTION - Continuous Batching:
- Process requests at token level
- Evict completed requests immediately
- Insert new requests on-the-fly

EXAMPLE:
========

Static Batching:
    Time 0: [Req1, Req2, Req3] start
    Time 10: Req1 done (waits)
    Time 20: Req2 done (waits)
    Time 50: Req3 done → ALL released
    
Continuous Batching:
    Time 0: [Req1, Req2, Req3] start
    Time 10: Req1 done → Req4 joins
    Time 20: Req2 done → Req5 joins
    ...
    
THROUGHPUT IMPROVEMENT: 2-5x typical
"""

@dataclass
class Request:
    """A generation request."""
    request_id: int
    prompt_tokens: int
    generated_tokens: int = 0
    max_tokens: int = 100
    is_complete: bool = False


class ContinuousBatchingScheduler:
    """
    Simplified continuous batching scheduler.
    
    Real implementations (vLLM, TGI) are more sophisticated.
    """
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.running: List[Request] = []
        self.waiting: List[Request] = []
    
    def add_request(self, request: Request):
        """Add a new request to the scheduler."""
        if len(self.running) < self.max_batch_size:
            self.running.append(request)
        else:
            self.waiting.append(request)
    
    def step(self) -> List[Request]:
        """
        Perform one decode step.
        
        Returns list of completed requests.
        """
        completed = []
        
        # Generate one token for each running request
        still_running = []
        for req in self.running:
            req.generated_tokens += 1
            
            if req.generated_tokens >= req.max_tokens:
                req.is_complete = True
                completed.append(req)
            else:
                still_running.append(req)
        
        self.running = still_running
        
        # Fill slots with waiting requests
        while self.waiting and len(self.running) < self.max_batch_size:
            self.running.append(self.waiting.pop(0))
        
        return completed
    
    def get_batch_size(self) -> int:
        """Current batch size."""
        return len(self.running)


def demonstrate_continuous_batching():
    """Compare static vs continuous batching."""
    print("\n" + "="*70)
    print(" CONTINUOUS BATCHING DEMONSTRATION")
    print("="*70)
    
    import random
    random.seed(42)
    
    # Create requests with varying lengths
    num_requests = 20
    requests = [
        Request(
            request_id=i,
            prompt_tokens=50,
            max_tokens=random.randint(10, 100),
        )
        for i in range(num_requests)
    ]
    
    print(f"\n Created {num_requests} requests")
    print(f" Output lengths: {[r.max_tokens for r in requests]}")
    
    # Static batching simulation
    print(f"\n STATIC BATCHING (batch_size=4):")
    
    static_batches = [requests[i:i+4] for i in range(0, len(requests), 4)]
    total_static_steps = 0
    
    for batch_idx, batch in enumerate(static_batches):
        max_len = max(r.max_tokens for r in batch)
        total_static_steps += max_len
        waste = sum(max_len - r.max_tokens for r in batch)
        print(f"   Batch {batch_idx}: max_len={max_len}, wasted_steps={waste}")
    
    print(f"   Total steps: {total_static_steps}")
    
    # Continuous batching simulation
    print(f"\n CONTINUOUS BATCHING (max_batch_size=4):")
    
    scheduler = ContinuousBatchingScheduler(max_batch_size=4)
    
    # Add initial requests
    for req in requests[:4]:
        scheduler.add_request(Request(
            request_id=req.request_id,
            prompt_tokens=req.prompt_tokens,
            max_tokens=req.max_tokens,
        ))
    
    # Add rest to waiting
    for req in requests[4:]:
        scheduler.add_request(Request(
            request_id=req.request_id,
            prompt_tokens=req.prompt_tokens,
            max_tokens=req.max_tokens,
        ))
    
    continuous_steps = 0
    completed_count = 0
    
    while scheduler.running or scheduler.waiting:
        completed = scheduler.step()
        continuous_steps += 1
        completed_count += len(completed)
        
        if completed:
            for req in completed:
                print(f"   Step {continuous_steps}: Request {req.request_id} "
                      f"completed ({req.generated_tokens} tokens)")
    
    print(f"   Total steps: {continuous_steps}")
    print(f"\n Efficiency improvement: {total_static_steps / continuous_steps:.2f}x")


# ============================================================================
# SECTION 5: SPECULATIVE DECODING
# ============================================================================
"""
SPECULATIVE DECODING:
=====================

PROBLEM:
- Decode is memory-bound (loading weights)
- Each token requires full model forward pass
- GPU compute underutilized

IDEA:
- Use small "draft" model to generate K tokens quickly
- Verify all K tokens in parallel with large "target" model
- Accept matching tokens, reject at first mismatch

ALGORITHM:
==========

1. Draft model generates: [t1, t2, t3, t4, t5] (fast)
2. Target model verifies in ONE forward pass:
   - P(t1|context) → accept if matches draft
   - P(t2|context, t1) → accept if matches
   - P(t3|context, t1, t2) → REJECT (different)
3. Accept [t1, t2], discard rest
4. Sample new token from target at position 3
5. Repeat

SPEEDUP:
========
If draft accepts rate = α, speculation length = K:
    Expected accepted tokens = 1 + α + α² + ... + α^K ≈ 1/(1-α)
    
With α=0.8, K=5: ~3-4 tokens per target forward pass
Speedup: 2-3x typical

REQUIREMENTS:
=============
- Draft model must be MUCH faster (smaller)
- Draft model should have similar distribution
- Common: 7B target + 1B draft
"""

def speculative_decoding_simulation():
    """Simulate speculative decoding."""
    print("\n" + "="*70)
    print(" SPECULATIVE DECODING SIMULATION")
    print("="*70)
    
    import random
    random.seed(42)
    
    # Simulation parameters
    num_tokens_to_generate = 100
    speculation_length = 5
    acceptance_rate = 0.7  # Draft matches target 70% of time
    
    # Cost model (relative)
    draft_cost = 1   # Small model forward pass
    target_cost = 10  # Large model forward pass
    
    print(f"\n Parameters:")
    print(f"   Speculation length (K): {speculation_length}")
    print(f"   Acceptance rate: {acceptance_rate}")
    print(f"   Draft/Target cost ratio: 1:{target_cost}")
    
    # Standard decoding
    standard_cost = num_tokens_to_generate * target_cost
    
    # Speculative decoding simulation
    generated = 0
    speculative_cost = 0
    num_iterations = 0
    
    while generated < num_tokens_to_generate:
        num_iterations += 1
        
        # Draft generates K tokens
        speculative_cost += speculation_length * draft_cost
        
        # Target verifies (one forward pass for K+1 positions)
        speculative_cost += target_cost
        
        # Simulate acceptance
        accepted = 0
        for i in range(speculation_length):
            if random.random() < acceptance_rate:
                accepted += 1
            else:
                break
        
        # Always get at least 1 token (from target)
        accepted = max(1, accepted)
        generated += accepted
    
    print(f"\n Results:")
    print(f"   Tokens generated: {generated}")
    print(f"   Iterations: {num_iterations}")
    print(f"   Avg tokens per iteration: {generated/num_iterations:.2f}")
    
    print(f"\n Cost Analysis:")
    print(f"   Standard decoding cost: {standard_cost}")
    print(f"   Speculative decoding cost: {speculative_cost}")
    print(f"   Speedup: {standard_cost/speculative_cost:.2f}x")
    
    # Theoretical analysis
    print(f"\n Theoretical Analysis:")
    expected_accepted = sum(acceptance_rate**i for i in range(speculation_length + 1))
    theoretical_speedup = expected_accepted * target_cost / \
                         (speculation_length * draft_cost + target_cost)
    print(f"   Expected tokens per iteration: {expected_accepted:.2f}")
    print(f"   Theoretical speedup: {theoretical_speedup:.2f}x")


# ============================================================================
# SECTION 6: INFERENCE QUANTIZATION FORMATS
# ============================================================================
"""
QUANTIZATION FORMATS FOR INFERENCE:
===================================

1. INT8 (W8A8):
   - Weights and activations in INT8
   - 4x memory reduction
   - Good quality, 1-2% accuracy drop
   
2. INT4 (W4A16):
   - Weights in INT4, activations in FP16
   - 8x memory reduction
   - Moderate quality loss

3. GPTQ:
   - Layer-wise quantization with calibration
   - Better quality than naive INT4
   - Requires calibration data

4. AWQ (Activation-aware Weight Quantization):
   - Protects important weights
   - Better quality than GPTQ
   - State-of-art for INT4

5. GGUF (llama.cpp):
   - Multiple quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0)
   - Optimized for CPU inference
   - K-quants use different bits for different weights

GGUF QUANTIZATION LEVELS:
=========================

| Type    | Bits/Weight | Quality | Size (7B) |
|---------|-------------|---------|-----------|
| F16     | 16          | Best    | 14 GB     |
| Q8_0    | 8           | Great   | 7 GB      |
| Q6_K    | 6.5         | Very Good| 5.5 GB   |
| Q5_K_M  | 5.5         | Good    | 4.8 GB    |
| Q4_K_M  | 4.5         | Good    | 4.1 GB    |
| Q4_0    | 4           | OK      | 3.8 GB    |
| Q3_K_M  | 3.5         | Fair    | 3.3 GB    |
| Q2_K    | 2.5         | Poor    | 2.7 GB    |
"""

def print_quantization_comparison():
    """Print quantization format comparison."""
    print("\n" + "="*70)
    print(" INFERENCE QUANTIZATION FORMATS")
    print("="*70)
    
    print("""
    QUANTIZATION DECISION GUIDE:
    ════════════════════════════
    
    For MAXIMUM QUALITY (production):
        → FP16 or BF16 (if memory allows)
        → INT8 (W8A16) with good calibration
    
    For BALANCED (most use cases):
        → AWQ INT4 (best quality/size)
        → GPTQ INT4 (well supported)
        → Q5_K_M GGUF (CPU inference)
    
    For MAXIMUM COMPRESSION (edge/mobile):
        → Q4_K_M GGUF
        → Q3_K_M GGUF (quality loss)
    
    LIBRARY SUPPORT:
    ════════════════
    
    vLLM:       AWQ, GPTQ, FP8 (Hopper)
    TGI:        GPTQ, AWQ, EETQ, bitsandbytes
    llama.cpp:  GGUF (all variants)
    Unsloth:    4-bit NF4/FP4 for training
    
    CHOOSING A FORMAT:
    ══════════════════
    
    GPU Inference (NVIDIA):
        A100/H100: FP8 (best), AWQ, GPTQ
        RTX 4090:  AWQ, GPTQ (INT4)
        RTX 3090:  GPTQ, AWQ
    
    CPU Inference:
        Apple M1/M2: GGUF Q4_K_M or Q5_K_M
        Intel/AMD:   GGUF Q4_K_M
    
    Edge Devices:
        GGUF Q3_K_M or Q2_K
    """)


# ============================================================================
# SECTION 7: FLASH DECODING
# ============================================================================
"""
FLASH DECODING:
===============

PROBLEM:
- Decode phase processes ONE query token
- Standard Flash Attention optimized for long queries
- Decode is memory-bound, not compute-bound

FLASH DECODING SOLUTION:
- Split KV cache across multiple thread blocks
- Each block computes partial attention
- Reduce partial results

For long sequences:
    Standard: 1 thread block, sequential
    Flash Decoding: N thread blocks, parallel
    
Speedup: 2-8x for long contexts (8K+ tokens)

IMPLEMENTATION INSIGHT:
=======================
- Parallelize over KV sequence length (not batch)
- Each block: compute partial softmax + weighted sum
- Final reduction: combine partial results with correction
"""

def print_flash_decoding_concept():
    """Explain Flash Decoding concept."""
    print("\n" + "="*70)
    print(" FLASH DECODING")
    print("="*70)
    
    print("""
    STANDARD ATTENTION DECODE:
    ══════════════════════════
    
    Query: (1, head_dim)  ← Single token
    K, V:  (seq_len, head_dim)  ← Full context
    
    Standard parallelization:
        - Parallelize over batch dimension
        - Each thread block handles one sequence
        - For seq_len=8192, single block does all 8192
        
    Problem: Low GPU utilization for small batch sizes!
    
    
    FLASH DECODING:
    ═══════════════
    
    Split KV cache into chunks:
        K, V chunks: [(0:1024), (1024:2048), ..., (7168:8192)]
        
    Each thread block:
        1. Compute partial attention for its chunk
        2. Return: partial_output, partial_max, partial_sum
        
    Final reduction:
        - Combine partials using logsumexp trick
        - Same as online softmax in Flash Attention
    
    
    SPEEDUP:
    ════════
    
    Context Length    Standard    Flash Decoding    Speedup
    ─────────────────────────────────────────────────────────
    512               1.0 ms      1.1 ms            0.9x
    2048              3.5 ms      1.8 ms            1.9x
    8192              14 ms       3.5 ms            4.0x
    32768             55 ms       8 ms              6.9x
    
    Flash Decoding shines for LONG contexts!
    """)


# ============================================================================
# MAIN
# ============================================================================

def print_summary():
    """Print inference optimization summary."""
    print("\n" + "="*70)
    print(" LLM INFERENCE OPTIMIZATION SUMMARY")
    print("="*70)
    
    print("""
    KEY TECHNIQUES:
    ═══════════════
    
    1. KV CACHING
       - Store K, V for previous tokens
       - Avoid recomputation
       - Memory grows with sequence length
    
    2. PAGEDATTENTION (vLLM)
       - Virtual memory for KV cache
       - Block-based allocation
       - Near-zero memory waste
       - Enables larger batch sizes
    
    3. CONTINUOUS BATCHING
       - Process requests at token level
       - No waiting for long requests
       - 2-5x throughput improvement
    
    4. SPECULATIVE DECODING
       - Draft model generates candidates
       - Target model verifies in parallel
       - 2-3x speedup typical
    
    5. QUANTIZATION
       - INT8/INT4 for memory reduction
       - AWQ/GPTQ for quality
       - GGUF for CPU inference
    
    6. FLASH DECODING
       - Parallelize over KV length
       - 2-8x speedup for long contexts
    
    
    RECOMMENDED STACK:
    ══════════════════
    
    Production (NVIDIA GPU):
        vLLM + PagedAttention + AWQ/GPTQ
        
    Research/Development:
        HuggingFace TGI + Flash Attention 2
        
    Local/Edge:
        llama.cpp + GGUF Q4_K_M
        
    Training + Inference:
        Unsloth (training) + vLLM (inference)
    
    
    CODE REFERENCES:
    ════════════════
    
    vLLM:        github.com/vllm-project/vllm
    TGI:         github.com/huggingface/text-generation-inference
    llama.cpp:   github.com/ggerganov/llama.cpp
    Flash Attn:  github.com/Dao-AILab/flash-attention
    """)


if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " LLM INFERENCE OPTIMIZATION ".center(68) + "║")
    print("║" + " KV Cache, PagedAttention, Speculative Decoding ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    analyze_memory_requirements()
    demonstrate_kv_cache()
    demonstrate_paged_attention()
    demonstrate_continuous_batching()
    speculative_decoding_simulation()
    print_quantization_comparison()
    print_flash_decoding_concept()
    print_summary()
