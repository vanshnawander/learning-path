# Flash Attention Deep Dive

Memory-efficient exact attention algorithm - the breakthrough that made
long-context transformers practical.

## 📚 Modules Created

| File | Description |
|------|-------------|
| `01_flash_attention_deep_dive.py` | Complete algorithm with implementation |
| `02_flex_attention_pytorch.py` | PyTorch 2.5+ Flex Attention API |

## The Breakthrough

| Metric | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory | O(N²) | O(N) |
| Speed | Baseline | 2-4x faster |
| Accuracy | Exact | Exact |

## 🔬 Topics Covered In Depth

### GPU Memory Hierarchy
- **HBM vs SRAM**: Bandwidth and latency differences
- **Memory-bound vs Compute-bound**: Why standard attention is slow
- **IO Complexity Analysis**: Quantifying memory access patterns

### Online Softmax Algorithm
- **Mathematical Derivation**: Incremental max and sum updates
- **Numerical Stability**: Maintaining precision across blocks
- **Implementation**: Step-by-step Python code

### Tiling Strategy
- **Block Size Selection**: Based on SRAM capacity
- **Loop Order**: Outer KV, inner Q for efficiency
- **Parallelization**: Thread block assignment

### Flash Attention Forward Pass
- **Complete Algorithm**: Pseudocode and implementation
- **Statistics Tracking**: Running max (m) and sum (l)
- **Output Accumulation**: Incremental weighted sum

### Flash Attention Backward Pass
- **Recomputation Strategy**: Don't save P, recompute it
- **Memory Savings**: Only save Q, K, V, O, and statistics
- **Gradient Computation**: dQ, dK, dV derivation

### Flash Attention Versions
```
FA1 (2022): Original algorithm, 2-4x speedup
FA2 (2023): Better parallelism, 2x over FA1
FA3 (2024): Hopper TMA, warp specialization, FP8
```

### Flex Attention (PyTorch 2.5+)
- **score_mod**: Custom score modification functions
- **block_mask**: Compile-time sparsity patterns
- **Common Patterns**: Causal, ALiBi, sliding window, document masks
- **Performance**: Near Flash Attention speed with full flexibility

## Algorithm Overview

### Standard Attention
```python
S = Q @ K^T / sqrt(d)   # Attention scores (N×N) - STORED IN HBM
P = softmax(S, dim=-1)  # Attention weights (N×N) - STORED IN HBM
O = P @ V               # Output (N×d)
```

**Problem**: S and P are N×N matrices = quadratic memory!

### Flash Attention
```python
# Never materialize full N×N matrix
for each K, V block:
    for each Q block:
        S_block = Q_block @ K_block^T / sqrt(d)  # In SRAM
        Update running max, sum, output          # Online softmax
```

## IO Complexity

| Operation | Standard | Flash |
|-----------|----------|-------|
| HBM reads | O(Nd + N²) | O(N²d²/M) |
| HBM writes | O(Nd + N²) | O(Nd) |
| Memory | O(N²) | O(N) |

Where M = SRAM size (~100KB). Flash Attention is **compute-bound**!

## 💻 Practical Exercises

1. Run online softmax demonstration
2. Implement tiled attention forward pass
3. Compare memory usage vs standard attention
4. Use Flex Attention for custom patterns
5. Benchmark different attention implementations

## 📖 Key Papers

- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism" (Dao, 2023)
- "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (2024)
- "FlexAttention: PyTorch Native Flexible Attention" (PyTorch Team, 2024)

## 🔧 Code References

- `flash-attention/csrc/flash_attn/` - CUDA implementation
- `flash-attention/flash_attn/flash_attn_interface.py` - Python API
- `torch.nn.attention.flex_attention` - PyTorch Flex Attention
- `torch.nn.functional.scaled_dot_product_attention` - PyTorch SDPA
