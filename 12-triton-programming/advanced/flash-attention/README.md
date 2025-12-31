# Flash Attention in Triton

Implementing memory-efficient attention from scratch.

## The Problem

Standard attention: O(N²) memory
```python
S = Q @ K.T          # N×N matrix
P = softmax(S)       # N×N matrix  
O = P @ V            # Output
```

For N=4096, S takes 64MB in FP32!

## Flash Attention Key Ideas

### 1. Tiling
- Process Q, K, V in blocks
- Never materialize full N×N matrix
- Memory: O(N) instead of O(N²)

### 2. Online Softmax
```python
# Traditional: need full row for softmax
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

# Online: update running statistics
m_new = max(m_old, block_max)
l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
```

### 3. Recomputation
- Don't save attention matrix for backward
- Recompute from Q, K, V
- Trade compute for memory

## Triton Implementation Skeleton

```python
@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    # ... more strides
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get block indices
    start_m = tl.program_id(0) * BLOCK_M
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Load Q block (stays in registers)
    q = tl.load(Q + ...)
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K + ...)
        v = tl.load(V + ...)
        
        # Compute attention scores
        s = tl.dot(q, tl.trans(k))
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        
        # Update accumulators
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
    
    # Final normalization
    out = acc / l_i[:, None]
    tl.store(Out + ..., out)
```

## Study Resources
- `flash-attention/flash_attn/` - Reference implementation
- `unsloth/unsloth/kernels/flex_attention.py` - Flex Attention
- Flash Attention papers (v1, v2, v3)
