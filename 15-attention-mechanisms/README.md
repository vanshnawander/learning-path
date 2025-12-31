# 15 - Attention Mechanisms - Comprehensive Deep Dive

**From Bahdanau (2014) to Mamba (2024)** - Complete coverage of attention mechanisms,
including latest research on RWKV, State Space Models, and hybrid architectures.

## 📚 Modules Created

### Python Files (with Profiling)

| File | Description |
|------|-------------|
| `01_attention_history_fundamentals.py` | Bahdanau, Luong, seq2seq bottleneck, QKV framework |
| `02_transformer_attention.py` | Multi-head attention, position encodings, encoder-decoder |
| `03_efficient_attention.py` | Flash Attention, linear attention, GQA, sliding window |
| `04_advanced_attention.py` | RWKV, Mamba, SSMs, gated attention, hybrids |
| `05_attention_pitfalls.py` | Numerical stability, memory issues, debugging |

### Interactive Notebooks

| File | Description |
|------|-------------|
| `attention_mechanisms_notebook.ipynb` | Hands-on attention implementation and visualization |

## 🔬 Research Timeline Covered

```
2014: Bahdanau Attention    → Solved seq2seq bottleneck
2015: Luong Attention       → Simplified variants (dot, general)
2017: Transformer           → "Attention Is All You Need"
2018: BERT, GPT             → Pre-training revolution
2020: Linear Attention      → Performer, Linformer attempts
2021: Flash Attention       → IO-aware exact attention
2022: S4, RWKV              → State Space Models emerge
2023: Mamba                 → Selective SSMs breakthrough
2024: Mamba-2, RWKV-6       → Hybrids, improved quality
```

## 🎯 Learning Path

```
1. History & Fundamentals (01_) → Why attention was invented
2. Transformer Attention (02_)  → Multi-head, position encodings
3. Efficient Attention (03_)    → Flash, linear, sparse
4. Advanced Mechanisms (04_)    → RWKV, Mamba, SSMs
5. Pitfalls & Debugging (05_)   → Common issues, solutions
6. Notebook                     → Hands-on practice
```

## 📚 Topics Covered

### Attention Fundamentals
- **Scaled Dot-Product Attention**: Q, K, V
- **Multi-Head Attention**: Parallel heads
- **Self-Attention**: Sequence modeling
- **Cross-Attention**: Encoder-decoder
- **Complexity**: O(n²) memory and compute

### Memory-Efficient Attention
- **Problem**: Quadratic memory growth
- **Chunked Attention**: Block-wise computation
- **Gradient Checkpointing**: Recomputation
- **Sparse Attention**: Reducing computation

### Flash Attention
- **Key Insight**: Online softmax, tiling
- **IO Complexity**: Memory-bound → compute-bound
- **Algorithm**: Block-wise, recomputation
- **Flash Attention 2**: Further optimizations
- **Flash Attention 3**: Hopper optimizations

### Implementation Details
- **Tiling Strategy**: Block sizes
- **Online Softmax**: Running statistics
- **Backward Pass**: Recomputation strategy
- **Fused Kernels**: QKV projection fusion
- **Causal Masking**: Efficient implementation

### Advanced Variants
- **Flash Decoding**: Inference optimization
- **Paged Attention**: vLLM's approach
- **Ring Attention**: Sequence parallelism
- **GQA/MQA**: Grouped/Multi-query attention
- **Sliding Window**: Local attention

### Flex Attention (PyTorch 2.5+)
- **Block Mask**: Custom attention patterns
- **Score Mod**: Modifying attention scores
- **Compilation**: torch.compile integration

## 🎯 Learning Objectives

- [ ] Understand Flash Attention algorithm
- [ ] Implement attention from scratch
- [ ] Use Flex Attention API
- [ ] Profile attention memory

## 💻 Practical Exercises

1. Implement vanilla attention
2. Study Flash Attention kernel
3. Use Flex Attention for custom patterns
4. Benchmark attention variants

## 📖 Resources

### Key Papers
- Bahdanau 2014: "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani 2017: "Attention Is All You Need"
- Dao 2022: "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Gu 2023: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

### Code References
- `flash-attention/` - Flash Attention implementation
- `pytorch/` - PyTorch SDPA internals

## 🔧 Running the Code

```bash
# Run Python files
python 01_attention_history_fundamentals.py
python 02_transformer_attention.py
python 03_efficient_attention.py
python 04_advanced_attention.py
python 05_attention_pitfalls.py

# Run notebook
jupyter notebook attention_mechanisms_notebook.ipynb
```

## 📊 Architecture Comparison

| Architecture | Training | Inference | Long Context | Quality |
|--------------|----------|-----------|--------------|---------|
| Transformer | O(N²) | O(N)+KV | Limited | Excellent |
| + Flash Attn | O(N²) | O(N)+KV | Better | Excellent |
| + Sliding Win | O(Nw) | O(w)+KV | Good | Very Good |
| RWKV | O(N) | O(1) | Excellent | Good |
| Mamba | O(N) | O(1) | Excellent | Very Good |
| Hybrid | O(N)~ | Mixed | Excellent | Excellent |

## ⏱️ Estimated Time

- Quick overview: 2-3 days
- Deep understanding: 2-3 weeks
- Implementation practice: 4-5 weeks
