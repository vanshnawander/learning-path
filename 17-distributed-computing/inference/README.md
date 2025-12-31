# LLM Inference Optimization

Comprehensive guide to optimizing LLM inference for production deployment.

## 📚 Topics Covered

### 01_llm_inference_optimization.py

| Topic | Description |
|-------|-------------|
| **KV Cache** | Fundamentals, memory analysis, implementation |
| **PagedAttention** | vLLM's virtual memory approach for KV cache |
| **Continuous Batching** | Dynamic request handling, in-flight batching |
| **Speculative Decoding** | Draft-verify paradigm for faster generation |
| **Flash Decoding** | Parallelized decode for long contexts |
| **Quantization** | INT8, INT4, GPTQ, AWQ, GGUF formats |

## 🎯 Key Concepts

### Memory-Bound vs Compute-Bound

```
PREFILL (Compute-bound):
    - Process all input tokens in parallel
    - GPU cores fully utilized
    - Time ∝ batch × seq_len × model_size

DECODE (Memory-bound):
    - Generate one token at a time
    - Limited by memory bandwidth
    - Time ∝ model_size + KV_cache_size
```

### Optimization Techniques Summary

| Technique | Speedup | Memory Savings | Complexity |
|-----------|---------|----------------|------------|
| KV Caching | 10-100x | - | Low |
| PagedAttention | 2-4x throughput | 50-90% | Medium |
| Continuous Batching | 2-5x throughput | - | Medium |
| Speculative Decoding | 2-3x | - | High |
| INT4 Quantization | - | 4x | Low |

## 📖 Resources

### Papers
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)

### Code
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention, continuous batching
- [TGI](https://github.com/huggingface/text-generation-inference) - HuggingFace serving
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF, CPU inference

## 🚀 Quick Start

```bash
# Run the comprehensive module
python 01_llm_inference_optimization.py

# Or explore interactively
python -i 01_llm_inference_optimization.py
>>> analyze_memory_requirements()
>>> demonstrate_paged_attention()
```

## 🔗 Related Modules

- `12-triton-programming/advanced/03_quantization_kernels.py` - Triton quantization
- `16-training-optimization/quantization/` - Training-time quantization
- `15-attention-mechanisms/` - Flash Attention fundamentals
