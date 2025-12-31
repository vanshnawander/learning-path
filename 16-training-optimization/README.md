# 16 - Training Optimization

Comprehensive techniques for efficient model training, from mixed precision
to quantization to compilation. Based on real-world implementations from
Unsloth, HuggingFace, and PyTorch.

## 📚 Modules Created

### Python Files

| File | Description |
|------|-------------|
| `mixed-precision/01_floating_point_formats.py` | FP32, FP16, BF16, FP8 deep dive |
| `mixed-precision/02_automatic_mixed_precision.py` | AMP, GradScaler, autocast |
| `memory/01_gradient_checkpointing.py` | Activation recomputation theory & practice |
| `memory/02_gradient_accumulation_8bit_optimizers.py` | Effective batch size, bitsandbytes |
| `fine-tuning/lora/01_lora_deep_dive.py` | LoRA, QLoRA, DoRA mathematics & implementation |
| `quantization/01_quantization_fundamentals.py` | PTQ, GPTQ, AWQ, inference formats |
| `fusion/01_operator_fusion.py` | RMSNorm, fused attention, fused MLP, Triton |
| `compilation/01_torch_compile.py` | TorchDynamo, Inductor, graph breaks |

## 🔬 Topics Covered In Depth

### Mixed Precision Training
- **FP32/FP16/BF16/FP8**: Bit representations, ranges, precision trade-offs
- **Loss Scaling**: Dynamic scaling, GradScaler internals
- **AMP**: autocast operation categories, common pitfalls
- **Hardware Support**: Tensor Cores, Ampere vs Hopper

### Memory Optimization
- **Gradient Checkpointing**: O(√N) memory algorithm, selective strategies
- **Gradient Accumulation**: Effective batch size, LR scaling rules
- **8-bit Optimizers**: bitsandbytes Adam8bit, dynamic quantization
- **Paged Optimizers**: CPU offloading for peak memory

### Efficient Fine-Tuning (LoRA/QLoRA)
- **Low-Rank Decomposition**: Mathematical foundations, rank selection
- **LoRA Implementation**: Forward/backward pass, scaling factor α/r
- **QLoRA**: 4-bit NF4 quantization, double quantization
- **Advanced Variants**: DoRA, LoRA+, rsLoRA, AdaLoRA
- **Hyperparameters**: Rank, alpha, target modules, learning rates

### Quantization
- **Quantization Theory**: Affine vs symmetric, per-tensor vs per-channel
- **GPTQ**: Optimal Brain Quantization, Hessian-based updates
- **AWQ**: Activation-aware scaling, salient weight protection
- **Inference Formats**: GGUF, GGML, ExLlama, vLLM integration

### Operator Fusion
- **Memory Bandwidth**: Why fusion matters, arithmetic intensity
- **RMSNorm**: Simpler than LayerNorm, Triton implementation
- **Fused Attention**: QKV projection, Flash Attention integration
- **Fused MLP**: SwiGLU/GeGLU gate+up fusion
- **Fused Cross-Entropy**: Chunked computation for large vocabularies

### Compilation (torch.compile)
- **PyTorch 2.0 Stack**: TorchDynamo → AOTAutograd → TorchInductor
- **Compilation Modes**: default, reduce-overhead, max-autotune
- **Graph Breaks**: Causes, debugging, solutions
- **Inductor Optimizations**: Fusion, memory planning, Triton codegen

## 🎯 Learning Objectives

- [x] Understand floating point formats and their trade-offs
- [x] Implement mixed precision training with AMP
- [x] Apply gradient checkpointing strategically
- [x] Use 8-bit optimizers for memory efficiency
- [x] Master LoRA/QLoRA fine-tuning
- [x] Understand quantization algorithms (GPTQ, AWQ)
- [x] Apply operator fusion principles
- [x] Use torch.compile effectively

## 💻 Practical Exercises

1. Compare FP32/FP16/BF16 precision and speed
2. Implement gradient accumulation training loop
3. Fine-tune LLM with QLoRA on consumer GPU
4. Quantize model with GPTQ/AWQ
5. Profile fused vs unfused operations
6. Debug torch.compile graph breaks

## 📖 Key Papers

- "Mixed Precision Training" (Micikevicius et al., 2017)
- "LoRA: Low-Rank Adaptation" (Hu et al., 2021)
- "QLoRA: Efficient Finetuning" (Dettmers et al., 2023)
- "GPTQ: Post-Training Quantization" (Frantar et al., 2022)
- "AWQ: Activation-aware Weight Quantization" (Lin et al., 2023)
- "8-bit Optimizers via Block-wise Quantization" (Dettmers et al., 2021)

## 🔧 Code References

- `unsloth/unsloth/kernels/` - Production fused kernels (RMSNorm, LoRA, CrossEntropy)
- `unsloth/unsloth/models/` - Optimized model implementations
- `bitsandbytes/` - 8-bit optimizers and quantization
- `peft/` - Parameter-efficient fine-tuning library

## 📁 Structure

```
16-training-optimization/
├── README.md
├── mixed-precision/
│   ├── 01_floating_point_formats.py      # FP32, FP16, BF16, FP8
│   └── 02_automatic_mixed_precision.py   # AMP, GradScaler
├── memory/
│   ├── 01_gradient_checkpointing.py      # Activation recomputation
│   └── 02_gradient_accumulation_8bit_optimizers.py
├── fine-tuning/
│   └── lora/
│       └── 01_lora_deep_dive.py          # LoRA, QLoRA, DoRA
├── quantization/
│   └── 01_quantization_fundamentals.py   # PTQ, GPTQ, AWQ
├── fusion/
│   └── 01_operator_fusion.py             # RMSNorm, fused ops
└── compilation/
    └── 01_torch_compile.py               # TorchDynamo, Inductor
```

## 🔄 Recommended Learning Path

```
1. Floating Point Formats     → Understand the basics
2. Mixed Precision (AMP)      → Apply to training
3. Gradient Checkpointing     → Reduce activation memory
4. Gradient Accumulation      → Scale batch size
5. 8-bit Optimizers          → Reduce optimizer memory
6. LoRA/QLoRA                → Efficient fine-tuning
7. Quantization              → Inference optimization
8. Operator Fusion           → Understanding Unsloth internals
9. torch.compile             → Automatic optimization
```

## ⏱️ Estimated Time

- Quick overview: 1-2 weeks
- Deep understanding: 4-5 weeks
- Hands-on mastery: 6-8 weeks
