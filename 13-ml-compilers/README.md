# 14 - ML Compilers - Comprehensive Guide

This directory provides an in-depth exploration of ML compilers - the essential technology that makes deep learning fast.

## 📚 Modules

### Python Files

| File | Description |
|------|-------------|
| `01_ml_compilers_overview.py` | Why compilers matter, compilation pipeline, comparison table |
| `02_pytorch_compiler_stack.py` | TorchDynamo, AOTAutograd, Inductor, Triton integration |
| `03_xla_jax_compiler.py` | XLA architecture, JAX fundamentals, XLA optimizations |
| `04_nvidia_compiler_ecosystem.py` | TensorRT, TensorRT-LLM, cuDNN, cuBLAS, CUTLASS |
| `05_tvm_mlir_compilers.py` | TVM, MLIR, IREE, ONNX Runtime, cross-platform compilers |
| `06_mojo_emerging_compilers.py` | Mojo language, MAX platform, emerging tools |

### Interactive Content

| File | Description |
|------|-------------|
| `ml_compilers_notebook.ipynb` | Hands-on experiments with torch.compile and profiling |
| `cuda_comparison/01_cuda_vs_compiled.cu` | Plain CUDA C++ for comparison with compiled code |

## 🎯 Compiler Comparison

| Compiler | Best For | Pros | Cons |
|----------|----------|------|------|
| **torch.compile** | PyTorch training | Easy, flexible | NVIDIA-focused |
| **XLA/JAX** | TPU, functional | Excellent optimization | Steep learning curve |
| **TensorRT** | NVIDIA inference | Best perf on NVIDIA | Inference only |
| **TVM** | Edge/multi-hardware | Portable | Tuning time |
| **Mojo** | Future projects | Fast + easy | Very new |

## 🔧 Running the Code

```bash
# Python files
python 01_ml_compilers_overview.py

# CUDA comparison (requires nvcc)
cd cuda_comparison
nvcc -O3 -o cuda_comparison 01_cuda_vs_compiled.cu
./cuda_comparison

# Notebook
jupyter notebook ml_compilers_notebook.ipynb
```

## 📖 Learning Path

1. Start with `01_ml_compilers_overview.py` for the big picture
2. Run `ml_compilers_notebook.ipynb` for hands-on experiments
3. Deep dive into your compiler of interest (02-06)
4. Study `cuda_comparison/` to understand what compilers optimize

## ⏱️ Expected Time

- Overview + Notebook: 2-3 hours
- Full deep dive: 2-3 days

Automatic optimization of ML models for hardware.

## 📚 Topics Covered

### Compiler Fundamentals
- **IR (Intermediate Representation)**: Graph representation
- **Optimization Passes**: Transformations
- **Code Generation**: Target-specific code
- **JIT vs AOT**: Compilation strategies

### Apache TVM
- **Relay**: High-level IR
- **TIR**: Low-level tensor IR
- **AutoTVM**: Auto-tuning
- **AutoScheduler (Ansor)**: Search-based tuning
- **Target Backends**: CUDA, LLVM, etc.

### XLA (Accelerated Linear Algebra)
- **HLO**: High-level operations
- **Fusion**: Kernel fusion
- **JAX Integration**: Functional ML
- **TPU Backend**: Google TPUs

### MLIR (Multi-Level IR)
- **Dialects**: Composable IRs
- **Transformations**: Passes
- **Torch-MLIR**: PyTorch to MLIR
- **IREE**: Runtime execution

### PyTorch Compilation
- **TorchScript**: Scripting and tracing
- **torch.compile**: PyTorch 2.0
- **TorchDynamo**: Graph capture
- **TorchInductor**: Code generation
- **Triton Backend**: Triton code gen

### Graph Optimizations
- **Operator Fusion**: Reducing memory traffic
- **Memory Planning**: Buffer reuse
- **Layout Optimization**: NCHW vs NHWC
- **Quantization**: Precision reduction
- **Pruning**: Sparsity

## 🎯 Learning Objectives

- [ ] Understand compiler pipeline
- [ ] Use TVM for model optimization
- [ ] Use torch.compile effectively
- [ ] Analyze generated code

## 💻 Practical Exercises

1. Compile model with TVM
2. Analyze torch.compile output
3. Write custom TVM schedule
4. Compare compilation backends

## 📖 Resources

### Papers
- "TVM: An Automated End-to-End Optimizing Compiler"
- "MLIR: A Compiler Infrastructure for the End of Moore's Law"

### Online
- TVM tutorials: tvm.apache.org
- PyTorch 2.0 documentation

### Code References
- `mirage/` - ML compiler research
- `pytorch/torch/_inductor/` - Inductor

## 📁 Structure

```
14-ml-compilers/
├── fundamentals/
│   ├── ir-basics/
│   ├── optimization-passes/
│   └── code-generation/
├── tvm/
│   ├── relay/
│   ├── tir/
│   ├── auto-tuning/
│   └── backends/
├── xla-jax/
│   ├── hlo/
│   ├── jax-basics/
│   └── tpu/
├── mlir/
│   ├── dialects/
│   ├── torch-mlir/
│   └── iree/
├── pytorch-compile/
│   ├── dynamo/
│   ├── inductor/
│   └── triton-backend/
└── optimizations/
    ├── fusion/
    ├── memory-planning/
    └── quantization/
```

## ⏱️ Estimated Time: 5-6 weeks
