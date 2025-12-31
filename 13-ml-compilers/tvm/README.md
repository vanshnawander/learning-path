# Apache TVM

Open-source ML compiler for any hardware.

## What is TVM?

Compiles ML models to optimized code for:
- CPUs (x86, ARM)
- GPUs (CUDA, ROCm, Vulkan)
- Accelerators (TPU, custom ASICs)

## Architecture

```
┌─────────────────┐
│ Framework Model │  (PyTorch, TF, ONNX)
└────────┬────────┘
         ▼
┌─────────────────┐
│     Relay       │  (High-level IR)
└────────┬────────┘
         ▼
┌─────────────────┐
│  TIR (Tensor IR)│  (Low-level IR)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Code Gen      │  (LLVM, CUDA, etc.)
└─────────────────┘
```

## Key Concepts

### Relay
High-level functional IR for graph optimization:
- Operator fusion
- Layout transformation
- Constant folding

### TIR (Tensor IR)
Low-level loop-based IR:
- Loop transformations
- Vectorization
- Memory layout

### AutoTVM / AutoScheduler
Automatic optimization:
- Search-based tuning
- Template-guided (AutoTVM)
- Search-based (Ansor/AutoScheduler)

## Basic Usage

```python
import tvm
from tvm import relay

# Import model
mod, params = relay.frontend.from_pytorch(model, shape_dict)

# Compile
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Run
dev = tvm.cuda()
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
runtime.run()
```

## When to Use TVM
- Deploying to edge devices
- Custom hardware support
- Maximum inference performance
- Cross-platform deployment

## Reference
- `mirage/` - Related compiler research
- tvm.apache.org - Official docs
