# 11 - GPU Architecture

Understanding NVIDIA GPU internals before CUDA programming.

## 📚 Topics Covered

### GPU vs CPU
- **Throughput vs Latency**: Design philosophies
- **SIMT**: Single Instruction Multiple Threads
- **Massive Parallelism**: Thousands of cores
- **Memory Bandwidth**: 10x+ more than CPU

### NVIDIA Architecture Evolution
- **Fermi**: First modern architecture
- **Kepler/Maxwell**: Power efficiency
- **Pascal**: Unified memory, NVLink
- **Volta/Turing**: Tensor Cores, mixed precision
- **Ampere**: 3rd gen Tensor Cores, sparsity
- **Hopper**: Transformer Engine, TMA
- **Blackwell**: Latest generation

### Streaming Multiprocessor (SM)
- **CUDA Cores**: FP32, INT32 units
- **Tensor Cores**: Matrix multiply accelerators
- **Warp Schedulers**: Instruction dispatch
- **Register File**: Per-SM registers
- **Shared Memory**: Programmer-managed cache
- **L1 Cache**: Combined with shared memory

### Memory Hierarchy
- **Registers**: Fastest, limited per thread
- **Shared Memory**: Per-block, explicit
- **L1/L2 Cache**: Automatic caching
- **Global Memory**: HBM, high bandwidth
- **Constant Memory**: Read-only, cached
- **Texture Memory**: Spatial locality

### Execution Model
- **Threads, Warps, Blocks**: Hierarchy
- **Warp Execution**: 32 threads SIMT
- **Occupancy**: Active warps per SM
- **Latency Hiding**: Many warps in flight

## 🎯 Learning Objectives

- [ ] Understand SM architecture
- [ ] Calculate theoretical performance
- [ ] Know memory hierarchy trade-offs
- [ ] Analyze occupancy

## 💻 Practical Exercises

1. Calculate roofline model limits
2. Analyze occupancy calculator
3. Compare GPU generations
4. Study architecture diagrams

## 📖 Resources

### Whitepapers
- NVIDIA GPU Architecture whitepapers
- "Dissecting the NVIDIA Volta/Turing Architecture"

### Online
- NVIDIA Developer Blog
- GPU Mode lectures

## 📁 Structure

```
11-gpu-architecture/
├── fundamentals/
│   ├── gpu-vs-cpu/
│   ├── simt/
│   └── history/
├── nvidia-architectures/
│   ├── ampere/
│   ├── hopper/
│   └── blackwell/
├── sm-internals/
│   ├── cuda-cores/
│   ├── tensor-cores/
│   └── warp-schedulers/
├── memory-hierarchy/
│   ├── registers/
│   ├── shared-memory/
│   ├── global-memory/
│   └── caches/
└── execution-model/
    ├── warps/
    ├── occupancy/
    └── latency-hiding/
```

## ⏱️ Estimated Time: 2-3 weeks
