# 12 - CUDA Programming

The foundation of GPU programming for NVIDIA GPUs.

## 📚 Topics Covered

### CUDA Basics
- **Kernels**: __global__, __device__, __host__
- **Thread Hierarchy**: threadIdx, blockIdx, blockDim
- **Memory Allocation**: cudaMalloc, cudaMemcpy
- **Error Handling**: cudaGetLastError
- **Synchronization**: cudaDeviceSynchronize

### Thread Organization
- **Grid and Block Dimensions**: 1D, 2D, 3D
- **Warp Size**: 32 threads
- **Cooperative Groups**: Flexible synchronization
- **Thread Divergence**: Performance impact

### Memory Management
- **Global Memory**: Coalesced access patterns
- **Shared Memory**: Bank conflicts
- **Constant Memory**: Broadcast reads
- **Texture Memory**: 2D locality
- **Unified Memory**: Automatic migration
- **Pinned Memory**: DMA transfers

### Optimization Techniques
- **Occupancy**: Maximizing active warps
- **Memory Coalescing**: Aligned, strided access
- **Bank Conflicts**: Shared memory access
- **Instruction-Level Parallelism**: ILP
- **Loop Unrolling**: Reducing overhead
- **Warp-Level Primitives**: Shuffle, vote

### Advanced CUDA
- **Streams**: Concurrent execution
- **Events**: Timing, synchronization
- **Dynamic Parallelism**: Kernels launching kernels
- **Multi-GPU**: Peer-to-peer, NVLink
- **PTX Assembly**: Low-level optimization

### cuBLAS, cuDNN
- **cuBLAS**: BLAS on GPU
- **cuDNN**: Deep learning primitives
- **CUTLASS**: Template library for GEMM
- **cuSPARSE**: Sparse operations

## 🎯 Learning Objectives

- [ ] Write correct CUDA kernels
- [ ] Optimize memory access patterns
- [ ] Use shared memory effectively
- [ ] Profile with Nsight

## 💻 Practical Exercises

1. Implement vector addition
2. Write optimized matrix multiply
3. Implement parallel reduction
4. Profile and optimize a kernel

## 📖 Resources

### Books
- "Programming Massively Parallel Processors" - Kirk & Hwu
- CUDA C Programming Guide (NVIDIA)

### Online
- NVIDIA CUDA samples
- GPU Mode lectures

## 📁 Structure

```
12-cuda-programming/
├── basics/
│   ├── hello-world/
│   ├── thread-hierarchy/
│   └── memory-allocation/
├── memory/
│   ├── global-memory/
│   ├── shared-memory/
│   ├── unified-memory/
│   └── pinned-memory/
├── optimization/
│   ├── coalescing/
│   ├── occupancy/
│   ├── bank-conflicts/
│   └── warp-primitives/
├── advanced/
│   ├── streams/
│   ├── multi-gpu/
│   └── ptx/
└── libraries/
    ├── cublas/
    ├── cudnn/
    └── cutlass/
```

## ⏱️ Estimated Time: 6-8 weeks
