# 19 - HPC & Performance Engineering

Measuring, analyzing, and optimizing performance.

## 📚 Topics Covered

### Performance Analysis
- **Roofline Model**: Compute vs memory bound
- **Arithmetic Intensity**: FLOPS/byte
- **Bandwidth Utilization**: Memory throughput
- **Latency Analysis**: Critical path

### Profiling Tools
- **NVIDIA Nsight Systems**: Timeline profiling
- **NVIDIA Nsight Compute**: Kernel profiling
- **PyTorch Profiler**: Python-level profiling
- **perf**: Linux performance counters
- **VTune**: Intel profiling tool

### CPU Optimization
- **Vectorization**: SIMD utilization
- **Cache Optimization**: Blocking, prefetch
- **Branch Prediction**: Avoiding mispredictions
- **Multi-threading**: Parallel efficiency

### GPU Optimization
- **Occupancy Analysis**: SM utilization
- **Memory Analysis**: Bandwidth, coalescing
- **Warp Efficiency**: Divergence
- **Instruction Throughput**: Compute utilization

### Benchmarking
- **Micro-benchmarks**: Isolated measurements
- **End-to-End**: Full application
- **Statistical Methods**: Variance, confidence
- **Reproducibility**: Consistent results

### Common Bottlenecks
- **Memory Bandwidth**: Most common in ML
- **Launch Overhead**: Kernel launch cost
- **Synchronization**: Barriers, locks
- **I/O**: Data loading
- **Communication**: Distributed overhead

## 🎯 Learning Objectives

- [ ] Use Nsight tools effectively
- [ ] Apply roofline model
- [ ] Identify bottlenecks
- [ ] Optimize real workloads

## 💻 Practical Exercises

1. Profile a training loop
2. Analyze kernel with Nsight Compute
3. Calculate roofline limits
4. Optimize identified bottleneck

## 📖 Resources

### Documentation
- NVIDIA Nsight documentation
- PyTorch Profiler guide

### Books
- "Introduction to High Performance Computing" - Hager & Wellein

## 📁 Structure

```
19-hpc-performance-engineering/
├── analysis/
│   ├── roofline/
│   ├── arithmetic-intensity/
│   └── bottleneck-identification/
├── profiling/
│   ├── nsight-systems/
│   ├── nsight-compute/
│   ├── pytorch-profiler/
│   └── perf/
├── cpu-optimization/
│   ├── vectorization/
│   ├── cache-optimization/
│   └── threading/
├── gpu-optimization/
│   ├── occupancy/
│   ├── memory-analysis/
│   └── warp-efficiency/
├── benchmarking/
│   ├── methodology/
│   ├── statistics/
│   └── reproducibility/
└── case-studies/
    ├── training-loop/
    ├── inference/
    └── data-loading/
```

## ⏱️ Estimated Time: 4-5 weeks
