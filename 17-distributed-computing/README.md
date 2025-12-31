# 17 - Distributed Computing

Comprehensive coverage of distributed deep learning: from GPU interconnects
to parallelism strategies to data loading optimization. Based on real-world
practices from NVIDIA, PyTorch, and the FFCV paper.

## 📚 Modules Created

| Directory | File | Description |
|-----------|------|-------------|
| `fundamentals/` | `01_gpu_interconnects.py` | NVLink, NVSwitch, PCIe, InfiniBand deep dive |
| `parallelism/` | `01_data_parallelism_ddp.py` | DDP architecture, all-reduce, debugging |
| `parallelism/` | `02_fsdp_zero.py` | ZeRO stages, FSDP configuration, hybrid sharding |
| `parallelism/` | `03_model_tensor_pipeline_parallel.py` | TP, PP, 3D parallelism, sequence parallel |
| `communication/` | `01_nccl_collectives.py` | Collective ops, ring/tree algorithms, tuning |
| `profiling/` | `01_distributed_profiling.py` | Bottleneck analysis, NCCL debug, Nsight |
| `data-loading/` | `01_ffcv_distributed_optimization.py` | FFCV paper concepts, .beton format |

## 🔬 Topics Covered In Depth

### GPU Interconnects (NVLink, NVSwitch, InfiniBand)
- **PCIe Limitations**: Bandwidth bottleneck, shared bus contention
- **NVLink Generations**: 1.0 (160 GB/s) → 5.0 (1.8 TB/s)
- **NVSwitch**: Full crossbar connectivity, uniform 900 GB/s
- **InfiniBand vs Ethernet**: RDMA, latency, bandwidth comparison
- **GPUDirect**: P2P, RDMA, Storage, Async technologies

### Data Parallelism (DDP)
- **DDP Architecture**: Bucket mechanism, gradient hooks
- **All-Reduce Algorithms**: Ring, tree, bandwidth formulas
- **Process Groups**: Backends (NCCL, Gloo), initialization
- **Launch Methods**: torchrun, mp.spawn, multi-node
- **Best Practices**: DistributedSampler, checkpointing, gradient accumulation
- **Debugging**: Common hangs, NCCL debug variables

### FSDP and ZeRO
- **Memory Analysis**: Why 7B model needs 112GB in DDP
- **ZeRO Stages**: Stage 0/1/2/3 memory breakdown
- **FSDP Configuration**: Sharding strategies, mixed precision, CPU offload
- **Wrapping Policies**: Transformer auto-wrap, size-based
- **Hybrid Sharding (HSDP)**: Intra-node FSDP, inter-node DDP
- **FSDP vs DeepSpeed**: Feature comparison, when to use each

### Model Parallelism (TP, PP)
- **Tensor Parallelism**: Column/row parallel linear, attention heads
- **Pipeline Parallelism**: Bubble problem, micro-batching, 1F1B schedule
- **3D Parallelism**: Combining DP + TP + PP for 175B+ models
- **Sequence Parallelism**: Activation memory reduction
- **Parallelism Decision Guide**: When to use which strategy

### Communication (NCCL)
- **Collective Operations**: Broadcast, reduce, all-reduce, all-gather, reduce-scatter
- **Bandwidth Formulas**: Data transferred per collective
- **Algorithm Selection**: Ring vs tree, message size thresholds
- **NCCL Tuning**: Environment variables, debugging, performance

### Profiling and Bottlenecks
- **Amdahl's Law**: Why communication kills scaling
- **Key Metrics**: GPU utilization, communication time, data loading
- **PyTorch Profiler**: Distributed profiling setup
- **Nsight Systems**: Multi-GPU analysis
- **Common Bottlenecks**: Data loading, communication, memory, load imbalance

### Data Loading (FFCV)
- **The Data Bottleneck**: Why processing takes 94% of epoch time
- **.beton File Format**: Header, data table, heap storage, pages
- **Memory Management**: OS cache, process cache, quasi-random
- **JIT Compilation**: Numba augmentation pipeline, threading vs multiprocessing
- **Distributed Loading**: Integration with DDP, shared OS cache

## 🎯 Learning Objectives

- [x] Understand GPU interconnect hierarchy (PCIe → NVLink → NVSwitch)
- [x] Implement DDP training with proper configuration
- [x] Configure FSDP for memory-efficient large model training
- [x] Choose appropriate parallelism strategy for model size
- [x] Debug distributed training issues
- [x] Profile and identify communication bottlenecks
- [x] Optimize data loading with FFCV

## 💻 Practical Exercises

1. Measure NVLink vs PCIe bandwidth
2. Implement DDP training with gradient accumulation
3. Configure FSDP with hybrid sharding
4. Profile all-reduce overhead with Nsight
5. Create FFCV .beton dataset for ImageNet
6. Debug a hanging distributed job

## 📖 Key Papers

- "FFCV: Accelerating Training by Removing Data Bottlenecks" (Leclerc et al., 2023)
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al., 2020)
- "Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al., 2019)
- "GPipe: Efficient Training of Giant Neural Networks" (Huang et al., 2019)

## 📁 Structure

```
17-distributed-computing/
├── README.md
├── fundamentals/
│   └── 01_gpu_interconnects.py     # NVLink, NVSwitch, PCIe, InfiniBand
├── parallelism/
│   ├── 01_data_parallelism_ddp.py  # DDP architecture, all-reduce
│   ├── 02_fsdp_zero.py             # ZeRO stages, FSDP config
│   └── 03_model_tensor_pipeline_parallel.py  # TP, PP, 3D
├── communication/
│   └── 01_nccl_collectives.py      # Collective ops, algorithms
├── profiling/
│   └── 01_distributed_profiling.py # Bottlenecks, debugging
├── data-loading/
│   └── 01_ffcv_distributed_optimization.py  # FFCV paper
└── fsdp/
    └── README.md
```

## 🔄 Recommended Learning Path

```
1. GPU Interconnects        → Understand hardware constraints
2. DDP Fundamentals        → Basic multi-GPU training
3. NCCL Collectives        → Communication primitives
4. FSDP/ZeRO              → Memory-efficient training
5. Tensor/Pipeline Parallel → Large model strategies
6. Profiling              → Find and fix bottlenecks
7. FFCV Data Loading      → Remove data bottlenecks
```

## ⏱️ Estimated Time

- Quick overview: 2-3 weeks
- Deep understanding: 5-6 weeks
- Hands-on mastery: 8-10 weeks
