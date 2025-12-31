# Master Learning Order

A comprehensive guide to navigating all modules in order.

## Phase 1: Foundations (Weeks 1-4)

### Week 1-2: Computer Architecture
**Folder**: `01-computer-architecture/`

| Order | Folder | Key Files |
|-------|--------|-----------|
| 1 | `01-binary-and-bits/` | Binary, floating point, bit ops |
| 2 | `02-memory-hierarchy/` | Cache, blocking, prefetch |
| 3 | `03-simd-vectorization/` | AVX basics |
| 4 | `04-memory-alignment/` | Alignment for CPU/GPU |
| 5 | `05-cpu-pipeline/` | ILP, branches |
| 6 | `06-data-layout/` | AoS vs SoA |
| 7 | `07-benchmarking/` | Measurement |

### Week 3-4: Operating Systems
**Folder**: `02-operating-systems/`

| Order | Folder | Key Files |
|-------|--------|-----------|
| 1 | `01-memory-mapping/` | **mmap** (FFCV foundation!) |
| 2 | `02-processes-threads/` | fork, pthreads |
| 3 | `03-file-io/` | I/O strategies |
| 4 | `04-virtual-memory/` | Pages, faults |
| 5 | `05-system-calls/` | Syscall overhead |
| 6 | `06-memory-allocators/` | malloc, caching |
| 7 | `07-synchronization/` | Atomics |
| 8 | `08-shared-memory-ipc/` | DataLoader IPC |

## Phase 2: Systems Programming (Weeks 5-8)

### Week 5-6: Assembly Programming
**Folder**: `03-assembly-programming/`

| Order | Folder | Focus |
|-------|--------|-------|
| 1 | `01-x86-64-basics/` | Registers, syscalls |
| 2 | `02-simd-avx/` | AVX for ML |
| 3 | `03-optimization-patterns/` | Quantized ops |
| 4 | `04-reading-compiler-output/` | Godbolt |

### Week 7-8: C Programming
**Folder**: `04-c-programming/`

| Order | Folder | Focus |
|-------|--------|-------|
| 1 | `01-pointers-deep-dive/` | Pointer mastery |
| 2 | `02-memory-management/` | Stack, heap, allocators |
| 3 | `03-mmap-advanced/` | mmap I/O, shared memory |
| 4 | `04-struct-patterns/` | Data-oriented design |
| 5 | `05-io-patterns/` | Buffered I/O |
| 6 | `06-ffcv-patterns/` | Real-world analysis |

## Phase 3: GPU Programming (Weeks 9-12)

### Week 9-10: GPU Architecture
**Folder**: `11-gpu-architecture/`

Focus on: SM internals, memory hierarchy, warp execution

### Week 11-12: CUDA Programming
**Folder**: `12-cuda-programming/`

Focus on: Kernels, thread hierarchy, memory management

## Phase 4: ML Systems (Weeks 13-16)

### Week 13: Data Formats
**Folder**: `09-data-formats-serialization/`

Focus on: .beton, WebDataset, Arrow

### Week 14: Data Loading
**Folder**: `10-data-loading-pipelines/`

Focus on: FFCV, PyTorch DataLoader

### Week 15-16: Triton & Flash Attention
**Folders**: `13-triton-programming/`, `16-attention-mechanisms/`

Focus on: Triton kernels, Flash Attention implementation

## Phase 5: Advanced Topics (Weeks 17-20)

- `14-ml-compilers/` - TVM, XLA
- `15-pytorch-internals/` - Autograd, Dispatcher
- `17-training-optimization/` - Mixed precision, LoRA
- `18-distributed-computing/` - FSDP, DDP

## Daily Study Pattern

```
Morning (2 hrs):
  - Read concept material
  - Study code examples
  
Afternoon (2 hrs):
  - Compile and run examples
  - Modify and experiment
  
Evening (1 hr):
  - Review, take notes
  - Connect to ML applications
```

## Progress Tracking

Use checkboxes in each module's README to track:
- [ ] Read theory
- [ ] Run all examples
- [ ] Complete exercises
- [ ] Connect to ML use case
