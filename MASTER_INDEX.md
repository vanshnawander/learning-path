# Master Index: Deep Foundations for ML Systems

Complete learning materials for understanding ML systems from first principles.

## 📚 Module Overview

| # | Module | Focus | Files |
|---|--------|-------|-------|
| 00 | Crucial Concepts | Often-ignored fundamentals | 3+ |
| 01 | Computer Architecture | CPU, cache, SIMD | 15+ |
| 02 | Operating Systems | mmap, processes, I/O | 15+ |
| 03 | Assembly Programming | x86-64, AVX, optimization | 13+ |
| 04 | C Programming | Pointers, memory, patterns | 17+ |
| 05 | C++ Programming | RAII, smart pointers, move | 4+ |
| 06 | Hardware Fundamentals | PCIe, latency, bandwidth | 5+ |
| 07 | Multimodal Data Formats | Text, image, audio, video | 8+ |
| 08 | Device I/O | Data acquisition, DMA | 2+ |
| 10 | GPU Architecture | Memory hierarchy, Tensor Cores | 5+ |
| 11 | CUDA Programming | Kernels, optimization | 6+ |
| 12 | Triton Programming | **Unsloth kernels, quantization** | 10+ |
| 13 | ML Compilers | TorchDynamo, Inductor | 6+ |
| 14 | PyTorch Internals | Dispatcher, autograd | 4+ |
| 15 | Attention Mechanisms | Flash Attention, efficient attention | 7+ |
| 16 | Training Optimization | **LoRA, quantization, fusion** | 10+ |

## 🎯 Learning Paths

### Path 1: Systems Foundations (4 weeks)
```
Week 1: 01-computer-architecture (binary, cache, SIMD)
Week 2: 02-operating-systems (mmap, processes)
Week 3: 04-c-programming (pointers, memory)
Week 4: 06-hardware-fundamentals (latency, PCIe)
```

### Path 2: Low-Level Optimization (4 weeks)
```
Week 1: 03-assembly-programming/01-x86-64-basics
Week 2: 03-assembly-programming/02-simd-avx
Week 3: 03-assembly-programming/03-optimization-patterns
Week 4: 00-crucial-concepts (bandwidth, precision)
```

### Path 3: Multimodal Pipeline (3 weeks)
```
Week 1: 07-multimodal-data-formats/01-text-encoding
        07-multimodal-data-formats/02-image-formats
Week 2: 07-multimodal-data-formats/03-audio-formats
        07-multimodal-data-formats/04-video-formats
Week 3: 08-device-io
        07-multimodal-data-formats/05-ml-data-formats
```

### Path 4: C++/Modern Systems (2 weeks)
```
Week 1: 05-cpp-programming/01-memory-management
Week 2: Advanced patterns and integration
```

### Path 5: GPU & Triton Mastery (6 weeks) ★ NEW
```
Week 1: 10-gpu-architecture (memory hierarchy, Tensor Cores)
Week 2: 11-cuda-programming (basics, optimization)
Week 3: 12-triton-programming/basics + puzzles
Week 4: 12-triton-programming/patterns (softmax, matmul)
Week 5: 12-triton-programming/advanced (Flash Attention, Unsloth kernels)
Week 6: 12-triton-programming/advanced (quantization kernels)
```

### Path 6: LLM Training Optimization (4 weeks) ★ NEW
```
Week 1: 16-training-optimization/mixed-precision + memory
Week 2: 16-training-optimization/fine-tuning (LoRA, QLoRA)
Week 3: 16-training-optimization/fusion + quantization
Week 4: 16-training-optimization/compilation (torch.compile)
```

## 📁 Complete File Listing

### 00-crucial-concepts/ (PROFILING FOCUSED)
```
├── README.md                           # Top 10 ignored concepts
├── 01_memory_bandwidth_bottleneck.md   # #1 performance issue
├── 02_floating_point_precision.md      # Numerical stability
├── 03_profiling_fundamentals.md        # How to profile (ESSENTIAL)
├── 04_data_movement_costs.c            # Data movement benchmarks
├── 05_profiling_multimodal_pipeline.py # Profile image/audio/video
├── 06_cpu_gpu_transfer_costs.py        # CPU↔GPU transfer profiling
└── 07_end_to_end_pipeline_profile.py   # Full training loop profiler
```

### 01-computer-architecture/
```
├── 01-binary-and-bits/
├── 02-memory-hierarchy/
├── 03-simd-vectorization/
├── 04-memory-alignment/
├── 05-cpu-pipeline/
├── 06-data-layout/
├── 07-benchmarking/
└── exercises/
```

### 02-operating-systems/
```
├── 01-memory-mapping/
├── 02-processes-threads/
├── 03-file-io/
├── 04-virtual-memory/
├── 05-system-calls/
├── 06-memory-allocators/
├── 07-synchronization/
├── 08-shared-memory-ipc/
└── exercises/
```

### 03-assembly-programming/
```
├── 01-x86-64-basics/
│   ├── README.md
│   ├── 01_hello_world.s
│   ├── 02_registers.s
│   └── 02_registers_main.c
├── 02-simd-avx/
│   ├── README.md
│   ├── 01_avx_basics.c
│   ├── 02_avx_dotproduct.s
│   └── 02_dotprod_main.c
├── 03-optimization-patterns/
│   ├── README.md
│   ├── 01_quantized_dot.c
│   └── 02_prefetch_patterns.c
├── 04-reading-compiler-output/
│   ├── README.md
│   └── 01_simple_functions.c
└── LEARNING_ORDER.md
```

### 04-c-programming/
```
├── 01-pointers-deep-dive/
│   ├── README.md
│   ├── 01_pointer_basics.c
│   ├── 02_pointer_arithmetic.c
│   ├── 03_void_pointers.c
│   └── 04_function_pointers.c
├── 02-memory-management/
│   ├── README.md
│   ├── 01_stack_vs_heap.c
│   └── 02_custom_allocator.c
├── 03-mmap-advanced/
│   ├── README.md
│   ├── 01_mmap_file_io.c
│   └── 02_shared_tensor.c
├── 04-struct-patterns/
│   ├── README.md
│   └── 01_data_oriented.c
├── 05-io-patterns/
│   ├── README.md
│   └── 01_buffered_io.c
└── 06-ffcv-patterns/
    └── 01_ffcv_analysis.md
```

### 05-cpp-programming/
```
├── 01-memory-management/
│   ├── README.md
│   ├── 01_raii.cpp
│   ├── 02_smart_pointers.cpp
│   └── 03_move_semantics.cpp
└── README.md
```

### 06-hardware-fundamentals/
```
├── README.md
├── 01-system-architecture/
│   ├── README.md
│   ├── 01_latency_numbers.c
│   └── 02_pcie_deep_dive.md
├── 02-memory-hierarchy-deep/
│   ├── README.md
│   ├── 01_cache_line_effects.c
│   └── 02_bandwidth_profiled.c      # Memory bandwidth profiling
```

### 07-multimodal-data-formats/ (ALL PROFILED)
```
├── README.md
├── 01-text-encoding/
│   └── 01_unicode_utf8.c
├── 02-image-formats/
│   ├── 01_image_fundamentals.md
│   └── 02_image_decode_profiled.c   # Image pipeline with timing
├── 03-audio-formats/
│   ├── 01_audio_fundamentals.md
│   └── 02_audio_processing_profiled.c # Audio pipeline with timing
├── 04-video-formats/
│   ├── 01_video_fundamentals.md
│   ├── 02_color_spaces.c
│   └── 03_video_decode_profiled.c   # Video pipeline with timing
└── 05-ml-data-formats/
    ├── 01_tensor_storage.md
    └── 02_multimodal_batch_profiled.py # Batch creation profiling
```

### 08-device-io/
```
└── README.md                        # DMA, ring buffers, interfaces
```

### 09-data-loading-pipelines/
```
├── 01_dataloader_profiling.py       # PyTorch DataLoader profiling
└── 02_ffcv_webdataset_comparison.md # FFCV vs WebDataset analysis
```

### 12-triton-programming/ ★ EXTENSIVELY UPDATED
```
├── README.md                        # Overview with Unsloth coverage
├── basics/
│   ├── README.md
│   └── 01_triton_fundamentals.py    # Core concepts, profiling
├── patterns/
│   ├── README.md
│   ├── 01_softmax_kernel.py         # Online softmax algorithm
│   └── 02_matmul_kernel.py          # Tiled matmul, auto-tuning
├── advanced/
│   ├── README.md
│   ├── 01_flash_attention.py        # Flash Attention deep dive
│   ├── 02_unsloth_kernels.py        # ★ NEW: Production kernels
│   │   ├── Fused RMSNorm + Residual
│   │   ├── Fused Cross-Entropy (chunked)
│   │   ├── Fused RoPE
│   │   ├── Fused SwiGLU
│   │   └── Fused LoRA forward
│   ├── 03_quantization_kernels.py   # ★ NEW: Quantization
│   │   ├── INT8 quantize/dequantize
│   │   ├── INT8 matmul with dequant
│   │   ├── NF4 (QLoRA) concepts
│   │   ├── FP8 (Hopper)
│   │   └── Dynamic quantization
│   └── flash-attention/
│       └── README.md
├── puzzles/                         # ★ NEW: Practice problems
│   └── 01_triton_puzzles.py         # 9 exercises with solutions
└── triton_programming_notebook.ipynb
```

### 16-training-optimization/ (COMPREHENSIVE)
```
├── README.md
├── mixed-precision/
│   ├── 01_floating_point_formats.py # FP32, FP16, BF16, FP8
│   └── 02_automatic_mixed_precision.py
├── memory/
│   ├── 01_gradient_checkpointing.py
│   └── 02_gradient_accumulation_8bit_optimizers.py
├── fine-tuning/
│   └── lora/
│       ├── README.md
│       └── 01_lora_deep_dive.py     # LoRA, QLoRA, DoRA
├── quantization/
│   └── 01_quantization_fundamentals.py
├── fusion/
│   └── 01_operator_fusion.py        # Unsloth-style fusion
└── compilation/
    └── 01_torch_compile.py          # TorchDynamo, Inductor
```

## 🔑 Key Concepts by Topic

### Memory & Performance
- Cache lines (64 bytes)
- Memory bandwidth bottleneck
- Data alignment
- False sharing
- NUMA effects

### Data Representation
- UTF-8 encoding
- IEEE 754 floating point
- YUV color spaces
- PCM audio
- Video codecs (H.264, H.265)

### System Interfaces
- mmap for zero-copy
- PCIe for GPU transfer
- DMA for async I/O
- Pinned memory

### Optimization Techniques
- SIMD/AVX vectorization
- Cache blocking/tiling
- Prefetching
- Kernel fusion
- Quantization (INT8/INT4)

### Triton & GPU Programming ★ NEW
- Block-based programming model
- Auto-tuning configurations
- Memory coalescing patterns
- Fused kernels (RMSNorm, CrossEntropy, RoPE)
- Online softmax algorithm
- Flash Attention tiling

### Unsloth Optimizations ★ NEW
- Fused RMSNorm + Residual (2-3x speedup)
- Chunked Cross-Entropy (10x memory reduction)
- Fused SwiGLU MLP
- Fused LoRA forward/backward
- NF4/INT8 quantization kernels

### LLM Inference Optimization ★ NEW
- KV Cache fundamentals and memory analysis
- PagedAttention (vLLM) virtual memory
- Continuous Batching / In-flight batching
- Speculative Decoding (draft-verify)
- Flash Decoding for long contexts
- GGUF/GPTQ/AWQ quantization formats

### Flash Attention 3 (Hopper) ★ NEW
- WGMMA and TMA hardware features
- Pingpong warpgroup scheduling
- Intra-warpgroup GEMM-softmax overlap
- FP8 with incoherent processing
- 740 TFLOPS (75% H100 peak)

## 📖 Reference Resources

### Books
- "Computer Systems: A Programmer's Perspective" (CS:APP)
- "What Every Programmer Should Know About Memory" (Drepper)
- "The C Programming Language" (K&R)

### Online
- Godbolt Compiler Explorer: https://godbolt.org
- Intel Intrinsics Guide
- NVIDIA CUDA Documentation
- PyTorch Internals

## ✅ Completion Checklist

- [ ] Module 00: Crucial Concepts
- [ ] Module 01: Computer Architecture
- [ ] Module 02: Operating Systems
- [ ] Module 03: Assembly Programming
- [ ] Module 04: C Programming
- [ ] Module 05: C++ Programming
- [ ] Module 06: Hardware Fundamentals
- [ ] Module 07: Multimodal Data Formats
- [ ] Module 08: Device I/O
