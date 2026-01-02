# Custom Binary Format Design: From Zero to Production

## Overview

This module teaches you how to design and implement your own optimized binary data format for machine learning - supporting **ANY modality** (audio, video, text, images, multimodal).

FFCV's .beton format is our reference implementation, but we generalize the patterns to work with any data type.

## Why Build Your Own Format?

| Existing Format | Limitation |
|-----------------|------------|
| PyTorch DataLoader | Python GIL, slow augmentations |
| TFRecord | Sequential only, TensorFlow ecosystem |
| WebDataset | No random access, streaming only |
| HDF5 | Lock contention, slower than mmap |
| FFCV .beton | Primarily image-focused |

**Your custom format can:**
- Support your specific modality (audio spectrograms, tokenized text, video frames)
- Optimize for your access patterns (sequential, random, quasi-random)
- Integrate with your preprocessing pipeline
- Match your hardware (NVMe, network storage, GPU direct)

## Module Structure

```
custom-format-design/
├── 01-fundamentals/
│   ├── 01_binary_file_anatomy.md          ✅ Headers, metadata, data regions
│   ├── 02_endianness_and_alignment.md     ✅ Little/big endian, padding
│   └── 03_numpy_structured_arrays.md      ✅ dtype system for formats
│
├── 02-format-specification/
│   └── 01_header_design.md                ✅ Complete format spec, header, metadata, alloc tables
│
├── 03-field-system/
│   └── 01_field_abstraction.md            ✅ Abstract base class pattern, type registry
│
├── 04-encoding-writing/
│   └── 01_page_based_allocation.md        ✅ Page allocator with parallel writing
│
├── 05-decoding-reading/
│   ├── 01_memory_mapped_reading.md        ✅ mmap, page faults, prefetching
│   ├── 02_decoder_architecture.md         ✅ Decoder base class, modality decoders
│   └── 03_asynchronous_loading.md         🏗️ Producer-consumer, double buffering
│
├── 06-os-hardware-concepts/
│   ├── 01_os_hardware_overview.md         ✅ Memory hierarchy, page cache, I/O patterns
│   └── 02_io_optimization_tricks.md       🏗️ madvise, hugepages, quasi-random access
│
├── 07-pipeline-system/
│   ├── 01_jit_compilation.md              ✅ Numba JIT for pipelines
│   ├── 02_transform_operations.md         ✅ Image/audio/text transforms
│   └── 03_pipeline_compilation.md         🏗️ AST-based code generation (The FFCV "Secret Sauce")
│
├── 08-modality-specific/
│   ├── audio/
│   │   └── 01_audio_field_design.md       ✅ 5 audio field types (waveform, compressed, mel, codec tokens)
│   ├── video/
│   │   └── 01_video_field_design.md       ✅ Pre-extracted frames, compressed, optical flow
│   ├── text/
│   │   └── 01_text_field_design.md        ✅ Raw, tokenized, packed, hierarchical
│   └── multimodal/
│       └── 01_multimodal_design.md        ✅ Unified samples, streams, video+audio sync
│
├── 09-projects/
│   ├── 01_complete_implementation.md      ✅ Full working format implementation
│   └── 02_advanced_optimizations.md       🏗️ C++ extensions, custom allocators
│
├── 10-deep-internals/                     (FOR EXPERTS)
│   ├── 01_libffcv_cpp_internals.md        💀 `libffcv.cpp` line-by-line + threads
│   ├── 02_graph_compiler_internals.md     💀 AST metaprogramming & Numba linking
│   └── 03_memory_allocator_internals.md   💀 The OS `mmap` & `malloc` mechanics
│
└── README.md                              ✅ This file
```

**Status Legend:** ✅ = Completed and ready to study

## Learning Path

### Week 1: Fundamentals
1. `01-fundamentals/` - Binary basics
2. `02-format-specification/` - Format design

### Week 2: Field System
3. `03-field-system/` - All field types
4. `04-encoding-writing/` - Writer implementation

### Week 3: Reading & Optimization
5. `05-decoding-reading/` - Reader implementation
6. `06-os-hardware-concepts/` - System optimization

### Week 4: Pipelines & Modalities
7. `07-pipeline-system/` - Transform pipelines
8. `08-modality-specific/` - Per-modality guides

### Week 5: Projects
9. `09-projects/` - Build complete formats

## Prerequisites

| Module | Why Needed |
|--------|------------|
| 01-computer-architecture | Memory hierarchy, alignment |
| 02-operating-systems | mmap, page cache, I/O |
| 04-c-programming | Pointers, structs, memory |
| 23-parallel-computing | Multiprocessing, locks |

## Key FFCV Files Reference

| File | Teaches |
|------|---------|
| `ffcv/types.py` | Format specification with numpy dtypes |
| `ffcv/fields/base.py` | Abstract field pattern |
| `ffcv/fields/*.py` | Concrete field implementations |
| `ffcv/writer.py` | Parallel page-based writing |
| `ffcv/memory_allocator.py` | Page allocation strategy |
| `ffcv/reader.py` | Header parsing, metadata loading |
| `ffcv/memory_managers/*.py` | OS cache vs process cache |
| `ffcv/pipeline/*.py` | Decode pipeline with JIT |
| `ffcv/loader/*.py` | Batch assembly, iteration |
| `libffcv/libffcv.cpp` | C++ optimized operations |
