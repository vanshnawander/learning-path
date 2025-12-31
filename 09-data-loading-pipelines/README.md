# Data Loading Pipelines - Complete Guide

**The most critical module for ML training performance.**

Data loading is often THE bottleneck in training. This module covers every major
data loading solution with profiling and multimodal support.

## 📊 Quick Comparison

| Loader | Speed | GPU Decode | Cloud | Multimodal | Best For |
|--------|-------|------------|-------|------------|----------|
| PyTorch DataLoader | ⭐⭐ | ❌ | ❌ | ✅ | Prototyping |
| FFCV | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ⚠️ | Local SSD training |
| NVIDIA DALI | ⭐⭐⭐⭐⭐ | ✅ | ✅ | ✅ | GPU decode, video |
| WebDataset | ⭐⭐⭐⭐ | ❌ | ✅ | ✅ | Cloud, multimodal |

## 📁 Files in This Module

| File | Description | Language |
|------|-------------|----------|
| `01_dataloader_profiling.py` | Profile PyTorch DataLoader bottlenecks | Python |
| `02_ffcv_webdataset_comparison.md` | Detailed FFCV vs WebDataset analysis | Markdown |
| `03_dataloader_comparison.md` | **Complete comparison of ALL loaders** | Markdown |
| `04_nvidia_dali_complete.py` | Full NVIDIA DALI guide (image/video/audio) | Python |
| `05_cpp_dataloader.cpp` | High-performance C++ data loader | C++ |
| `06_c_mmap_loader.c` | Pure C memory-mapped loader (FFCV-style) | C |
| `07_webdataset_multimodal.py` | WebDataset for video+audio+text | Python |
| `08_beton_format_deep_dive.md` | FFCV .beton file format internals | Markdown |

## 🎯 Learning Path

```
Week 1: Fundamentals
├── 01_dataloader_profiling.py    ← Start here! Find YOUR bottleneck
├── 03_dataloader_comparison.md   ← Understand all options
└── 02_ffcv_webdataset_comparison.md

Week 2: High-Performance Loading
├── 04_nvidia_dali_complete.py    ← GPU decode, video, audio
├── 08_beton_format_deep_dive.md  ← Understand .beton internals
└── 07_webdataset_multimodal.py   ← Cloud + multimodal

Week 3: Low-Level Implementation
├── 05_cpp_dataloader.cpp         ← Build your own loader
└── 06_c_mmap_loader.c            ← Memory mapping fundamentals
```

## 🔥 Key Insights

### Why Data Loading Matters
```
Typical GPU Utilization:
  Naive loading:    ████░░░░░░ 40%   ← GPU starving!
  Optimized:        █████████░ 90%   ← GPU busy
  
The difference: 2-5x training speed
```

### Choosing the Right Loader
```
Local SSD + Images?          → FFCV
Cloud Storage?               → WebDataset
Video Training?              → NVIDIA DALI
Multimodal (V+A+T)?          → WebDataset or DALI
Quick Prototyping?           → PyTorch DataLoader
Maximum Control?             → C++ custom loader
```

## 💻 Compilation Commands

```bash
# C++ DataLoader
g++ -std=c++17 -O3 -pthread -o cpp_loader 05_cpp_dataloader.cpp

# C mmap Loader
gcc -O3 -o mmap_loader 06_c_mmap_loader.c -lpthread -lrt

# Python (install dependencies)
pip install ffcv webdataset nvidia-dali-cuda120 torch torchvision
```

## 📖 External Resources

- **FFCV**: https://docs.ffcv.io/
- **NVIDIA DALI**: https://docs.nvidia.com/deeplearning/dali/
- **WebDataset**: https://github.com/webdataset/webdataset
- **Mosaic StreamingDataset**: https://docs.mosaicml.com/

## ⏱️ Estimated Time: 2-3 weeks
