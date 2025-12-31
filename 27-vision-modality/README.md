# Vision Modality: Complete Learning Path

A comprehensive curriculum for understanding computer vision, from classical image processing to cutting-edge Vision Transformers and multimodal models. Built from foundational concepts to state-of-the-art research (SAM2, DINOv2, GPT-4V).

## Curriculum Philosophy

- **Core Depth Over Abstractions**: Every concept includes profiled implementations
- **Research-Grounded**: Each module references seminal papers and latest research
- **Practical Focus**: Runnable code, benchmarks, and production considerations

---

## Module Overview (55+ Files Planned)

```
27-vision-modality/
├── 01-foundations/                      # Core vision concepts (8 files)
│   ├── 00_cv_history_classical_to_deep.md       # SIFT → CNN → Transformers
│   ├── 01_image_fundamentals.md                 # Pixels, channels, color spaces
│   ├── 02_image_io_profiled.py                  # Image loading benchmarks
│   ├── 03_color_spaces_deep_dive.md             # RGB, HSV, LAB, YUV
│   ├── 04_image_processing.c                    # Pure C image operations
│   ├── 05_convolution_fundamentals.md           # Kernels, filters, edge detection
│   ├── 06_convolution_cuda.cu                   # CUDA convolution kernel
│   └── 07_classical_features.md                 # SIFT, SURF, ORB, HOG
│
├── 02-preprocessing-augmentation/       # Data preprocessing (8 files)
│   ├── 01_image_preprocessing.md                # Normalization, resizing, cropping
│   ├── 02_preprocessing_profiled.py             # Benchmarked implementations
│   ├── 03_data_augmentation_theory.md           # Why augmentation works
│   ├── 04_torchvision_transforms.py             # torchvision augmentations
│   ├── 05_albumentations_guide.py               # Albumentations library
│   ├── 06_augmentation_comparison.py            # Speed/quality benchmarks
│   ├── 07_advanced_augmentations.md             # MixUp, CutMix, RandAugment
│   └── 08_gpu_augmentation.py                   # DALI, Kornia GPU transforms
│
├── 03-convolutional-networks/           # CNN architectures (10 files)
│   ├── 01_cnn_fundamentals.md                   # Convolution, pooling, receptive field
│   ├── 02_lenet_alexnet.md                      # LeNet-5, AlexNet history
│   ├── 03_vgg_architecture.py                   # VGG implementation
│   ├── 04_resnet_deep_dive.md                   # ResNet, skip connections
│   ├── 05_resnet_implementation.py              # ResNet from scratch
│   ├── 06_inception_efficientnet.md             # Inception, EfficientNet
│   ├── 07_mobilenet_shufflenet.md               # Mobile architectures
│   ├── 08_convnext.md                           # ConvNeXt modernized CNN
│   ├── 09_cnn_visualization.ipynb               # Feature map visualization
│   └── 10_cnn_profiled.py                       # Architecture benchmarks
│
├── 04-vision-transformers/              # ViT architectures (10 files)
│   ├── 01_vit_fundamentals.md                   # Patch embedding, position encoding
│   ├── 02_vit_original_paper.md                 # "An Image is Worth 16x16 Words"
│   ├── 03_vit_implementation.py                 # ViT from scratch
│   ├── 04_deit_training.md                      # DeiT data-efficient training
│   ├── 05_swin_transformer.md                   # Shifted window attention
│   ├── 06_swin_implementation.py                # Swin Transformer implementation
│   ├── 07_vit_variants.md                       # BEiT, MAE, EVA, ViT-22B
│   ├── 08_cnn_vs_vit_comparison.md              # When to use which
│   ├── 09_hybrid_architectures.md               # CoAtNet, ViT-CoMer
│   └── 10_vit_profiled.py                       # ViT performance analysis
│
├── 05-self-supervised-learning/         # SSL for vision (6 files)
│   ├── 01_ssl_fundamentals.md                   # Contrastive, generative, masked
│   ├── 02_simclr_moco.md                        # SimCLR, MoCo contrastive learning
│   ├── 03_dino_dinov2.md                        # DINO self-distillation
│   ├── 04_mae_masked_autoencoder.md             # MAE pretraining
│   ├── 05_ssl_implementation.py                 # Contrastive learning from scratch
│   └── 06_ssl_comparison.py                     # SSL method benchmarks
│
├── 06-object-detection/                 # Detection architectures (10 files)
│   ├── 01_detection_fundamentals.md             # IoU, NMS, anchor boxes
│   ├── 02_rcnn_family.md                        # R-CNN → Fast R-CNN → Faster R-CNN
│   ├── 03_faster_rcnn_implementation.py         # Faster R-CNN from scratch
│   ├── 04_yolo_evolution.md                     # YOLOv1 → YOLOv8 → YOLOv11
│   ├── 05_yolov8_implementation.py              # YOLOv8 analysis
│   ├── 06_anchor_free_detection.md              # FCOS, CenterNet
│   ├── 07_detr_rt_detr.md                       # DETR, RT-DETR transformers
│   ├── 08_detection_comparison.py               # Speed/accuracy tradeoffs
│   ├── 09_yolo_world_open_vocab.md              # Open-vocabulary detection
│   └── 10_detection_profiled.py                 # Detection benchmarks
│
├── 07-segmentation/                     # Segmentation methods (10 files)
│   ├── 01_segmentation_fundamentals.md          # Semantic, instance, panoptic
│   ├── 02_fcn_unet.md                           # FCN, U-Net architectures
│   ├── 03_unet_implementation.py                # U-Net from scratch
│   ├── 04_mask_rcnn.md                          # Instance segmentation
│   ├── 05_deeplabv3.md                          # DeepLab series, ASPP
│   ├── 06_segment_anything.md                   # SAM architecture
│   ├── 07_sam_sam2_analysis.py                  # SAM/SAM2 implementation
│   ├── 08_panoptic_segmentation.md              # Panoptic FPN, MaskFormer
│   ├── 09_medical_segmentation.md               # Medical imaging specifics
│   └── 10_segmentation_profiled.py              # Segmentation benchmarks
│
├── 08-vision-language-models/           # VLMs (8 files)
│   ├── 01_vlm_fundamentals.md                   # Image-text alignment
│   ├── 02_clip_architecture.md                  # CLIP contrastive learning
│   ├── 03_clip_implementation.py                # CLIP from scratch
│   ├── 04_blip_blip2.md                         # BLIP bootstrapping
│   ├── 05_llava_architecture.md                 # LLaVA visual instruction tuning
│   ├── 06_gpt4v_analysis.md                     # GPT-4V capabilities analysis
│   ├── 07_vlm_comparison.py                     # VLM benchmarks
│   └── 08_multimodal_fusion.md                  # Fusion architectures
│
├── 09-generative-vision/                # Image generation (6 files)
│   ├── 01_generative_fundamentals.md            # VAE, GAN, diffusion overview
│   ├── 02_vae_architecture.md                   # Variational autoencoders
│   ├── 03_gan_architectures.md                  # GAN, DCGAN, StyleGAN
│   ├── 04_diffusion_models.md                   # DDPM, DDIM, LDM
│   ├── 05_stable_diffusion.md                   # Stable Diffusion architecture
│   └── 06_generation_profiled.py                # Generation benchmarks
│
├── 10-optimization-profiling/           # Performance engineering (5 files)
│   ├── 01_vision_data_loading.md                # Efficient image datasets
│   ├── 02_ffcv_vision_loader.py                 # FFCV for images
│   ├── 03_dali_vision_pipeline.py               # NVIDIA DALI
│   ├── 04_model_optimization.md                 # TensorRT, ONNX, pruning
│   └── 05_inference_benchmarks.py               # Inference optimization
│
├── 11-practical-notebooks/              # Hands-on experiments (6 files)
│   ├── 01_image_classification.ipynb            # End-to-end classification
│   ├── 02_object_detection.ipynb                # Detection pipeline
│   ├── 03_segmentation_sam.ipynb                # SAM usage
│   ├── 04_vit_from_scratch.ipynb                # Build ViT
│   ├── 05_exercises_and_solutions.py            # Graded exercises
│   └── 06_transfer_learning.ipynb               # Fine-tuning pretrained
│
├── 12-advanced-topics/                  # Cutting-edge research (5 files)
│   ├── 01_3d_vision.md                          # 3D reconstruction, NeRF
│   ├── 02_video_understanding.md                # Video transformers, action recognition
│   ├── 03_embodied_vision.md                    # Robotics vision
│   ├── 04_efficient_vision.md                   # Mobile, edge deployment
│   └── 05_latest_research_2025.md               # Most recent developments
│
├── papers/                              # Reference materials
│   └── paper_summaries.md                       # All papers summarized
│
├── resources/                           # Learning resources
│   ├── glossary.md                              # 100+ CV terms defined
│   └── external_links.md                        # Datasets, tools, community
│
└── README.md                            # This file
```

---

## Learning Progression

### Phase 1: Foundations (Week 1-2)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 01 | Image Fundamentals | - |
| 01 | Convolution Operations | - |
| 02 | Preprocessing & Augmentation | - |
| 03 | CNN Fundamentals | [LeNet 1998](http://yann.lecun.com/exdb/lenet/) |

### Phase 2: CNN Architectures (Week 3-4)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 03 | AlexNet, VGG | [AlexNet 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) |
| 03 | ResNet | [ResNet 2015](https://arxiv.org/abs/1512.03385) |
| 03 | EfficientNet | [EfficientNet 2019](https://arxiv.org/abs/1905.11946) |
| 03 | ConvNeXt | [ConvNeXt 2022](https://arxiv.org/abs/2201.03545) |

### Phase 3: Vision Transformers (Week 5-7)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 04 | ViT Architecture | [ViT 2020](https://arxiv.org/abs/2010.11929) |
| 04 | DeiT Training | [DeiT 2020](https://arxiv.org/abs/2012.12877) |
| 04 | Swin Transformer | [Swin 2021](https://arxiv.org/abs/2103.14030) |
| 05 | DINO, DINOv2 | [DINOv2 2023](https://arxiv.org/abs/2304.07193) |

### Phase 4: Detection & Segmentation (Week 8-10)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 06 | YOLO Evolution | [YOLOv1 2015](https://arxiv.org/abs/1506.02640) |
| 06 | DETR | [DETR 2020](https://arxiv.org/abs/2005.12872) |
| 07 | U-Net | [U-Net 2015](https://arxiv.org/abs/1505.04597) |
| 07 | SAM | [SAM 2023](https://arxiv.org/abs/2304.02643) |

### Phase 5: Vision-Language (Week 11-12)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 08 | CLIP | [CLIP 2021](https://arxiv.org/abs/2103.00020) |
| 08 | LLaVA | [LLaVA 2023](https://arxiv.org/abs/2304.08485) |
| 09 | Diffusion Models | [DDPM 2020](https://arxiv.org/abs/2006.11239) |
| 09 | Stable Diffusion | [LDM 2021](https://arxiv.org/abs/2112.10752) |

### Phase 6: Production (Week 13-14)
| Module | Topic | Resources |
|--------|-------|-----------|
| 10 | Data Loading | DALI, FFCV |
| 10 | Model Optimization | TensorRT |
| 11 | Practical Projects | - |

---

## Key Research Papers

### Foundational CNN Era (2012-2016)
1. **AlexNet** (2012) - Deep CNNs for ImageNet
2. **VGG** (2014) - Very deep networks
3. **GoogLeNet/Inception** (2014) - Inception modules
4. **ResNet** (2015) - Skip connections, depth
5. **YOLO** (2015) - Real-time object detection
6. **U-Net** (2015) - Medical image segmentation

### Modern CNNs (2017-2021)
7. **Mask R-CNN** (2017) - Instance segmentation
8. **EfficientNet** (2019) - Compound scaling
9. **YOLOv4/v5** (2020) - Production detection
10. **ConvNeXt** (2022) - Modernized CNNs

### Vision Transformers (2020-2023)
11. **ViT** (2020) - Image transformers
12. **DeiT** (2020) - Data-efficient training
13. **Swin** (2021) - Hierarchical ViT
14. **MAE** (2021) - Masked autoencoders
15. **DINOv2** (2023) - Self-supervised foundation

### Vision-Language & Foundation (2021-2025)
16. **CLIP** (2021) - Image-text contrastive
17. **BLIP** (2022) - Bootstrapped pretraining
18. **SAM** (2023) - Segment anything
19. **LLaVA** (2023) - Visual instruction tuning
20. **SAM2** (2024) - Video segmentation

---

## Profiling Focus Areas

### Memory & Bandwidth
- Image sizes: 224² vs 384² vs 512² impact
- Batch processing: Memory scaling
- Feature maps: Resolution vs channels tradeoff
- Data loading: Decode overhead (JPEG vs raw)

### Computation
- Convolution: FLOP counts, im2col vs direct
- Attention: O(n²) for ViT, O(n) for linear attention
- Detection: Backbone vs neck vs head breakdown

### Architecture Comparisons
| Model | Params | ImageNet Top-1 | Throughput |
|-------|--------|----------------|------------|
| ResNet-50 | 25M | 76.1% | 1200 img/s |
| EfficientNet-B0 | 5.3M | 77.1% | 1100 img/s |
| ViT-B/16 | 86M | 77.9% | 900 img/s |
| Swin-B | 88M | 83.5% | 750 img/s |
| ConvNeXt-B | 89M | 83.8% | 800 img/s |
| DINOv2-B | 86M | 84.5% | 850 img/s |

### Detection Speed/Accuracy
| Model | COCO mAP | FPS (T4) | Use Case |
|-------|----------|----------|----------|
| YOLOv8-n | 37.3 | 1200 | Edge/mobile |
| YOLOv8-x | 53.9 | 280 | Accuracy |
| RT-DETR-L | 53.0 | 114 | Transformer |
| DINO-4scale | 49.4 | 23 | Research |

---

## Prerequisites

1. **Linear Algebra**: Convolution, matrix operations
2. **Python**: Intermediate level
3. **PyTorch**: Basic tensor operations
4. **Computer Vision Basics**: Images, pixels, channels

---

## Quick Start

```bash
# Setup environment
pip install torch torchvision
pip install albumentations opencv-python
pip install timm  # PyTorch Image Models
pip install segment-anything  # SAM
pip install ultralytics  # YOLO

# Clone reference implementations
git clone https://github.com/facebookresearch/detectron2
git clone https://github.com/facebookresearch/segment-anything
git clone https://github.com/facebookresearch/dinov2
```

---

## Status Tracker

| Module | Status | Last Updated |
|--------|--------|--------------|
| 01-foundations | 🟡 Planned | Dec 2024 |
| 02-preprocessing-augmentation | 🟡 Planned | Dec 2024 |
| 03-convolutional-networks | 🟡 Planned | Dec 2024 |
| 04-vision-transformers | 🟡 Planned | Dec 2024 |
| 05-self-supervised-learning | 🟡 Planned | Dec 2024 |
| 06-object-detection | 🟡 Planned | Dec 2024 |
| 07-segmentation | 🟡 Planned | Dec 2024 |
| 08-vision-language-models | 🟡 Planned | Dec 2024 |
| 09-generative-vision | 🟡 Planned | Dec 2024 |
| 10-optimization-profiling | 🟡 Planned | Dec 2024 |
| 11-practical-notebooks | 🟡 Planned | Dec 2024 |
| 12-advanced-topics | 🟡 Planned | Dec 2024 |

---

## Estimated Time: 14-16 weeks
