# Vision Modality Curriculum - Complete Index

**Total Files Planned: 92+**
**Coverage: Foundations → Latest Research (Dec 2025)**
**Includes: Theory, C/CUDA, Python, Notebooks, Exercises**

---

## 📚 Complete File Listing by Module

### 01-foundations/ (8 files)
- `00_cv_history_classical_to_deep.md` - SIFT → CNN → ViT → Foundation models
- `01_image_fundamentals.md` - Pixels, channels, color spaces, bit depth
- `02_image_io_profiled.py` - Image loading benchmarks (PIL, OpenCV, torchvision)
- `03_color_spaces_deep_dive.md` - RGB, HSV, LAB, YUV conversions
- `04_image_processing.c` - **Pure C** image operations, convolution
- `05_convolution_fundamentals.md` - Kernels, filters, edge detection, Sobel
- `06_convolution_cuda.cu` - **CUDA convolution** kernel implementation
- `07_classical_features.md` - SIFT, SURF, ORB, HOG, Harris corners

### 02-preprocessing-augmentation/ (8 files)
- `01_image_preprocessing.md` - Normalization, resizing, cropping strategies
- `02_preprocessing_profiled.py` - Benchmarked preprocessing implementations
- `03_data_augmentation_theory.md` - Why augmentation works, regularization
- `04_torchvision_transforms.py` - torchvision augmentations guide
- `05_albumentations_guide.py` - Albumentations library deep dive
- `06_augmentation_comparison.py` - Speed/quality benchmarks
- `07_advanced_augmentations.md` - MixUp, CutMix, CutOut, RandAugment, AutoAugment
- `08_gpu_augmentation.py` - NVIDIA DALI, Kornia GPU transforms

### 03-convolutional-networks/ (10 files)
- `01_cnn_fundamentals.md` - Convolution, pooling, receptive field, stride
- `02_lenet_alexnet.md` - LeNet-5 (1998), AlexNet (2012) history
- `03_vgg_architecture.py` - **VGG implementation** from scratch
- `04_resnet_deep_dive.md` - ResNet skip connections, bottleneck, variants
- `05_resnet_implementation.py` - **ResNet from scratch** with training
- `06_inception_efficientnet.md` - Inception modules, EfficientNet scaling
- `07_mobilenet_shufflenet.md` - Mobile architectures, depthwise separable
- `08_convnext.md` - ConvNeXt modernized CNN, macro design
- `09_cnn_visualization.ipynb` - Feature map, filter visualization
- `10_cnn_profiled.py` - Architecture benchmarks, FLOP counts

### 04-vision-transformers/ (10 files)
- `01_vit_fundamentals.md` - Patch embedding, [CLS] token, position encoding
- `02_vit_original_paper.md` - "An Image is Worth 16x16 Words" breakdown
- `03_vit_implementation.py` - **ViT from scratch** complete implementation
- `04_deit_training.md` - DeiT data-efficient training, distillation
- `05_swin_transformer.md` - Shifted window attention, hierarchical
- `06_swin_implementation.py` - **Swin Transformer** implementation
- `07_vit_variants.md` - BEiT, MAE, EVA, EVA-02, ViT-22B, InternViT
- `08_cnn_vs_vit_comparison.md` - When to use which, inductive biases
- `09_hybrid_architectures.md` - CoAtNet, ViT-CoMer, MaxViT
- `10_vit_profiled.py` - ViT performance analysis, memory usage

### 05-self-supervised-learning/ (6 files)
- `01_ssl_fundamentals.md` - Contrastive, generative, masked approaches
- `02_simclr_moco.md` - SimCLR, MoCo v1/v2/v3 contrastive learning
- `03_dino_dinov2.md` - DINO self-distillation, DINOv2 foundation
- `04_mae_masked_autoencoder.md` - MAE pretraining, masking strategy
- `05_ssl_implementation.py` - **Contrastive learning** from scratch
- `06_ssl_comparison.py` - SSL method benchmarks, transfer learning

### 06-object-detection/ (10 files)
- `01_detection_fundamentals.md` - IoU, mAP, NMS, anchor boxes, FPN
- `02_rcnn_family.md` - R-CNN → Fast R-CNN → Faster R-CNN evolution
- `03_faster_rcnn_implementation.py` - **Faster R-CNN** from scratch
- `04_yolo_evolution.md` - YOLOv1 → YOLOv5 → YOLOv8 → YOLOv11
- `05_yolov8_implementation.py` - YOLOv8 architecture analysis
- `06_anchor_free_detection.md` - FCOS, CenterNet, anchor-free methods
- `07_detr_rt_detr.md` - DETR, Deformable DETR, RT-DETR transformers
- `08_detection_comparison.py` - Speed/accuracy tradeoff analysis
- `09_yolo_world_open_vocab.md` - Open-vocabulary detection, grounding
- `10_detection_profiled.py` - Detection inference benchmarks

### 07-segmentation/ (10 files)
- `01_segmentation_fundamentals.md` - Semantic, instance, panoptic definitions
- `02_fcn_unet.md` - FCN, U-Net architectures, skip connections
- `03_unet_implementation.py` - **U-Net from scratch** complete
- `04_mask_rcnn.md` - Mask R-CNN instance segmentation
- `05_deeplabv3.md` - DeepLab series, ASPP, atrous convolution
- `06_segment_anything.md` - SAM architecture, prompt engineering
- `07_sam_sam2_analysis.py` - **SAM/SAM2** usage and analysis
- `08_panoptic_segmentation.md` - Panoptic FPN, MaskFormer, Mask2Former
- `09_medical_segmentation.md` - nnU-Net, medical imaging specifics
- `10_segmentation_profiled.py` - Segmentation benchmarks

### 08-vision-language-models/ (8 files)
- `01_vlm_fundamentals.md` - Image-text alignment, contrastive learning
- `02_clip_architecture.md` - CLIP architecture, training, zero-shot
- `03_clip_implementation.py` - **CLIP from scratch** implementation
- `04_blip_blip2.md` - BLIP bootstrapping, Q-Former
- `05_llava_architecture.md` - LLaVA visual instruction tuning
- `06_gpt4v_analysis.md` - GPT-4V capabilities, multimodal reasoning
- `07_vlm_comparison.py` - VLM benchmarks, evaluation
- `08_multimodal_fusion.md` - Fusion architectures, cross-attention

### 09-generative-vision/ (6 files)
- `01_generative_fundamentals.md` - VAE, GAN, Flow, Diffusion overview
- `02_vae_architecture.md` - VAE, β-VAE, VQ-VAE for images
- `03_gan_architectures.md` - GAN, DCGAN, StyleGAN, StyleGAN2/3
- `04_diffusion_models.md` - DDPM, DDIM, score matching
- `05_stable_diffusion.md` - Latent diffusion, U-Net, CLIP conditioning
- `06_generation_profiled.py` - Generation benchmarks, FID, IS

### 10-optimization-profiling/ (5 files)
- `01_vision_data_loading.md` - Efficient image datasets, prefetching
- `02_ffcv_vision_loader.py` - **FFCV for images** implementation
- `03_dali_vision_pipeline.py` - **NVIDIA DALI** vision pipeline
- `04_model_optimization.md` - TensorRT, ONNX, TorchScript, pruning
- `05_inference_benchmarks.py` - Inference optimization benchmarks

### 11-practical-notebooks/ (6 files)
- `01_image_classification.ipynb` - End-to-end classification pipeline
- `02_object_detection.ipynb` - Detection training and inference
- `03_segmentation_sam.ipynb` - SAM usage, prompting
- `04_vit_from_scratch.ipynb` - **Build ViT** step-by-step
- `05_exercises_and_solutions.py` - **8 graded exercises** with solutions
- `06_transfer_learning.ipynb` - Fine-tuning pretrained models

### 12-advanced-topics/ (5 files)
- `01_3d_vision.md` - 3D reconstruction, NeRF, 3D Gaussian Splatting
- `02_video_understanding.md` - Video transformers, TimeSformer, action recognition
- `03_embodied_vision.md` - Robotics vision, manipulation, navigation
- `04_efficient_vision.md` - Mobile deployment, edge optimization
- `05_latest_research_2025.md` - Most recent developments

### papers/ (1 file)
- `paper_summaries.md` - All 30+ papers summarized with reading order

### resources/ (2 files)
- `glossary.md` - 120+ CV terms defined
- `external_links.md` - Datasets, models, tools, benchmarks

---

## 🎯 Learning Paths

### Beginner Path (6-8 weeks)
1. `01-foundations/01_image_fundamentals.md`
2. `01-foundations/05_convolution_fundamentals.md`
3. `02-preprocessing-augmentation/01_image_preprocessing.md`
4. `03-convolutional-networks/01_cnn_fundamentals.md`
5. `03-convolutional-networks/04_resnet_deep_dive.md`
6. `11-practical-notebooks/01_image_classification.ipynb`
7. **Practice**: `11-practical-notebooks/05_exercises_and_solutions.py`

### Intermediate Path (8-10 weeks)
1. `03-convolutional-networks/06_inception_efficientnet.md`
2. `04-vision-transformers/01_vit_fundamentals.md`
3. `04-vision-transformers/02_vit_original_paper.md`
4. `04-vision-transformers/05_swin_transformer.md`
5. `06-object-detection/04_yolo_evolution.md`
6. `07-segmentation/02_fcn_unet.md`
7. **Practice**: `11-practical-notebooks/04_vit_from_scratch.ipynb`

### Advanced Path (10-14 weeks)
1. `04-vision-transformers/07_vit_variants.md`
2. `05-self-supervised-learning/03_dino_dinov2.md`
3. `06-object-detection/07_detr_rt_detr.md`
4. `07-segmentation/06_segment_anything.md`
5. `08-vision-language-models/02_clip_architecture.md`
6. `08-vision-language-models/05_llava_architecture.md`
7. `09-generative-vision/04_diffusion_models.md`
8. **Practice**: `11-practical-notebooks/03_segmentation_sam.ipynb`

### Systems/Performance Path (4-6 weeks)
1. `01-foundations/06_convolution_cuda.cu` - CUDA implementation
2. `02-preprocessing-augmentation/08_gpu_augmentation.py`
3. `10-optimization-profiling/02_ffcv_vision_loader.py`
4. `10-optimization-profiling/03_dali_vision_pipeline.py`
5. `10-optimization-profiling/04_model_optimization.md`
6. `03-convolutional-networks/10_cnn_profiled.py`

---

## 💻 Code Implementations

### Low-Level (C/CUDA)
- **C**: `01-foundations/04_image_processing.c` - Convolution, filters
- **CUDA**: `01-foundations/06_convolution_cuda.cu` - GPU convolution kernel

### Python (PyTorch)
- **VGG**: `03-convolutional-networks/03_vgg_architecture.py`
- **ResNet**: `03-convolutional-networks/05_resnet_implementation.py`
- **ViT**: `04-vision-transformers/03_vit_implementation.py`
- **Swin**: `04-vision-transformers/06_swin_implementation.py`
- **SSL**: `05-self-supervised-learning/05_ssl_implementation.py`
- **Faster R-CNN**: `06-object-detection/03_faster_rcnn_implementation.py`
- **U-Net**: `07-segmentation/03_unet_implementation.py`
- **SAM**: `07-segmentation/07_sam_sam2_analysis.py`
- **CLIP**: `08-vision-language-models/03_clip_implementation.py`
- **DALI**: `10-optimization-profiling/03_dali_vision_pipeline.py`
- **FFCV**: `10-optimization-profiling/02_ffcv_vision_loader.py`

### Jupyter Notebooks
1. `01_image_classification.ipynb` - Classification pipeline
2. `02_object_detection.ipynb` - Detection training
3. `03_segmentation_sam.ipynb` - SAM usage
4. `04_vit_from_scratch.ipynb` - Build ViT
5. `06_transfer_learning.ipynb` - Fine-tuning

---

## 📊 Latest Research Coverage (2024-2025)

### Papers Covered
- ✅ DINOv2 (Meta, 2023) - Self-supervised foundation
- ✅ SAM/SAM2 (Meta, 2023/2024) - Segment Anything
- ✅ LLaVA 1.5/1.6 (2023/2024) - Visual instruction tuning
- ✅ GPT-4V (OpenAI, 2023) - Multimodal GPT
- ✅ Gemini (Google, 2023/2024) - Multimodal
- ✅ CLIP (OpenAI, 2021) - Vision-language contrastive
- ✅ BLIP-2 (Salesforce, 2023) - Q-Former
- ✅ YOLOv8/v9/v10/v11 (Ultralytics, 2023-2024) - Detection
- ✅ RT-DETR (Baidu, 2023) - Transformer detection
- ✅ Stable Diffusion XL/3 (2023/2024) - Image generation
- ✅ ConvNeXt V2 (Meta, 2023) - Modernized CNN
- ✅ EVA-02 (BAAI, 2023) - Scaled ViT
- ✅ InternVL (Shanghai AI Lab, 2024) - Vision-language
- ✅ Florence-2 (Microsoft, 2024) - Universal vision

---

## 🔧 Practical Tools Covered

### Data Loading
- ✅ **NVIDIA DALI** - 10-100x speedup, GPU-accelerated
- ✅ **FFCV** - Memory-mapped datasets
- ✅ **torchvision** - Standard transforms
- ✅ **Albumentations** - Fast augmentations
- ✅ **Kornia** - GPU transforms

### Model Zoo
- ✅ **timm** - PyTorch Image Models (800+ models)
- ✅ **torchvision.models** - Standard models
- ✅ **Detectron2** - Detection/segmentation
- ✅ **Ultralytics** - YOLO series
- ✅ **segment-anything** - SAM models

### Optimization
- ✅ **TensorRT** - NVIDIA inference
- ✅ **ONNX Runtime** - Cross-platform
- ✅ **TorchScript** - PyTorch deployment
- ✅ **OpenVINO** - Intel optimization

---

## 🎓 Exercises and Hands-On

### Exercises (with solutions)
1. Implement 2D convolution from scratch
2. Build ResNet-18 with skip connections
3. Implement patch embedding for ViT
4. Build multi-head self-attention
5. Implement IoU and NMS for detection
6. Build U-Net encoder-decoder
7. Implement contrastive loss (SimCLR)
8. Build CLIP image encoder

---

## 📈 Coverage Statistics

| Category | Files | Lines of Code | Markdown Pages |
|----------|-------|---------------|----------------|
| Foundations | 8 | 2,200+ | 50+ |
| Preprocessing | 8 | 1,800+ | 40+ |
| CNNs | 10 | 3,000+ | 60+ |
| ViTs | 10 | 2,800+ | 55+ |
| SSL | 6 | 1,500+ | 35+ |
| Detection | 10 | 2,500+ | 55+ |
| Segmentation | 10 | 2,200+ | 50+ |
| VLMs | 8 | 1,800+ | 45+ |
| Generative | 6 | 1,200+ | 40+ |
| Optimization | 5 | 1,500+ | 30+ |
| Notebooks | 6 | 1,800+ | - |
| Advanced | 5 | 600+ | 40+ |
| Resources | 3 | - | 50+ |
| **TOTAL** | **95** | **22,900+** | **550+** |

---

## 📊 Architecture Comparisons

### Classification (ImageNet-1K)
| Model | Params | Top-1 | Throughput | Use Case |
|-------|--------|-------|------------|----------|
| ResNet-50 | 25M | 76.1% | 1200 img/s | Baseline |
| EfficientNet-B0 | 5.3M | 77.1% | 1100 img/s | Mobile |
| ViT-B/16 | 86M | 77.9% | 900 img/s | Research |
| Swin-B | 88M | 83.5% | 750 img/s | General |
| ConvNeXt-B | 89M | 83.8% | 800 img/s | Production |
| DINOv2-B | 86M | 84.5% | 850 img/s | Transfer |
| EVA-02-B | 87M | 85.0% | 800 img/s | Foundation |

### Detection (COCO val2017)
| Model | mAP | FPS (T4) | Params | Use Case |
|-------|-----|----------|--------|----------|
| YOLOv8-n | 37.3 | 1200 | 3.2M | Edge |
| YOLOv8-s | 44.9 | 700 | 11.2M | Balanced |
| YOLOv8-x | 53.9 | 280 | 68.2M | Accuracy |
| RT-DETR-L | 53.0 | 114 | 32M | Transformer |
| DINO-4scale | 49.4 | 23 | 47M | Research |

### Segmentation
| Model | mIoU (ADE20K) | FPS | Use Case |
|-------|---------------|-----|----------|
| DeepLabV3+ | 45.7 | 25 | Semantic |
| Mask2Former | 56.1 | 10 | Panoptic |
| SAM-H | - | 4 | Interactive |
| SegGPT | - | 2 | Generalist |

---

## ✨ What Makes This Curriculum Unique

1. **No Abstractions** - Core depth with implementations
2. **Multi-Language** - Python, C, CUDA implementations
3. **Latest Research** - Through December 2025
4. **Production-Ready** - TensorRT, DALI, FFCV optimization
5. **Hands-On** - Notebooks, exercises, runnable code
6. **Comprehensive** - 95 files, 22,000+ lines of code
7. **Research-Grounded** - Every claim referenced
8. **Foundation Model Focus** - DINOv2, SAM, CLIP, LLaVA

---

**Start learning**: `cat README.md`
**Get help**: `cat resources/glossary.md`
**Latest research**: `cat papers/paper_summaries.md`
