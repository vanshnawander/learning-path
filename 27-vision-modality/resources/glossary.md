# Computer Vision Glossary

## Image Fundamentals

| Term | Definition |
|------|------------|
| **Pixel** | Smallest addressable element |
| **Channel** | Color component (R, G, B) |
| **Resolution** | Image dimensions (width × height) |
| **Aspect Ratio** | Width to height ratio |
| **Bit Depth** | Bits per channel (8-bit = 256 levels) |
| **Color Space** | RGB, HSV, LAB, YUV |
| **Normalization** | Scaling pixel values (e.g., 0-1) |

## Convolution Terms

| Term | Definition |
|------|------------|
| **Kernel/Filter** | Small weight matrix for convolution |
| **Stride** | Step size for sliding kernel |
| **Padding** | Border handling (same, valid) |
| **Receptive Field** | Input region affecting output |
| **Feature Map** | Output of convolutional layer |
| **Pooling** | Downsampling operation (max, avg) |
| **Depthwise Conv** | Per-channel convolution |
| **Pointwise Conv** | 1×1 convolution |

## CNN Architectures

| Term | Definition |
|------|------------|
| **LeNet** | Early CNN for digits |
| **AlexNet** | 2012 ImageNet winner |
| **VGG** | Very deep uniform architecture |
| **ResNet** | Residual connections |
| **Inception** | Multi-scale parallel convolutions |
| **EfficientNet** | Compound scaling |
| **MobileNet** | Efficient mobile architecture |
| **ConvNeXt** | Modernized CNN |

## Vision Transformer Terms

| Term | Definition |
|------|------------|
| **ViT** | Vision Transformer |
| **Patch Embedding** | Image to sequence conversion |
| **[CLS] Token** | Classification token |
| **Position Embedding** | Spatial position encoding |
| **DeiT** | Data-efficient Image Transformer |
| **Swin** | Shifted window transformer |
| **MAE** | Masked Autoencoder |
| **DINO** | Self-distillation with no labels |

## Detection Terms

| Term | Definition |
|------|------------|
| **Bounding Box** | Rectangle around object |
| **IoU** | Intersection over Union |
| **NMS** | Non-Maximum Suppression |
| **Anchor Box** | Predefined box templates |
| **FPN** | Feature Pyramid Network |
| **RPN** | Region Proposal Network |
| **RoI Pooling** | Region of Interest pooling |
| **mAP** | Mean Average Precision |
| **YOLO** | You Only Look Once |
| **DETR** | Detection Transformer |

## Segmentation Terms

| Term | Definition |
|------|------------|
| **Semantic Seg** | Per-pixel class labels |
| **Instance Seg** | Per-object masks |
| **Panoptic Seg** | Combined semantic + instance |
| **FCN** | Fully Convolutional Network |
| **U-Net** | Encoder-decoder with skip connections |
| **Mask R-CNN** | Instance segmentation |
| **SAM** | Segment Anything Model |
| **mIoU** | Mean Intersection over Union |

## Self-Supervised Learning

| Term | Definition |
|------|------------|
| **Contrastive Learning** | Similar pairs closer |
| **SimCLR** | Simple contrastive learning |
| **MoCo** | Momentum Contrast |
| **BYOL** | Bootstrap Your Own Latent |
| **SwAV** | Swapping Assignments |
| **Masked Image Modeling** | Predict masked patches |

## Vision-Language

| Term | Definition |
|------|------------|
| **VLM** | Vision-Language Model |
| **CLIP** | Contrastive Language-Image Pretraining |
| **BLIP** | Bootstrapping Language-Image Pretraining |
| **LLaVA** | Large Language and Vision Assistant |
| **Image Captioning** | Generate text from image |
| **VQA** | Visual Question Answering |
| **Zero-shot** | No task-specific training |

## Generative Models

| Term | Definition |
|------|------------|
| **VAE** | Variational Autoencoder |
| **GAN** | Generative Adversarial Network |
| **Diffusion Model** | Iterative denoising |
| **DDPM** | Denoising Diffusion Probabilistic |
| **Stable Diffusion** | Latent diffusion model |
| **FID** | Fréchet Inception Distance |
| **IS** | Inception Score |

## Data Augmentation

| Term | Definition |
|------|------------|
| **RandomCrop** | Random region extraction |
| **RandomFlip** | Horizontal/vertical flip |
| **ColorJitter** | Random color adjustments |
| **MixUp** | Blend two images |
| **CutMix** | Cut and paste regions |
| **CutOut** | Random rectangular mask |
| **RandAugment** | Random augmentation policy |
| **AutoAugment** | Learned augmentation policy |

## Metrics

| Term | Definition |
|------|------------|
| **Top-1 Accuracy** | Correct if top prediction matches |
| **Top-5 Accuracy** | Correct if label in top 5 |
| **mAP** | Mean Average Precision |
| **mIoU** | Mean Intersection over Union |
| **FPS** | Frames Per Second |
| **FLOPS** | Floating Point Operations |
| **Params** | Number of parameters |
