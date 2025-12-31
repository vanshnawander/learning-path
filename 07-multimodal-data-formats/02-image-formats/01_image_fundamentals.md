# Image Formats Fundamentals

How images are stored, compressed, and loaded for ML.

## Raw Image Representation

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW IMAGE IN MEMORY                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image: 224 × 224 pixels, RGB (3 channels)                  │
│                                                              │
│  Memory Layout (HWC - Height × Width × Channels):           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ R G B R G B R G B ... (224 pixels) ... R G B            ││
│  │ ─────────────── Row 0 ───────────────────               ││
│  │ R G B R G B R G B ... (224 pixels) ... R G B            ││
│  │ ─────────────── Row 1 ───────────────────               ││
│  │ ...                                                      ││
│  │ ─────────────── Row 223 ─────────────────               ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  Total size: 224 × 224 × 3 = 150,528 bytes (~147 KB)        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Common Image Formats

| Format | Compression | Lossy? | Alpha? | Use Case |
|--------|-------------|--------|--------|----------|
| BMP | None/RLE | No | Yes | Raw, Windows |
| PNG | DEFLATE | No | Yes | Screenshots, logos |
| JPEG | DCT | Yes | No | Photos |
| WebP | VP8/VP9 | Both | Yes | Web images |
| TIFF | Various | Both | Yes | Professional |
| GIF | LZW | No | 1-bit | Animations |

## JPEG Compression Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    JPEG ENCODING                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  RGB Image                                                   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────┐                                           │
│  │ RGB → YCbCr  │  Y = Luminance, Cb/Cr = Chrominance       │
│  └──────────────┘                                           │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────┐                                           │
│  │ Chroma      │  Cb,Cr downsampled (4:2:0 = 1/4 size)     │
│  │ Subsampling │  Humans less sensitive to color detail     │
│  └──────────────┘                                           │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────┐                                           │
│  │ 8×8 Block   │  Process in 8×8 pixel blocks              │
│  │ DCT         │  Discrete Cosine Transform                 │
│  └──────────────┘                                           │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────┐                                           │
│  │ Quantization │  Divide by quality table (LOSSY STEP!)    │
│  └──────────────┘                                           │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────┐                                           │
│  │ Entropy     │  Huffman or arithmetic coding              │
│  │ Coding      │                                            │
│  └──────────────┘                                           │
│      │                                                       │
│      ▼                                                       │
│  JPEG File (10-20x smaller!)                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Memory Layouts for ML

### HWC (Height × Width × Channels) - Common in storage
```
Memory: [R₀₀ G₀₀ B₀₀] [R₀₁ G₀₁ B₀₁] ... [R₀ₙ G₀ₙ B₀ₙ]
        [R₁₀ G₁₀ B₁₀] [R₁₁ G₁₁ B₁₁] ...
        ...
```

### CHW (Channels × Height × Width) - PyTorch default
```
Memory: [R₀₀ R₀₁ R₀₂ ... Rₘₙ]  # All red values
        [G₀₀ G₀₁ G₀₂ ... Gₘₙ]  # All green values
        [B₀₀ B₀₁ B₀₂ ... Bₘₙ]  # All blue values
```

### NHWC vs NCHW (Batch dimension)
```python
# NCHW (PyTorch default)
tensor.shape = [batch, channels, height, width]
# Channels contiguous - good for CPU

# NHWC (TensorFlow default, GPU Tensor Cores)
tensor.shape = [batch, height, width, channels]
# Spatial positions contiguous - better for GPU!

# Convert in PyTorch:
x = x.to(memory_format=torch.channels_last)
```

## Data Types

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| uint8 | 8 | 0-255 | Storage, display |
| float32 | 32 | ±3.4e38 | Training |
| float16 | 16 | ±65504 | Mixed precision |
| bfloat16 | 16 | ±3.4e38 | TPU/GPU training |

## Image Loading Pipeline

```python
# SLOW: Standard PIL loading
from PIL import Image
img = Image.open("image.jpg")       # Decode JPEG
arr = np.array(img)                 # Copy to numpy
tensor = torch.from_numpy(arr)      # Copy to tensor
tensor = tensor.permute(2, 0, 1)    # HWC → CHW
tensor = tensor.float() / 255.0     # Normalize

# FAST: TurboJPEG + direct tensor
import turbojpeg
jpeg = turbojpeg.TurboJPEG()
img = jpeg.decode(open("image.jpg", "rb").read())  # 2-3x faster!

# FASTEST: FFCV / NVIDIA DALI
# - Hardware decode (NVJPEG)
# - Direct GPU transfer
# - Fused operations
```

## Compression Ratios (224×224 RGB)

| Format | Size | Ratio |
|--------|------|-------|
| Raw | 147 KB | 1x |
| PNG | 80-150 KB | 1-2x |
| JPEG Q=95 | 30-50 KB | 3-5x |
| JPEG Q=75 | 10-20 KB | 7-15x |
| WebP | 8-15 KB | 10-20x |

## ML Training Considerations

1. **Decode on GPU**: NVJPEG, NVIDIA DALI
2. **Pre-resize**: Store at training resolution
3. **Progressive JPEG**: Faster partial decode
4. **Batch decode**: Amortize overhead
5. **Cache decoded**: FFCV .beton format
