# Video Formats Fundamentals

Understanding video encoding is critical for multimodal ML at scale.

## Video = Images + Time + Audio

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO STRUCTURE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frame 0    Frame 1    Frame 2    Frame 3    ...            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                    │
│  │ IMG  │  │ IMG  │  │ IMG  │  │ IMG  │                    │
│  └──────┘  └──────┘  └──────┘  └──────┘                    │
│     │         │         │         │                         │
│  t=0.00    t=0.033   t=0.067   t=0.100  (30 fps)           │
│                                                              │
│  + Audio Track: [samples...]                                │
│  + Metadata: resolution, codec, timestamps                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Raw Video Size (Why Compression Matters!)

```
4K Video, 60 fps, 1 minute:

Frames: 3840 × 2160 pixels × 3 channels × 8 bits
      = 24.88 MB per frame

Total: 24.88 MB × 60 fps × 60 sec = 89.6 GB per minute!

With H.265: ~50-100 MB per minute (900-1800x compression!)
```

## Video Codecs

| Codec | Year | Compression | Patent | Use Case |
|-------|------|-------------|--------|----------|
| H.264/AVC | 2003 | Good | Yes | Universal |
| H.265/HEVC | 2013 | 50% better | Yes | 4K, streaming |
| VP9 | 2013 | ~H.265 | No | YouTube |
| AV1 | 2018 | Best | No | Next-gen streaming |

## Video Compression Concepts

### I-Frames, P-Frames, B-Frames

```
I-Frame (Intra): Complete image (keyframe)
P-Frame (Predicted): Changes from previous frame
B-Frame (Bidirectional): Changes from prev AND next

GOP (Group of Pictures):
  I  P  B  B  P  B  B  P  B  B  I  P  B  B  ...
  └────────── GOP ──────────┘

I-frames: Large (~100 KB)
P-frames: Medium (~30 KB)  
B-frames: Small (~15 KB)
```

### Motion Compensation

```
┌─────────────┐     ┌─────────────┐
│  Frame N    │     │  Frame N+1  │
│    ┌───┐    │     │       ┌───┐ │
│    │ A │    │ ──▶ │       │ A │ │  Motion vector: (dx, dy)
│    └───┘    │     │       └───┘ │  Only store the movement!
└─────────────┘     └─────────────┘
```

## Color Spaces

### RGB vs YUV

```
RGB: Red, Green, Blue (display native)
YUV: Luminance (Y), Chrominance (U, V)

Y = 0.299R + 0.587G + 0.114B  (brightness)
U = 0.492(B - Y)               (blue diff)
V = 0.877(R - Y)               (red diff)

Why YUV?
- Human eyes more sensitive to brightness than color
- Can subsample U,V with minimal quality loss
```

### Chroma Subsampling

```
4:4:4 - Full resolution for Y, U, V (no savings)
4:2:2 - U,V at half horizontal resolution
4:2:0 - U,V at half both dimensions (most common!)

4:2:0 example (4x2 pixel block):
  Y: ████████     Full resolution (8 samples)
  U: ████         Quarter resolution (2 samples)  
  V: ████         Quarter resolution (2 samples)

Data reduction: 8 + 2 + 2 = 12 samples vs 24 (2x!)
```

## Container Formats vs Codecs

```
Container = Box that holds video + audio + metadata
Codec = How video/audio is compressed

Container: .mp4, .mkv, .avi, .mov, .webm
Codec: H.264, H.265, VP9, AV1 (video)
       AAC, MP3, Opus, FLAC (audio)

Example: MP4 file with H.264 video and AAC audio
```

## Video Decoding for ML

### CPU Decoding (Slow)
```python
import cv2
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()  # Decode one frame
# ~30-100 fps depending on resolution
```

### GPU Decoding (Fast)
```python
# NVIDIA NVDEC via PyNvVideoCodec or DALI
# Up to 1000+ fps for 1080p!

# With NVIDIA DALI:
import nvidia.dali as dali
video_input = dali.fn.readers.video(
    device="gpu",
    filenames=["video.mp4"]
)
```

### Seeking Challenges

```
Problem: Can only decode from I-frames!

To get frame 500:
  1. Find nearest I-frame before 500 (say, frame 480)
  2. Decode frames 480-500
  3. Return frame 500

Random access is SLOW!

Solution for ML:
  - Extract frames to images (pre-processing)
  - Use video datasets with frame indices
  - Sample clips, not random frames
```

## ML Video Tensor Shapes

```python
# Single video clip
clip.shape = [T, C, H, W]  # T=frames, C=3, H=height, W=width

# Batch of clips
batch.shape = [B, T, C, H, W]  # B=batch size

# Common dimensions:
# - T: 8, 16, 32 frames
# - H, W: 224, 256, 384
# - fps: 1-8 for understanding, 24-30 for generation
```

## Video Loading Libraries

| Library | GPU Decode | Strengths |
|---------|------------|-----------|
| OpenCV | No | Universal |
| decord | Yes | ML-focused |
| PyAV | No | FFmpeg wrapper |
| torchvision | Yes* | PyTorch native |
| DALI | Yes | NVIDIA optimized |

## Video ML Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Storage │───▶│ Decode  │───▶│ Sample  │───▶│ Process │
│ .mp4    │    │ NVDEC   │    │ Frames  │    │ Augment │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
                                            ┌─────────┐
                                            │  Model  │
                                            │ [B,T,C, │
                                            │   H,W]  │
                                            └─────────┘
```

## Performance Tips

1. **Pre-extract keyframes** for random access
2. **Use hardware decode** (NVDEC, VAAPI)
3. **Sample clips** instead of random frames
4. **Lower resolution** if quality allows
5. **Pre-resize** videos to training resolution
6. **Use efficient containers** (mp4 > avi)
