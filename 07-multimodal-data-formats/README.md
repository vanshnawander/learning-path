# Multimodal Data Formats

Understanding how different data types are stored and processed is essential for building efficient multimodal pipelines.

## Why Data Formats Matter

```
Raw 1-minute 4K video:
  4096 × 2160 pixels × 3 channels × 8 bits × 60 fps × 60 sec
  = 95 GB (uncompressed!)

H.265 encoded: ~100 MB (950x compression)

YOU MUST understand encoding to build efficient pipelines!
```

## Data Modalities Overview

| Modality | Raw Format | Common Encodings | ML Input |
|----------|------------|------------------|----------|
| Text | Unicode | UTF-8, tokenized | Token IDs |
| Image | Pixel array | JPEG, PNG, WebP | Tensor [C,H,W] |
| Audio | PCM samples | WAV, MP3, FLAC | Mel spectrogram |
| Video | Frame sequence | H.264, H.265, AV1 | Tensors [T,C,H,W] |

## Directory Structure

```
07-multimodal-data-formats/
├── 01-text-encoding/
│   ├── 01_unicode_utf8.c
│   ├── 02_tokenization.py
│   └── 03_text_formats.md
├── 02-image-formats/
│   ├── 01_raw_pixels.c
│   ├── 02_jpeg_basics.md
│   ├── 03_image_decoding.c
│   └── 04_tensor_layouts.md
├── 03-audio-formats/
│   ├── 01_pcm_basics.c
│   ├── 02_sampling_theory.md
│   ├── 03_audio_codecs.md
│   └── 04_mel_spectrograms.py
├── 04-video-formats/
│   ├── 01_video_fundamentals.md
│   ├── 02_color_spaces.c
│   ├── 03_video_codecs.md
│   └── 04_frame_extraction.c
└── 05-ml-data-formats/
    ├── 01_numpy_internals.py
    ├── 02_tensor_storage.md
    ├── 03_dataset_formats.md
    └── 04_streaming_formats.md
```

## The Multimodal Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL TRAINING PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Disk    │──▶│  Decode  │──▶│ Transform│──▶│  Model   │     │
│  │ Storage  │   │          │   │          │   │  Input   │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                  │
│  VIDEO:                                                          │
│  .mp4 ─▶ H.264 decode ─▶ YUV→RGB ─▶ Resize ─▶ [T,3,224,224]   │
│                                                                  │
│  AUDIO:                                                          │
│  .mp3 ─▶ MP3 decode ─▶ Resample ─▶ Mel spec ─▶ [1,80,T]        │
│                                                                  │
│  IMAGE:                                                          │
│  .jpg ─▶ JPEG decode ─▶ Resize/Aug ─▶ Normalize ─▶ [3,224,224] │
│                                                                  │
│  TEXT:                                                           │
│  .txt ─▶ UTF-8 decode ─▶ Tokenize ─▶ Pad/Truncate ─▶ [seq_len] │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Bottlenecks

| Stage | Bottleneck | Solution |
|-------|------------|----------|
| Disk I/O | 7 GB/s max (NVMe) | Pre-decode, mmap |
| Video decode | CPU-bound | Hardware decode (NVDEC) |
| Image decode | JPEG is slow | TurboJPEG, pre-resize |
| Data transfer | PCIe bandwidth | pin_memory, prefetch |

## Bits vs Bytes Reminder

```
1 byte = 8 bits

Image pixel (RGB):  3 bytes = 24 bits
Float32:            4 bytes = 32 bits
Float16:            2 bytes = 16 bits
BFloat16:           2 bytes = 16 bits

Unicode codepoint:  up to 4 bytes (UTF-8)
Audio sample (CD):  2 bytes = 16 bits per channel
```
