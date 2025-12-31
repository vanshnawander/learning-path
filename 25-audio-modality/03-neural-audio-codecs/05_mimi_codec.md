# Mimi: Neural Audio Codec for Speech LLMs

**Paper**: [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) (Kyutai, 2024)

Mimi is Kyutai's neural audio codec designed specifically for audio language models. It achieves extreme compression (1.1 kbps) while maintaining speech quality through **semantic token distillation**.

## Table of Contents
1. [Design Goals](#design-goals)
2. [Architecture Overview](#architecture-overview)
3. [Semantic Token Distillation](#semantic-token-distillation)
4. [Extreme Compression](#extreme-compression)
5. [Comparison with EnCodec](#comparison-with-encodec)
6. [Using Mimi](#using-mimi)
7. [Profiling and Benchmarks](#profiling-and-benchmarks)

---

## Design Goals

### Why a New Codec?

```
EnCodec/SoundStream were designed for:
├── Audio compression (storage, streaming)
├── Music and general audio
└── Reconstruction quality as primary metric

Mimi is designed for:
├── Audio LANGUAGE MODELS
├── Speech-focused (not music)
├── Semantic understanding as primary goal
└── Extreme compression for efficient LLM modeling
```

### Key Requirements

```
1. LOW TOKEN RATE
   - Fewer tokens = shorter context for LLM
   - Faster training and inference
   - Target: 12.5 Hz (vs 75 Hz for EnCodec)

2. SEMANTIC TOKENS
   - First token level encodes MEANING
   - Invariant to speaker identity, prosody
   - Distilled from WavLM (self-supervised speech model)

3. STREAMABLE
   - Low latency for real-time dialogue
   - Causal encoder/decoder
   - No future lookahead

4. GOOD SPEECH QUALITY
   - Despite 80x lower bitrate than CD audio
   - Acceptable for conversational AI
```

---

## Architecture Overview

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     MIMI ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 24 kHz waveform                                         │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │   Streaming Encoder     │  Stride: 1920 (80ms)               │
│  │   (Causal Convolutions) │  Output: 12.5 Hz                   │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Transformer Encoder   │  8 layers, causal attention        │
│  │   (Optional, for        │  Captures longer context           │
│  │    temporal modeling)   │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              SEMANTIC + RVQ QUANTIZATION                 │    │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │    │
│  │  │ Semantic Token  │  │  Acoustic RVQ (7 levels)    │   │    │
│  │  │ (WavLM distill) │  │  Standard residual VQ       │   │    │
│  │  │    Level 0      │  │  Levels 1-7                 │   │    │
│  │  └─────────────────┘  └─────────────────────────────┘   │    │
│  └───────────┬────────────────────────┬────────────────────┘    │
│              │                        │                          │
│              └────────────┬───────────┘                          │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────┐                                    │
│  │   Transformer Decoder   │  8 layers                          │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐                                    │
│  │   Streaming Decoder     │  Transposed convolutions           │
│  │   (Causal)              │  Upsample 1920x                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  Output: 24 kHz waveform                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Encoder Architecture

```python
# Mimi Encoder (simplified)
class MimiEncoder(nn.Module):
    def __init__(self):
        # Convolutional encoder with large stride
        self.conv_layers = nn.Sequential(
            # Initial conv
            CausalConv1d(1, 64, kernel_size=7, stride=1),
            
            # Downsampling blocks (total stride = 1920)
            DownsampleBlock(64, 128, stride=8),    # 3 kHz
            DownsampleBlock(128, 256, stride=6),   # 500 Hz
            DownsampleBlock(256, 512, stride=5),   # 100 Hz  
            DownsampleBlock(512, 1024, stride=4),  # 25 Hz
            DownsampleBlock(1024, 1536, stride=2), # 12.5 Hz
        )
        
        # Optional transformer for temporal modeling
        self.transformer = CausalTransformer(
            dim=1536, depth=8, heads=12
        )
```

### Key Differences from EnCodec

| Aspect | EnCodec | Mimi |
|--------|---------|------|
| Stride | 320 | 1920 (6x more) |
| Frame rate | 75 Hz | 12.5 Hz |
| Latency | ~13 ms | ~80 ms |
| Semantic tokens | No | Yes (level 0) |
| Transformer | No | Yes |
| Target use | General audio | Speech LLMs |

---

## Semantic Token Distillation

### The Problem with Pure Acoustic Tokens

```
Standard RVQ tokens encode:
├── What is said (phonemes, words)
├── Who says it (speaker identity)
├── How it's said (prosody, emotion)
└── Recording quality (noise, room acoustics)

For LLM:
- We care most about WHAT is said
- Other info is useful but secondary
- Mixing semantic/acoustic makes modeling harder
```

### WavLM Distillation

```
WavLM is a self-supervised speech model (like BERT for audio):
- Trained on 94k hours of speech
- Masked prediction objective
- Learns rich speech representations
- Features encode semantic content

Mimi distills WavLM into its first token level:

┌─────────────────┐
│    WavLM        │ ──── Teacher (frozen)
│  (pretrained)   │        │
└─────────────────┘        │
         ↓                 │ Distillation
    WavLM features         │ Loss
         ↓                 │
┌─────────────────┐        │
│ Mimi Semantic   │ ◄──────┘
│ Quantizer       │
│  (Level 0)      │
└─────────────────┘
```

### Distillation Loss

```python
def semantic_distillation_loss(mimi_features, audio):
    """
    Align Mimi's first-level representation with WavLM.
    """
    # Extract WavLM features (frozen teacher)
    with torch.no_grad():
        wavlm_features = wavlm_model(audio)
        # Use features from layer 7 (empirically chosen)
        wavlm_features = wavlm_features[7]
    
    # Upsample/downsample to match dimensions
    mimi_features = resample_features(mimi_features, wavlm_features.shape)
    
    # Cosine similarity loss
    loss = 1 - F.cosine_similarity(
        mimi_features, wavlm_features, dim=-1
    ).mean()
    
    return loss
```

### Why This Works

```
Semantic tokens from Mimi level 0:
├── Capture phonetic content (what is said)
├── Invariant to speaker (same words → same tokens)
├── Invariant to prosody (mostly)
└── Similar words → similar token sequences

Experiments show:
- Same sentence, different speakers → nearly identical level 0 tokens
- Different sentences, same speaker → different level 0 tokens

This is EXACTLY what we want for language modeling!
```

---

## Extreme Compression

### Token Rate Calculation

```
Mimi configuration:
- Sample rate: 24,000 Hz
- Stride: 1,920
- Frame rate: 24000 / 1920 = 12.5 Hz

Tokens per second:
- 8 quantizer levels
- 12.5 frames × 8 levels = 100 tokens/second

Bitrate:
- Codebook size: 2048 (11 bits)
- 12.5 Hz × 8 levels × 11 bits = 1100 bps = 1.1 kbps

Compare:
- EnCodec @ 6kbps: 75 Hz × 8 levels × 10 bits = 6000 bps
- CD audio: 1,411 kbps
- MP3 128kbps: 128,000 bps

Mimi is 116x more compressed than MP3!
```

### Quality at Low Bitrate

```
Subjective evaluation (MOS on speech):

Codec          | Bitrate  | MOS Score
---------------|----------|----------
Original       | 1411 kbps| 4.5
EnCodec        | 6 kbps   | 4.1
EnCodec        | 3 kbps   | 3.7
Mimi           | 1.1 kbps | 3.9  ← Better than EnCodec @ 3kbps!

Key insight: Semantic tokens improve subjective quality
             because the content is preserved correctly.
```

---

## Comparison with EnCodec

### Architecture Differences

```python
# EnCodec: Pure convolutional
class EnCodecEncoder(nn.Module):
    def __init__(self):
        self.layers = nn.Sequential(
            Conv1d(1, 32, 7),
            ResBlock(32), Conv1d(32, 64, stride=2),
            ResBlock(64), Conv1d(64, 128, stride=4),
            ResBlock(128), Conv1d(128, 256, stride=5),
            ResBlock(256), Conv1d(256, 512, stride=8),
            LSTM(512, 512, bidirectional=True),  # Limited temporal
        )

# Mimi: Conv + Transformer
class MimiEncoder(nn.Module):
    def __init__(self):
        self.conv = ConvEncoder(stride=1920)
        self.transformer = CausalTransformer(depth=8)  # Rich temporal
```

### Token Structure

```
EnCodec tokens (per frame):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  All levels are "acoustic" - encoding reconstruction info

Mimi tokens (per frame):
┌──────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ SEMANTIC │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │ RVQ │
│  (WavLM) │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
└──────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  Level 0 encodes MEANING, levels 1-7 encode acoustics
```

### Use Case Comparison

| Use Case | Best Codec | Why |
|----------|------------|-----|
| Music streaming | EnCodec | Full frequency range, stereo |
| Audio archival | EnCodec | Higher quality at reasonable bitrate |
| Speech LLM | **Mimi** | Semantic tokens, extreme compression |
| Real-time dialogue | **Mimi** | Designed for Moshi |
| Voice cloning | EnCodec | Preserves speaker details better |

---

## Using Mimi

### Installation and Basic Usage

```python
# Install
pip install moshi

# Load model
from moshi import MimiCodec
import torch

codec = MimiCodec.from_pretrained("kyutai/mimi")
codec = codec.cuda()

# Encode audio to tokens
audio = torch.randn(1, 1, 24000 * 5).cuda()  # 5 seconds @ 24kHz
with torch.no_grad():
    tokens = codec.encode(audio)
    # tokens shape: (1, 8, 62)  # 8 levels, 62 frames for 5 seconds

# Decode tokens to audio
reconstructed = codec.decode(tokens)
# reconstructed shape: (1, 1, 119040)  # ~5 seconds
```

### Token Analysis

```python
def analyze_mimi_tokens(audio, codec):
    """Analyze semantic vs acoustic token behavior"""
    
    # Encode
    tokens = codec.encode(audio)
    
    # Semantic tokens (level 0)
    semantic = tokens[:, 0, :]
    
    # Acoustic tokens (levels 1-7)
    acoustic = tokens[:, 1:, :]
    
    print(f"Semantic tokens: {semantic.shape}")
    print(f"Acoustic tokens: {acoustic.shape}")
    
    # Unique tokens per level (codebook utilization)
    for level in range(8):
        unique = len(tokens[:, level, :].unique())
        print(f"Level {level}: {unique} unique tokens / 2048")
    
    return semantic, acoustic
```

### Partial Decoding (Quality Control)

```python
def decode_with_levels(tokens, codec, num_levels):
    """Decode using only first N levels"""
    # Zero out unused levels
    partial_tokens = tokens.clone()
    partial_tokens[:, num_levels:, :] = 0  # Or use padding token
    
    return codec.decode(partial_tokens)

# Quality comparison
for n in [1, 2, 4, 8]:
    audio = decode_with_levels(tokens, codec, n)
    # Save and listen to compare quality
```

---

## Profiling and Benchmarks

### Latency Analysis

```python
import time
import torch

def profile_mimi(codec, audio_seconds=1.0):
    """Profile encoding and decoding latency"""
    
    audio = torch.randn(1, 1, int(24000 * audio_seconds)).cuda()
    
    # Warmup
    for _ in range(5):
        tokens = codec.encode(audio)
        _ = codec.decode(tokens)
    
    torch.cuda.synchronize()
    
    # Encode timing
    start = time.perf_counter()
    for _ in range(100):
        tokens = codec.encode(audio)
    torch.cuda.synchronize()
    encode_time = (time.perf_counter() - start) / 100
    
    # Decode timing
    start = time.perf_counter()
    for _ in range(100):
        _ = codec.decode(tokens)
    torch.cuda.synchronize()
    decode_time = (time.perf_counter() - start) / 100
    
    print(f"Audio duration: {audio_seconds:.1f}s")
    print(f"Encode time: {encode_time*1000:.2f}ms (RTF: {encode_time/audio_seconds:.3f})")
    print(f"Decode time: {decode_time*1000:.2f}ms (RTF: {decode_time/audio_seconds:.3f})")
    print(f"Total RTF: {(encode_time + decode_time)/audio_seconds:.3f}")
```

### Memory Footprint

```
Mimi model size:
├── Encoder: ~50M parameters
├── Decoder: ~50M parameters
├── RVQ codebooks: 8 × 2048 × 1536 × 4 bytes = 96 MB
└── Total: ~300M parameters, ~1.2 GB GPU memory

EnCodec model size:
├── Total: ~85M parameters, ~340 MB GPU memory

Mimi is larger but processes fewer frames (12.5 Hz vs 75 Hz)
Net result: Similar compute per second of audio
```

### Throughput Comparison

```
Single GPU throughput (A100):

Codec    | Encode RTF | Decode RTF | Total RTF
---------|------------|------------|----------
EnCodec  | 0.02       | 0.03       | 0.05
Mimi     | 0.03       | 0.04       | 0.07

Both are ~15-20x faster than real-time
Mimi slightly slower due to Transformer layers
```

---

## Key Takeaways

```
1. SEMANTIC TOKENS are Mimi's key innovation
   - Level 0 distilled from WavLM
   - Encodes MEANING, not just sound

2. EXTREME COMPRESSION (1.1 kbps)
   - 12.5 Hz frame rate
   - 6x fewer tokens than EnCodec
   - Enables longer LLM context

3. DESIGNED FOR SPEECH LLMs
   - Not for music or general audio
   - Trade-offs optimized for dialogue

4. STREAMING COMPATIBLE
   - Causal architecture
   - ~80ms latency

5. OPEN SOURCE
   - Available on Hugging Face
   - Part of Moshi ecosystem
```

---

## Next Steps

- `06_codec_comparison_benchmark.py` - Quantitative comparison of all codecs
- `../04-speech-representations/02_wavlm_architecture.md` - Deep dive into WavLM
- `../06-audio-language-models/02_moshi_architecture.md` - How Mimi is used in Moshi
