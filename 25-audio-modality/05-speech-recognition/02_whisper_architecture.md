# Whisper Architecture: Robust Speech Recognition

**Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (OpenAI, 2022)

Whisper is a general-purpose speech recognition model trained on 680,000 hours of multilingual data. It achieves robust performance across domains, accents, and noise conditions.

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Architecture](#architecture)
3. [Input Processing](#input-processing)
4. [Training Data](#training-data)
5. [Multitask Format](#multitask-format)
6. [Model Sizes](#model-sizes)
7. [Using Whisper](#using-whisper)
8. [Profiling](#profiling)

---

## Design Philosophy

### Why Whisper is Different

```
Traditional ASR:
├── Train on clean, transcribed speech (~1-10k hours)
├── Fine-tune for specific domain
├── Struggles with: accents, noise, domain shift
└── High accuracy on benchmark, poor generalization

Whisper:
├── Train on diverse internet audio (680k hours)
├── Weak supervision (noisy labels from web)
├── Zero-shot generalization
└── Works on: accents, noise, multiple languages, any domain
```

### Key Insight

```
Scale + Diversity > Perfect Labels

680,000 hours of noisy web data
beats
1,000 hours of perfect studio recordings

The model learns to handle real-world audio
because it was trained on real-world audio.
```

---

## Architecture

### Encoder-Decoder Transformer

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHISPER ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Input (30 seconds max)                                   │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────┐                        │
│  │      Log Mel Spectrogram            │                        │
│  │      (80 bins × 3000 frames)        │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │      Conv Stem (2 layers)           │                        │
│  │      kernel=3, stride=(1,2)         │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │      Sinusoidal Position Encoding   │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │                                             │
│                    ▼                                             │
│  ┌─────────────────────────────────────┐                        │
│  │           ENCODER                    │                        │
│  │   N Transformer Blocks               │                        │
│  │   - Self-attention                   │                        │
│  │   - Feed-forward                     │                        │
│  └─────────────────┬───────────────────┘                        │
│                    │                                             │
│                    │ Encoder Output                              │
│                    │                                             │
│  ┌─────────────────┼───────────────────┐                        │
│  │           DECODER                    │                        │
│  │                 ▼                    │                        │
│  │   ┌─────────────────────────────┐   │                        │
│  │   │ Causal Self-Attention       │   │                        │
│  │   └──────────────┬──────────────┘   │                        │
│  │                  ▼                   │                        │
│  │   ┌─────────────────────────────┐   │                        │
│  │   │ Cross-Attention             │◄──┼── Encoder output       │
│  │   └──────────────┬──────────────┘   │                        │
│  │                  ▼                   │                        │
│  │   ┌─────────────────────────────┐   │                        │
│  │   │ Feed-Forward                │   │                        │
│  │   └──────────────┬──────────────┘   │                        │
│  │                  │                   │                        │
│  │         (× N decoder layers)         │                        │
│  └──────────────────┬──────────────────┘                        │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────┐                        │
│  │      Output Projection              │                        │
│  │      → Token Probabilities          │                        │
│  └─────────────────────────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

```python
# Simplified Whisper architecture
class Whisper(nn.Module):
    def __init__(self, config):
        # Audio encoder
        self.encoder = AudioEncoder(
            n_mels=80,
            n_ctx=1500,  # 30 seconds at 50Hz
            n_state=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_encoder_layers
        )
        
        # Text decoder
        self.decoder = TextDecoder(
            n_vocab=config.vocab_size,  # 51864 for multilingual
            n_ctx=448,  # Max output tokens
            n_state=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_decoder_layers
        )
    
    def forward(self, mel, tokens):
        encoder_output = self.encoder(mel)
        logits = self.decoder(tokens, encoder_output)
        return logits
```

---

## Input Processing

### Mel Spectrogram Configuration

```python
# Whisper's exact preprocessing
SAMPLE_RATE = 16000
N_FFT = 400          # 25ms window
HOP_LENGTH = 160     # 10ms hop
N_MELS = 80
CHUNK_LENGTH = 30    # seconds

# Feature extraction
def log_mel_spectrogram(audio):
    # Pad or trim to 30 seconds
    audio = pad_or_trim(audio, SAMPLE_RATE * CHUNK_LENGTH)
    
    # STFT
    stft = torch.stft(
        audio, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT),
        return_complex=True
    )
    
    # Magnitude spectrogram
    magnitudes = stft.abs() ** 2
    
    # Mel filterbank
    mel_spec = mel_filters @ magnitudes
    
    # Log compression
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec  # Shape: (80, 3000) for 30s
```

### Fixed 30-Second Chunks

```
Whisper processes exactly 30 seconds at a time:

Input audio:  [...........60 seconds...........]
              ↓
Chunk 1:      [....30 seconds....]
Chunk 2:                         [....30 seconds....]
              ↓
Process each chunk independently
              ↓
Concatenate transcriptions
```

---

## Training Data

### Data Composition

```
Total: 680,000 hours of audio

By language:
├── English: 438,000 hours (65%)
├── Non-English: 242,000 hours (35%)
└── 99 languages total

By type:
├── Transcription (ASR): ~75%
├── Translation (X→English): ~25%
└── Includes timestamps

Quality:
├── Automated from web (weak supervision)
├── Filtered for quality
├── Human-level accuracy not required
```

### Data Processing Pipeline

```
Web audio/video
       ↓
Language detection (audio)
       ↓
Existing caption extraction
       ↓
Quality filtering
       ↓
Deduplication
       ↓
Training data
```

---

## Multitask Format

### Special Tokens

```
Whisper uses special tokens to control output:

<|startoftranscript|>  Start of output
<|en|>                 Target language (e.g., English)
<|transcribe|>         Task: transcribe in source language
<|translate|>          Task: translate to English
<|notimestamps|>       Don't output timestamps
<|0.00|>               Timestamp tokens
