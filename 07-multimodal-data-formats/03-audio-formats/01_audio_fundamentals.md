# Audio Formats Fundamentals

How sound is digitized, stored, and processed for ML.

## Sound to Digital: The Basics

```
┌─────────────────────────────────────────────────────────────┐
│                  AUDIO DIGITIZATION                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Analog Sound Wave                                          │
│       ∿∿∿∿∿∿∿∿∿∿∿∿∿∿                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    SAMPLING     │  Take measurements at regular intervals│
│  │  (Sample Rate)  │  CD: 44,100 samples/second             │
│  └─────────────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  QUANTIZATION   │  Convert each sample to integer        │
│  │  (Bit Depth)    │  CD: 16-bit = 65,536 levels            │
│  └─────────────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  Digital Samples: [1024, 1056, 1089, 1120, ...]            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Nyquist Theorem

```
To capture frequency F, you need sample rate ≥ 2F

Human hearing: 20 Hz - 20,000 Hz
CD sample rate: 44,100 Hz (captures up to 22,050 Hz)

Speech recognition: 16,000 Hz is often enough
  (speech < 8000 Hz mostly)
```

## Common Sample Rates

| Rate | Use Case | Nyquist Limit |
|------|----------|---------------|
| 8,000 Hz | Telephone | 4 kHz |
| 16,000 Hz | Speech ML | 8 kHz |
| 22,050 Hz | Low-quality | 11 kHz |
| 44,100 Hz | CD quality | 22 kHz |
| 48,000 Hz | Video/Pro | 24 kHz |
| 96,000 Hz | High-res | 48 kHz |

## Bit Depth

| Bits | Levels | Dynamic Range | Use |
|------|--------|---------------|-----|
| 8 | 256 | 48 dB | Telephone |
| 16 | 65,536 | 96 dB | CD, most ML |
| 24 | 16.7M | 144 dB | Professional |
| 32 float | Huge | ~1500 dB | ML processing |

## Raw Audio Size Calculation

```
Size = Sample Rate × Bit Depth × Channels × Duration

1 minute CD audio:
  44,100 × 16 bits × 2 channels × 60 sec
  = 44,100 × 2 bytes × 2 × 60
  = 10.58 MB (uncompressed)

1 minute 16kHz mono (speech ML):
  16,000 × 2 bytes × 1 × 60 = 1.92 MB
```

## Audio Formats

| Format | Compression | Lossy | Typical Ratio |
|--------|-------------|-------|---------------|
| WAV/PCM | None | No | 1x |
| FLAC | Lossless | No | 2-3x |
| MP3 | Psychoacoustic | Yes | 10-12x |
| AAC | Psychoacoustic | Yes | 10-15x |
| Opus | Psychoacoustic | Yes | 10-20x |
| Vorbis | Psychoacoustic | Yes | 10-15x |

## PCM (Pulse Code Modulation)

```
Raw samples stored sequentially:

Mono:    [S0] [S1] [S2] [S3] ...
Stereo:  [L0 R0] [L1 R1] [L2 R2] [L3 R3] ...

Interleaved (common):
  Memory: L0 R0 L1 R1 L2 R2 ...

Planar (some frameworks):
  Memory: L0 L1 L2 ... R0 R1 R2 ...
```

## ML Audio Representations

### 1. Raw Waveform
```python
# Shape: [batch, channels, samples]
waveform.shape = [32, 1, 16000]  # 1 sec @ 16kHz mono
```

### 2. Spectrogram (STFT)
```python
# Short-Time Fourier Transform
# Shape: [batch, freq_bins, time_frames]
spec = torch.stft(waveform, n_fft=512, hop_length=160)
# Shape: [32, 257, 101] for 1 sec @ 16kHz
```

### 3. Mel Spectrogram (Most Common for Speech)
```python
# Human-perception frequency scale
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    n_mels=80  # 80 mel frequency bins
)(waveform)
# Shape: [32, 80, 101]
```

### 4. MFCCs (Mel-Frequency Cepstral Coefficients)
```python
# Compact representation, traditional ASR
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=13
)(waveform)
# Shape: [32, 13, 101]
```

## Spectrogram Parameters

```
n_fft: FFT window size
  - Larger = better frequency resolution
  - Smaller = better time resolution
  - Common: 512, 1024, 2048

hop_length: Samples between windows
  - Smaller = more time frames
  - Common: n_fft / 4

n_mels: Number of mel bins
  - Speech: 40-80
  - Music: 80-128

win_length: Window size (usually = n_fft)
```

## Audio Loading Pipeline

```python
# Standard (slow)
import librosa
audio, sr = librosa.load("file.mp3", sr=16000)

# Faster: torchaudio with SoX/FFmpeg backend
import torchaudio
audio, sr = torchaudio.load("file.mp3")
audio = torchaudio.functional.resample(audio, sr, 16000)

# Fastest: Pre-convert to WAV, use soundfile
import soundfile as sf
audio, sr = sf.read("file.wav")  # Direct PCM read
```

## ML Training Considerations

1. **Resample once**: Store at target sample rate
2. **Normalize**: [-1, 1] or zero-mean unit-variance
3. **Augmentation**: Time stretch, pitch shift, noise
4. **Chunking**: Fixed-length segments for batching
5. **Padding**: Handle variable lengths
