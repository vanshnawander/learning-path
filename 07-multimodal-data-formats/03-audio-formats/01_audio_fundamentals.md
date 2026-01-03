# Audio Formats Fundamentals

How sound is captured, stored, transformed into spectral features such as STFTs and mel spectrograms, and finally consumed by PyTorch dataloaders for machine learning workflows.

**Table of Contents:**
1. [From Air Pressure to Tensors](#1-from-air-pressure-to-tensors)
2. [Audio Containers & Codecs](#2-audio-containers--codecs)
3. [Time- vs Frequency-Domain Representations](#3-time--vs-frequency-domain-representations)
4. [Short-Time Fourier Transform (STFT) in Depth](#4-short-time-fourier-transform-stft-in-depth)
5. [Mel Spectrograms Explained](#5-mel-spectrograms-explained)
6. [MFCCs and Derived Features](#6-mfccs-and-derived-features)
7. [Preparing Audio for ML Pipelines](#7-preparing-audio-for-ml-pipelines)
8. [Delivering Audio to PyTorch Dataloaders](#8-delivering-audio-to-pytorch-dataloaders)
9. [End-to-End Speech Feature Pipeline](#9-end-to-end-speech-feature-pipeline)
10. [Spectrogram Inversion & Reconstruction](#10-spectrogram-inversion--reconstruction)
11. [Advanced Topics & Practical Tips](#11-advanced-topics--practical-tips)
12. [Summary Cheat Sheet](#12-summary-cheat-sheet)

---

## 1. From Air Pressure to Tensors

### The digitization pipeline

```
Analog waveform (air pressure variations)
    ↓  Microphone transducer converts pressure → voltage
    ↓  Anti-alias low-pass filter removes > Nyquist content
    ↓  Sample & hold circuit measures voltage every Ts seconds
    ↓  Analog-to-Digital Converter (ADC) quantizes to N bits
    ↓  PCM samples stored as integers/floats → tensors in ML
```

### Key stages

| Stage | Concept | Notes |
|-------|---------|-------|
| Sampling | Sample rate (Hz) controls how often the waveform is measured. | Too low ⇒ aliasing (high frequencies masquerade as lower ones). |
| Quantization | Bit depth controls how many amplitude levels are available. | Fewer bits ⇒ higher quantization noise; dithering randomizes error. |
| Encoding | Samples arranged in PCM (raw) or compressed via codecs. | Containers (.wav, .flac, .mp3) store metadata + encoded samples. |
| ML ingestion | Samples decoded → tensors (float32/float16). | Additional transforms (resample, normalize, spectrogram). |

### Nyquist–Shannon sampling theorem

**Theorem:** To reconstruct frequencies up to `F_max`, the sampling rate must be ≥ `2 × F_max`.

**Examples:**
- Human hearing range: 20 Hz – 20,000 Hz → need ≥ 40,000 Hz sample rate
- Speech (mostly < 8,000 Hz): 16,000 Hz is sufficient (2× 8,000)
- Telephone (4,000 Hz bandwidth): 8,000 Hz sample rate

**Key points:**
- Anti-alias filters attenuate energy above Nyquist before sampling
- Oversampling (recording above the minimum) simplifies filter design and improves SNR by noise shaping

### Typical sample rates and justification

| Rate | Nyquist | Typical use | Rationale |
|------|---------|-------------|-----------|
| 8 kHz | 4 kHz | Telephony | Narrowband speech; limited intelligibility. |
| 16 kHz | 8 kHz | Modern ASR | Captures most speech energy incl. consonants. |
| 22.05 kHz | 11.025 kHz | Low-quality music | Half of CD; still acceptable for demos. |
| 44.1 kHz | 22.05 kHz | CD audio | Covers human hearing range. |
| 48 kHz | 24 kHz | Video/streaming | Matches video frame timing; pro audio standard. |
| 96 kHz+ | 48 kHz+ | High-res production | Allows headroom for processing, pitch shifting. |

### Bit depth, dynamic range, and quantization

- **Bit depth**: number of bits per sample.
- **Dynamic range (dB)** ≈ `6.02 × bits + 1.76`
  - Example: 16-bit → `6.02 × 16 + 1.76 ≈ 98 dB`
- **Quantization noise** is approximately white and uniformly distributed between ±½ LSB if dithering is applied.

| Bit depth | Levels | Dynamic range | Usage |
|-----------|--------|---------------|-------|
| 8-bit | 256 | ~50 dB | Telephony, low-power devices. |
| 16-bit | 65 536 | ~96 dB | CD, most public datasets. |
| 24-bit | 16.7M | ~144 dB | Studio/mastering, headroom for processing. |
| 32-bit float | 8.5e+37 | ~1500 dB theoretical | ML pipelines, DAWs for non-destructive edits. |

### Raw storage sizing

**Formula:**
```
Size (bytes) = sample_rate × (bit_depth / 8) × channels × duration
```

**Example calculations:**

- 1 minute of CD audio (44.1 kHz, 16-bit, stereo):
  - `44,100 × (16 / 8) × 2 × 60 = 10,584,000 bytes ≈ 10.1 MB`

- 1 minute of speech ML audio (16 kHz, 16-bit, mono):
  - `16,000 × (16 / 8) × 1 × 60 = 1,920,000 bytes ≈ 1.83 MB`

- 10 seconds of high-res audio (48 kHz, 24-bit, stereo):
  - `48,000 × (24 / 8) × 2 × 10 = 2,880,000 bytes ≈ 2.75 MB`

---

## 2. Audio Containers & Codecs

### Overview of common formats

| Container / Codec | Compression | Lossy? | Typical ratio | Notes |
|-------------------|-------------|--------|---------------|-------|
| WAV + PCM | None | No | 1× | Simple header + raw PCM. Largest but fastest to decode. |
| FLAC | Lossless | No | 2–3× | Linear prediction + entropy coding. |
| ALAC | Lossless | No | ~2× | Apple ecosystem. |
| Opus | Transform (hybrid) | Yes | 10–20× | Great for speech/music; low latency. |
| MP3 / AAC / Vorbis | Psychoacoustic | Yes | 8–15× | Frequency masking, perceptual models. |
| Ogg / MP4 | Containers | Depends | — | Wrap codecs + metadata (chapters, tags). |

**ML pipeline note:** In ML pipelines we typically decode everything to PCM (float32) once, optionally cache the tensors (e.g., WebDataset, Hugging Face datasets) to avoid repeated lossy decompress.

---

## 3. Time- vs Frequency-Domain Representations

### Comparison of audio representations for ML

| Representation | Shape (mono) | Pros | Cons | Typical models |
|----------------|--------------|------|------|----------------|
| Raw waveform | `[samples]` | Preserves phase; simple; differentiable (ConvNets). | Requires more data; harder optimization. | Wav2Vec 2.0, raw-spectrogram CNNs. |
| STFT spectrogram | `[freq_bins, frames]` (complex) | Exposes local frequency content; interpretable. | Need to pick window; complex numbers. | CNNs, conformers. |
| Mel spectrogram | `[mel_bins, frames]` | Matches human perception; compact. | Loses precise frequency info & phase. | ASR, TTS, voice cloning. |
| MFCC | `[n_mfcc, frames]` | Very compact; decorrelated features. | Hand-crafted; loses detail. | Classical ASR, small models. |

---

## 4. Short-Time Fourier Transform (STFT) in Depth

### Mathematical definition

The STFT computes the Fourier Transform over short, overlapping windows of the signal:

```
X(m, k) = Σ x[n + mH] × w[n] × e^(-j × 2π × k × n / N)
           n=0 to N-1
```

**Where:**
- `x[n]` = discrete-time signal (your audio samples)
- `w[n]` = window function (e.g., Hann, Hamming)
- `N` = `n_fft` = window length in samples
- `H` = `hop_length` = stride between adjacent windows
- `m` = frame index (which window we're in)
- `k` = frequency bin index (which frequency we're measuring)

**Intuition:** You slide a window across the audio, compute FFT at each position, and get time-frequency representation.

### Practical implementation details

1. **Window choice**: Hann and Hamming reduce spectral leakage. Rectangular window maintains amplitude but leaks more.
2. **Overlap**: hop_length commonly `N/4` (75% overlap) for speech to balance time/frequency detail.
   - Example: with `n_fft=400`, `hop_length=100` means each 400-sample window overlaps by 300 samples
   - This gives 4× more frames than non-overlapping windows, smoother time resolution
3. **Zero padding**: If `n_fft > win_length`, zero padding improves interpolated frequency resolution (more bins) but not actual info.
   - Example: `win_length=400`, `n_fft=512` adds 112 zeros before FFT
   - Result: 257 frequency bins instead of 201, but same actual frequency resolution
4. **Complex output**: `torch.stft` returns either complex tensor or stacked real/imag.
   - Magnitude = `sqrt(real² + imag²)`
   - Power spectrogram uses magnitude²
5. **Scaling options:**
   - **Amplitude spectrogram**: `|X|` (linear scale)
   - **Power spectrogram**: `|X|²` (energy)
   - **Log/Decibel scaling**: `10 × log10(power + epsilon)` to compress dynamic range
   - Example: 80 dB dynamic range becomes ~8.9 after log10

### Understanding the time–frequency trade-off

**Key trade-off:** You can't have perfect time AND frequency resolution simultaneously (uncertainty principle).

| n_fft | Window duration @16kHz | Frequency resolution | Frames per second @hop=160 | Use case |
|-------|----------------------|---------------------|----------------------------|----------|
| 256 | 16 ms | 62.5 Hz | 100 | Fast events, consonants |
| 400 | 25 ms | 40 Hz | 100 | Standard speech ASR |
| 512 | 32 ms | 31.25 Hz | 100 | Balanced |
| 1024 | 64 ms | 15.6 Hz | 100 | Pitch analysis, music |
| 2048 | 128 ms | 7.8 Hz | 100 | Fine pitch, slow changes |

**Examples:**
- Detecting a quick plosive like "p" or "t" → smaller `n_fft` (256-400) for better time resolution
- Analyzing vowel formants or musical pitch → larger `n_fft` (1024-2048) for better frequency resolution
- Larger `n_fft` → narrower frequency bins (better frequency resolution) but fewer frames (worse time localization)
- Smaller `n_fft` → coarse frequency bins but better temporal detail
- Multiresolution approaches (wavelets, constant-Q) handle both ends but are heavier

### PyTorch implementation example

```python
import torch

# waveform: [batch, channels, samples]
spec_complex = torch.stft(
    waveform,
    n_fft=512,
    hop_length=160,
    win_length=400,
    window=torch.hann_window(400),
    center=True,
    pad_mode="reflect",
    normalized=False,
    return_complex=True,
)
spec_mag = spec_complex.abs()  # magnitude
spec_db = 10 * torch.log10(spec_mag.pow(2) + 1e-10)
```

---

## 5. Mel Spectrograms Explained

### What are mel spectrograms?

Mel spectrograms remap linear frequency bins to mel-spaced bands that approximate human loudness perception.

### The mel scale conversion formulas

**Convert Hz to mel:**
```
mel = 2595 × log10(1 + frequency_hz / 700)
```

**Convert mel to Hz:**
```
frequency_hz = 700 × (10^(mel / 2595) - 1)
```

**Examples:**
- 1000 Hz → `2595 × log10(1 + 1000/700) ≈ 1000 mel`
- 2000 Hz → `2595 × log10(1 + 2000/700) ≈ 1545 mel`
- 4000 Hz → `2595 × log10(1 + 4000/700) ≈ 2146 mel`

Notice: doubling frequency from 1000→2000 Hz adds ~545 mel, but 2000→4000 Hz adds only ~601 mel. This reflects human perception where pitch discrimination is finer at low frequencies.

### Building a mel spectrogram: step-by-step

**Step-by-step with numbers (example: 16 kHz audio, n_fft=400, hop=160, n_mels=80):**

1. **STFT**: compute magnitude or power spectrogram on a linear frequency grid.
   - Input: 1 second of audio = 16,000 samples
   - Output: `n_fft//2 + 1 = 201` frequency bins × `ceil(16000/160) = 100` time frames
   - Shape: `[201, 100]`

2. **Mel filter bank**:
   - Choose `n_mels = 80`
   - Map mel centers (0-80) to Hz (0-8000), then to FFT bin indices (0-200)
   - Create triangular (or Slaney-style) filters that sum to one across frequency
   - Filter bank matrix shape: `[80, 201]`

3. **Apply filter bank**: matrix multiply `[n_mels, n_fft//2 + 1]` with spectrogram to aggregate energy per mel bin.
   - `mel_spec = filter_bank @ linear_spec`
   - Shape: `[80, 100]` (80 mel bins × 100 time frames)

4. **Log/DB compression**: `log10` or natural log after adding epsilon, making distribution Gaussian-like.
   - `log_mel = log10(mel_spec + 1e-6)`
   - Dynamic range compressed from ~60-80 dB to ~1.8-1.9

5. **Optional normalization**: mean/var per feature or per utterance.
   - `normalized = (log_mel - mean) / std`

### Why use the mel scale?

**Human perception reality:**
- Human pitch discrimination is finer at low frequencies and coarser at high ones
- We can distinguish ~1 Hz difference at 100 Hz, but need ~10 Hz difference at 1000 Hz
- Mel scale approximates this perceptual non-linearity

**Dimensionality reduction:**
- Reduces dimensionality (~80 bins vs 257 linear bins for 512 FFT)
- 80 mel bins capture most speech-relevant information
- Fewer parameters → faster training, less overfitting

**Empirical benefits:**
- Empirically improves speech model convergence since energy is concentrated in relevant bins
- Log-mel features are approximately Gaussian-distributed → better for neural networks
- Standard in ASR (LibriSpeech, Common Voice) and TTS (LJSpeech, VCTK)

### Torchaudio implementation

```python
import torchaudio

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    window_fn=torch.hann_window,
    power=2.0,  # power spectrogram
    center=True,
    pad_mode="reflect",
    mel_scale="htk",  # or "slaney"
)
mel_spec = mel_transform(waveform)  # [batch, n_mels, frames]
mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel_spec)
```

### Log-mel best practices

**Why log compression?**
- Audio has huge dynamic range: whispers (~-60 dB) to shouts (~0 dB)
- Linear scale would be dominated by loud sounds
- Log compression makes the distribution more Gaussian-like

**Implementation tips:**
- Add small floor before log: `log(mel + 1e-6)` to avoid `-inf`
- Normalize per utterance (mean-variance) or dataset-level (global stats)
- Save as float16 to reduce disk footprint without big quality loss

**Example normalization:**
```python
# Per-utterance normalization
mean = log_mel.mean(dim=1, keepdim=True)  # [80, 1]
std = log_mel.std(dim=1, keepdim=True)    # [80, 1]
normalized = (log_mel - mean) / (std + 1e-8)
```

---

## 6. MFCCs and Derived Features

### What are MFCCs?

Mel-Frequency Cepstral Coefficients capture the envelope of the log-mel spectrum.

### MFCC computation pipeline

1. Compute mel spectrogram (power)
2. Take log magnitude: `log10(mel_power + epsilon)`
3. Apply Discrete Cosine Transform (DCT-II) along mel axis
4. Keep first `N_mfcc` coefficients (usually 13 or 20)
   - Coefficient 0 = log-energy (overall loudness)
   - Coefficients 1-12 = spectral envelope shape
5. Optionally append:
   - **Delta (Δ)**: first derivative (how features change over time)
   - **Delta-delta (Δ²)**: acceleration (how the change changes)

**Example:** If you have 80 mel bins and compute 13 MFCCs, you get 13 static + 13 delta + 13 delta-delta = 39 features per frame.

MFCCs work well for classical speech recognition with GMM/HMMs or small neural networks. Modern deep models often operate directly on log-mel spectrograms to avoid losing detail.

### Other spectral features

- **Spectral centroid** (brightness).
- **Spectral bandwidth** / roll-off.
- **Chroma** (pitch class energy, music applications).
- **Zero-crossing rate** (voicing detection).

---

## 7. Preparing Audio for ML Pipelines

### Preprocessing checklist

1. **Resampling**: Convert every clip to the target rate (typically 16 kHz mono for speech). Use high-quality polyphase filters (`torchaudio.functional.resample` or `soxio`).
2. **Channel handling**: Downmix stereo to mono via weighted sum unless spatial info is important.
3. **Silence trimming**: Remove leading/trailing silence for efficient training or keep along with VAD labels depending on objective.
4. **Clipping check**: Ensure samples lie within [-1, 1] after normalization. Apply limiter or `librosa.util.normalize`.
5. **Length normalization**:
   - **Chunking**: slice long recordings into fixed windows (e.g., 1–5 s).
   - **Padding**: zero-pad shorter clips, store actual lengths for masking.
6. **Augmentation**:
   - Time stretching (speed perturbation ±5%).
   - Pitch shifting (±2 semitones).
   - Additive noise (room impulse responses, background loops).
   - SpecAugment (masking frequency/time stripes on mel spectrograms).
7. **Caching/transcoding**:
   - Store intermediate tensors (e.g., log-mel) in LMDB, WebDataset tar, or Hugging Face dataset to avoid repeated STFTs.
   - Use float16 or int16 to balance space/performance.
8. **Metadata**:
   - Keep transcripts, speaker IDs, sampling rate, durations in JSON/CSV.
   - Track original file paths for debugging.

---

## 8. Delivering Audio to PyTorch Dataloaders

### Dataset file structure

```
dataset/
  manifest.jsonl  # each line: {"audio": "clips/123.wav", "text": "...", "speaker": "..."}
  clips/
    123.wav
    124.wav
```

Manifests allow streaming-based datasets (esp. WebDataset, Lhotse, torchaudio’s `SpeechCommands`, `LibriLight`).

### Building a Custom PyTorch Dataset

```python
import json
from pathlib import Path
import torch
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, sample_rate=16000, transform=None):
        self.entries = [json.loads(l) for l in Path(manifest_path).read_text().splitlines()]
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        waveform, sr = torchaudio.load(item["audio"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:  # multi-channel → mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.transform:
            features = self.transform(waveform)  # mel, mfcc, raw, etc.
        else:
            features = waveform
        return {
            "features": features,          # tensor [C, T] or [mel, frames]
            "waveform": waveform,
            "length": features.shape[-1],
            "text": item.get("text", ""),
        }
```

### Handling variable-length sequences

**The problem:** Raw audio/spectrograms vary in time dimension. A collate function pads to the max length within the batch and keeps the original lengths for masking.

**Example:**
- Clip 1: 3 seconds → 300 frames @ 10ms hop
- Clip 2: 5 seconds → 500 frames
- Clip 3: 2 seconds → 200 frames
- After padding: all become 500 frames, with zeros filling the gaps
- Lengths tensor: `[300, 500, 200]` tells the model which frames are real

```python
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    feats = [b["features"].squeeze(0).transpose(0, 1) for b in batch]  # [frames, feat_dim]
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    padded = pad_sequence(feats, batch_first=True)  # [B, max_frames, feat_dim]
    texts = [b["text"] for b in batch]
    return {
        "features": padded.transpose(1, 2),  # [B, feat_dim, max_frames]
        "lengths": lengths,
        "texts": texts,
    }
```

### Setting up the DataLoader

```python
from torch.utils.data import DataLoader

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
)

dataset = SpeechDataset("dataset/manifest.jsonl", sample_rate=16000, transform=mel_transform)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,         # parallel decode + transform
    pin_memory=True,
    collate_fn=collate_batch,
    persistent_workers=True,
)

for batch in loader:
    logits = model(batch["features"], batch["lengths"])
    ...
```

**Important notes:**

- **Transforms inside `__getitem__`** keep them on CPU; for GPU transforms use `torchaudio.transforms` on CUDA tensors or transform after batching.
- **num_workers** should match CPU cores / storage speed. Too high leads to thrashing.
- **Prefetching**: frameworks like `torchdata`, `webdataset`, and `lhotse` support streaming shards, shuffling at chunk level, etc.
- **Mixed precision**: convert features to `float16` after normalization to halve memory usage.

### Large-scale and streaming setups

1. **Shard** dataset into tar files (~100–500 MB) to keep dataloader balanced.
2. **WebDataset**: `WebDataset("shard-{000000..000999}.tar").decode().to_tuple("wav", "txt")`.
3. **On-the-fly feature caching**: use `functools.lru_cache`, memmap, or distributed feature stores (e.g., HF dataset with `set_format(type="torch")`).
4. **Chunked transcripts**: when transcripts align at frame-level (CTC models), store alignment metadata, or compute `lengths` for blank tokens.

---

## 9. End-to-End Speech Feature Pipeline

### Complete workflow from raw audio to model input

1. **Manifest preparation**: store (path, text, speaker, duration, original_sr, checksum).

2. **Offline preprocessing (optional)**:
   - Resample & convert to 16-bit PCM WAV.
   - Loudness-normalize (EBU R128) for consistent amplitude.
   - Trim silences or keep plus VAD labels.
3. **Runtime pipeline**:
   - Load PCM → convert to float32 tensor in `[−1, 1]`.
   - Apply augmentation transforms (speed perturb, noise, RIR convolution).
   - Compute STFT → log-mel features.
   - Normalize features: global stats or per-utterance (z-score).
4. **Batching**:
   - Use collate_fn to pad, store lengths.
   - Optionally create attention masks for transformer/conformer models (1 where valid, 0 where padded).

5. **Model consumption**:
   - Feed `[B, n_mels, T]` into CNN/Conformer or `[B, T, n_mels]` depending on architecture.
   - For CTC/Transducer, convert lengths to frame counts: `lengths_frames = lengths // hop_length`.

6. **Post-processing**:
   - For synthesis tasks (e.g., TTS), invert features via Griffin-Lim or neural vocoder (WaveGlow, HiFi-GAN).

---

## 10. Spectrogram Inversion & Reconstruction

### Why STFT phase matters

- STFT stores both magnitude and phase. Mel spectrograms discard phase, so reconstruction requires estimating it.
- **Griffin–Lim algorithm** iteratively estimates phase to match target magnitude (typically 30-100 iterations).
- **Neural vocoders** (WaveNet, WaveGlow, HiFi-GAN) learn to map log-mel spectrograms to waveforms, bypassing explicit phase estimation.

### Practical implications

- When evaluating features, remember that some losses (e.g., L2) are computed on log-mel magnitude only; phase is handled implicitly by the vocoder or ignored for ASR.
- For TTS, you need a vocoder to convert mel spectrograms back to audio.
- For ASR, phase is often ignored since models learn from magnitude patterns.

---

## 11. Advanced Topics & Practical Tips

### STFT configuration heuristics

1. **Speech ASR**: `n_fft=400` (25 ms window @ 16 kHz), `hop=160` (10 ms stride), `n_mels=80`.
2. **Music**: larger window (1024–4096) for better pitch resolution.

### Pre-emphasis filtering

Apply `y[n] = x[n] - α × x[n-1]` with `α ≈ 0.97` to boost high-frequency consonants before STFT.

**Example:** if `x = [0.5, 0.6, 0.4, 0.3]` and `α = 0.97`:
- `y[0] = 0.5 - 0 = 0.5`
- `y[1] = 0.6 - 0.97×0.5 ≈ 0.115`
- `y[2] = 0.4 - 0.97×0.6 ≈ -0.182`
- `y[3] = 0.3 - 0.97×0.4 ≈ -0.088`

### Dynamic range compression

- Log/DB ensures features roughly Gaussian. Use `torchaudio.transforms.AmplitudeToDB`.

### Silence weighting strategies

- Keep track of VAD (voice activity detection) masks to avoid overfitting to silence.

### Feature standardization approaches

- Compute global mean/std over training set and store for inference to ensure identical scaling.

### Precision and storage trade-offs

- Storing features as float16 halves disk usage; convert back to float32 when feeding to model if needed.

### Determinism vs speed trade-offs

- For reproducibility: set `torch.set_num_threads`, fix RNG seeds, disable multi-threaded resamplers.
- For throughput: allow multi-threaded SoX backend.

### Data quality monitoring

- Visualize spectrogram batches, check loudness histograms, verify transcripts align with durations (e.g., WER < 2% on transcript vs TTS).

---

## 12. Summary Cheat Sheet

| Concept | Key formula / API | Notes |
|---------|-------------------|-------|
| Nyquist | `sample_rate >= 2 * max_freq` | Use anti-alias filters before downsampling. |
| STFT | `torch.stft(waveform, n_fft, hop_length, return_complex=True)` | Choose window & hop to balance time/frequency. |
| Mel scale | `mel = 2595*log10(1 + f/700)` | Filter bank aggregates FFT bins into perceptual bands. |
| Mel spectrogram | `torchaudio.transforms.MelSpectrogram(...)(waveform)` | Most ASR/TTS models consume log-mel features. |
| MFCC | `torchaudio.transforms.MFCC(...)` | DCT of log-mel, often 13 coeffs + deltas. |
| Data pipeline | Custom `Dataset` + `collate_fn` | Handle resampling, padding, augmentation, lengths. |
| Feature shapes | `[batch, channels, samples]` → `[batch, n_mels, frames]` | Track hop size to convert samples ↔ frames. |

---

With these details you can build a reproducible audio ingestion pipeline—from raw files on disk, through spectral feature extraction, and finally into PyTorch dataloaders ready for large-scale speech or audio models.
