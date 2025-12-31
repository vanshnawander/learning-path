# Signal Processing Fundamentals for Audio ML

Core concepts that underpin all neural audio systems. Every audio model—from WaveNet to Moshi—builds on these fundamentals.

## Table of Contents
1. [The Nature of Sound](#the-nature-of-sound)
2. [Sampling Theory Deep Dive](#sampling-theory-deep-dive)
3. [Quantization and Bit Depth](#quantization-and-bit-depth)
4. [Fourier Analysis](#fourier-analysis)
5. [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
6. [Window Functions](#window-functions)
7. [Phase and Magnitude](#phase-and-magnitude)
8. [Practical Considerations](#practical-considerations)

---

## The Nature of Sound

### Physical Representation
```
Sound = Pressure variations over time

Analog waveform:
    Amplitude
        │
    +1  │    ∿∿∿
        │   ∿    ∿
     0  │──∿──────∿──────▶ Time
        │           ∿    ∿
    -1  │            ∿∿∿
```

### Key Properties
| Property | Definition | Range |
|----------|------------|-------|
| **Frequency** | Oscillations per second (Hz) | 20 Hz - 20 kHz (human hearing) |
| **Amplitude** | Pressure magnitude | Measured in dB SPL |
| **Phase** | Position in oscillation cycle | 0° - 360° |
| **Wavelength** | Physical length of one cycle | ~17m (20 Hz) to ~17mm (20 kHz) |

### Speech vs Music Characteristics
```
Speech:
├── Fundamental frequency (F0): 85-255 Hz (male), 165-255 Hz (female)
├── Formants: resonant frequencies of vocal tract
├── Most energy: 100 Hz - 4 kHz
└── Bandwidth needed: ~8 kHz sufficient

Music:
├── Frequency range: 20 Hz - 20 kHz
├── Dynamic range: up to 120 dB
├── Complex harmonics and overtones
└── Bandwidth needed: full 20 kHz
```

---

## Sampling Theory Deep Dive

### Nyquist-Shannon Theorem

**Theorem**: To perfectly reconstruct a signal, sample at ≥ 2× the highest frequency component.

```
f_sample ≥ 2 × f_max

Example:
- Human hearing: 20 kHz max
- CD quality: 44.1 kHz (captures up to 22.05 kHz)
- Speech ML: 16 kHz (captures up to 8 kHz - sufficient for speech)
```

### Why Aliasing Matters

```
Aliasing: When f_signal > f_nyquist, high frequencies fold back

Original signal at 10 kHz, sampled at 8 kHz:
- Nyquist: 4 kHz
- 10 kHz aliases to: |10 - 8| = 2 kHz (WRONG frequency!)

┌─────────────────────────────────────────────┐
│  True Signal      Aliased Signal            │
│                                             │
│    /\      /\         /\                    │
│   /  \    /  \       /  \                   │
│  /    \  /    \     /    \                  │
│ /      \/      \   /      \                 │
│                                             │
│  High frequency    Appears as LOW frequency │
└─────────────────────────────────────────────┘

Solution: Anti-aliasing filter before sampling
```

### Sample Rate Selection

| Sample Rate | Nyquist | Use Case | Memory/sec (16-bit mono) |
|-------------|---------|----------|--------------------------|
| 8,000 Hz | 4 kHz | Telephone, basic ASR | 16 KB |
| 16,000 Hz | 8 kHz | **Most speech ML** | 32 KB |
| 22,050 Hz | 11 kHz | Low-quality audio | 44 KB |
| 24,000 Hz | 12 kHz | **Mimi codec** | 48 KB |
| 44,100 Hz | 22 kHz | CD quality | 88 KB |
| 48,000 Hz | 24 kHz | Video, professional | 96 KB |

### Resampling Costs (Profiled)

```python
# Resampling is EXPENSIVE - avoid at runtime!
# See 02_spectral_analysis_profiled.py for benchmarks

# Common resampling approaches:
1. Polyphase filtering (scipy.signal.resample_poly) - Best quality
2. FFT-based (scipy.signal.resample) - Fast for large factors  
3. Linear interpolation - Fast but poor quality
4. Kaiser windowed sinc - Balance of quality/speed

# Rule: Pre-resample your dataset to target sample rate
```

---

## Quantization and Bit Depth

### Linear PCM Quantization

```
Bit Depth → Number of amplitude levels

8-bit:  256 levels    → 48 dB dynamic range
16-bit: 65,536 levels → 96 dB dynamic range  (CD standard)
24-bit: 16.7M levels  → 144 dB dynamic range (professional)
32-bit float: ~1500 dB → ML processing standard
```

### Quantization Noise

```
Each bit ≈ 6 dB of dynamic range

Signal-to-Quantization-Noise Ratio (SQNR):
SQNR ≈ 6.02n + 1.76 dB, where n = bit depth

16-bit: SQNR ≈ 98 dB (excellent for most applications)
```

### μ-law Companding (Important for WaveNet!)

```python
# μ-law compresses dynamic range for 8-bit storage
# Used in WaveNet to get 256 discrete levels

μ = 255  # Standard telephony

# Compression:
def mu_law_encode(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

# Expansion:
def mu_law_decode(y, mu=255):
    return np.sign(y) * (np.exp(np.abs(y) * np.log1p(mu)) - 1) / mu

# Why μ-law?
# - Matches human perception (logarithmic sensitivity)
# - 8-bit μ-law ≈ 13-bit linear PCM quality
# - WaveNet uses 256 μ-law quantization levels
```

---

## Fourier Analysis

### Discrete Fourier Transform (DFT)

```
Time domain → Frequency domain

X[k] = Σ x[n] · e^(-j·2π·k·n/N)
       n=0 to N-1

Where:
- x[n]: input signal (N samples)
- X[k]: frequency bin k
- k: frequency index (0 to N-1)
```

### Fast Fourier Transform (FFT)

```
FFT reduces DFT from O(N²) to O(N log N)

Computational cost for 1 second @ 16 kHz:
- DFT: 256,000,000 operations
- FFT: ~240,000 operations (1000x faster!)

FFT is THE reason spectrograms are practical.
```

### Frequency Resolution

```
Δf = f_sample / N

Example: 16 kHz sample rate, 512-point FFT
Δf = 16000 / 512 = 31.25 Hz per bin

Trade-off:
- Larger N → Better frequency resolution
- Larger N → Worse time resolution
- This is the uncertainty principle of signal processing!
```

---

## Short-Time Fourier Transform (STFT)

### The Core of Audio ML

```
STFT = Windowed FFT applied at regular intervals

┌────────────────────────────────────────────────────┐
│  Audio Waveform                                    │
│  ▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃        │
│                                                    │
│  Sliding windows with hop_length:                  │
│  [═══════]                    ← Window 0           │
│      [═══════]                ← Window 1           │
│          [═══════]            ← Window 2           │
│              [═══════]        ← Window 3           │
│  ↔                                                 │
│  hop_length                                        │
│                                                    │
│  Each window → FFT → One column of spectrogram     │
└────────────────────────────────────────────────────┘
```

### STFT Parameters

| Parameter | Symbol | Typical Values | Effect |
|-----------|--------|----------------|--------|
| **n_fft** | N | 512, 1024, 2048 | Frequency resolution |
| **hop_length** | H | N/4 | Time resolution |
| **win_length** | W | = n_fft | Window size |

### Output Dimensions

```python
# Input: audio of length T samples
# Output: complex spectrogram

num_freq_bins = n_fft // 2 + 1
num_time_frames = (T - n_fft) // hop_length + 1

# Example: 1 second @ 16 kHz, n_fft=512, hop_length=160
num_freq_bins = 257
num_time_frames = (16000 - 512) // 160 + 1 = 97

# Shape: [257, 97] (complex64) = 50 KB per second
```

### Memory Analysis

```
Raw audio (1 sec, 16 kHz, 16-bit): 32 KB
Spectrogram (n_fft=512): 257 × 97 × 8 bytes = 200 KB (6x larger!)
Mel spectrogram (80 bins): 80 × 97 × 4 bytes = 31 KB (comparable)

Lesson: Mel spectrograms are memory-efficient representations
```

---

## Window Functions

### Why Windows?

```
Problem: FFT assumes periodic signal
Reality: We analyze finite segments

Without windowing:
┌─────────────────┐
│ █████           │  Abrupt edges cause
│█████████████████│  spectral leakage
│                 │  (false frequencies)
└─────────────────┘

With Hann window:
┌─────────────────┐
│    ▄████▄       │  Smooth edges
│  ▄████████▄     │  reduce leakage
│▁▂▃▄▅▆▇██▇▆▅▄▃▂▁│
└─────────────────┘
```

### Common Windows

```python
import numpy as np

def hann_window(N):
    """Most common for audio ML"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

def hamming_window(N):
    """Slightly better sidelobe suppression"""
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

def blackman_window(N):
    """Best sidelobe suppression, wider main lobe"""
    n = np.arange(N)
    return 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08 * np.cos(4*np.pi*n/(N-1))
```

### Window Trade-offs

| Window | Main Lobe Width | Sidelobe Level | Use Case |
|--------|-----------------|----------------|----------|
| Rectangular | Narrow | -13 dB | Never for audio |
| Hann | Medium | -31 dB | **Default for ML** |
| Hamming | Medium | -42 dB | ASR features |
| Blackman | Wide | -58 dB | High dynamic range |

---

## Phase and Magnitude

### Complex STFT Output

```python
# STFT output is complex: X = |X| · e^(jφ)

stft_output = librosa.stft(audio)  # Complex64

magnitude = np.abs(stft_output)    # |X| - WHAT frequencies
phase = np.angle(stft_output)      # φ - WHERE in cycle

# For most ML: we only use magnitude!
# Phase is hard to model and often ignored
```

### Power vs Amplitude Spectrogram

```python
amplitude_spec = np.abs(stft_output)
power_spec = np.abs(stft_output) ** 2

# dB conversion (log scale - matches human perception):
db_spec = 20 * np.log10(amplitude_spec + 1e-10)  # amplitude
db_spec = 10 * np.log10(power_spec + 1e-10)      # power (same result)
```

### Phase Reconstruction Problem

```
For audio generation, we need to reconstruct waveform from spectrogram.
But we usually only predict magnitude!

Solutions:
1. Griffin-Lim algorithm (iterative, slow, imperfect)
2. Neural vocoders (WaveNet, HiFi-GAN) - learn to generate waveform
3. Complex spectrogram prediction (harder to train)
4. End-to-end models (neural codecs) - bypass spectrograms entirely
```

---

## Practical Considerations

### Common Pitfalls

1. **Wrong sample rate assumption**
   ```python
   # ALWAYS check sample rate on load!
   audio, sr = librosa.load(path, sr=None)  # sr=None preserves original
   if sr != 16000:
       audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
   ```

2. **Forgetting to normalize**
   ```python
   # Neural networks expect [-1, 1] or zero-mean unit-variance
   audio = audio / np.max(np.abs(audio))  # Peak normalization
   # OR
   audio = (audio - audio.mean()) / audio.std()  # Statistical normalization
   ```

3. **Integer overflow in raw audio**
   ```python
   # int16 range: [-32768, 32767]
   # Always convert to float32 for processing
   audio_float = audio.astype(np.float32) / 32768.0
   ```

### Profiling Checklist

- [ ] Audio loading time vs processing time ratio
- [ ] Memory footprint: raw vs spectrogram vs mel-spectrogram
- [ ] Resampling overhead (should be 0 at inference!)
- [ ] FFT vs direct DFT (always use FFT)
- [ ] Batch processing vs sample-by-sample

---

## Next: Spectral Analysis Implementation

See `02_spectral_analysis_profiled.py` for profiled implementations of all concepts.
