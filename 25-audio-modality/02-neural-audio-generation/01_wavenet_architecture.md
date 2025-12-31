# WaveNet Architecture: The Foundation of Neural Audio

**Paper**: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (DeepMind, 2016)

WaveNet was revolutionary—the first neural network to generate high-quality raw audio waveforms. Every modern audio codec and vocoder traces back to WaveNet's innovations.

## Table of Contents
1. [The Challenge](#the-challenge)
2. [Core Architecture](#core-architecture)
3. [Dilated Causal Convolutions](#dilated-causal-convolutions)
4. [Gated Activation Units](#gated-activation-units)
5. [Residual and Skip Connections](#residual-and-skip-connections)
6. [Conditioning Mechanisms](#conditioning-mechanisms)
7. [Output Distribution](#output-distribution)
8. [The Speed Problem](#the-speed-problem)
9. [Legacy and Impact](#legacy-and-impact)

---

## The Challenge

### Audio Generation is HARD

```
1 second of audio at 16 kHz = 16,000 samples
Each sample depends on ALL previous samples

Compare to text:
- Average word: 5 characters
- 1 second of speech: ~3 words = 15 characters
- Audio is 1000x more samples for same content!

Traditional approach (concatenative TTS):
- Cut and paste pre-recorded audio segments
- Sounds robotic, limited vocabulary

Statistical parametric TTS:
- Model acoustic features, vocoder generates audio
- Better flexibility, but still unnatural
```

### WaveNet's Innovation

```
Model the raw waveform directly as:
P(x) = ∏ P(xₜ | x₁, x₂, ..., xₜ₋₁)

Each sample is predicted based on ALL previous samples.
Autoregressive generation, one sample at a time.
```

---

## Core Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     WAVENET ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: x[t-1], x[t-2], ..., x[1]  (previous samples)           │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────┐                                    │
│  │   Causal Convolution    │  (Initial embedding)               │
│  │   1×1 conv, μ-law input │                                    │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐ ──┐                                │
│  │   Residual Block 1      │   │                                │
│  │   (dilated conv, d=1)   │   │                                │
│  └───────────┬─────────────┘   │                                │
│              │                 │  Stack of                      │
│              ▼                 │  residual                      │
│  ┌─────────────────────────┐   │  blocks                        │
│  │   Residual Block 2      │   │                                │
│  │   (dilated conv, d=2)   │   │                                │
│  └───────────┬─────────────┘   │                                │
│              │                 │                                │
│              ▼                 │                                │
│            ...                 │                                │
│              │                 │                                │
│              ▼                 │                                │
│  ┌─────────────────────────┐   │                                │
│  │   Residual Block N      │   │                                │
│  │   (dilated conv, d=512) │ ──┘                                │
│  └───────────┬─────────────┘                                    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────┐     ┌────────────────────┐         │
│  │   Skip Connections      │ ──▶ │   Post-processing  │         │
│  │   (sum from all blocks) │     │   1×1 convs + ReLU │         │
│  └─────────────────────────┘     └──────────┬─────────┘         │
│                                             │                    │
│                                             ▼                    │
│                                  ┌────────────────────┐         │
│                                  │   Softmax (256)    │         │
│                                  │   → P(x[t])        │         │
│                                  └────────────────────┘         │
│                                                                  │
│  Output: Probability distribution over 256 μ-law values         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dilated Causal Convolutions

### The Receptive Field Problem

```
Standard convolution with kernel size 3:

Layer 1: Each output sees 3 inputs
Layer 2: Each output sees 5 inputs  (3 + 2)
Layer 3: Each output sees 7 inputs  (5 + 2)

To get receptive field of 16,000 (1 second @ 16kHz):
Need ~8,000 layers! ❌ Not practical

Solution: DILATED convolutions
```

### Dilated Convolution Explained

```
Dilation rate d = spacing between kernel elements

d=1 (normal):   ●─●─●     (looks at indices 0, 1, 2)
d=2:            ●───●───●  (looks at indices 0, 2, 4)
d=4:            ●───────●───────●  (looks at indices 0, 4, 8)

Receptive field grows EXPONENTIALLY with depth!

Layer 1 (d=1):    3 samples
Layer 2 (d=2):    7 samples
Layer 3 (d=4):    15 samples
Layer 4 (d=8):    31 samples
...
Layer 10 (d=512): 2047 samples

Just 10 layers for ~2000 samples receptive field!
```

### Causal Constraint

```
CAUSAL: Output at time t only depends on inputs at times ≤ t

Non-causal (bidirectional):
     ●───●───●───●───●
         ↓
     output[t] uses future samples ❌

Causal (WaveNet):
●───●───●
    ↓
output[t] only uses past samples ✓

Implementation: Pad on the LEFT, not symmetrically
```

### WaveNet's Dilation Pattern

```
Standard pattern: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

Receptive field = Σ(d) × (kernel_size - 1) + 1
                = 1023 × 2 + 1 = 2047 samples

Repeat this stack multiple times:
- 4 stacks × 10 layers = 40 residual blocks
- Receptive field: ~8000 samples (0.5 sec @ 16kHz)
```

---

## Gated Activation Units

### Why Gating?

```
Standard ReLU: y = ReLU(conv(x))
- Works for images
- Audio has complex temporal dynamics

LSTM-inspired gating: learn WHAT to pass through

WaveNet gating:
z = tanh(W_f * x) ⊙ σ(W_g * x)

Where:
- tanh(W_f * x): "filter" - what information
- σ(W_g * x): "gate" - how much to let through
- ⊙: element-wise multiplication
```

### Gated Activation Unit

```
                Input x
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   ┌─────────┐           ┌─────────┐
   │ Dilated │           │ Dilated │
   │ Conv Wf │           │ Conv Wg │
   └────┬────┘           └────┬────┘
        │                     │
        ▼                     ▼
   ┌─────────┐           ┌─────────┐
   │  tanh   │           │ sigmoid │
   └────┬────┘           └────┬────┘
        │                     │
        └─────────┬───────────┘
                  │
                  ▼
               ⊙ (multiply)
                  │
                  ▼
             Gated Output
```

---

## Residual and Skip Connections

### Residual Connections

```
Enable deeper networks by providing gradient shortcuts.

     Input x
        │
        ├───────────────┐
        │               │
        ▼               │
  ┌───────────┐         │
  │  Gated    │         │
  │  Block    │         │
  └─────┬─────┘         │
        │               │
        ▼               │
  ┌───────────┐         │
  │ 1×1 Conv  │         │
  └─────┬─────┘         │
        │               │
        ▼               │
       (+) ◄────────────┘
        │
        ▼
   To next layer
```

### Skip Connections

```
Collect features from ALL layers for final prediction.
Each layer contributes its "view" of the input.

Skip connections aggregate information at different time scales:
- Early layers: fine-grained local features
- Later layers: coarse long-range dependencies

┌─────────┐     ┌─────────┐     ┌─────────┐
│ Layer 1 │     │ Layer 2 │     │ Layer N │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │    skip       │    skip       │    skip
     └───────────┬───┴───────────┬───┴───────────┐
                 │               │               │
                 └───────────────┴───────────────┘
                               │
                               ▼
                       ┌───────────┐
                       │   SUM     │
                       └─────┬─────┘
                             │
                             ▼
                      Post-processing
```

---

## Conditioning Mechanisms

### Global Conditioning

```
Same condition for entire sequence (e.g., speaker ID)

Condition c → Embedding → Broadcast across time

z = tanh(W_f * x + V_f * c) ⊙ σ(W_g * x + V_g * c)

Use case: Multi-speaker TTS
- Embed speaker ID
- Add to every gated unit
- Single WaveNet generates multiple voices
```

### Local Conditioning

```
Time-varying condition (e.g., linguistic features, mel spectrogram)

Condition h (at lower sample rate) → Upsample → Add to each sample

z = tanh(W_f * x + V_f * h) ⊙ σ(W_g * x + V_g * h)

Use case: Text-to-Speech
- Text → Mel spectrogram (slower rate, e.g., 100 Hz)
- Upsample mel to audio rate (e.g., 16 kHz)
- WaveNet generates audio conditioned on mel
```

### Upsampling for Local Conditioning

```
Mel spectrogram: 100 frames/sec
Audio: 16,000 samples/sec
Ratio: 160:1

Options:
1. Repeat each frame 160 times
2. Transposed convolution (learnable upsampling)
3. Interpolation + refinement network

WaveNet uses transposed convolution stacks.
```

---

## Output Distribution

### μ-law Quantization

```
Raw audio is continuous. Neural networks output discrete distributions.

μ-law companding (μ=255):
F(x) = sign(x) · ln(1 + μ|x|) / ln(1 + μ)

Benefits:
- 256 discrete levels (8-bit)
- Logarithmic: more levels for quiet sounds
- Matches human perception
- Enables softmax output

Training: Cross-entropy loss on 256-class classification
```

### Softmax Output

```
Output layer: 256-way softmax

P(x_t = k | x_<t) = softmax(W · features)_k

For each timestep:
- Predict probability of each of 256 μ-law values
- Sample from distribution during generation
- Use argmax for greedy decoding
```

### Mixture of Logistics (Later Enhancement)

```
Discretized Mixture of Logistics (PixelCNN++, also used in WaveNet variants):

P(x) = Σ πᵢ · Logistic(μᵢ, sᵢ)

Benefits:
- Continuous output (no quantization)
- Better gradient flow
- Used in Parallel WaveNet and later vocoders
```

---

## The Speed Problem

### Autoregressive Generation is SLOW

```
Generation process:
1. Predict P(x_1)
2. Sample x_1
3. Predict P(x_2 | x_1)
4. Sample x_2
...
16,000 forward passes for 1 second of audio!

Reported inference time (2016):
- 90 minutes to generate 1 second of audio
- Completely impractical for real-time
```

### Why Can't We Parallelize?

```
Each sample depends on the previous:
x_t = f(x_1, x_2, ..., x_{t-1})

Can't compute x_5 without knowing x_1 through x_4.

This is the FUNDAMENTAL limitation of autoregressive models.
Solved by: Parallel WaveNet, Flow-based models, GANs
```

---

## Legacy and Impact

### Direct Descendants

| Model | Year | Key Innovation |
|-------|------|----------------|
| Parallel WaveNet | 2017 | Knowledge distillation for real-time |
| WaveRNN | 2018 | Efficient single-layer RNN |
| WaveGlow | 2018 | Flow-based, parallel generation |
| MelGAN | 2019 | GAN-based, 100x faster than WaveNet |
| HiFi-GAN | 2020 | Multi-scale discriminators |

### Architectural Contributions to Codecs

```
WaveNet innovations used in neural codecs:

SoundStream (2021):
├── Dilated convolutions in encoder/decoder
├── Residual connections
└── Gated activations

EnCodec (2022):
├── Similar convolutional architecture
├── Skip connections
└── LSTM for temporal modeling

Mimi (2024):
├── Builds on EnCodec
├── Transformer attention (replacing some convolutions)
└── Semantic token injection from WavLM
```

### Key Lessons

```
1. Raw waveform modeling IS possible with deep learning
2. Dilated convolutions enable large receptive fields efficiently
3. Gating mechanisms improve temporal modeling
4. Autoregressive = high quality but slow
5. Conditioning mechanisms enable controllable generation
```

---

## Next Steps

- `02_wavenet_implementation.py` - PyTorch implementation with profiling
- `03_parallel_wavenet.md` - How to make WaveNet real-time
- `04_melgan_hifigan.md` - GAN-based alternatives
