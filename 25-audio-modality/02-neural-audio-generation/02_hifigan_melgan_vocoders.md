# HiFi-GAN and MelGAN: GAN-Based Vocoders

**Papers**:
- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) (2019)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) (2020)

GAN-based vocoders that replaced WaveNet for real-time audio generation.

## Table of Contents
1. [Why GANs for Audio?](#why-gans-for-audio)
2. [MelGAN Architecture](#melgan-architecture)
3. [HiFi-GAN Architecture](#hifigan-architecture)
4. [Multi-Period Discriminator](#multi-period-discriminator)
5. [Multi-Scale Discriminator](#multi-scale-discriminator)
6. [Training Details](#training-details)
7. [Comparison](#comparison)
8. [Code Implementation](#code-implementation)

---

## Why GANs for Audio?

### The Speed Problem

```
WaveNet (2016):
├── Autoregressive: generate one sample at a time
├── 16,000 forward passes for 1 second @ 16kHz
├── 90 minutes to generate 1 second
└── Impossible for real-time use

GAN Solution:
├── Non-autoregressive: generate all samples in parallel
├── Single forward pass for entire audio
├── ~10ms to generate 1 second
└── Real-time on CPU!
```

### GAN for Waveform Generation

```
Traditional GAN:
├── Generator: random noise → image
├── Discriminator: real vs fake classification

Vocoder GAN:
├── Generator: mel spectrogram → waveform
├── Discriminator: real vs fake waveform
├── Conditional generation (not random)
└── Focus on perceptual quality
```

---

## MelGAN Architecture

### Generator

```
┌─────────────────────────────────────────────────────────────────┐
│                    MELGAN GENERATOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Mel spectrogram (80, T)                                 │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Conv1d(80, 512, k=7, p=3)    │                            │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│  For each upsample ratio in [8, 8, 2, 2]:                       │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Upsample Block:               │                            │
│  │   - LeakyReLU(0.2)             │                            │
│  │   - ConvTranspose1d(C, C/2)    │                            │
│  │   - 4× ResBlock(C/2, k=[3,7,11]) │                          │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   LeakyReLU + Conv1d(32, 1)    │                            │
│  │   Tanh activation               │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  Output: Waveform (1, T × 256)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total upsampling: 8 × 8 × 2 × 2 = 256 (matches hop length)
```

### Multi-Scale Discriminator

```
MelGAN introduced the multi-scale discriminator:

┌─────────────────────────────────────────────────────────────────┐
│              MULTI-SCALE DISCRIMINATOR (MSD)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input waveform x                                               │
│         │                                                        │
│         ├──────────────────┬──────────────────┐                 │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│     D₁(x)           D₂(↓₂x)           D₃(↓₄x)                  │
│   (original)      (downsample 2x)   (downsample 4x)            │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                 │
│                            │                                     │
│                   Multi-scale features                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Each D_k is a stack of strided convolutions.
Downsampling: average pooling.
```

### Key MelGAN Innovations

```
1. WEIGHT NORMALIZATION
   - Applied to all conv layers
   - More stable training than batch norm

2. NO NORMALIZATION IN GENERATOR
   - Allows for better fine-grained control
   - Avoids "washing out" artifacts

3. FEATURE MATCHING LOSS
   - Match discriminator intermediate features
   - Stabilizes training significantly
```

---

## HiFi-GAN Architecture

### Generator

```
HiFi-GAN uses Multi-Receptive Field Fusion (MRF):

┌─────────────────────────────────────────────────────────────────┐
│                    HIFI-GAN GENERATOR                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Mel spectrogram                                         │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Conv1d(80, 512, k=7)         │                            │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│  For upsample in [8, 8, 2, 2]:                                  │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   LeakyReLU                     │                            │
│  │   ConvTranspose1d (upsample)    │                            │
│  │   MRF Block (multiple kernels)  │  ← Key innovation         │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   LeakyReLU + Conv1d → Tanh    │                            │
│  └─────────────────────────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Receptive Field Fusion (MRF)

```python
class MRFBlock(nn.Module):
    """
    Multi-Receptive Field Fusion block.
    
    Processes input with multiple kernel sizes and dilations,
    then sums the outputs. Captures patterns at different scales.
    """
    
    def __init__(self, channels: int,
                 kernel_sizes: List[int] = [3, 7, 11],
                 dilations: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]]):
        super().__init__()
        
        self.resblocks = nn.ModuleList()
        for k, d_list in zip(kernel_sizes, dilations):
            self.resblocks.append(
                ResBlockStack(channels, kernel_size=k, dilations=d_list)
            )
    
    def forward(self, x):
        # Sum outputs from all kernel sizes
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)
        return output / len(self.resblocks)


class ResBlockStack(nn.Module):
    """Stack of residual blocks with same kernel, different dilations."""
    
    def __init__(self, channels: int, kernel_size: int, 
                 dilations: List[int]):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, 
                              padding=(kernel_size * dilation - dilation) // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size,
                              padding=(kernel_size - 1) // 2),
                )
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x
```

---

## Multi-Period Discriminator

### Key HiFi-GAN Innovation

```
┌─────────────────────────────────────────────────────────────────┐
│            MULTI-PERIOD DISCRIMINATOR (MPD)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Insight: Audio has periodic structure at multiple scales       │
│           Reshape 1D waveform to 2D and apply 2D convolutions   │
│                                                                  │
│  Period 2:   [a,b,c,d,e,f,g,h] → [[a,b], [c,d], [e,f], [g,h]]  │
│  Period 3:   [a,b,c,d,e,f] → [[a,b,c], [d,e,f]]                 │
│  Period 5:   [a,b,c,d,e,f,g,h,i,j] → [[a,b,c,d,e], [f,g,h,i,j]]│
│  ...                                                             │
│                                                                  │
│  Use primes to avoid overlap: [2, 3, 5, 7, 11]                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class PeriodDiscriminator(nn.Module):
    """Single period discriminator."""
    
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        
        # 2D convolutions after reshaping
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)),
        ])
    
    def forward(self, x):
        # Reshape to 2D: (B, 1, T) → (B, 1, T/period, period)
        b, c, t = x.shape
        
        # Pad to make divisible by period
        if t % self.period != 0:
            pad_size = self.period - (t % self.period)
            x = F.pad(x, (0, pad_size), mode='reflect')
            t = t + pad_size
        
        # Reshape
        x = x.view(b, c, t // self.period, self.period)
        
        # Apply 2D convolutions
        features = []
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.convs[-1](x)
        features.append(x)
        
        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    """MPD with multiple periods."""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, x):
        outputs = []
        features = []
        for d in self.discriminators:
            out, feat = d(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features
```

---

## Multi-Scale Discriminator

### Combined with MPD in HiFi-GAN

```python
class ScaleDiscriminator(nn.Module):
    """Single scale discriminator (1D convolutions)."""
    
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
            nn.Conv1d(1024, 1, 3, 1, padding=1),
        ])
    
    def forward(self, x):
        features = []
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.convs[-1](x)
        features.append(x)
        return x.flatten(1, -1), features


class MultiScaleDiscriminator(nn.Module):
    """MSD with multiple scales."""
    
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)
    
    def forward(self, x):
        outputs = []
        features = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling(x)
            out, feat = d(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features
```

---

## Training Details

### Loss Functions

```python
def discriminator_loss(disc_real, disc_fake):
    """
    Least squares GAN loss for discriminator.
    """
    loss = 0
    for dr, df in zip(disc_real, disc_fake):
        loss += torch.mean((dr - 1) ** 2)  # Real should be 1
        loss += torch.mean(df ** 2)         # Fake should be 0
    return loss


def generator_loss(disc_fake):
    """
    Generator adversarial loss.
    """
    loss = 0
    for df in disc_fake:
        loss += torch.mean((df - 1) ** 2)  # Fool disc: fake should be 1
    return loss


def feature_matching_loss(features_real, features_fake):
    """
    L1 loss between discriminator features.
    """
    loss = 0
    for fr_list, ff_list in zip(features_real, features_fake):
        for fr, ff in zip(fr_list, ff_list):
            loss += F.l1_loss(ff, fr.detach())
    return loss


def mel_spectrogram_loss(audio, audio_hat, mel_transform):
    """
    L1 loss on mel spectrograms.
    """
    mel_real = mel_transform(audio)
    mel_fake = mel_transform(audio_hat)
    return F.l1_loss(mel_fake, mel_real)
```

### Training Configuration

```
HiFi-GAN:
├── Optimizer: AdamW (β₁=0.8, β₂=0.99)
├── Learning rate: 2e-4 (generator), 2e-4 (discriminator)
├── LR decay: 0.999 per epoch
├── Batch size: 16
├── Segment length: 8192 samples
├── Training: 2500 epochs on LJSpeech
└── λ_fm = 2, λ_mel = 45
```

---

## Comparison

### Quality vs Speed Tradeoff

| Model | MOS | RTF (CPU) | RTF (GPU) | Params |
|-------|-----|-----------|-----------|--------|
| WaveNet | 4.5 | 0.001 | 0.02 | 25M |
| MelGAN | 4.0 | 10x | 100x | 4.3M |
| HiFi-GAN v1 | 4.5 | 15x | 200x | 13.9M |
| HiFi-GAN v2 | 4.4 | 40x | 400x | 0.9M |
| HiFi-GAN v3 | 4.3 | 80x | 600x | 1.5M |

### When to Use What

```
MelGAN:
├── Fastest option
├── Good for real-time on limited hardware
└── Acceptable for TTS, not music

HiFi-GAN v1:
├── Best quality
├── Still real-time on modern hardware
└── Recommended for most use cases

HiFi-GAN v2/v3:
├── Smaller models
├── Better for edge deployment
└── Slight quality tradeoff
```

---

## Code Implementation

### Minimal HiFi-GAN Generator

```python
class HiFiGANGenerator(nn.Module):
    def __init__(self, 
                 in_channels: int = 80,
                 upsample_rates: List[int] = [8, 8, 2, 2],
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                 resblock_kernel_sizes: List[int] = [3, 7, 11],
                 resblock_dilations: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]],
                 initial_channels: int = 512):
        super().__init__()
        
        self.num_upsamples = len(upsample_rates)
        
        # Initial conv
        self.conv_pre = nn.Conv1d(in_channels, initial_channels, 7, 1, 3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        ch = initial_channels
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
            )
            self.mrfs.append(
                MRFBlock(ch // 2, resblock_kernel_sizes, resblock_dilations)
            )
            ch = ch // 2
        
        # Output conv
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3)
    
    def forward(self, x):
        x = self.conv_pre(x)
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
```

---

## Key Takeaways

```
1. GAN VOCODERS ARE FAST
   - Non-autoregressive generation
   - Single forward pass for entire audio
   - Real-time on CPU

2. MULTI-SCALE/PERIOD DISCRIMINATORS
   - Capture different audio structures
   - MPD for periodic patterns (pitch)
   - MSD for overall quality

3. FEATURE MATCHING STABILIZES TRAINING
   - Match discriminator features
   - More stable than adversarial loss alone

4. MRF IMPROVES QUALITY
   - Multiple kernel sizes/dilations
   - Captures patterns at different scales

5. MEL SPECTROGRAM LOSS HELPS
   - Perceptually meaningful
   - Stabilizes training
```

---

## Further Reading

- `01_wavenet_architecture.md` - The original neural vocoder
- `../03-neural-audio-codecs/` - How these ideas inform codecs
- HiFi-GAN source: github.com/jik876/hifi-gan
