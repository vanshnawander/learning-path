# SoundStream and EnCodec Architecture Deep Dive

**Papers**:
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) (Google, 2021)
- [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) (Meta, 2022)

Detailed architectural analysis of the two foundational neural audio codecs.

## Table of Contents
1. [SoundStream Architecture](#soundstream-architecture)
2. [EnCodec Architecture](#encodec-architecture)
3. [Key Differences](#key-differences)
4. [Discriminator Architectures](#discriminator-architectures)
5. [Loss Functions](#loss-functions)
6. [Training Details](#training-details)
7. [Implementation Notes](#implementation-notes)

---

## SoundStream Architecture

### Overview

```
SoundStream (Google, 2021):
├── First end-to-end neural audio codec
├── Introduced RVQ for audio
├── Variable bitrate: 3-18 kbps
├── Real-time on mobile devices
└── Works for speech, music, general audio
```

### Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                  SOUNDSTREAM ENCODER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 24 kHz mono waveform                                    │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Conv1d(1, C, k=7, s=1)       │  C = 32                    │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│  For each stride in [2, 4, 5, 8]:                               │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   3× ResBlock(C, dilation=[1,3,9])                          │
│  │   ELU activation                │                            │
│  │   Conv1d(C, 2C, k=2s, stride=s) │  Downsample               │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   3× ResBlock(C_final)          │                            │
│  │   Conv1d(C_final, D, k=3)       │  D = 128 (latent dim)     │
│  └─────────────────────────────────┘                            │
│                                                                  │
│  Output: 75 Hz latent (24000 / 320 = 75)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total stride: 2 × 4 × 5 × 8 = 320
Frame rate: 24000 / 320 = 75 Hz
```

### Residual Block Design

```python
class SoundStreamResBlock(nn.Module):
    """
    Residual block with dilated convolutions.
    
    Dilation pattern [1, 3, 9] gives receptive field of 27 samples
    per block without increasing parameters.
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=3, 
                      dilation=dilation, padding=dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
    
    def forward(self, x):
        return x + self.block(x)
```

### Decoder

```
Mirror of encoder with transposed convolutions:

Latent (75 Hz) → Upsample blocks → Waveform (24 kHz)

Strides: [8, 5, 4, 2] (reverse of encoder)
Each block: ConvTranspose1d + ResBlocks
```

---

## EnCodec Architecture

### Overview

```
EnCodec (Meta, 2022):
├── Open-source (github.com/facebookresearch/encodec)
├── Improved discriminator (multi-scale STFT)
├── Balancer for loss weighting
├── Stereo support
├── 24 kHz and 48 kHz models
└── Bitrates: 1.5-24 kbps
```

### Key Architectural Differences from SoundStream

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENCODEC INNOVATIONS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LSTM LAYER (temporal modeling)                              │
│     ┌──────────────────────┐                                    │
│     │  After final conv:   │                                    │
│     │  LSTM(D, D, layers=2)│                                    │
│     │  Bidirectional       │                                    │
│     └──────────────────────┘                                    │
│     Captures long-range temporal dependencies                   │
│                                                                  │
│  2. SNAKE ACTIVATION (instead of ELU)                           │
│     snake(x) = x + sin²(αx) / α                                 │
│     Better for periodic signals (audio)                         │
│                                                                  │
│  3. WEIGHT NORMALIZATION                                        │
│     All conv layers use weight norm                             │
│     More stable training                                        │
│                                                                  │
│  4. MULTI-SCALE STFT DISCRIMINATOR                             │
│     Multiple STFT resolutions                                   │
│     Complex-valued discrimination                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### EnCodec Encoder Architecture

```python
class EnCodecEncoder(nn.Module):
    """Simplified EnCodec encoder."""
    
    def __init__(
        self,
        channels: int = 32,
        latent_dim: int = 128,
        strides: List[int] = [2, 4, 5, 8],
    ):
        super().__init__()
        
        layers = []
        
        # Initial conv
        layers.append(
            nn.Conv1d(1, channels, kernel_size=7, padding=3)
        )
        
        # Encoder blocks
        in_ch = channels
        for stride in strides:
            out_ch = min(in_ch * 2, 512)
            
            # Residual blocks
            for dilation in [1, 3, 9]:
                layers.append(ResBlock(in_ch, dilation=dilation))
            
            # Downsample
            layers.append(Snake(in_ch))  # Snake activation
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=stride*2, 
                          stride=stride, padding=stride//2)
            )
            in_ch = out_ch
        
        # Final residual blocks
        for dilation in [1, 3, 9]:
            layers.append(ResBlock(in_ch, dilation=dilation))
        
        # LSTM for temporal modeling
        self.conv_layers = nn.Sequential(*layers)
        self.lstm = nn.LSTM(in_ch, in_ch, num_layers=2, 
                            bidirectional=True, batch_first=True)
        self.proj = nn.Conv1d(in_ch * 2, latent_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        
        # LSTM (needs B, T, C format)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        
        return self.proj(x)
```

### Snake Activation

```python
class Snake(nn.Module):
    """
    Snake activation: x + sin²(αx) / α
    
    Better than ReLU/ELU for periodic signals.
    Learned α per channel.
    """
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha_init)
    
    def forward(self, x):
        return x + (torch.sin(self.alpha * x) ** 2) / self.alpha
```

---

## Key Differences

### Architecture Comparison

| Feature | SoundStream | EnCodec |
|---------|-------------|---------|
| Activation | ELU | Snake |
| Temporal | None | LSTM |
| Normalization | None | Weight norm |
| Discriminator | MSD | MSD + MSSTFT |
| Stereo | No | Yes |
| Open source | No | Yes |

### Quantization

```
Both use RVQ with similar configuration:

SoundStream:
├── Codebook size: 1024 (10 bits)
├── Codebook dim: 128
├── 3-12 quantizers (3-12 kbps)
└── EMA codebook update

EnCodec:
├── Codebook size: 1024 (10 bits)  
├── Codebook dim: 128
├── 2-32 quantizers (1.5-24 kbps)
└── EMA codebook update + commitment loss
```

---

## Discriminator Architectures

### Multi-Scale Discriminator (MSD)

```
Used by both SoundStream and EnCodec:

┌───────────────────────────────────────────────────────────────┐
│              MULTI-SCALE DISCRIMINATOR                         │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Scale 1: D(x)           Original waveform                    │
│  Scale 2: D(↓2(x))       Downsampled 2x                       │
│  Scale 3: D(↓4(x))       Downsampled 4x                       │
│                                                                │
│  Each scale: Stack of strided convolutions                    │
│              Output: multi-resolution features                │
│                                                                │
└───────────────────────────────────────────────────────────────┘

Purpose: Capture both fine and coarse audio structure
```

### Multi-Period Discriminator (MPD)

```
From HiFi-GAN, used in some variants:

┌───────────────────────────────────────────────────────────────┐
│              MULTI-PERIOD DISCRIMINATOR                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Reshape waveform into 2D with different periods:             │
│                                                                │
│  Period 2:  [.......] → [[..][..][..][..]]                   │
│  Period 3:  [.......] → [[...][...][..]]                     │
│  Period 5:  [.......] → [[.....][..]]                        │
│  Period 7:  [.......] → [[.......]]                          │
│  Period 11: [.......] → [[...........]]                      │
│                                                                │
│  Apply 2D convolutions to each                                │
│  Captures different periodic structures                       │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Multi-Scale STFT Discriminator (EnCodec)

```python
class MultiScaleSTFTDiscriminator(nn.Module):
    """
    EnCodec's discriminator operates on STFT.
    
    Key insight: Discriminate in frequency domain
    Catches spectral artifacts that waveform discriminators miss.
    """
    
    def __init__(self, 
                 n_ffts: List[int] = [1024, 2048, 512],
                 hop_lengths: List[int] = [256, 512, 128]):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft, hop)
            for n_fft, hop in zip(n_ffts, hop_lengths)
        ])
    
    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
        return outputs


class STFTDiscriminator(nn.Module):
    """Single-scale STFT discriminator."""
    
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 2D conv on complex STFT
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, (3, 9), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, (3, 3), padding=(1, 1)),
        )
    
    def forward(self, x):
        # Compute STFT
        stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length,
                          return_complex=True)
        # Stack real and imaginary as channels
        stft = torch.stack([stft.real, stft.imag], dim=1)
        return self.conv(stft)
```

---

## Loss Functions

### Reconstruction Losses

```python
def reconstruction_loss(x, x_hat):
    """
    Multi-scale spectral reconstruction loss.
    """
    loss = 0
    
    # Time domain L1
    loss += F.l1_loss(x, x_hat)
    
    # Multi-scale STFT
    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        
        x_stft = torch.stft(x.squeeze(1), n_fft, hop, return_complex=True)
        x_hat_stft = torch.stft(x_hat.squeeze(1), n_fft, hop, return_complex=True)
        
        # Magnitude L1
        loss += F.l1_loss(x_stft.abs(), x_hat_stft.abs())
        
        # Log magnitude L2
        loss += F.mse_loss(
            torch.log(x_stft.abs() + 1e-5),
            torch.log(x_hat_stft.abs() + 1e-5)
        )
    
    return loss
```

### Adversarial Losses

```python
def discriminator_loss(real_outputs, fake_outputs):
    """Hinge loss for discriminator."""
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean(F.relu(1 - real))
        loss += torch.mean(F.relu(1 + fake))
    return loss

def generator_adversarial_loss(fake_outputs):
    """Generator tries to fool discriminator."""
    loss = 0
    for fake in fake_outputs:
        loss += -torch.mean(fake)
    return loss

def feature_matching_loss(real_features, fake_features):
    """
    Match intermediate features from discriminator.
    Stabilizes GAN training.
    """
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss
```

### EnCodec Balancer

```python
class Balancer:
    """
    Automatically balance loss weights based on gradient magnitudes.
    
    Key EnCodec contribution for stable training.
    """
    def __init__(self, weights: dict, rescale_grads: bool = True):
        self.weights = weights
        self.rescale_grads = rescale_grads
        self.ema_grads = {}
    
    def backward(self, losses: dict, model_output: torch.Tensor):
        """
        Compute balanced gradients.
        """
        # Compute gradient magnitude for each loss
        grad_norms = {}
        for name, loss in losses.items():
            grad = torch.autograd.grad(
                loss, model_output, retain_graph=True
            )[0]
            grad_norms[name] = grad.norm()
        
        # Balance based on gradient magnitudes
        total_norm = sum(grad_norms.values())
        
        balanced_loss = 0
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            if self.rescale_grads:
                # Rescale to have equal gradient contribution
                scale = total_norm / (grad_norms[name] + 1e-8)
                balanced_loss += weight * scale * loss
            else:
                balanced_loss += weight * loss
        
        return balanced_loss
```

---

## Training Details

### Training Configuration

```
SoundStream:
├── Batch size: 64
├── Learning rate: 3e-4 (Adam)
├── Training steps: 400k
├── Discriminator: 1 update per generator update
└── EMA decay: 0.99 for codebook

EnCodec:
├── Batch size: 64
├── Learning rate: 3e-4 (Adam)
├── Training steps: 300 epochs
├── Balancer: Automatic loss weighting
├── Audio length: 1 second during training
└── Discriminator warmup: 20k steps
```

### Training Tips

```
1. CODEBOOK INITIALIZATION
   - Random init or kmeans from first batch
   - Monitor codebook usage (avoid dead entries)

2. DISCRIMINATOR WARMUP
   - Train generator alone for initial steps
   - Prevents discriminator from dominating

3. GRADIENT CLIPPING
   - Clip generator gradients to prevent explosion
   - Typical: max_norm=1.0

4. LEARNING RATE SCHEDULE
   - Constant or cosine decay
   - Discriminator can use lower LR

5. DATA AUGMENTATION
   - Random crops (1-2 seconds)
   - Gain normalization
   - Optional: noise injection
```

---

## Implementation Notes

### Memory Optimization

```python
# Use gradient checkpointing for large models
class CheckpointedEncoder(nn.Module):
    def forward(self, x):
        # Checkpoint each residual block
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x)
        return x
```

### Streaming Inference

```python
class StreamingCodec:
    """
    Process audio in chunks for real-time use.
    """
    def __init__(self, codec, chunk_size=320):
        self.codec = codec
        self.chunk_size = chunk_size
        self.buffer = []
    
    def encode_chunk(self, audio_chunk):
        self.buffer.extend(audio_chunk)
        
        if len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            
            with torch.no_grad():
                return self.codec.encode(chunk)
        return None
```

---

## Key Takeaways

```
1. ARCHITECTURE MATTERS
   - LSTM improves temporal coherence (EnCodec)
   - Snake activation better for periodic signals
   - Weight normalization stabilizes training

2. DISCRIMINATORS ARE CRUCIAL
   - Multi-scale catches different artifacts
   - STFT discriminator for spectral quality
   - Feature matching stabilizes training

3. LOSS BALANCING IS HARD
   - EnCodec's Balancer automates this
   - Without balance, one loss dominates

4. RVQ IS THE KEY INNOVATION
   - Enables high quality at low bitrate
   - Variable bitrate by using fewer levels

5. BOTH ARE PRODUCTION QUALITY
   - SoundStream: Google's internal use
   - EnCodec: Open source, widely adopted
```

---

## Further Reading

- `05_mimi_codec.md` - Next-generation codec for LLMs
- `../09-codec-benchmarks/` - Quantitative comparisons
- EnCodec source code: github.com/facebookresearch/encodec
