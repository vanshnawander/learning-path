# Music Generation with MusicGen and AudioCraft

**Paper**: [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) (Meta, 2023)
**Code**: [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)

MusicGen is Meta's text-to-music generation model, part of the AudioCraft ecosystem. Latest developments through 2025.

## Table of Contents
1. [MusicGen Overview](#musicgen-overview)
2. [Architecture](#architecture)
3. [Conditioning Mechanisms](#conditioning-mechanisms)
4. [Training Strategy](#training-strategy)
5. [AudioCraft Ecosystem](#audiocraft-ecosystem)
6. [Latest Developments (2024-2025)](#latest-developments-2024-2025)
7. [Practical Usage](#practical-usage)
8. [Comparison with Competitors](#comparison-with-competitors)

---

## MusicGen Overview

### What Makes Music Generation Hard?

```
Music vs Speech challenges:

Speech:
├── Relatively simple structure
├── Limited frequency range (300-3400 Hz for telephony)
├── Phonetic units well-defined
└── Shorter context needed

Music:
├── Complex harmonic structure
├── Full frequency range (20-20000 Hz)
├── Multiple instruments simultaneously
├── Long-term structure (verse, chorus, bridge)
├── Rhythm and tempo consistency
└── Emotional/stylistic coherence
```

### MusicGen's Approach

```
Key innovations:
├── Single-stage transformer (no cascading)
├── Efficient token interleaving pattern
├── Text and melody conditioning
├── 32 kHz high-quality audio
└── Up to 30 seconds generation
```

---

## Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    MUSICGEN ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Text: "upbeat electronic dance music"                          │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │  T5 Text Encoder │  Frozen pretrained                        │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  Text embeddings (conditioning)                                 │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              TRANSFORMER DECODER                         │    │
│  │  - Causal attention                                      │    │
│  │  - Cross-attention to text                               │    │
│  │  - Predicts audio tokens                                 │    │
│  │  - 48 layers, 2048 dim (large model)                     │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        │                                         │
│                        ▼                                         │
│  Audio tokens (EnCodec, 4 RVQ levels)                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │  EnCodec Decoder │  32 kHz audio                             │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Token Interleaving Pattern

```
MusicGen uses DELAY PATTERN for RVQ tokens:

Standard (naive):
Time 0: [L0_0, L1_0, L2_0, L3_0]
Time 1: [L0_1, L1_1, L2_1, L3_1]
...
Problem: Long sequences, hard to model

MusicGen delay pattern:
Time 0: [L0_0, ___, ___, ___]
Time 1: [L0_1, L1_0, ___, ___]
Time 2: [L0_2, L1_1, L2_0, ___]
Time 3: [L0_3, L1_2, L2_1, L3_0]
...

Benefits:
├── Shorter sequences
├── Hierarchical modeling (coarse to fine)
├── Better quality
└── Faster training
```

### Delay Pattern Implementation

```python
def apply_delay_pattern(codes, num_codebooks=4):
    """
    Apply delay pattern to RVQ codes.
    
    Args:
        codes: (batch, num_codebooks, time)
    Returns:
        delayed_codes: (batch, time * num_codebooks)
    """
    B, K, T = codes.shape
    
    # Create delayed sequence
    delayed = []
    for t in range(T + K - 1):
        timestep_tokens = []
        for k in range(K):
            # Token from codebook k appears at time t-k
            if t >= k and t - k < T:
                timestep_tokens.append(codes[:, k, t - k])
            else:
                timestep_tokens.append(torch.zeros(B, dtype=torch.long))  # Padding
        delayed.append(torch.stack(timestep_tokens, dim=1))
    
    # Stack all timesteps
    delayed_codes = torch.cat(delayed, dim=1)
    
    return delayed_codes
```

---

## Conditioning Mechanisms

### Text Conditioning

```python
class TextConditioner(nn.Module):
    """
    Text conditioning using T5 embeddings.
    """
    def __init__(self, d_model=2048):
        super().__init__()
        from transformers import T5EncoderModel
        
        # Frozen T5 encoder
        self.t5 = T5EncoderModel.from_pretrained("t5-base")
        for param in self.t5.parameters():
            param.requires_grad = False
        
        # Project T5 embeddings to model dimension
        self.proj = nn.Linear(768, d_model)
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: Tokenized text
        Returns:
            text_embeddings: (batch, seq_len, d_model)
        """
        with torch.no_grad():
            t5_output = self.t5(text_tokens).last_hidden_state
        
        return self.proj(t5_output)
```

### Melody Conditioning

```
MusicGen can condition on melody:

Input melody: Audio file or MIDI
Process:
1. Extract chromagram (pitch content)
2. Encode with separate network
3. Add to transformer as additional conditioning

Use case: Generate arrangement of existing melody
Example: "Jazz version of [melody.mp3]"
```

### Classifier-Free Guidance

```python
def classifier_free_guidance(
    model,
    audio_tokens,
    text_condition,
    guidance_scale=3.0
):
    """
    Classifier-free guidance for better text adherence.
    
    Generate with and without conditioning, interpolate.
    """
    # Conditional generation
    logits_cond = model(audio_tokens, text_condition)
    
    # Unconditional generation (no text)
    logits_uncond = model(audio_tokens, text_condition=None)
    
    # Guided logits
    logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
    
    return logits
```

---

## Training Strategy

### Dataset

```
MusicGen training data:
├── 20,000 hours of licensed music
├── Diverse genres and styles
├── High-quality recordings
└── Text descriptions (manual + automatic)

Data sources:
├── ShutterStock music library
├── Pond5 music library
└── Internal Meta music data
```

### Training Configuration

```
Model sizes:
├── Small: 300M parameters
├── Medium: 1.5B parameters
└── Large: 3.3B parameters

Training:
├── Optimizer: AdamW
├── Learning rate: 1e-4
├── Batch size: 192 (across GPUs)
├── Sequence length: 30 seconds
├── Training time: Several weeks on cluster
└── Hardware: 64-96 A100 GPUs
```

### Loss Function

```python
def musicgen_loss(model, audio_tokens, text_condition):
    """
    Autoregressive cross-entropy loss.
    """
    # Apply delay pattern
    delayed_tokens = apply_delay_pattern(audio_tokens)
    
    # Shift for autoregressive prediction
    input_tokens = delayed_tokens[:, :-1]
    target_tokens = delayed_tokens[:, 1:]
    
    # Forward pass
    logits = model(input_tokens, text_condition)
    
    # Cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        target_tokens.view(-1),
        ignore_index=PAD_TOKEN
    )
    
    return loss
```

---

## AudioCraft Ecosystem

### Components

```
AudioCraft (Meta's audio generation suite):

1. EnCodec
   ├── Neural audio codec
   ├── 24 kHz and 48 kHz models
   └── Used by MusicGen for tokenization

2. MusicGen
   ├── Text-to-music generation
   ├── Melody conditioning
   └── Multiple model sizes

3. AudioGen
   ├── Text-to-audio (sound effects)
   ├── Environmental sounds
   └── Shares architecture with MusicGen

4. MAGNeT (2024)
   ├── Non-autoregressive music generation
   ├── Masked generation (like SoundStorm)
   └── 10x faster than MusicGen
```

### Using AudioCraft

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-large')

# Set generation parameters
model.set_generation_params(
    duration=30,  # seconds
    temperature=1.0,
    top_k=250,
    top_p=0.0,
    cfg_coef=3.0,  # Classifier-free guidance
)

# Generate from text
descriptions = [
    "upbeat electronic dance music with heavy bass",
    "calm acoustic guitar melody",
    "epic orchestral soundtrack"
]

wav = model.generate(descriptions)

# Save
for idx, one_wav in enumerate(wav):
    audio_write(
        f'generated_{idx}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness"
    )
```

---

## Latest Developments (2024-2025)

### MAGNeT (Masked Audio Generation using Non-autoregressive Transformers)

```
Released: 2024

Key improvements over MusicGen:
├── Non-autoregressive generation
├── 10x faster than MusicGen
├── Comparable quality
├── Iterative refinement (like SoundStorm)
└── Better for real-time applications

Architecture:
├── Masked token prediction
├── Rescoring mechanism
├── Span masking strategy
└── Single-stage generation
```

### MusicGen Stereo

```
Extension to stereo:
├── Stereo EnCodec
├── Left/right channel modeling
├── Spatial audio generation
└── Better immersion

Released: Late 2023
```

### Long-Form Music Generation

```
Challenge: Maintain coherence beyond 30 seconds

Solutions (2024-2025):
├── Hierarchical generation
├── Structure-aware models
├── Segment conditioning
└── Memory-augmented transformers

Current state:
├── Up to 5 minutes with coherence
├── Active research area
└── Production systems use stitching
```

---

## Practical Usage

### Basic Generation

```python
import torchaudio
from audiocraft.models import MusicGen

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Generate
descriptions = ["80s pop track with synthesizer"]
wav = model.generate(descriptions, progress=True)

# Save
torchaudio.save('output.wav', wav[0].cpu(), model.sample_rate)
```

### Melody Conditioning

```python
# Load melody
melody, sr = torchaudio.load('melody.mp3')

# Generate with melody conditioning
model.set_generation_params(duration=30)
wav = model.generate_with_chroma(
    descriptions=["jazz piano arrangement"],
    melody_wavs=melody,
    melody_sample_rate=sr,
    progress=True
)
```

### Continuation

```python
# Continue existing audio
prompt, sr = torchaudio.load('prompt.wav')

# Generate continuation
wav = model.generate_continuation(
    prompt=prompt,
    prompt_sample_rate=sr,
    descriptions=["continue in same style"],
    progress=True
)
```

---

## Comparison with Competitors

### Music Generation Landscape (2024-2025)

| Model | Company | Quality | Speed | Controllability | Open Source |
|-------|---------|---------|-------|-----------------|-------------|
| MusicGen | Meta | High | Medium | Good (text+melody) | Yes |
| MAGNeT | Meta | High | Fast | Good | Yes |
| MusicLM | Google | High | Slow | Excellent | No |
| Stable Audio | Stability AI | High | Fast | Good | Partial |
| Suno | Suno AI | Very High | Fast | Excellent | No |
| Udio | Udio | Very High | Medium | Excellent | No |

### Quality Metrics

```
Frechet Audio Distance (FAD):
├── MusicGen: 5.1 (lower is better)
├── MusicLM: 4.8
├── Riffusion: 8.2
└── Ground truth: 0.0

Subjective evaluation (MOS):
├── MusicGen: 3.9/5.0
├── MusicLM: 4.1/5.0
├── Suno v3: 4.3/5.0
└── Real music: 4.5/5.0
```

---

## Key Takeaways

```
1. MUSICGEN IS PRODUCTION-READY
   - Open source, well-documented
   - Multiple model sizes
   - Good quality/speed balance

2. DELAY PATTERN IS KEY
   - Efficient RVQ token modeling
   - Enables single-stage generation
   - Better than cascaded approaches

3. TEXT + MELODY CONDITIONING
   - Flexible control
   - Natural interface
   - Practical for users

4. AUDIOCRAFT ECOSYSTEM
   - Complete toolkit
   - EnCodec, MusicGen, AudioGen
   - Easy to use

5. ACTIVE DEVELOPMENT
   - MAGNeT for speed
   - Stereo support
   - Longer generation coming
```

---

## Further Reading

- AudioCraft documentation
- MusicGen paper
- MAGNeT paper (2024)
- `../03-neural-audio-codecs/03_soundstream_encodec_architecture.md`
