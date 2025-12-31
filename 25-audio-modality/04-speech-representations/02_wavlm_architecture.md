# WavLM: Universal Speech Representations (MUST READ)

**Paper**: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) (Microsoft, 2021)

WavLM is the foundation of semantic audio understanding. It's used in Mimi for semantic token distillation, and understanding WavLM is crucial for modern audio LLMs.

## Table of Contents
1. [Why Self-Supervised Speech?](#why-self-supervised-speech)
2. [Evolution: wav2vec → HuBERT → WavLM](#evolution-wav2vec--hubert--wavlm)
3. [WavLM Architecture](#wavlm-architecture)
4. [Masked Speech Prediction](#masked-speech-prediction)
5. [Denoising Pretraining](#denoising-pretraining)
6. [What WavLM Learns](#what-wavlm-learns)
7. [Using WavLM Features](#using-wavlm-features)
8. [Applications](#applications)

---

## Why Self-Supervised Speech?

### The Data Problem

```
Supervised speech learning requires:
├── Transcribed audio (expensive to create)
├── Speaker labels
├── Emotion annotations
└── Hours of human labeling effort

Available data:
├── Transcribed: ~100k hours (expensive)
├── Untranscribed: MILLIONS of hours (free!)

Solution: Learn from raw audio, no labels needed
         → Self-Supervised Learning (SSL)
```

### The Vision-Language Parallel

```
NLP:    BERT → GPT → Modern LLMs
        Self-supervised on text, then fine-tune

Vision: SimCLR → CLIP → Modern vision models
        Self-supervised on images, then fine-tune

Speech: wav2vec → HuBERT → WavLM → Modern audio
        Self-supervised on audio, then fine-tune

WavLM is the "BERT moment" for speech.
```

---

## Evolution: wav2vec → HuBERT → WavLM

### wav2vec 2.0 (Facebook, 2020)

```
Key innovation: Contrastive learning on speech

┌─────────────────────────────────────────────┐
│              wav2vec 2.0                     │
├─────────────────────────────────────────────┤
│                                              │
│  Audio ──▶ CNN Encoder ──▶ Quantizer ──▶ Targets
│                │                              │
│                ▼                              │
│        Transformer                           │
│                │                              │
│                ▼                              │
│        Contrastive Loss                      │
│        (match correct quantized target)      │
│                                              │
└─────────────────────────────────────────────┘

Problem: Quantization is learned jointly
         → Unstable training, mode collapse
```

### HuBERT (Facebook, 2021)

```
Key innovation: Offline clustering for targets

Step 1: Cluster audio features (k-means on MFCC/prior model)
Step 2: Use cluster IDs as pseudo-labels
Step 3: Predict masked cluster IDs

┌─────────────────────────────────────────────┐
│                 HuBERT                       │
├─────────────────────────────────────────────┤
│                                              │
│  Audio ──▶ CNN Encoder ──▶ [MASK] some frames
│                │                              │
│                ▼                              │
│        Transformer                           │
│                │                              │
│                ▼                              │
│   Predict cluster ID (cross-entropy loss)   │
│                                              │
│  Targets: k-means clusters (offline)        │
│                                              │
└─────────────────────────────────────────────┘

Advantage: Stable training, simpler objective
Problem: Optimized only for ASR-like tasks
```

### WavLM (Microsoft, 2021)

```
Key innovations:
1. Denoising objective (handles noisy speech)
2. Speaker preservation (learns speaker info)
3. Gated relative position bias (better attention)

┌─────────────────────────────────────────────┐
│                 WavLM                        │
├─────────────────────────────────────────────┤
│                                              │
│  Audio ──▶ Optional: Add noise/overlap      │
│     │                                        │
│     ▼                                        │
│  CNN Encoder ──▶ [MASK] some frames         │
│     │                                        │
│     ▼                                        │
│  Transformer (with gated rel pos bias)      │
│     │                                        │
│     ▼                                        │
│  Predict ORIGINAL cluster ID                │
│  (must denoise AND predict content)         │
│                                              │
└─────────────────────────────────────────────┘

Result: Best performance across ALL speech tasks
        - ASR, speaker ID, emotion, etc.
```

---

## WavLM Architecture

### Model Specifications

```
WavLM Base:
├── CNN encoder: 7 layers, 512 channels
├── Transformer: 12 layers, 768 dim, 8 heads
├── Parameters: 94.7M
└── Training: 960h LibriSpeech

WavLM Base+:
├── Same architecture as Base
├── Parameters: 94.7M
└── Training: 60k hours (Libri-Light + VoxPopuli)

WavLM Large:
├── CNN encoder: 7 layers, 512 channels
├── Transformer: 24 layers, 1024 dim, 16 heads
├── Parameters: 316M
└── Training: 94k hours
```

### CNN Feature Encoder

```python
# Converts raw waveform to frame-level features
class CNNEncoder(nn.Module):
    def __init__(self):
        # 7 convolutional layers
        # Total stride: 320 (20ms at 16kHz)
        self.layers = nn.Sequential(
            Conv1d(1, 512, 10, stride=5),     # 16kHz → 3.2kHz
            *[Conv1d(512, 512, 3, stride=2)   # 3.2kHz → 50Hz
              for _ in range(6)]
        )
        self.output_dim = 512
        # Output: 50 frames per second (20ms each)
```

### Transformer with Gated Relative Position Bias

```python
class WavLMTransformerLayer(nn.Module):
    def __init__(self, dim, heads):
        self.attention = MultiHeadAttention(dim, heads)
        self.ffn = FeedForward(dim)
        
        # WavLM's innovation: Gated relative position bias
        self.rel_pos_bias = GatedRelativePositionBias(dim, heads)
    
    def forward(self, x):
        # Compute relative position bias
        rel_pos = self.rel_pos_bias(x)
        
        # Attention with position info
        attn_out = self.attention(x, rel_pos_bias=rel_pos)
        x = x + attn_out
        
        x = x + self.ffn(x)
        return x

class GatedRelativePositionBias(nn.Module):
    """
    Key innovation: Position bias is GATED based on content.
    
    Standard: bias = f(relative_position)
    WavLM:    bias = gate(content) * f(relative_position)
    
    This allows the model to dynamically weight positional
    information based on what it's attending to.
    """
    def __init__(self, dim, heads):
        self.pos_conv = nn.Conv1d(dim, dim, 128, groups=16)
        self.gate = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid()
        )
```

---

## Masked Speech Prediction

### Masking Strategy

```
Similar to BERT, but for audio:

1. Randomly select starting positions (p=0.065)
2. Mask spans of M=10 consecutive frames
3. Replace with learned [MASK] embedding
4. Predict original cluster ID for masked frames

Example (frames):
Original:   [A][B][C][D][E][F][G][H][I][J][K][L]
Masked:     [A][B][M][M][M][M][M][G][H][I][J][K]
                 ↑ ↑ ↑ ↑ ↑
               Predict C,D,E,F (original cluster IDs)
```

### Target Generation (k-means Clustering)

```
Two-stage target creation:

Stage 1 (first iteration):
├── Extract MFCC features from all audio
├── Run k-means (k=100 or 500)
└── Use cluster IDs as targets

Stage 2+ (iterative refinement):
├── Extract features from PREVIOUS WavLM model
├── Run k-means on those features
└── Use new cluster IDs as better targets

This iterative refinement improves target quality.
```

---

## Denoising Pretraining

### The Key WavLM Innovation

```
Most SSL models train on CLEAN audio only.

WavLM trains on:
├── Clean audio (original)
├── Noisy audio (added noise)
├── Overlapping speech (2 speakers mixed)
└── Reverberated audio

But targets are ALWAYS from the clean version!

This forces the model to:
1. Denoise the input
2. Separate overlapping speakers
3. STILL understand the content
```

### Data Augmentation

```python
def wavlm_augmentation(clean_audio, noise_db_range=(-5, 20)):
    """WavLM-style denoising augmentation"""
    
    aug_type = random.choice(['clean', 'noise', 'overlap', 'reverb'])
    
    if aug_type == 'clean':
        return clean_audio
    
    elif aug_type == 'noise':
        # Add background noise
        noise = sample_noise()
        snr = random.uniform(*noise_db_range)
        return mix_at_snr(clean_audio, noise, snr)
    
    elif aug_type == 'overlap':
        # Mix with another utterance
        other = sample_other_utterance()
        ratio = random.uniform(0.25, 0.75)
        return clean_audio * ratio + other * (1 - ratio)
    
    elif aug_type == 'reverb':
        # Apply room impulse response
        rir = sample_room_impulse_response()
        return convolve(clean_audio, rir)
```

### Why This Matters

```
Real-world audio is NEVER clean:
├── Background noise (traffic, AC, music)
├── Overlapping speakers (meetings, calls)
├── Room acoustics (echo, reverb)
└── Recording quality variations

Standard SSL models fail on noisy input.
WavLM handles real-world audio because it was TRAINED on it.

Benchmark results (noisy ASR):
├── wav2vec 2.0: WER 15.2%
├── HuBERT: WER 13.8%
└── WavLM: WER 10.1% ← Much better!
```

---

## What WavLM Learns

### Layer-wise Representations

```
Different layers capture different information:

Layers 1-4:   Acoustic features
              - Formants, pitch, prosody
              - Similar to traditional features

Layers 5-8:   Phonetic features
              - Phone boundaries
              - Phoneme identity
              
Layers 9-12:  Lexical/semantic features
              - Word-level information
              - Used in Mimi for semantic tokens!

Layers 13+:   High-level features (Large model)
              - Sentence structure
              - Abstract representations
```

### Visualization of Learned Features

```
t-SNE of WavLM features:

Layer 1 (acoustic):           Layer 7 (phonetic):
    .   .   .                     AAA   EEE
  .   .   .   .                 AAA     EEE
    .   .   .                     AAA   EEE
  (no clear structure)           III   OOO
                                III     OOO
                               (clusters by phoneme!)

Layer 12 (semantic):
    hello_hello_hello
    hi_hi_hi_hi_hi
    greetings_greetings
    (similar words cluster together!)
```

### Task-Specific Layer Selection

```
Best layer for each task (from ablations):

Task                  | Best Layer | Why
----------------------|------------|------------------
Phone recognition     | 4-6        | Low-level phonetic
ASR                   | 6-9        | Phonetic + lexical
Speaker verification  | 4-6        | Acoustic properties
Emotion recognition   | 8-12       | Prosodic + semantic
Intent classification | 10-12      | High-level semantic

For Mimi: Uses layer 7 (balance of phonetic + early semantic)
```

---

## Using WavLM Features

### Feature Extraction

```python
import torch
from transformers import WavLMModel, Wav2Vec2Processor

# Load model
model = WavLMModel.from_pretrained("microsoft/wavlm-large")
processor = Wav2Vec2Processor.from_pretrained("microsoft/wavlm-large")

# Process audio
audio = load_audio("speech.wav", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# Extract features
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# All layer outputs
hidden_states = outputs.hidden_states  # Tuple of (batch, seq, dim)

# Last layer output
last_hidden = outputs.last_hidden_state  # (batch, seq, 1024)

# Specific layer (e.g., layer 7 for Mimi-style semantic features)
layer_7_features = hidden_states[7]  # (batch, seq, 1024)
```

### Weighted Layer Aggregation (SUPERB style)

```python
class WeightedLayerAggregation(nn.Module):
    """
    Learn optimal layer weights for downstream task.
    Used in SUPERB benchmark.
    """
    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, hidden_states):
        # hidden_states: tuple of (batch, seq, dim)
        stacked = torch.stack(hidden_states, dim=0)  # (layers, batch, seq, dim)
        weights = F.softmax(self.weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(0)
        return weighted  # (batch, seq, dim)
```

### Memory and Compute Considerations

```
WavLM Large inference:
├── Input: 10 seconds @ 16kHz = 160,000 samples
├── CNN output: 500 frames × 512 dim
├── Transformer output: 500 frames × 1024 dim
├── GPU memory: ~2 GB for single utterance
└── Time: ~50ms on A100

For batch processing:
├── Batch size 32, 10 sec each: ~8 GB GPU
├── Throughput: ~640 seconds audio / second
└── 160x faster than real-time

Optimization tips:
├── Use fp16/bf16 for 2x memory reduction
├── Chunk long audio (WavLM handles up to ~30s well)
└── Cache features for repeated use
```

---

## Applications

### SUPERB Benchmark Results

```
WavLM achieves SOTA on SUPERB (Speech Understanding Evaluation):

Task                    | Metric    | WavLM Large | Previous SOTA
------------------------|-----------|-------------|---------------
ASR (LibriSpeech)       | WER ↓     | 1.9%        | 2.1%
Phone Recognition       | PER ↓     | 3.6%        | 4.8%
Speaker Identification  | Acc ↑     | 94.3%       | 92.1%
Speaker Verification    | EER ↓     | 2.9%        | 4.1%
Emotion Recognition     | Acc ↑     | 67.6%       | 65.4%
Intent Classification   | Acc ↑     | 98.5%       | 97.8%
Keyword Spotting        | Acc ↑     | 97.8%       | 97.2%

WavLM is the best across ALL tasks (as of 2021).
```

### Use in Modern Systems

```
Mimi (Kyutai):
├── Distill WavLM layer 7 into semantic tokens
├── First RVQ level encodes WavLM-like features
└── Enables content-aware audio generation

Whisper (OpenAI):
├── Different approach (encoder-decoder)
├── But influenced by SSL speech research
└── WavLM pretraining could improve Whisper

Speech Translation:
├── WavLM → Translation model
├── Better than mel spectrogram input
└── Captures more linguistic information

Voice Conversion:
├── Extract WavLM content features
├── Combine with target speaker embedding
└── Better content preservation
```

---

## Key Takeaways

```
1. SELF-SUPERVISED learning enables massive scale
   - 94k hours of unlabeled audio
   - No transcription needed

2. DENOISING pretraining is crucial
   - Real-world audio is noisy
   - WavLM handles this natively

3. LAYER-WISE representations matter
   - Different layers = different information
   - Choose based on downstream task

4. UNIVERSAL representations work
   - One model for ASR, speaker ID, emotion, etc.
   - Fine-tune or extract features

5. FOUNDATION for modern audio LLMs
   - Mimi distills WavLM
   - Enables semantic audio understanding
```

---

## Next Steps

- `03_semantic_vs_acoustic_tokens.md` - Deep dive into token types
- `04_speech_representation_extraction.py` - Practical feature extraction
- `../03-neural-audio-codecs/05_mimi_codec.md` - How Mimi uses WavLM
