# wav2vec Evolution: Self-Supervised Speech Learning

The wav2vec family revolutionized speech processing through self-supervised learning. Understanding this evolution is crucial for modern audio ML.

## Table of Contents
1. [The Self-Supervised Revolution](#the-self-supervised-revolution)
2. [wav2vec (2019)](#wav2vec-2019)
3. [vq-wav2vec (2019)](#vq-wav2vec-2019)
4. [wav2vec 2.0 (2020)](#wav2vec-20-2020)
5. [HuBERT (2021)](#hubert-2021)
6. [WavLM (2021)](#wavlm-2021)
7. [Comparative Analysis](#comparative-analysis)
8. [Implementation Guide](#implementation-guide)

---

## The Self-Supervised Revolution

### The Data Problem

```
Traditional supervised speech learning:
├── Requires: Transcribed audio
├── Cost: $10-50 per hour of transcription
├── Available: ~10k-100k hours
└── Limitation: Expensive, time-consuming

Self-supervised learning:
├── Requires: Raw audio only (no labels)
├── Cost: Free (scrape from internet)
├── Available: MILLIONS of hours
└── Advantage: Unlimited scale
```

### Key Insight

```
Learn representations from the STRUCTURE of speech itself:

Pretext task: Predict masked audio from context
Real task: Use learned features for ASR, speaker ID, etc.

Similar to BERT for text:
├── BERT: Mask words, predict from context
├── wav2vec: Mask audio frames, predict from context
└── Both learn rich representations
```

---

## wav2vec (2019)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAV2VEC (Original)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw waveform                                                    │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │  CNN Encoder     │  5 layers, stride 10                      │
│  │  (feature ext.)  │  Output: 100 Hz features                  │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │  Context Network │  9-layer CNN                              │
│  │  (aggregation)   │  Receptive field: ~210ms                  │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  Contrastive Loss                                               │
│  (predict future frames)                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Contrastive Predictive Coding (CPC)

```python
def wav2vec_loss(context, future_frames, negative_samples):
    """
    Contrastive loss for wav2vec.
    
    Goal: Distinguish true future frame from negatives.
    """
    # context: (batch, context_dim)
    # future_frames: (batch, feature_dim) - true future
    # negative_samples: (batch, num_negatives, feature_dim)
    
    # Positive score
    pos_score = torch.sum(context * future_frames, dim=-1)
    
    # Negative scores
    neg_scores = torch.bmm(
        negative_samples,
        context.unsqueeze(-1)
    ).squeeze(-1)
    
    # Contrastive loss (InfoNCE)
    logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    
    loss = F.cross_entropy(logits, labels)
    return loss
```

### Limitations

```
1. FUTURE PREDICTION ONLY
   - Predicts future frames from past
   - Cannot use future context
   - Suboptimal for non-causal tasks

2. CONTINUOUS TARGETS
   - Predicting continuous features
   - Harder to learn than discrete
   - Less stable training

3. NEGATIVE SAMPLING
   - Requires careful sampling strategy
   - Sensitive to hyperparameters
   - Can be unstable
```

---

## vq-wav2vec (2019)

### Adding Discretization

```
Key innovation: Quantize features before prediction

Pipeline:
Raw audio → CNN Encoder → VQ → Discrete codes
                          ↓
                    Context Network
                          ↓
              Predict future discrete codes
```

### Vector Quantization Module

```python
class VQModule(nn.Module):
    """
    Vector quantization for vq-wav2vec.
    """
    def __init__(self, num_vars: int = 320, num_groups: int = 2):
        super().__init__()
        self.num_vars = num_vars
        self.num_groups = num_groups
        
        # Codebook: (num_groups, num_vars, dim)
        self.codebook = nn.Parameter(
            torch.randn(num_groups, num_vars, 512)
        )
    
    def forward(self, x):
        """
        x: (batch, dim, time)
        """
        # Split into groups
        x = x.view(x.shape[0], self.num_groups, -1, x.shape[2])
        
        # Quantize each group
        codes = []
        for g in range(self.num_groups):
            # Compute distances
            dist = torch.cdist(
                x[:, g].transpose(1, 2),
                self.codebook[g]
            )
            # Nearest neighbor
            code = dist.argmin(dim=-1)
            codes.append(code)
        
        # Combine groups
        codes = torch.stack(codes, dim=1)
        return codes
```

### Advantages

```
1. DISCRETE TARGETS
   - Classification instead of regression
   - More stable training
   - Better representations

2. GUMBEL-SOFTMAX
   - Differentiable sampling
   - End-to-end training
   - No straight-through estimator needed

3. BETTER DOWNSTREAM PERFORMANCE
   - Improved ASR accuracy
   - Better phone recognition
   - More robust features
```

---

## wav2vec 2.0 (2020)

### Major Breakthrough

```
wav2vec 2.0 = BERT for speech

Key innovations:
├── Masked prediction (like BERT)
├── Transformer architecture
├── Contrastive + diversity loss
└── Massive scale (960h → 60k hours)

Result: Near human-level ASR with minimal labels
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAV2VEC 2.0                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw waveform (16 kHz)                                          │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │  CNN Encoder     │  7 layers, stride 320                     │
│  │  (feature ext.)  │  Output: 50 Hz features                   │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ├────────────────┐                                    │
│           │                │                                    │
│           ▼                ▼                                    │
│  ┌──────────────┐   ┌──────────────┐                           │
│  │  Quantizer   │   │  Mask        │                           │
│  │  (targets)   │   │  (input)     │                           │
│  └────────┬─────┘   └──────┬───────┘                           │
│           │                │                                    │
│           │                ▼                                    │
│           │      ┌──────────────────┐                           │
│           │      │  Transformer     │  24 layers                │
│           │      │  (context)       │  1024 dim                 │
│           │      └────────┬─────────┘                           │
│           │               │                                     │
│           └───────────────┼─────────────┐                       │
│                           │             │                       │
│                           ▼             ▼                       │
│                    Contrastive Loss  Diversity Loss            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Masking Strategy

```python
def mask_spans(features, mask_prob=0.065, mask_length=10):
    """
    Mask contiguous spans of features.
    
    Args:
        features: (batch, time, dim)
        mask_prob: Probability of starting a mask
        mask_length: Length of each mask span
    """
    batch, time, dim = features.shape
    
    # Sample mask starting positions
    mask_starts = torch.rand(batch, time) < mask_prob
    
    # Extend to spans
    mask = torch.zeros(batch, time, dtype=torch.bool)
    for b in range(batch):
        starts = torch.where(mask_starts[b])[0]
        for start in starts:
            end = min(start + mask_length, time)
            mask[b, start:end] = True
    
    # Replace masked positions with learned mask embedding
    masked_features = features.clone()
    mask_embedding = nn.Parameter(torch.randn(dim))
    masked_features[mask] = mask_embedding
    
    return masked_features, mask
```

### Contrastive Loss

```python
def wav2vec2_contrastive_loss(
    transformer_output,
    quantized_targets,
    mask,
    num_negatives=100
):
    """
    Contrastive loss for wav2vec 2.0.
    
    Predict quantized target from transformer output.
    """
    # Only compute loss on masked positions
    masked_output = transformer_output[mask]  # (num_masked, dim)
    masked_targets = quantized_targets[mask]  # (num_masked, dim)
    
    # Sample negatives from other timesteps
    negatives = sample_negatives(
        quantized_targets, num_negatives
    )  # (num_masked, num_negatives, dim)
    
    # Cosine similarity
    pos_sim = F.cosine_similarity(masked_output, masked_targets, dim=-1)
    neg_sim = F.cosine_similarity(
        masked_output.unsqueeze(1),
        negatives,
        dim=-1
    )
    
    # Contrastive loss
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    
    return F.cross_entropy(logits, labels)
```

### Diversity Loss

```python
def diversity_loss(quantized_codes, num_codebooks=2, num_entries=320):
    """
    Encourage uniform usage of codebook entries.
    
    Prevents codebook collapse.
    """
    # quantized_codes: (batch, time, num_codebooks)
    
    # Compute usage distribution
    usage = torch.zeros(num_codebooks, num_entries)
    for g in range(num_codebooks):
        codes = quantized_codes[:, :, g].flatten()
        usage[g] = torch.bincount(codes, minlength=num_entries).float()
    
    # Normalize to probabilities
    usage = usage / usage.sum(dim=1, keepdim=True)
    
    # Entropy (higher = more uniform)
    entropy = -(usage * torch.log(usage + 1e-7)).sum(dim=1)
    
    # Loss: negative entropy (encourage high entropy)
    return -entropy.mean()
```

### Training Scale

```
wav2vec 2.0 configurations:

BASE:
├── 12 transformer layers
├── 768 dim
├── 95M parameters
└── Trained on 960h LibriSpeech

LARGE:
├── 24 transformer layers
├── 1024 dim
├── 317M parameters
└── Trained on 60k hours Libri-Light

Results with fine-tuning:
├── 10 min labels: WER 4.8%
├── 1 hour labels: WER 2.9%
├── 100 hours labels: WER 1.8%
└── Near human-level with minimal supervision!
```

---

## HuBERT (2021)

### Offline Clustering Approach

```
Problem with wav2vec 2.0:
- Online quantization during training
- Can be unstable
- Codebook collapse issues

HuBERT solution:
- Offline clustering (k-means)
- Use cluster IDs as targets
- More stable training
```

### Two-Stage Training

```
Stage 1: MFCC clustering
├── Extract MFCC features from all audio
├── Run k-means (k=100 or 500)
├── Use cluster IDs as targets
├── Train HuBERT to predict clusters
└── Result: First iteration model

Stage 2: Iterative refinement
├── Extract features from Stage 1 model
├── Re-run k-means on these features
├── Use new cluster IDs as targets
├── Train HuBERT again
└── Result: Improved model

Can repeat Stage 2 multiple times
```

### Architecture

```python
class HuBERT(nn.Module):
    """
    HuBERT architecture (similar to wav2vec 2.0).
    """
    def __init__(
        self,
        num_clusters: int = 100,
        d_model: int = 768,
        num_layers: int = 12,
    ):
        super().__init__()
        
        # CNN feature extractor (same as wav2vec 2.0)
        self.feature_extractor = CNNFeatureExtractor()
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
        )
        
        # Prediction head
        self.pred_head = nn.Linear(d_model, num_clusters)
    
    def forward(self, audio, mask=None):
        # Extract features
        features = self.feature_extractor(audio)
        
        # Apply mask
        if mask is not None:
            features = apply_mask(features, mask)
        
        # Transformer
        hidden = self.transformer(features)
        
        # Predict cluster IDs
        logits = self.pred_head(hidden)
        
        return logits
```

### Advantages

```
1. STABLE TRAINING
   - No online quantization
   - No codebook collapse
   - More predictable

2. SIMPLER LOSS
   - Just cross-entropy
   - No contrastive sampling
   - No diversity loss needed

3. ITERATIVE IMPROVEMENT
   - Each iteration improves targets
   - Better final representations
   - Flexible framework

4. STRONG PERFORMANCE
   - Matches or beats wav2vec 2.0
   - Better on some tasks
   - More robust
```

---

## WavLM (2021)

### Denoising Innovation

```
Key insight: Real audio is NOISY

WavLM additions:
├── Train on noisy/overlapped speech
├── Predict CLEAN targets
├── Gated relative position bias
└── Better for real-world audio

See: 02_wavlm_architecture.md for full details
```

---

## Comparative Analysis

### Performance Comparison

| Model | LibriSpeech test-clean WER | Parameters | Training Data |
|-------|----------------------------|------------|---------------|
| wav2vec | 5.2% | 33M | 960h |
| vq-wav2vec | 4.8% | 34M | 960h |
| wav2vec 2.0 Base | 3.4% | 95M | 960h |
| wav2vec 2.0 Large | 1.9% | 317M | 60k h |
| HuBERT Base | 3.3% | 95M | 960h |
| HuBERT Large | 1.9% | 316M | 60k h |
| WavLM Base+ | 3.2% | 95M | 60k h |
| WavLM Large | 1.8% | 316M | 94k h |

### Key Differences

```
wav2vec 2.0:
├── Online quantization
├── Contrastive + diversity loss
├── Best for clean speech
└── Slightly unstable training

HuBERT:
├── Offline clustering
├── Simple cross-entropy loss
├── More stable training
└── Slightly better on some tasks

WavLM:
├── Denoising objective
├── Gated position bias
├── Best for noisy speech
└── SOTA on SUPERB benchmark
```

---

## Implementation Guide

### Using Pretrained Models

```python
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
import torch

# Load models
wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")

# Extract features
audio = torch.randn(1, 16000)  # 1 second @ 16kHz

with torch.no_grad():
    w2v_features = wav2vec2(audio).last_hidden_state
    hubert_features = hubert(audio).last_hidden_state
    wavlm_features = wavlm(audio).last_hidden_state

print(f"wav2vec 2.0: {w2v_features.shape}")
print(f"HuBERT: {hubert_features.shape}")
print(f"WavLM: {wavlm_features.shape}")
```

### Fine-tuning for ASR

```python
from transformers import Wav2Vec2ForCTC, Trainer

# Load pretrained model with CTC head
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h",
    vocab_size=32  # Your vocabulary size
)

# Freeze feature extractor (optional)
model.freeze_feature_encoder()

# Fine-tune with your data
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # ... training args
)

trainer.train()
```

---

## Key Takeaways

```
1. SELF-SUPERVISED LEARNING WORKS
   - Learn from structure of speech
   - Scales to millions of hours
   - Matches supervised with minimal labels

2. EVOLUTION PATH
   - wav2vec: Contrastive future prediction
   - vq-wav2vec: Add discretization
   - wav2vec 2.0: Masked prediction + scale
   - HuBERT: Offline clustering
   - WavLM: Denoising + real-world focus

3. PRACTICAL IMPACT
   - Pretrained models widely used
   - Foundation for modern ASR
   - Enables low-resource languages

4. CHOOSE BASED ON USE CASE
   - Clean speech: wav2vec 2.0 or HuBERT
   - Noisy speech: WavLM
   - Stability: HuBERT
   - Latest features: WavLM
```

---

## Further Reading

- `02_wavlm_architecture.md` - Deep dive into WavLM
- wav2vec 2.0 paper: [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)
- HuBERT paper: [arxiv.org/abs/2106.07447](https://arxiv.org/abs/2106.07447)
- WavLM paper: [arxiv.org/abs/2110.13900](https://arxiv.org/abs/2110.13900)
