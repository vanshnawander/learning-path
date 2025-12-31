# VQ-VAE Fundamentals: The Foundation of Neural Audio Codecs

**Paper**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (DeepMind, 2017)

VQ-VAE is the architectural foundation for ALL modern neural audio codecs (SoundStream, EnCodec, Mimi). Understanding VQ-VAE is essential for understanding audio tokenization.

## Table of Contents
1. [Why Discrete Representations?](#why-discrete-representations)
2. [Autoencoder Basics](#autoencoder-basics)
3. [Vector Quantization](#vector-quantization)
4. [The Straight-Through Estimator](#the-straight-through-estimator)
5. [Codebook Learning](#codebook-learning)
6. [Loss Functions](#loss-functions)
7. [From Images to Audio](#from-images-to-audio)
8. [Code Implementation](#code-implementation)

---

## Why Discrete Representations?

### The LLM Compatibility Problem

```
LLMs operate on DISCRETE tokens:
- Text: vocabulary of ~50,000 tokens
- Each token is an integer index
- Softmax predicts probability over vocabulary

Audio is CONTINUOUS:
- Waveform: float values in [-1, 1]
- 16,000 values per second
- Can't directly apply cross-entropy loss

Solution: QUANTIZE audio into discrete tokens
         Then model with standard LLM techniques
```

### Compression Benefits

```
Discrete representations enable:

1. COMPRESSION
   - 16 kHz audio: 256 kbps (16-bit)
   - Quantized: 1.5-6 kbps (50-170x compression)

2. LLM INTEGRATION
   - Treat audio like text tokens
   - Standard transformer architectures work

3. GENERATION
   - Sample discrete tokens autoregressively
   - Decode to continuous audio

4. STORAGE
   - Integer indices vs float arrays
   - Much more efficient for large datasets
```

---

## Autoencoder Basics

### Standard Autoencoder

```
Compress then reconstruct:

Input x ──▶ Encoder ──▶ Latent z ──▶ Decoder ──▶ Reconstruction x̂

Goal: x̂ ≈ x with z being much smaller than x

Loss: L = ||x - x̂||² (reconstruction loss)
```

### Variational Autoencoder (VAE)

```
VAE adds probabilistic latent space:

Input x ──▶ Encoder ──▶ μ, σ ──▶ z ~ N(μ, σ²) ──▶ Decoder ──▶ x̂

Benefits:
- Smooth latent space (can interpolate)
- Regularized representations

Problem for audio:
- Continuous latents still need discretization for LLMs
- KL divergence can hurt reconstruction quality
```

### VQ-VAE: Discrete Latents

```
Replace continuous latents with CODEBOOK lookup:

Input x ──▶ Encoder ──▶ z_e ──▶ Quantize ──▶ z_q ──▶ Decoder ──▶ x̂

Quantize: Find nearest codebook entry
z_q = e_k where k = argmin_j ||z_e - e_j||²

Codebook: {e_1, e_2, ..., e_K} where K = vocabulary size (e.g., 1024)
```

---

## Vector Quantization

### The Codebook

```
Codebook E ∈ R^{K × D}
- K: number of entries (vocabulary size), typically 1024-8192
- D: embedding dimension, typically 64-512

Each entry e_k is a D-dimensional vector.
These are LEARNED during training.

┌──────────────────────────────────────────────┐
│                 CODEBOOK                      │
├──────────────────────────────────────────────┤
│  Index 0:  [0.23, -0.15, 0.87, ..., 0.42]   │
│  Index 1:  [-0.56, 0.33, 0.12, ..., -0.78]  │
│  Index 2:  [0.91, 0.05, -0.34, ..., 0.15]   │
│  ...                                         │
│  Index K-1: [0.12, -0.89, 0.45, ..., 0.67]  │
└──────────────────────────────────────────────┘
```

### Quantization Process

```
For each encoder output vector z_e:

1. Compute distance to all codebook entries:
   d_k = ||z_e - e_k||² for k = 0, 1, ..., K-1

2. Find nearest neighbor:
   k* = argmin_k d_k

3. Replace with codebook entry:
   z_q = e_{k*}

The output is:
- Index k* (discrete token for LLM)
- Vector z_q (for decoder)
```

### Visual Example

```
Encoder output z_e = [0.5, 0.3]

Codebook:
  e_0 = [0.1, 0.2]  d_0 = (0.5-0.1)² + (0.3-0.2)² = 0.17
  e_1 = [0.6, 0.4]  d_1 = (0.5-0.6)² + (0.3-0.4)² = 0.02  ← NEAREST
  e_2 = [-0.3, 0.8] d_2 = (0.5+0.3)² + (0.3-0.8)² = 0.89

Quantized: z_q = e_1 = [0.6, 0.4], token index = 1
```

---

## The Straight-Through Estimator

### The Gradient Problem

```
argmin is NOT differentiable!

Forward pass:  z_e ──▶ argmin ──▶ z_q ──▶ Decoder ──▶ Loss

Backward pass: How to compute ∂Loss/∂z_e?
               argmin has zero gradient almost everywhere!

Without gradients, encoder can't learn.
```

### Straight-Through Estimator (STE)

```
TRICK: Pretend quantization didn't happen during backward pass.

Forward:  z_q = e_{k*}  (use quantized value)
Backward: ∂Loss/∂z_e = ∂Loss/∂z_q  (copy gradient through)

Implementation in PyTorch:
z_q = z_e + (z_q - z_e).detach()

This means:
- Forward: z_q gets the quantized value
- Backward: gradient flows directly to z_e
```

### Why This Works

```
Intuition: If z_e is close to z_q, then moving z_e 
           in the direction that helps z_q also helps z_e.

The encoder learns to:
1. Output vectors close to codebook entries
2. Pick "good" entries that help reconstruction

The commitment loss (below) encourages this closeness.
```

---

## Codebook Learning

### The EMA Update (Exponential Moving Average)

```
Alternative to gradient descent for codebook.
Update codebook entries toward their assigned vectors.

For each codebook entry e_k:
1. Find all encoder outputs assigned to e_k
2. Update e_k toward their centroid

N_k = (1 - γ) * N_k + γ * n_k           (count update)
m_k = (1 - γ) * m_k + γ * Σ z_e         (sum update)  
e_k = m_k / N_k                          (normalize)

where:
- γ: decay rate (e.g., 0.99)
- n_k: number of assignments to e_k in current batch
- Σ z_e: sum of encoder outputs assigned to e_k
```

### Codebook Collapse Problem

```
PROBLEM: Some codebook entries never get used.

Why:
- Random initialization puts entries far from data
- They never get selected → never get updated
- "Dead" entries waste codebook capacity

SOLUTIONS:

1. Random restart:
   If e_k unused for N batches, reinitialize to random z_e

2. Kmeans initialization:
   Initialize codebook with kmeans on first batch

3. Codebook entropy regularization:
   Encourage uniform usage of all entries
```

---

## Loss Functions

### Total VQ-VAE Loss

```
L_total = L_recon + L_codebook + β * L_commit

Where:
- L_recon: Reconstruction loss (main objective)
- L_codebook: Moves codebook toward encoder outputs
- L_commit: Moves encoder outputs toward codebook
- β: Commitment loss weight (typically 0.25)
```

### Reconstruction Loss

```
L_recon = ||x - x̂||² or other perceptual loss

For audio, typically:
- Multi-scale spectrogram loss
- Mel spectrogram loss
- Adversarial loss (discriminator)
```

### Codebook Loss (VQ Loss)

```
L_codebook = ||sg[z_e] - e||²

sg[·] = stop gradient (detach in PyTorch)

This loss only affects the CODEBOOK, not encoder.
Pulls codebook entries toward encoder outputs.

With EMA updates, this loss term is often removed.
```

### Commitment Loss

```
L_commit = ||z_e - sg[e]||²

This loss only affects the ENCODER.
Encourages encoder to "commit" to codebook entries.

Without this:
- Encoder could output vectors far from codebook
- Large quantization error
- Poor reconstruction
```

---

## From Images to Audio

### Key Differences

```
Images (Original VQ-VAE):
├── Input: 256×256×3 image
├── Encoder: Strided convolutions
├── Latent: 32×32×D (spatial)
├── Quantize: Each spatial position independently
└── Decoder: Transposed convolutions

Audio (SoundStream, EnCodec):
├── Input: T samples waveform
├── Encoder: 1D strided convolutions
├── Latent: T/S×D (temporal, S=stride product)
├── Quantize: Each timestep independently
└── Decoder: 1D transposed convolutions
```

### Temporal Compression Ratios

```
Audio codecs aim for high temporal compression:

SoundStream:
- Input: 24 kHz audio
- Stride: 320
- Latent rate: 75 Hz (75 vectors per second)

EnCodec:
- Input: 24 kHz audio  
- Stride: 320
- Latent rate: 75 Hz

Mimi:
- Input: 24 kHz audio
- Stride: 1920
- Latent rate: 12.5 Hz (very high compression!)
```

### Why RVQ? (Preview)

```
Single codebook with K=1024 entries:
- 10 bits per vector
- At 75 Hz: 750 bits/second

Not enough for high-quality audio!

Solution: Residual Vector Quantization (next module)
- Multiple codebooks
- Quantize the "residual" error
- Much higher effective codebook size
```

---

## Code Implementation

### Basic VQ Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: Encoder output, shape (B, D, T) or (B, T, D)
        Returns:
            z_q: Quantized vectors
            loss: VQ + commitment loss
            indices: Codebook indices
        """
        # Reshape to (B*T, D)
        z_e_flat = z_e.permute(0, 2, 1).contiguous()  # (B, T, D)
        z_e_flat = z_e_flat.view(-1, self.embedding_dim)  # (B*T, D)
        
        # Compute distances to all codebook entries
        # ||z_e - e||² = ||z_e||² + ||e||² - 2*z_e·e
        d = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * z_e_flat @ self.embedding.weight.t()
        )
        
        # Find nearest codebook entry
        indices = d.argmin(dim=1)  # (B*T,)
        
        # Lookup quantized vectors
        z_q_flat = self.embedding(indices)  # (B*T, D)
        
        # Reshape back
        B, D, T = z_e.shape
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)  # (B, D, T)
        indices = indices.view(B, T)
        
        # Compute losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())  # Move codebook toward z_e
        commitment_loss = F.mse_loss(z_e, z_q.detach())  # Move z_e toward codebook
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, loss, indices
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices back to vectors"""
        return self.embedding(indices).permute(0, 2, 1)
```

### EMA-Updated VQ Layer

```python
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook (not trained by gradient)
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.clone())
    
    def forward(self, z_e: torch.Tensor):
        # Same distance computation and index lookup as above
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
        d = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * z_e_flat @ self.embedding.t()
        )
        indices = d.argmin(dim=1)
        
        # One-hot encoding for EMA update
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        # EMA update (only during training)
        if self.training:
            # Update cluster sizes
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            
            # Update embedding averages
            embed_sum = encodings.t() @ z_e_flat
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )
            
            # Normalize
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))
        
        # Quantize
        z_q_flat = self.embedding[indices]
        
        B, D, T = z_e.shape
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)
        indices = indices.view(B, T)
        
        # Only commitment loss with EMA
        loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, loss, indices
```

---

## Key Takeaways

```
1. VQ-VAE enables DISCRETE representation of continuous signals
   → Critical for LLM integration

2. Straight-through estimator is the key trick
   → Allows gradient flow through non-differentiable argmin

3. Codebook learning requires care
   → EMA updates, restart dead entries

4. Audio needs temporal compression
   → High stride ratios (320-1920)

5. Single codebook has limited capacity
   → Leads to Residual Vector Quantization (RVQ)
```

---

## Next: Residual Vector Quantization

See `02_residual_vector_quantization.md` for how SoundStream and EnCodec achieve high-quality audio with multiple codebook levels.
