# Residual Vector Quantization (RVQ)

**Key Paper**: [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) (Google, 2021)

RVQ is THE technique that makes neural audio codecs practical. It's used in SoundStream, EnCodec, Mimi, and virtually every modern audio tokenizer.

## Table of Contents
1. [The Capacity Problem](#the-capacity-problem)
2. [RVQ Algorithm](#rvq-algorithm)
3. [Bitrate Control](#bitrate-control)
4. [Training RVQ](#training-rvq)
5. [Hierarchical Information](#hierarchical-information)
6. [Implementation](#implementation)
7. [Profiling Considerations](#profiling-considerations)

---

## The Capacity Problem

### Single Codebook Limitations

```
Single VQ with K=1024 entries:
- 10 bits per quantization (log2(1024) = 10)
- At 75 Hz frame rate: 750 bits/second

Quality requirements:
- Telephone: ~8 kbps
- Good speech: ~16 kbps  
- High-quality audio: ~64 kbps

Gap: 750 bps vs 8000+ bps needed!
```

### Naive Solution: Bigger Codebook

```
If K=1024 gives 10 bits, why not K=2^20 = 1M entries?

Problems:
1. MEMORY: 1M × 512 dims × 4 bytes = 2 GB just for codebook!
2. COMPUTE: Distance to 1M vectors per quantization
3. LEARNING: Hard to learn so many entries from data

This doesn't scale.
```

### RVQ Solution: Multiple Smaller Codebooks

```
Instead of one 2^20 codebook:
Use 8 codebooks of 2^10 each!

Total combinations: (2^10)^8 = 2^80 >> 2^20
But only 8 × 1024 = 8192 vectors to store and search

This is RESIDUAL Vector Quantization.
```

---

## RVQ Algorithm

### Core Idea: Quantize the Error

```
Step 1: Quantize input with first codebook
        r₀ = z (input)
        q₁ = Quantize(r₀, Codebook₁)
        r₁ = r₀ - q₁  (residual = what we missed)

Step 2: Quantize the residual with second codebook
        q₂ = Quantize(r₁, Codebook₂)
        r₂ = r₁ - q₂

Step 3: Continue for N levels
        ...
        
Final quantized value:
        z_q = q₁ + q₂ + q₃ + ... + qₙ
```

### Visual Representation

```
Input z ●────────────────────────────────────────────────────●
        │                                            Target  │
        │                                                     │
Level 1 │  z ────▶ VQ₁ ────▶ q₁                              │
        │         ↓                                           │
        │      residual r₁ = z - q₁                          │
        │         ↓                                           │
Level 2 │  r₁ ───▶ VQ₂ ────▶ q₂                              │
        │         ↓                                           │
        │      residual r₂ = r₁ - q₂                         │
        │         ↓                                           │
Level 3 │  r₂ ───▶ VQ₃ ────▶ q₃                              │
        │         ↓                                           │
        │      ...                                            │
        │                                                     │
Output  │  z_q = q₁ + q₂ + q₃ + ... + qₙ ●───────────────────●
        │                                  Approximation      │

With each level, z_q gets closer to z!
```

### Mathematical Formulation

```python
def rvq_encode(z, codebooks):
    """
    Args:
        z: Input vectors, shape (B, D, T)
        codebooks: List of N codebook modules
    Returns:
        codes: List of N index tensors, each (B, T)
        z_q: Quantized output (B, D, T)
    """
    codes = []
    residual = z
    z_q = torch.zeros_like(z)
    
    for codebook in codebooks:
        # Quantize current residual
        q, indices = codebook.quantize(residual)
        codes.append(indices)
        
        # Accumulate quantized value
        z_q = z_q + q
        
        # Compute new residual
        residual = residual - q
    
    return codes, z_q

def rvq_decode(codes, codebooks):
    """
    Args:
        codes: List of N index tensors
        codebooks: List of N codebook modules
    Returns:
        z_q: Reconstructed quantized vectors
    """
    z_q = torch.zeros(...)
    for code, codebook in zip(codes, codebooks):
        z_q = z_q + codebook.lookup(code)
    return z_q
```

---

## Bitrate Control

### Bits Per Second Calculation

```
Bitrate = frame_rate × num_levels × log2(codebook_size)

Example (SoundStream):
- Frame rate: 75 Hz
- Codebook size: 1024 (10 bits)
- 4 levels: 75 × 4 × 10 = 3000 bps = 3 kbps
- 8 levels: 75 × 8 × 10 = 6000 bps = 6 kbps
- 12 levels: 75 × 12 × 10 = 9000 bps = 9 kbps

Example (Mimi):
- Frame rate: 12.5 Hz (much lower!)
- Codebook size: 2048 (11 bits)  
- 8 levels: 12.5 × 8 × 11 = 1100 bps = 1.1 kbps
```

### Variable Bitrate

```
RVQ naturally supports variable bitrate:
- Use fewer levels → Lower bitrate, lower quality
- Use more levels → Higher bitrate, higher quality

At decode time, you can use:
- All 8 levels for full quality
- Only first 4 levels for streaming/low bandwidth
- Only first 1 level for very low quality preview
```

### Bitrate Comparison

| Codec | Frame Rate | Levels | Codebook | Bitrate |
|-------|------------|--------|----------|---------|
| SoundStream | 75 Hz | 3-12 | 1024 | 3-12 kbps |
| EnCodec | 75 Hz | 2-32 | 1024 | 1.5-24 kbps |
| Mimi | 12.5 Hz | 8 | 2048 | 1.1 kbps |

---

## Training RVQ

### Joint Training

```
All codebooks trained together end-to-end:

Loss = Reconstruction_Loss + Σ VQ_Loss(level_i)

Each level contributes:
1. Its quantization error to the residual chain
2. Its VQ/commitment loss

Gradients flow through all levels via straight-through estimator.
```

### RVQ Dropout (Important for Mimi!)

```
Problem: Model becomes dependent on all levels
         Can't gracefully degrade with fewer levels

Solution: RVQ Dropout
- During training, randomly use only first K levels
- K sampled uniformly from [1, N]
- Model learns to work with any number of levels

Implementation:
if training:
    num_active_levels = random.randint(1, num_levels)
    codes = codes[:num_active_levels]
    z_q = sum(codebook.lookup(c) for c in codes)
```

### Codebook Initialization

```
Good initialization is critical:

1. Random uniform (simple, works okay)
   embedding.weight.uniform_(-1/K, 1/K)

2. Kmeans initialization (better)
   - Run kmeans on first batch
   - Initialize each level's codebook from encoder outputs/residuals

3. Hierarchical initialization
   - Level 1: Kmeans on encoder outputs
   - Level 2+: Kmeans on residuals from previous level
```

---

## Hierarchical Information

### What Each Level Captures

```
Level 1 (Coarse): 
├── Overall energy/volume
├── Basic frequency content
└── Rough phonetic content

Level 2-4 (Mid):
├── Finer spectral details
├── Pitch information
└── Speaker characteristics

Level 5+ (Fine):
├── High-frequency details
├── Subtle timbre
└── Noise/breath sounds
```

### Semantic vs Acoustic Tokens

```
Observation: First few levels contain more "semantic" info
             Later levels contain more "acoustic" details

This is exploited in:
- AudioLM: Separate semantic and acoustic modeling
- Mimi: Dedicated semantic token from WavLM distillation
- MusicGen: Different modeling for different levels
```

### Ablation Study Results (from papers)

```
SoundStream ablation (subjective quality):
Levels  | MOS Score
--------|----------
1       | 2.1 (poor)
2       | 3.2
4       | 3.8
8       | 4.1
12      | 4.3 (near original)

First few levels matter most!
Diminishing returns after ~8 levels.
```

---

## Implementation

### Complete RVQ Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization module.
    
    Used in SoundStream, EnCodec, Mimi, etc.
    """
    
    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        # Create separate codebook for each level
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                num_embeddings=codebook_size,
                embedding_dim=codebook_dim,
                commitment_cost=commitment_weight,
                decay=ema_decay,
            )
            for _ in range(num_quantizers)
        ])
        
        self.threshold_ema_dead_code = threshold_ema_dead_code
    
    def forward(
        self, 
        z: torch.Tensor,
        num_quantizers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            z: Input tensor (B, D, T)
            num_quantizers: Number of levels to use (for RVQ dropout)
            
        Returns:
            z_q: Quantized tensor (B, D, T)
            total_loss: Sum of all VQ losses
            codes: List of code indices for each level
        """
        # RVQ dropout during training
        if num_quantizers is None:
            if self.training:
                # Random number of quantizers
                num_quantizers = torch.randint(
                    1, self.num_quantizers + 1, (1,)
                ).item()
            else:
                num_quantizers = self.num_quantizers
        
        residual = z
        z_q = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device)
        codes = []
        
        for i in range(num_quantizers):
            # Quantize residual
            quantized, loss, indices = self.quantizers[i](residual)
            
            # Accumulate
            z_q = z_q + quantized
            total_loss = total_loss + loss
            codes.append(indices)
            
            # Update residual (with straight-through)
            residual = residual - quantized
        
        return z_q, total_loss, codes
    
    def encode(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Encode to indices only (for storage/transmission)"""
        _, _, codes = self.forward(z, num_quantizers=self.num_quantizers)
        return codes
    
    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """Decode from indices"""
        z_q = torch.zeros(
            codes[0].shape[0],  # batch
            self.codebook_dim,
            codes[0].shape[1],  # time
            device=codes[0].device
        )
        
        for i, code in enumerate(codes):
            z_q = z_q + self.quantizers[i].decode_indices(code)
        
        return z_q
    
    def get_codebook_usage(self) -> List[float]:
        """Return percentage of codebook entries used per level"""
        usage = []
        for q in self.quantizers:
            used = (q.cluster_size > self.threshold_ema_dead_code).float().mean()
            usage.append(used.item() * 100)
        return usage


class VectorQuantizerEMA(nn.Module):
    """Single-level VQ with EMA updates (from previous module)"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.clone())
        
    def forward(self, z_e):
        # Flatten
        B, D, T = z_e.shape
        z_e_flat = z_e.permute(0, 2, 1).reshape(-1, D)
        
        # Distances
        d = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * z_e_flat @ self.embedding.t()
        )
        indices = d.argmin(dim=1)
        
        # EMA update
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).float()
            self.cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1-self.decay)
            embed_sum = encodings.t() @ z_e_flat
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))
        
        # Quantize
        z_q_flat = self.embedding[indices]
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)
        indices = indices.view(B, T)
        
        # Loss
        loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, loss, indices
    
    def decode_indices(self, indices):
        B, T = indices.shape
        flat_indices = indices.view(-1)
        z_q = self.embedding[flat_indices].view(B, T, -1).permute(0, 2, 1)
        return z_q
```

---

## Profiling Considerations

### Memory Usage

```
Per quantizer:
- Codebook: K × D × 4 bytes
- Example: 1024 × 256 × 4 = 1 MB

For 8 quantizers: 8 MB
For 32 quantizers (EnCodec max): 32 MB

Negligible compared to encoder/decoder networks.
```

### Compute Cost

```
Per quantizer, per timestep:
- Distance computation: O(K × D)
- Argmin: O(K)

Total per forward: O(N × T × K × D)

Example: N=8, T=100 (1.3s), K=1024, D=256
Operations: 8 × 100 × 1024 × 256 = 200M ops

This is fast! Not a bottleneck.
```

### Bottleneck Analysis

```
Typical audio codec timing breakdown:

Encoder (conv layers):     40%
Decoder (conv layers):     45%
RVQ (quantization):         5%  ← Very fast!
Discriminator (training):  10%

Optimization priority:
1. Encoder/decoder architecture
2. FFT computations in loss
3. RVQ is rarely the bottleneck
```

---

## Key Takeaways

```
1. RVQ enables high-fidelity quantization with small codebooks
   → Exponential effective vocabulary

2. Variable bitrate is built-in
   → Use fewer levels for lower bitrate

3. Hierarchical structure captures different information
   → Coarse-to-fine representation

4. RVQ dropout improves generalization
   → Model works with any number of levels

5. Very efficient to compute
   → Not a training/inference bottleneck
```

---

## Next: SoundStream Architecture

See `03_soundstream_architecture.md` for how Google combined RVQ with encoder/decoder networks and adversarial training.
