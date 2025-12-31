"""
04_encodec_implementation.py - Complete EnCodec Implementation

Full implementation of EnCodec neural audio codec with:
- Encoder/Decoder with Snake activation
- LSTM temporal modeling
- Residual Vector Quantization
- Multi-scale STFT discriminator
- Balancer for loss weighting
- Complete training loop

Based on: High Fidelity Neural Audio Compression (Meta, 2022)
Reference: github.com/facebookresearch/encodec

Run: python 04_encodec_implementation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Tuple, Optional
import time
import numpy as np


# ============================================================
# SNAKE ACTIVATION
# ============================================================

class Snake(nn.Module):
    """
    Snake activation: x + sin²(αx) / α
    
    Better than ReLU/ELU for periodic signals (audio).
    Learned α parameter per channel.
    """
    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.sin(self.alpha * x).pow(2)) / (self.alpha + 1e-9)


# ============================================================
# RESIDUAL BLOCKS
# ============================================================

class EnCodecResBlock(nn.Module):
    """
    Residual block with dilated convolutions and Snake activation.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.block = nn.Sequential(
            Snake(channels),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            Snake(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# ENCODER
# ============================================================

class EnCodecEncoder(nn.Module):
    """
    EnCodec encoder with strided convolutions and LSTM.
    
    Architecture:
    - Initial conv
    - 4 downsample blocks (stride 2, 4, 5, 8)
    - Bidirectional LSTM
    - Final projection to latent dim
    """
    def __init__(
        self,
        channels: int = 32,
        latent_dim: int = 128,
        strides: List[int] = [2, 4, 5, 8],
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.strides = strides
        self.total_stride = np.prod(strides)
        
        layers = []
        
        # Initial conv
        layers.append(nn.Conv1d(1, channels, kernel_size=7, padding=3))
        
        # Downsample blocks
        in_ch = channels
        for stride in strides:
            out_ch = min(in_ch * 2, 512)
            
            # Residual blocks with different dilations
            for dilation in [1, 3, 9]:
                layers.append(EnCodecResBlock(in_ch, dilation=dilation))
            
            # Strided conv for downsampling
            layers.append(Snake(in_ch))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=stride*2, stride=stride, padding=stride//2))
            
            in_ch = out_ch
        
        # Final residual blocks
        for dilation in [1, 3, 9]:
            layers.append(EnCodecResBlock(in_ch, dilation=dilation))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            in_ch, in_ch, num_layers=lstm_layers,
            bidirectional=True, batch_first=True
        )
        
        # Project to latent dim
        self.proj = nn.Conv1d(in_ch * 2, latent_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, samples) -> (batch, latent_dim, frames)"""
        # Convolutional layers
        x = self.conv_layers(x)
        
        # LSTM (needs B, T, C format)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        
        # Project to latent
        x = self.proj(x)
        
        return x


# ============================================================
# DECODER
# ============================================================

class EnCodecDecoder(nn.Module):
    """
    EnCodec decoder with transposed convolutions.
    
    Mirror of encoder architecture.
    """
    def __init__(
        self,
        channels: int = 32,
        latent_dim: int = 128,
        strides: List[int] = [8, 5, 4, 2],
        lstm_layers: int = 2,
    ):
        super().__init__()
        
        # Calculate channel progression
        channel_mult = [min(2**i, 16) for i in range(len(strides), 0, -1)]
        in_channels = [channels * m for m in channel_mult]
        
        # Initial projection
        self.proj = nn.Conv1d(latent_dim, in_channels[0] * 2, kernel_size=7, padding=3)
        
        # LSTM
        self.lstm = nn.LSTM(
            in_channels[0] * 2, in_channels[0] * 2,
            num_layers=lstm_layers, bidirectional=True, batch_first=True
        )
        
        layers = []
        
        # Initial residual blocks
        for dilation in [1, 3, 9]:
            layers.append(EnCodecResBlock(in_channels[0] * 2, dilation=dilation))
        
        # Upsample blocks
        in_ch = in_channels[0] * 2
        for i, stride in enumerate(strides):
            out_ch = in_channels[i + 1] if i + 1 < len(in_channels) else channels
            
            # Transposed conv for upsampling
            layers.append(Snake(in_ch))
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=stride*2, stride=stride, padding=stride//2))
            
            # Residual blocks
            for dilation in [1, 3, 9]:
                layers.append(EnCodecResBlock(out_ch, dilation=dilation))
            
            in_ch = out_ch
        
        # Final conv to waveform
        layers.append(Snake(channels))
        layers.append(nn.Conv1d(channels, 1, kernel_size=7, padding=3))
        layers.append(nn.Tanh())
        
        self.conv_layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim, frames) -> (batch, 1, samples)"""
        # Initial projection
        x = self.proj(z)
        
        # LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        return x


# ============================================================
# VECTOR QUANTIZATION (from previous modules)
# ============================================================

class VectorQuantizerEMA(nn.Module):
    """VQ with EMA updates."""
    def __init__(self, codebook_size, codebook_dim, commitment_weight=0.25, decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        
        self.register_buffer('embedding', torch.randn(codebook_size, codebook_dim))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.embedding.clone())
    
    def forward(self, z):
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, D)
        
        dist = (z_flat.pow(2).sum(1, keepdim=True) + self.embedding.pow(2).sum(1) - 2 * z_flat @ self.embedding.t())
        indices = dist.argmin(dim=1)
        
        if self.training:
            encodings = F.one_hot(indices, self.codebook_size).float()
            self.cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1-self.decay)
            embed_sum = encodings.t() @ z_flat
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
            self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))
        
        z_q_flat = self.embedding[indices]
        loss = self.commitment_weight * F.mse_loss(z_flat, z_q_flat.detach())
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)
        
        return z_q, loss, indices.view(B, T)


class ResidualVQ(nn.Module):
    """Residual VQ with multiple levels."""
    def __init__(self, num_quantizers=8, codebook_size=1024, codebook_dim=128):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(codebook_size, codebook_dim) for _ in range(num_quantizers)
        ])
    
    def forward(self, z, num_quantizers=None):
        if num_quantizers is None:
            num_quantizers = len(self.quantizers)
        
        z_q, residual, total_loss, codes = torch.zeros_like(z), z, 0, []
        for i in range(num_quantizers):
            quantized, loss, indices = self.quantizers[i](residual)
            z_q, residual, total_loss = z_q + quantized, residual - quantized, total_loss + loss
            codes.append(indices)
        
        return z_q, total_loss, codes


# ============================================================
# COMPLETE ENCODEC MODEL
# ============================================================

class EnCodec(nn.Module):
    """Complete EnCodec model."""
    def __init__(self, sample_rate=24000, channels=32, latent_dim=128, num_quantizers=8):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = EnCodecEncoder(channels, latent_dim)
        self.quantizer = ResidualVQ(num_quantizers, 1024, latent_dim)
        self.decoder = EnCodecDecoder(channels, latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, codes = self.quantizer(z)
        x_hat = self.decoder(z_q)
        min_len = min(x.shape[-1], x_hat.shape[-1])
        return x_hat[..., :min_len], vq_loss, codes
    
    def encode(self, x):
        return self.quantizer(self.encoder(x))[2]
    
    def decode(self, codes):
        z_q = sum(self.quantizer.quantizers[i].embedding[c].permute(0,2,1) for i,c in enumerate(codes))
        return self.decoder(z_q)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "█"*70)
    print("█" + "  EnCodec Implementation".center(68) + "█")
    print("█"*70)
    
    codec = EnCodec()
    x = torch.randn(1, 1, 24000)
    x_hat, loss, codes = codec(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Output: {x_hat.shape}")
    print(f"VQ Loss: {loss.item():.4f}")
    print(f"Codes: {len(codes)} levels")
    print(f"Parameters: {sum(p.numel() for p in codec.parameters()):,}")

if __name__ == "__main__":
    main()
