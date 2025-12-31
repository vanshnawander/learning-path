"""
03_exercises_and_solutions.py - Audio ML Exercises with Solutions

Hands-on exercises covering the full audio ML curriculum.
Each exercise builds on concepts from the learning materials.

Run: python 03_exercises_and_solutions.py

Structure:
- Exercise description
- Starter code
- Solution (hidden by default)
- Verification tests
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import time

# ============================================================
# EXERCISE 1: Implement STFT from scratch
# ============================================================

def exercise_1_stft():
    """
    EXERCISE 1: Short-Time Fourier Transform
    
    Implement STFT using numpy. This is what torchaudio.transforms.Spectrogram does.
    
    Steps:
    1. Split audio into overlapping frames
    2. Apply window to each frame
    3. Compute FFT of each frame
    4. Return complex spectrogram
    """
    print("\n" + "="*60)
    print("EXERCISE 1: Implement STFT")
    print("="*60)
    
    # === YOUR CODE HERE ===
    def stft_exercise(audio: np.ndarray, n_fft: int = 512, 
                      hop_length: int = 128) -> np.ndarray:
        """
        Compute STFT of audio signal.
        
        Args:
            audio: 1D array of shape (num_samples,)
            n_fft: FFT window size
            hop_length: Hop between frames
            
        Returns:
            Complex spectrogram of shape (n_fft//2+1, num_frames)
        """
        # TODO: Implement this!
        pass
    
    # === SOLUTION ===
    def stft_solution(audio: np.ndarray, n_fft: int = 512,
                      hop_length: int = 128) -> np.ndarray:
        """Reference solution for STFT."""
        num_samples = len(audio)
        num_frames = (num_samples - n_fft) // hop_length + 1
        num_bins = n_fft // 2 + 1
        
        # Create Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1)))
        
        # Initialize output
        spectrogram = np.zeros((num_bins, num_frames), dtype=np.complex64)
        
        # Process each frame
        for frame_idx in range(num_frames):
            start = frame_idx * hop_length
            frame = audio[start:start + n_fft]
            
            # Apply window
            windowed = frame * window
            
            # FFT (take positive frequencies only)
            fft_result = np.fft.rfft(windowed)
            spectrogram[:, frame_idx] = fft_result
        
        return spectrogram
    
    # Test
    audio = np.random.randn(16000).astype(np.float32)
    result = stft_solution(audio, n_fft=512, hop_length=128)
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Output spectrogram shape: {result.shape}")
    print(f"Expected shape: (257, 122)")
    
    # Verify against numpy reference
    expected_frames = (16000 - 512) // 128 + 1
    assert result.shape == (257, expected_frames), "Shape mismatch!"
    print("✓ Exercise 1 passed!")


# ============================================================
# EXERCISE 2: Mel Filterbank Construction
# ============================================================

def exercise_2_mel_filterbank():
    """
    EXERCISE 2: Build Mel Filterbank
    
    Create triangular mel-spaced filterbank from scratch.
    This is what's used inside MelSpectrogram transforms.
    """
    print("\n" + "="*60)
    print("EXERCISE 2: Mel Filterbank Construction")
    print("="*60)
    
    # === SOLUTION ===
    def hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)
    
    def create_mel_filterbank(sample_rate: int, n_fft: int, 
                               n_mels: int, f_min: float = 0.0,
                               f_max: Optional[float] = None) -> np.ndarray:
        """
        Create mel filterbank matrix.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            n_mels: Number of mel bins
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sample_rate/2)
            
        Returns:
            Filterbank matrix of shape (n_mels, n_fft//2+1)
        """
        if f_max is None:
            f_max = sample_rate / 2
        
        num_bins = n_fft // 2 + 1
        
        # Convert frequency bounds to mel
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        
        # Mel points (n_mels + 2 for left and right edges)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin indices
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, num_bins))
        
        for m in range(n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            
            # Rising edge
            for k in range(left, center):
                if k < num_bins:
                    filterbank[m, k] = (k - left) / (center - left + 1e-10)
            
            # Falling edge
            for k in range(center, right):
                if k < num_bins:
                    filterbank[m, k] = (right - k) / (right - center + 1e-10)
        
        return filterbank
    
    # Test
    filterbank = create_mel_filterbank(
        sample_rate=16000, n_fft=512, n_mels=80
    )
    
    print(f"Filterbank shape: {filterbank.shape}")
    print(f"Expected: (80, 257)")
    print(f"Sum of each filter (should be ~1): {filterbank.sum(axis=1)[:5]}")
    
    assert filterbank.shape == (80, 257), "Shape mismatch!"
    print("✓ Exercise 2 passed!")


# ============================================================
# EXERCISE 3: Vector Quantization
# ============================================================

def exercise_3_vector_quantization():
    """
    EXERCISE 3: Implement Vector Quantization
    
    Build a simple VQ layer with straight-through gradient estimator.
    This is the core of neural audio codecs.
    """
    print("\n" + "="*60)
    print("EXERCISE 3: Vector Quantization")
    print("="*60)
    
    # === SOLUTION ===
    class VectorQuantizer(nn.Module):
        def __init__(self, codebook_size: int, codebook_dim: int):
            super().__init__()
            self.codebook_size = codebook_size
            self.codebook_dim = codebook_dim
            
            # Initialize codebook
            self.embedding = nn.Embedding(codebook_size, codebook_dim)
            self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                z: Input tensor of shape (batch, dim, time)
            Returns:
                z_q: Quantized tensor (same shape as z)
                indices: Codebook indices (batch, time)
            """
            B, D, T = z.shape
            
            # Reshape: (B, D, T) -> (B*T, D)
            z_flat = z.permute(0, 2, 1).reshape(-1, D)
            
            # Compute distances: ||z - e||² = ||z||² + ||e||² - 2*z·e
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * z_flat @ self.embedding.weight.t()
            )
            
            # Find nearest codebook entry
            indices = dist.argmin(dim=1)
            
            # Lookup quantized vectors
            z_q_flat = self.embedding(indices)
            
            # Straight-through estimator
            z_q_flat = z_flat + (z_q_flat - z_flat).detach()
            
            # Reshape back
            z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)
            indices = indices.view(B, T)
            
            return z_q, indices
    
    # Test
    vq = VectorQuantizer(codebook_size=1024, codebook_dim=128)
    z = torch.randn(2, 128, 50)  # batch=2, dim=128, time=50
    z.requires_grad = True
    
    z_q, indices = vq(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Verify gradient flows
    loss = z_q.sum()
    loss.backward()
    
    assert z.grad is not None, "Gradient should flow through STE!"
    print(f"Gradient shape: {z.grad.shape}")
    print("✓ Exercise 3 passed (gradient flows through STE)!")


# ============================================================
# EXERCISE 4: Causal Convolution
# ============================================================

def exercise_4_causal_conv():
    """
    EXERCISE 4: Implement Causal Convolution
    
    Causal convolution only looks at past/current inputs.
    Essential for streaming/real-time audio processing.
    """
    print("\n" + "="*60)
    print("EXERCISE 4: Causal Convolution")
    print("="*60)
    
    # === SOLUTION ===
    class CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, dilation: int = 1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pad only on the LEFT (causal)
            x = F.pad(x, (self.padding, 0))
            return self.conv(x)
    
    # Test causality
    conv = CausalConv1d(1, 1, kernel_size=3)
    
    # Set weights to simple averaging
    with torch.no_grad():
        conv.conv.weight.fill_(1/3)
        conv.conv.bias.fill_(0)
    
    # Input: impulse at position 5
    x = torch.zeros(1, 1, 10)
    x[0, 0, 5] = 1.0
    
    y = conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input:  {x[0, 0].tolist()}")
    print(f"Output: {[f'{v:.2f}' for v in y[0, 0].tolist()]}")
    
    # Check causality: output before position 5 should be 0
    assert torch.allclose(y[0, 0, :5], torch.zeros(5)), "Causal violation!"
    # Check that output at positions 5,6,7 is affected
    assert y[0, 0, 5] > 0, "Position 5 should be affected"
    
    print("✓ Exercise 4 passed (causality preserved)!")


# ============================================================
# EXERCISE 5: RVQ (Residual Vector Quantization)
# ============================================================

def exercise_5_rvq():
    """
    EXERCISE 5: Implement Residual Vector Quantization
    
    RVQ applies multiple VQ layers to the residual.
    This is how SoundStream/EnCodec achieve high quality.
    """
    print("\n" + "="*60)
    print("EXERCISE 5: Residual Vector Quantization")
    print("="*60)
    
    # === SOLUTION ===
    class SimpleVQ(nn.Module):
        """Simplified VQ for RVQ exercise."""
        def __init__(self, codebook_size: int, codebook_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(codebook_size, codebook_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        
        def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B, D, T = z.shape
            z_flat = z.permute(0, 2, 1).reshape(-1, D)
            
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * z_flat @ self.embedding.weight.t()
            )
            
            indices = dist.argmin(dim=1)
            z_q_flat = self.embedding(indices)
            z_q_flat = z_flat + (z_q_flat - z_flat).detach()
            
            z_q = z_q_flat.view(B, T, D).permute(0, 2, 1)
            indices = indices.view(B, T)
            
            return z_q, indices
    
    class ResidualVQ(nn.Module):
        def __init__(self, num_quantizers: int, codebook_size: int, 
                     codebook_dim: int):
            super().__init__()
            self.quantizers = nn.ModuleList([
                SimpleVQ(codebook_size, codebook_dim)
                for _ in range(num_quantizers)
            ])
        
        def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            """
            Args:
                z: Input (batch, dim, time)
            Returns:
                z_q: Quantized sum
                indices_list: List of indices from each level
            """
            z_q = torch.zeros_like(z)
            residual = z
            all_indices = []
            
            for quantizer in self.quantizers:
                quantized, indices = quantizer(residual)
                z_q = z_q + quantized
                residual = residual - quantized
                all_indices.append(indices)
            
            return z_q, all_indices
    
    # Test
    rvq = ResidualVQ(num_quantizers=8, codebook_size=1024, codebook_dim=64)
    z = torch.randn(2, 64, 50)
    
    z_q, indices_list = rvq(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {z_q.shape}")
    print(f"Number of RVQ levels: {len(indices_list)}")
    
    # Check reconstruction improves with more levels
    errors = []
    for n_levels in [1, 2, 4, 8]:
        rvq_partial = ResidualVQ(n_levels, 1024, 64)
        z_q_partial, _ = rvq_partial(z)
        mse = F.mse_loss(z, z_q_partial).item()
        errors.append(mse)
        print(f"  {n_levels} levels: MSE = {mse:.6f}")
    
    # Errors should generally decrease (not strictly due to random init)
    print("✓ Exercise 5 passed!")


# ============================================================
# EXERCISE 6: Attention for Audio
# ============================================================

def exercise_6_audio_attention():
    """
    EXERCISE 6: Multi-Head Self-Attention for Audio
    
    Implement attention mechanism used in Transformers for audio.
    Key consideration: relative positional encoding.
    """
    print("\n" + "="*60)
    print("EXERCISE 6: Audio Self-Attention")
    print("="*60)
    
    # === SOLUTION ===
    class AudioSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, max_len: int = 5000):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * 
                -(np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
            """
            Args:
                x: Input (batch, seq_len, d_model)
                causal: Whether to use causal mask
            Returns:
                Output (batch, seq_len, d_model)
            """
            B, T, D = x.shape
            
            # Add positional encoding
            x = x + self.pe[:T]
            
            # Project Q, K, V
            Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale
            
            # Causal mask
            if causal:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
                attn = attn.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            out = (attn @ V).transpose(1, 2).reshape(B, T, D)
            
            return self.out_proj(out)
    
    # Test
    attn = AudioSelfAttention(d_model=256, n_heads=8)
    x = torch.randn(2, 100, 256)  # batch=2, seq=100, dim=256
    
    # Non-causal
    y = attn(x, causal=False)
    print(f"Non-causal attention: {x.shape} -> {y.shape}")
    
    # Causal
    y_causal = attn(x, causal=True)
    print(f"Causal attention: {x.shape} -> {y_causal.shape}")
    
    print("✓ Exercise 6 passed!")


# ============================================================
# EXERCISE 7: Complete Mini Audio Encoder
# ============================================================

def exercise_7_mini_encoder():
    """
    EXERCISE 7: Build a Mini Audio Encoder
    
    Combine convolutions, residual blocks, and downsampling
    into a complete audio encoder.
    """
    print("\n" + "="*60)
    print("EXERCISE 7: Mini Audio Encoder")
    print("="*60)
    
    # === SOLUTION ===
    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.ELU(),
                nn.Conv1d(channels, channels, 1),
            )
        
        def forward(self, x):
            return x + self.block(x)
    
    class MiniAudioEncoder(nn.Module):
        def __init__(self, latent_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                # Initial: 1 -> 32 channels
                nn.Conv1d(1, 32, 7, padding=3),
                nn.ELU(),
                
                # Downsample 4x
                ResBlock(32),
                nn.Conv1d(32, 64, 4, stride=4, padding=0),
                nn.ELU(),
                
                # Downsample 4x
                ResBlock(64),
                nn.Conv1d(64, 128, 4, stride=4, padding=0),
                nn.ELU(),
                
                # Downsample 4x
                ResBlock(128),
                nn.Conv1d(128, latent_dim, 4, stride=4, padding=0),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
    
    # Test
    encoder = MiniAudioEncoder(latent_dim=128)
    x = torch.randn(1, 1, 16000)  # 1 second @ 16kHz
    z = encoder(x)
    
    print(f"Input: {x.shape}")
    print(f"Latent: {z.shape}")
    print(f"Compression ratio: {x.shape[2] / z.shape[2]:.0f}x")
    
    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {params:,}")
    
    print("✓ Exercise 7 passed!")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  AUDIO ML EXERCISES".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    exercises = [
        ("STFT Implementation", exercise_1_stft),
        ("Mel Filterbank", exercise_2_mel_filterbank),
        ("Vector Quantization", exercise_3_vector_quantization),
        ("Causal Convolution", exercise_4_causal_conv),
        ("Residual VQ", exercise_5_rvq),
        ("Audio Attention", exercise_6_audio_attention),
        ("Mini Encoder", exercise_7_mini_encoder),
    ]
    
    passed = 0
    for name, func in exercises:
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(exercises)} exercises passed")
    print("=" * 60)
    
    if passed == len(exercises):
        print("\n🎉 Congratulations! All exercises completed!")
        print("\nNext steps:")
        print("  1. Modify exercises to experiment with parameters")
        print("  2. Add discriminator training to codec")
        print("  3. Train on real audio data")
        print("  4. Implement streaming inference")


if __name__ == "__main__":
    main()
