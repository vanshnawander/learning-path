"""
02_spectral_analysis_profiled.py - Spectral Analysis with Profiling

Every operation timed. Understand the real costs of audio feature extraction.
This is what happens inside torchaudio, librosa, and whisper preprocessing.

Run: python 02_spectral_analysis_profiled.py
"""

import time
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from functools import wraps

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/torchaudio not available. CPU-only benchmarks.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ============================================================
# TIMING INFRASTRUCTURE
# ============================================================

@dataclass
class TimingResult:
    name: str
    duration_ms: float
    throughput: Optional[float] = None  # samples/sec or frames/sec
    memory_mb: Optional[float] = None

def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper

def print_timing(name: str, duration_ms: float, extra: str = ""):
    """Pretty print timing result"""
    print(f"⏱️  {name:<40} {duration_ms:>8.3f} ms  {extra}")


# ============================================================
# AUDIO GENERATION (for benchmarking)
# ============================================================

@timer
def generate_test_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic audio for benchmarking"""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float32)
    
    # Multi-frequency signal with harmonics (speech-like)
    audio = (
        0.5 * np.sin(2 * np.pi * 220 * t) +      # Fundamental
        0.3 * np.sin(2 * np.pi * 440 * t) +      # 2nd harmonic
        0.15 * np.sin(2 * np.pi * 880 * t) +     # 4th harmonic
        0.05 * np.random.randn(num_samples)       # Noise
    ).astype(np.float32)
    
    return audio


# ============================================================
# FOURIER TRANSFORM IMPLEMENTATIONS
# ============================================================

@timer
def numpy_fft(signal: np.ndarray) -> np.ndarray:
    """NumPy FFT - baseline"""
    return np.fft.rfft(signal)

@timer
def numpy_stft(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    """Manual STFT implementation with NumPy"""
    num_frames = (len(signal) - n_fft) // hop_length + 1
    window = np.hanning(n_fft).astype(np.float32)
    
    # Pre-allocate output
    stft_matrix = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)
    
    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(frame)
    
    return stft_matrix

@timer
def vectorized_stft(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    """Vectorized STFT - faster than loop-based"""
    num_frames = (len(signal) - n_fft) // hop_length + 1
    window = np.hanning(n_fft).astype(np.float32)
    
    # Create frame indices
    frame_indices = np.arange(num_frames)[:, np.newaxis]
    sample_indices = np.arange(n_fft)
    indices = frame_indices * hop_length + sample_indices
    
    # Extract all frames at once
    frames = signal[indices] * window
    
    # FFT all frames
    return np.fft.rfft(frames, axis=1).T

if LIBROSA_AVAILABLE:
    @timer
    def librosa_stft(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
        """Librosa STFT - uses optimized FFT"""
        return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

if TORCH_AVAILABLE:
    @timer
    def torch_stft_cpu(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
        """PyTorch STFT on CPU"""
        signal_t = torch.from_numpy(signal)
        window = torch.hann_window(n_fft)
        return torch.stft(signal_t, n_fft=n_fft, hop_length=hop_length, 
                         window=window, return_complex=True)
    
    @timer
    def torch_stft_gpu(signal: np.ndarray, n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
        """PyTorch STFT on GPU"""
        device = torch.device('cuda')
        signal_t = torch.from_numpy(signal).to(device)
        window = torch.hann_window(n_fft).to(device)
        result = torch.stft(signal_t, n_fft=n_fft, hop_length=hop_length,
                           window=window, return_complex=True)
        torch.cuda.synchronize()  # Ensure GPU computation is complete
        return result


# ============================================================
# MEL SPECTROGRAM IMPLEMENTATIONS
# ============================================================

def create_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 80, 
                          fmin: float = 0, fmax: float = None) -> np.ndarray:
    """Create mel filterbank matrix"""
    if fmax is None:
        fmax = sample_rate / 2
    
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Create mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert to FFT bins
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # Create filterbank
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # Rising edge
        for j in range(left, center):
            filterbank[i, j] = (j - left) / (center - left)
        # Falling edge
        for j in range(center, right):
            filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank

@timer
def numpy_mel_spectrogram(signal: np.ndarray, sample_rate: int = 16000,
                          n_fft: int = 512, hop_length: int = 160, 
                          n_mels: int = 80) -> np.ndarray:
    """Complete mel spectrogram pipeline with NumPy"""
    # STFT
    stft_result, _ = vectorized_stft(signal, n_fft, hop_length)
    
    # Power spectrogram
    power_spec = np.abs(stft_result) ** 2
    
    # Mel filterbank
    mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels)
    
    # Apply filterbank
    mel_spec = mel_fb @ power_spec
    
    # Log scale
    log_mel_spec = np.log(mel_spec + 1e-10)
    
    return log_mel_spec

if LIBROSA_AVAILABLE:
    @timer
    def librosa_mel_spectrogram(signal: np.ndarray, sample_rate: int = 16000,
                                n_fft: int = 512, hop_length: int = 160,
                                n_mels: int = 80) -> np.ndarray:
        """Librosa mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sample_rate, n_fft=n_fft, 
            hop_length=hop_length, n_mels=n_mels
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

if TORCH_AVAILABLE:
    @timer
    def torch_mel_spectrogram_cpu(signal: np.ndarray, sample_rate: int = 16000,
                                   n_fft: int = 512, hop_length: int = 160,
                                   n_mels: int = 80) -> torch.Tensor:
        """torchaudio mel spectrogram on CPU"""
        signal_t = torch.from_numpy(signal).unsqueeze(0)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, 
            hop_length=hop_length, n_mels=n_mels
        )
        mel_spec = transform(signal_t)
        return torchaudio.functional.amplitude_to_DB(mel_spec, 10, 1e-10, 0, 80)
    
    @timer  
    def torch_mel_spectrogram_gpu(signal: np.ndarray, sample_rate: int = 16000,
                                   n_fft: int = 512, hop_length: int = 160,
                                   n_mels: int = 80) -> torch.Tensor:
        """torchaudio mel spectrogram on GPU"""
        device = torch.device('cuda')
        signal_t = torch.from_numpy(signal).unsqueeze(0).to(device)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        ).to(device)
        mel_spec = transform(signal_t)
        result = torchaudio.functional.amplitude_to_DB(mel_spec, 10, 1e-10, 0, 80)
        torch.cuda.synchronize()
        return result


# ============================================================
# RESAMPLING BENCHMARKS
# ============================================================

@timer
def numpy_resample_linear(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling"""
    ratio = target_sr / orig_sr
    new_length = int(len(signal) * ratio)
    indices = np.arange(new_length) / ratio
    left_indices = np.floor(indices).astype(int)
    right_indices = np.minimum(left_indices + 1, len(signal) - 1)
    fractions = indices - left_indices
    
    return signal[left_indices] * (1 - fractions) + signal[right_indices] * fractions

if LIBROSA_AVAILABLE:
    @timer
    def librosa_resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Librosa resampling (polyphase filter)"""
        return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)

if TORCH_AVAILABLE:
    @timer
    def torch_resample_cpu(signal: np.ndarray, orig_sr: int, target_sr: int) -> torch.Tensor:
        """torchaudio resampling on CPU"""
        signal_t = torch.from_numpy(signal).unsqueeze(0)
        return torchaudio.functional.resample(signal_t, orig_sr, target_sr)
    
    @timer
    def torch_resample_gpu(signal: np.ndarray, orig_sr: int, target_sr: int) -> torch.Tensor:
        """torchaudio resampling on GPU"""
        device = torch.device('cuda')
        signal_t = torch.from_numpy(signal).unsqueeze(0).to(device)
        result = torchaudio.functional.resample(signal_t, orig_sr, target_sr)
        torch.cuda.synchronize()
        return result


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_stft_benchmark(duration_sec: float = 10.0, sample_rate: int = 16000):
    """Compare STFT implementations"""
    print("\n" + "=" * 70)
    print(f"STFT BENCHMARK - {duration_sec}s audio @ {sample_rate} Hz")
    print("=" * 70 + "\n")
    
    audio, gen_time = generate_test_audio(duration_sec, sample_rate)
    print_timing("Generate test audio", gen_time, f"({len(audio):,} samples)")
    print()
    
    n_fft = 512
    hop_length = 160
    expected_frames = (len(audio) - n_fft) // hop_length + 1
    
    print(f"Parameters: n_fft={n_fft}, hop_length={hop_length}")
    print(f"Expected output: {n_fft//2 + 1} freq bins × {expected_frames} frames")
    print("-" * 70)
    
    # NumPy loop-based
    result, time_ms = numpy_stft(audio, n_fft, hop_length)
    print_timing("NumPy STFT (loop)", time_ms, f"shape: {result.shape}")
    
    # NumPy vectorized
    result, time_ms = vectorized_stft(audio, n_fft, hop_length)
    print_timing("NumPy STFT (vectorized)", time_ms, f"shape: {result.shape}")
    
    if LIBROSA_AVAILABLE:
        result, time_ms = librosa_stft(audio, n_fft, hop_length)
        print_timing("Librosa STFT", time_ms, f"shape: {result.shape}")
    
    if TORCH_AVAILABLE:
        result, time_ms = torch_stft_cpu(audio, n_fft, hop_length)
        print_timing("PyTorch STFT (CPU)", time_ms, f"shape: {tuple(result.shape)}")
        
        if torch.cuda.is_available():
            # Warmup
            _ = torch_stft_gpu(audio, n_fft, hop_length)
            result, time_ms = torch_stft_gpu(audio, n_fft, hop_length)
            print_timing("PyTorch STFT (GPU)", time_ms, f"shape: {tuple(result.shape)}")

def run_mel_benchmark(duration_sec: float = 10.0, sample_rate: int = 16000):
    """Compare mel spectrogram implementations"""
    print("\n" + "=" * 70)
    print(f"MEL SPECTROGRAM BENCHMARK - {duration_sec}s audio @ {sample_rate} Hz")
    print("=" * 70 + "\n")
    
    audio, _ = generate_test_audio(duration_sec, sample_rate)
    
    n_fft = 512
    hop_length = 160
    n_mels = 80
    
    print(f"Parameters: n_fft={n_fft}, hop_length={hop_length}, n_mels={n_mels}")
    print("-" * 70)
    
    result, time_ms = numpy_mel_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels)
    print_timing("NumPy mel spectrogram", time_ms, f"shape: {result.shape}")
    
    if LIBROSA_AVAILABLE:
        result, time_ms = librosa_mel_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels)
        print_timing("Librosa mel spectrogram", time_ms, f"shape: {result.shape}")
    
    if TORCH_AVAILABLE:
        result, time_ms = torch_mel_spectrogram_cpu(audio, sample_rate, n_fft, hop_length, n_mels)
        print_timing("PyTorch mel spectrogram (CPU)", time_ms, f"shape: {tuple(result.shape)}")
        
        if torch.cuda.is_available():
            # Warmup
            _ = torch_mel_spectrogram_gpu(audio, sample_rate, n_fft, hop_length, n_mels)
            result, time_ms = torch_mel_spectrogram_gpu(audio, sample_rate, n_fft, hop_length, n_mels)
            print_timing("PyTorch mel spectrogram (GPU)", time_ms, f"shape: {tuple(result.shape)}")

def run_resample_benchmark(duration_sec: float = 10.0):
    """Compare resampling implementations"""
    print("\n" + "=" * 70)
    print(f"RESAMPLING BENCHMARK - {duration_sec}s audio")
    print("=" * 70 + "\n")
    
    orig_sr = 44100
    target_sr = 16000
    
    audio, _ = generate_test_audio(duration_sec, orig_sr)
    expected_length = int(len(audio) * target_sr / orig_sr)
    
    print(f"Resample: {orig_sr} Hz → {target_sr} Hz")
    print(f"Samples: {len(audio):,} → {expected_length:,}")
    print("-" * 70)
    
    result, time_ms = numpy_resample_linear(audio, orig_sr, target_sr)
    print_timing("NumPy linear interpolation", time_ms, f"length: {len(result):,}")
    
    if LIBROSA_AVAILABLE:
        result, time_ms = librosa_resample(audio, orig_sr, target_sr)
        print_timing("Librosa (polyphase)", time_ms, f"length: {len(result):,}")
    
    if TORCH_AVAILABLE:
        result, time_ms = torch_resample_cpu(audio, orig_sr, target_sr)
        print_timing("PyTorch resample (CPU)", time_ms, f"length: {result.shape[-1]:,}")
        
        if torch.cuda.is_available():
            # Warmup
            _ = torch_resample_gpu(audio, orig_sr, target_sr)
            result, time_ms = torch_resample_gpu(audio, orig_sr, target_sr)
            print_timing("PyTorch resample (GPU)", time_ms, f"length: {result.shape[-1]:,}")

def run_memory_analysis():
    """Analyze memory usage for different representations"""
    print("\n" + "=" * 70)
    print("MEMORY ANALYSIS - 1 minute of audio at different sample rates")
    print("=" * 70 + "\n")
    
    duration = 60  # 1 minute
    
    print(f"{'Sample Rate':<15} {'Raw (16-bit)':<15} {'Raw (float32)':<15} {'Mel Spec':<15}")
    print("-" * 60)
    
    for sr in [8000, 16000, 24000, 44100, 48000]:
        samples = sr * duration
        raw_16bit = samples * 2  # 2 bytes per sample
        raw_float = samples * 4  # 4 bytes per sample
        
        # Mel spectrogram: n_mels × num_frames × 4 bytes
        n_fft = 512
        hop_length = 160
        n_mels = 80
        num_frames = (samples - n_fft) // hop_length + 1
        mel_size = n_mels * num_frames * 4
        
        print(f"{sr:>10} Hz   {raw_16bit/1024/1024:>10.2f} MB   "
              f"{raw_float/1024/1024:>10.2f} MB   {mel_size/1024/1024:>10.2f} MB")


def print_summary():
    """Print key takeaways"""
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. STFT PERFORMANCE
   - Vectorized NumPy is 5-10x faster than loop-based
   - Librosa uses optimized FFTW backend
   - GPU provides 10-50x speedup for batch processing

2. MEL SPECTROGRAM
   - Most time spent in FFT computation
   - Filterbank application is cheap
   - Pre-compute and cache for training!

3. RESAMPLING
   - Polyphase filtering (librosa) is high quality but slow
   - Linear interpolation is fast but poor quality
   - ALWAYS pre-resample your dataset to target sample rate

4. MEMORY
   - Mel spectrograms are MORE compact than raw audio at 16kHz
   - Higher sample rates = proportionally more memory
   - 16kHz is the sweet spot for speech ML

5. GPU ACCELERATION
   - Worth it for batch processing (>= 8 samples)
   - Data transfer overhead for single samples
   - Keep data on GPU through the pipeline
""")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SPECTRAL ANALYSIS PROFILING".center(68) + "█")
    print("█" + "  Understanding the real costs of audio preprocessing".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    run_stft_benchmark(duration_sec=10.0)
    run_mel_benchmark(duration_sec=10.0)
    run_resample_benchmark(duration_sec=10.0)
    run_memory_analysis()
    print_summary()
