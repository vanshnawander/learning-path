"""
03_mel_spectrograms_deep_dive.py - Complete Mel Spectrogram Implementation

From first principles: Hz → Mel scale → Filterbanks → Feature extraction
Every step explained with profiling.

This is what Whisper, WavLM, and every speech model uses for input features.
"""

import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# MEL SCALE: WHY AND HOW
# ============================================================

"""
THE MEL SCALE - Human Frequency Perception

Humans perceive pitch logarithmically, not linearly:
- Difference between 100 Hz and 200 Hz sounds like an octave
- Difference between 1000 Hz and 1100 Hz sounds like a small change
- Yet both are 100 Hz apart!

The Mel scale compresses high frequencies, matching human perception.

Standard formula (O'Shaughnessy, 1987):
    mel = 2595 * log10(1 + hz / 700)
    
Alternative (Slaney, 1998 - used by librosa):
    mel = (hz / f_sp) for hz < min_log_hz
    mel = min_log_mel + log(hz / min_log_hz) / log_step for hz >= min_log_hz
    where f_sp = 200/3, min_log_hz = 1000, log_step = 0.068751777
"""

def hz_to_mel_htk(hz: np.ndarray) -> np.ndarray:
    """HTK-style mel conversion (O'Shaughnessy formula)
    
    This is the standard used in:
    - torchaudio (default)
    - Kaldi
    - Most speech recognition systems
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz_htk(mel: np.ndarray) -> np.ndarray:
    """Inverse HTK mel conversion"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    """Slaney-style mel conversion (librosa default)
    
    Linear below 1000 Hz, logarithmic above.
    Slightly different from HTK but empirically similar.
    """
    f_sp = 200.0 / 3  # ~66.67 Hz
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0) / f_sp
    logstep = np.log(6.4) / 27.0  # step size for log region
    
    hz = np.asarray(hz)
    mel = np.zeros_like(hz)
    
    linear_region = hz < min_log_hz
    mel[linear_region] = hz[linear_region] / f_sp
    mel[~linear_region] = min_log_mel + np.log(hz[~linear_region] / min_log_hz) / logstep
    
    return mel

def mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    """Inverse Slaney mel conversion"""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    
    mel = np.asarray(mel)
    hz = np.zeros_like(mel)
    
    linear_region = mel < min_log_mel
    hz[linear_region] = f_sp * mel[linear_region]
    hz[~linear_region] = min_log_hz * np.exp(logstep * (mel[~linear_region] - min_log_mel))
    
    return hz


# ============================================================
# MEL FILTERBANK CONSTRUCTION
# ============================================================

@dataclass
class MelFilterbankConfig:
    """Configuration for mel filterbank"""
    sample_rate: int = 16000
    n_fft: int = 512
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None  # Default: sample_rate / 2
    htk: bool = True  # Use HTK formula (True) or Slaney (False)
    norm: Optional[str] = 'slaney'  # 'slaney' normalizes area, None for no norm

def create_mel_filterbank(config: MelFilterbankConfig) -> np.ndarray:
    """
    Create triangular mel filterbank matrix.
    
    Returns:
        filterbank: shape (n_mels, n_fft // 2 + 1)
        
    Each row is one triangular filter centered at a mel frequency.
    Multiply with power spectrum to get mel-band energies.
    """
    f_max = config.f_max or config.sample_rate / 2
    n_freqs = config.n_fft // 2 + 1
    
    # Choose mel conversion function
    hz_to_mel = hz_to_mel_htk if config.htk else hz_to_mel_slaney
    mel_to_hz = mel_to_hz_htk if config.htk else mel_to_hz_slaney
    
    # Create n_mels + 2 points (including edges)
    mel_min = hz_to_mel(np.array([config.f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz to FFT bin indices
    # bin = round(hz * n_fft / sample_rate)
    bin_points = np.floor((config.n_fft + 1) * hz_points / config.sample_rate).astype(int)
    
    # Create filterbank matrix
    filterbank = np.zeros((config.n_mels, n_freqs), dtype=np.float32)
    
    for i in range(config.n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # Rising edge: left to center
        if center > left:
            filterbank[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        
        # Falling edge: center to right
        if right > center:
            filterbank[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    
    # Slaney-style normalization: each filter has unit area
    if config.norm == 'slaney':
        enorm = 2.0 / (hz_points[2:config.n_mels + 2] - hz_points[:config.n_mels])
        filterbank *= enorm[:, np.newaxis]
    
    return filterbank


def visualize_mel_filterbank(filterbank: np.ndarray, sample_rate: int, n_fft: int,
                             save_path: str = 'mel_filterbank.png'):
    """Visualize mel filterbank as overlapping triangular filters"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return
    
    n_mels, n_freqs = filterbank.shape
    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Linear frequency axis
    ax1 = axes[0]
    for i in range(n_mels):
        ax1.plot(freqs, filterbank[i], alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Mel Filterbank ({n_mels} filters) - Linear Frequency')
    ax1.set_xlim(0, sample_rate / 2)
    
    # Log frequency axis (shows mel spacing better)
    ax2 = axes[1]
    for i in range(n_mels):
        ax2.semilogx(freqs[1:], filterbank[i, 1:], alpha=0.7)  # Skip 0 Hz for log
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Mel Filterbank ({n_mels} filters) - Log Frequency')
    ax2.set_xlim(20, sample_rate / 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved filterbank visualization to {save_path}")


# ============================================================
# COMPLETE MEL SPECTROGRAM PIPELINE
# ============================================================

class MelSpectrogramExtractor:
    """
    Complete mel spectrogram feature extractor with profiling.
    
    Pipeline:
    1. Pre-emphasis (optional)
    2. Framing with windowing
    3. FFT → Power spectrum
    4. Mel filterbank application
    5. Log compression
    6. Normalization (optional)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pre_emphasis: float = 0.97,
        log_offset: float = 1e-6,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.pre_emphasis = pre_emphasis
        self.log_offset = log_offset
        self.normalize = normalize
        
        # Pre-compute window and filterbank
        self.window = np.hanning(self.win_length).astype(np.float32)
        self.mel_filterbank = create_mel_filterbank(MelFilterbankConfig(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max
        ))
        
        # Profiling stats
        self.timing = {}
    
    def _time_step(self, name: str):
        """Context manager for timing steps"""
        class Timer:
            def __init__(self, timing_dict, step_name):
                self.timing_dict = timing_dict
                self.step_name = step_name
                self.start = None
            
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            
            def __exit__(self, *args):
                elapsed = (time.perf_counter() - self.start) * 1000
                self.timing_dict[self.step_name] = elapsed
        
        return Timer(self.timing, name)
    
    def extract(self, audio: np.ndarray, return_timing: bool = False) -> np.ndarray:
        """
        Extract mel spectrogram features.
        
        Args:
            audio: 1D numpy array of audio samples (float32, normalized to [-1, 1])
            return_timing: If True, return (features, timing_dict)
            
        Returns:
            mel_spec: shape (n_mels, num_frames)
        """
        self.timing = {}
        
        # Step 1: Pre-emphasis
        with self._time_step("1_pre_emphasis"):
            if self.pre_emphasis > 0:
                audio = np.concatenate([audio[:1], audio[1:] - self.pre_emphasis * audio[:-1]])
        
        # Step 2: Pad to ensure we get at least one frame
        with self._time_step("2_padding"):
            if len(audio) < self.n_fft:
                audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        # Step 3: Frame the signal
        with self._time_step("3_framing"):
            num_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            frame_indices = np.arange(num_frames)[:, np.newaxis] * self.hop_length
            sample_indices = np.arange(self.win_length)
            indices = frame_indices + sample_indices
            
            # Pad audio to handle edge case
            padded_audio = np.pad(audio, (0, max(0, indices.max() + 1 - len(audio))))
            frames = padded_audio[indices]  # shape: (num_frames, win_length)
        
        # Step 4: Apply window
        with self._time_step("4_windowing"):
            windowed_frames = frames * self.window
        
        # Step 5: Zero-pad to n_fft if win_length < n_fft
        with self._time_step("5_zero_padding"):
            if self.win_length < self.n_fft:
                pad_amount = self.n_fft - self.win_length
                windowed_frames = np.pad(windowed_frames, ((0, 0), (0, pad_amount)))
        
        # Step 6: FFT → Power spectrum
        with self._time_step("6_fft"):
            spectrum = np.fft.rfft(windowed_frames, n=self.n_fft, axis=1)
            power_spectrum = np.abs(spectrum) ** 2  # shape: (num_frames, n_fft//2 + 1)
        
        # Step 7: Apply mel filterbank
        with self._time_step("7_mel_filterbank"):
            mel_spec = np.dot(power_spectrum, self.mel_filterbank.T)  # shape: (num_frames, n_mels)
        
        # Step 8: Log compression
        with self._time_step("8_log_compression"):
            log_mel_spec = np.log(mel_spec + self.log_offset)
        
        # Step 9: Normalize (optional - per-utterance)
        with self._time_step("9_normalize"):
            if self.normalize:
                log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Transpose to (n_mels, num_frames) - standard format
        features = log_mel_spec.T
        
        if return_timing:
            return features, self.timing
        return features
    
    def print_timing(self):
        """Print timing breakdown"""
        if not self.timing:
            print("No timing data. Run extract() first.")
            return
        
        total = sum(self.timing.values())
        print("\n" + "=" * 60)
        print("MEL SPECTROGRAM EXTRACTION TIMING")
        print("=" * 60)
        for step, time_ms in self.timing.items():
            pct = 100 * time_ms / total
            bar = "█" * int(pct / 2)
            print(f"{step:<25} {time_ms:>8.3f} ms ({pct:>5.1f}%) {bar}")
        print("-" * 60)
        print(f"{'TOTAL':<25} {total:>8.3f} ms")


# ============================================================
# COMPARISON WITH REFERENCE IMPLEMENTATIONS
# ============================================================

def compare_with_torchaudio(audio: np.ndarray, sample_rate: int = 16000):
    """Compare our implementation with torchaudio"""
    if not TORCH_AVAILABLE:
        print("torchaudio not available for comparison")
        return
    
    print("\n" + "=" * 60)
    print("COMPARISON: Custom vs torchaudio")
    print("=" * 60)
    
    # Our implementation
    extractor = MelSpectrogramExtractor(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        pre_emphasis=0,  # Disable for fair comparison
        normalize=False
    )
    
    start = time.perf_counter()
    our_result = extractor.extract(audio)
    our_time = (time.perf_counter() - start) * 1000
    
    # torchaudio
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )
    
    start = time.perf_counter()
    torch_mel = mel_transform(audio_t)
    torch_log_mel = torch.log(torch_mel + 1e-6)
    torch_result = torch_log_mel.squeeze(0).numpy()
    torch_time = (time.perf_counter() - start) * 1000
    
    print(f"\nShape comparison:")
    print(f"  Custom:     {our_result.shape}")
    print(f"  torchaudio: {torch_result.shape}")
    
    print(f"\nTiming:")
    print(f"  Custom:     {our_time:.3f} ms")
    print(f"  torchaudio: {torch_time:.3f} ms")
    
    # Check similarity (they may differ slightly due to implementation details)
    min_frames = min(our_result.shape[1], torch_result.shape[1])
    diff = np.abs(our_result[:, :min_frames] - torch_result[:, :min_frames])
    print(f"\nNumerical difference (first {min_frames} frames):")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")


# ============================================================
# WHISPER-STYLE FEATURES
# ============================================================

def create_whisper_mel_extractor():
    """
    Create mel extractor matching Whisper's configuration.
    
    Whisper uses:
    - 80 mel bins
    - 16 kHz sample rate
    - 25ms window (400 samples)
    - 10ms hop (160 samples)
    - Hann window
    - Log mel spectrogram
    """
    return MelSpectrogramExtractor(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        f_min=0,
        f_max=8000,
        pre_emphasis=0,  # Whisper doesn't use pre-emphasis
        log_offset=1e-10,
        normalize=False  # Whisper normalizes differently
    )


# ============================================================
# DEMO
# ============================================================

def run_demo():
    """Demonstrate mel spectrogram extraction with profiling"""
    print("\n" + "█" * 60)
    print("█" + " MEL SPECTROGRAM DEEP DIVE ".center(58) + "█")
    print("█" * 60)
    
    # Generate test audio
    duration = 5.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    
    # Speech-like signal: varying frequency with harmonics
    f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Varying fundamental
    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.1 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)
    
    print(f"\nTest audio: {duration}s @ {sample_rate} Hz ({len(audio):,} samples)")
    
    # Create extractor and extract features
    extractor = MelSpectrogramExtractor(
        sample_rate=sample_rate,
        n_fft=512,
        hop_length=160,
        n_mels=80
    )
    
    features, timing = extractor.extract(audio, return_timing=True)
    
    print(f"Output shape: {features.shape} (n_mels × num_frames)")
    print(f"Feature rate: {features.shape[1] / duration:.1f} frames/second")
    
    extractor.print_timing()
    
    # Visualize filterbank
    print("\n" + "-" * 60)
    print("Creating mel filterbank visualization...")
    visualize_mel_filterbank(
        extractor.mel_filterbank, 
        sample_rate, 
        extractor.n_fft,
        'mel_filterbank.png'
    )
    
    # Compare with torchaudio
    compare_with_torchaudio(audio, sample_rate)
    
    # Memory analysis
    print("\n" + "=" * 60)
    print("MEMORY ANALYSIS")
    print("=" * 60)
    raw_bytes = audio.nbytes
    feature_bytes = features.nbytes
    compression = raw_bytes / feature_bytes
    print(f"Raw audio:     {raw_bytes / 1024:.2f} KB")
    print(f"Mel features:  {feature_bytes / 1024:.2f} KB")
    print(f"Compression:   {compression:.1f}x")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. FFT dominates computation (~60-80% of time)
   → Use GPU or optimized libraries for large batches

2. Mel filterbank is pre-computed and cheap to apply
   → Matrix multiplication, very fast

3. Normalization affects downstream model performance
   → Whisper: global mean/std from training data
   → Per-utterance: (x - mean) / std

4. Memory: Mel features are compact representation
   → Good for storage and training efficiency

5. Feature rate: 100 frames/sec is common (10ms hop)
   → Matches typical speech analysis resolution
""")


if __name__ == "__main__":
    run_demo()
