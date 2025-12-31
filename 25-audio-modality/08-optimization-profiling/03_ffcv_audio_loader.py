"""
03_ffcv_audio_loader.py - FFCV Data Loading for Audio

FFCV (Fast Forward Computer Vision) can be adapted for audio data loading,
providing significant speedups over standard PyTorch DataLoader.

This module demonstrates:
1. Creating .beton files from audio datasets
2. Custom audio field for FFCV
3. Audio-specific decoders and transforms
4. Benchmarking against standard approaches

Requirements:
    pip install ffcv torch torchaudio numpy

Run: python 03_ffcv_audio_loader.py
"""

import os
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torchaudio

# Check FFCV availability
try:
    from ffcv.writer import DatasetWriter
    from ffcv.reader import Reader
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import Field, NDArrayField, IntField, FloatField
    from ffcv.fields.decoders import NDArrayDecoder, IntDecoder, FloatDecoder
    from ffcv.transforms import ToTensor, ToDevice, Squeeze
    from ffcv.pipeline.operation import Operation
    from ffcv.pipeline.allocation_query import AllocationQuery
    from ffcv.pipeline.state import State
    FFCV_AVAILABLE = True
except ImportError:
    FFCV_AVAILABLE = False
    print("FFCV not available. Install with: pip install ffcv")


# ============================================================
# AUDIO DATASET FOR FFCV
# ============================================================

class AudioDataset:
    """
    Simple audio dataset that returns preprocessed numpy arrays.
    
    FFCV requires datasets to return tuples of numpy arrays or scalars.
    Audio is pre-processed (loaded, resampled, padded) before writing to .beton.
    """
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        compute_mel: bool = False,
        n_mels: int = 80,
    ):
        self.audio_files = sorted(Path(audio_dir).glob("*.wav"))
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.compute_mel = compute_mel
        self.n_mels = n_mels
        
        if compute_mel:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=160,
                n_mels=n_mels,
            )
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Return (audio_or_mel, label) as numpy arrays."""
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        # Convert to mono
        audio = audio.mean(dim=0)
        
        # Pad or trim to fixed length
        if audio.shape[0] > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.max_samples - audio.shape[0])
            )
        
        if self.compute_mel:
            # Compute mel spectrogram
            mel = self.mel_transform(audio.unsqueeze(0))
            mel = torch.log(mel.clamp(min=1e-10))
            data = mel.squeeze(0).numpy().astype(np.float32)
        else:
            data = audio.numpy().astype(np.float32)
        
        # Label (using file index for demo)
        label = idx % 10  # Fake 10 classes
        
        return data, label


# ============================================================
# FFCV WRITER FOR AUDIO
# ============================================================

def write_audio_beton(
    audio_dir: str,
    output_path: str,
    sample_rate: int = 16000,
    max_duration: float = 10.0,
    compute_mel: bool = False,
    n_mels: int = 80,
    num_workers: int = 4,
):
    """
    Write audio dataset to FFCV .beton format.
    
    Pre-computes features (waveform or mel) and stores efficiently.
    """
    if not FFCV_AVAILABLE:
        raise RuntimeError("FFCV not available")
    
    print(f"Creating FFCV dataset from {audio_dir}")
    
    # Create dataset
    dataset = AudioDataset(
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
        compute_mel=compute_mel,
        n_mels=n_mels,
    )
    
    # Determine field shapes
    sample_data, sample_label = dataset[0]
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample label: {sample_label}")
    
    # Define fields
    fields = {
        'audio': NDArrayField(dtype=np.dtype('float32'), shape=sample_data.shape),
        'label': IntField(),
    }
    
    # Write dataset
    writer = DatasetWriter(
        output_path,
        fields,
        num_workers=num_workers,
    )
    
    writer.from_indexed_dataset(dataset)
    
    print(f"Written to {output_path}")
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"File size: {file_size_mb:.2f} MB")
    
    return output_path


# ============================================================
# FFCV LOADER FOR AUDIO
# ============================================================

def create_audio_loader(
    beton_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: int = 0,
    shuffle: bool = True,
) -> 'Loader':
    """
    Create FFCV loader for audio data.
    
    Returns batches directly on GPU for maximum efficiency.
    """
    if not FFCV_AVAILABLE:
        raise RuntimeError("FFCV not available")
    
    # Define pipeline for each field
    pipelines = {
        'audio': [
            NDArrayDecoder(),
            ToTensor(),
            ToDevice(torch.device(f'cuda:{device}'), non_blocking=True),
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            ToDevice(torch.device(f'cuda:{device}'), non_blocking=True),
            Squeeze(),
        ],
    }
    
    # Create loader
    loader = Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
        pipelines=pipelines,
        drop_last=True,
    )
    
    return loader


# ============================================================
# CUSTOM AUDIO TRANSFORMS FOR FFCV
# ============================================================

if FFCV_AVAILABLE:
    
    class AudioNormalize(Operation):
        """Normalize audio to [-1, 1] range."""
        
        def generate_code(self) -> Callable:
            def normalize(audio, dst):
                max_val = np.abs(audio).max(axis=-1, keepdims=True)
                np.divide(audio, max_val + 1e-8, out=dst)
                return dst
            return normalize
        
        def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
            return previous_state, AllocationQuery(previous_state.shape, previous_state.dtype)
    
    
    class AudioRandomGain(Operation):
        """Apply random gain to audio."""
        
        def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0):
            super().__init__()
            self.min_gain = 10 ** (min_gain_db / 20)
            self.max_gain = 10 ** (max_gain_db / 20)
        
        def generate_code(self) -> Callable:
            min_gain = self.min_gain
            max_gain = self.max_gain
            
            def apply_gain(audio, dst):
                gain = np.random.uniform(min_gain, max_gain)
                np.multiply(audio, gain, out=dst)
                return dst
            
            return apply_gain
        
        def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
            return previous_state, AllocationQuery(previous_state.shape, previous_state.dtype)
    
    
    class AudioAddNoise(Operation):
        """Add Gaussian noise to audio."""
        
        def __init__(self, noise_level: float = 0.01):
            super().__init__()
            self.noise_level = noise_level
        
        def generate_code(self) -> Callable:
            noise_level = self.noise_level
            
            def add_noise(audio, dst):
                noise = np.random.randn(*audio.shape).astype(audio.dtype) * noise_level
                np.add(audio, noise, out=dst)
                return dst
            
            return add_noise
        
        def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
            return previous_state, AllocationQuery(previous_state.shape, previous_state.dtype)


# ============================================================
# BENCHMARKING
# ============================================================

def create_test_audio_files(directory: str, num_files: int = 100,
                            duration: float = 10.0, sample_rate: int = 16000):
    """Create synthetic WAV files for benchmarking."""
    os.makedirs(directory, exist_ok=True)
    
    for i in range(num_files):
        num_samples = int(duration * sample_rate)
        audio = torch.randn(1, num_samples) * 0.5
        filepath = os.path.join(directory, f"audio_{i:04d}.wav")
        torchaudio.save(filepath, audio, sample_rate)
    
    print(f"Created {num_files} test audio files")


def benchmark_ffcv_vs_pytorch(
    audio_dir: str,
    beton_path: str,
    batch_size: int = 32,
    num_batches: int = 50,
):
    """Compare FFCV loader with standard PyTorch DataLoader."""
    
    print("\n" + "=" * 70)
    print("FFCV vs PyTorch DataLoader Benchmark")
    print("=" * 70)
    
    results = {}
    
    # === PyTorch DataLoader ===
    print("\n--- PyTorch DataLoader ---")
    
    dataset = AudioDataset(audio_dir)
    pytorch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Warmup
    for i, batch in enumerate(pytorch_loader):
        if i >= 3:
            break
    
    # Benchmark
    start = time.perf_counter()
    batches_processed = 0
    for batch in pytorch_loader:
        audio = batch[0].cuda(non_blocking=True)
        torch.cuda.synchronize()
        batches_processed += 1
        if batches_processed >= num_batches:
            break
    
    pytorch_time = time.perf_counter() - start
    results["pytorch"] = pytorch_time
    print(f"Time: {pytorch_time:.3f}s for {batches_processed} batches")
    print(f"Throughput: {batches_processed * batch_size / pytorch_time:.1f} samples/sec")
    
    # === FFCV Loader ===
    if FFCV_AVAILABLE:
        print("\n--- FFCV Loader ---")
        
        ffcv_loader = create_audio_loader(
            beton_path,
            batch_size=batch_size,
            num_workers=4,
        )
        
        # Warmup
        for i, batch in enumerate(ffcv_loader):
            if i >= 3:
                break
        
        # Benchmark
        start = time.perf_counter()
        batches_processed = 0
        for batch in ffcv_loader:
            audio = batch[0]  # Already on GPU
            torch.cuda.synchronize()
            batches_processed += 1
            if batches_processed >= num_batches:
                break
        
        ffcv_time = time.perf_counter() - start
        results["ffcv"] = ffcv_time
        print(f"Time: {ffcv_time:.3f}s for {batches_processed} batches")
        print(f"Throughput: {batches_processed * batch_size / ffcv_time:.1f} samples/sec")
        
        # Speedup
        speedup = pytorch_time / ffcv_time
        print(f"\nFFCV Speedup: {speedup:.2f}x")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  FFCV AUDIO DATA LOADING".center(68) + "█")
    print("█" + "  Fast data loading for audio training".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\nNo GPU available!")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    if not FFCV_AVAILABLE:
        print("\nFFCV not available. Install with: pip install ffcv")
        print("Note: FFCV requires specific system dependencies.")
        print("See: https://docs.ffcv.io/installation.html")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        beton_path = os.path.join(temp_dir, "audio.beton")
        
        # Create test dataset
        print("\nCreating test audio files...")
        create_test_audio_files(audio_dir, num_files=200, duration=10.0)
        
        # Write to FFCV format
        print("\nWriting FFCV dataset...")
        write_audio_beton(
            audio_dir=audio_dir,
            output_path=beton_path,
            sample_rate=16000,
            max_duration=10.0,
        )
        
        # Benchmark
        benchmark_ffcv_vs_pytorch(
            audio_dir=audio_dir,
            beton_path=beton_path,
            batch_size=32,
            num_batches=30,
        )
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: FFCV FOR AUDIO")
    print("=" * 70)
    print("""
1. FFCV excels when data is pre-processed
   - Write mel spectrograms to .beton, not raw audio
   - Avoids runtime feature extraction

2. Memory-mapped files enable fast random access
   - No need to load entire dataset into memory
   - Efficient shuffling

3. Best for fixed-length data
   - FFCV works best with uniform shapes
   - Pad audio to max length before writing

4. Complementary to DALI
   - FFCV: Pre-compute features, fast loading
   - DALI: Runtime GPU processing

5. Storage trade-off
   - Mel spectrograms may be larger than compressed audio
   - But loading is much faster
   
WHEN TO USE:
- Training with static preprocessing
- Large datasets that don't fit in memory
- When data loading is the bottleneck
""")


if __name__ == "__main__":
    main()
