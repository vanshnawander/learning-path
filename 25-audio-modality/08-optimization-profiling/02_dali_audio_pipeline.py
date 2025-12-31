"""
02_dali_audio_pipeline.py - Complete NVIDIA DALI Audio Pipeline

Production-ready DALI pipelines for high-throughput audio training.
Includes benchmarking, profiling, and integration with PyTorch.

Requirements:
    pip install nvidia-dali-cuda120  # Adjust CUDA version
    pip install torch torchaudio

Run: python 02_dali_audio_pipeline.py
"""

import os
import time
import tempfile
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

# Check DALI availability
try:
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("NVIDIA DALI not available. Install with: pip install nvidia-dali-cuda120")

import torch
import torchaudio


# ============================================================
# DALI PIPELINE DEFINITIONS
# ============================================================

if DALI_AVAILABLE:
    
    @pipeline_def
    def basic_audio_pipeline(
        audio_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
    ):
        """
        Basic audio loading and preprocessing pipeline.
        
        Operations:
        1. Read audio files from directory
        2. Decode audio
        3. Resample to target rate
        4. Pad/trim to fixed length
        """
        # Read audio file bytes
        audio_bytes, labels = fn.readers.file(
            file_root=audio_dir,
            random_shuffle=True,
            name="AudioReader"
        )
        
        # Decode audio (can be GPU or CPU)
        audio, sample_rate_out = fn.decoders.audio(
            audio_bytes,
            dtype=types.FLOAT,
            downmix=True,  # Convert to mono
            device="cpu"   # Audio decode on CPU (common)
        )
        
        # Resample to target sample rate
        audio = fn.audio_resample(
            audio,
            in_rate=sample_rate_out,
            out_rate=sample_rate,
            device="cpu"
        )
        
        # Pad or trim to fixed length
        target_length = int(sample_rate * max_duration)
        audio = fn.pad(audio, axes=[0], shape=[target_length], fill_value=0.0)
        audio = fn.slice(audio, start=[0], shape=[target_length], axes=[0])
        
        return audio, labels


    @pipeline_def
    def mel_spectrogram_pipeline(
        audio_dir: str,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        max_duration: float = 30.0,
    ):
        """
        Complete mel spectrogram extraction pipeline (Whisper-style).
        
        All operations on GPU for maximum throughput.
        """
        # Read and decode
        audio_bytes, labels = fn.readers.file(
            file_root=audio_dir,
            random_shuffle=True,
            name="AudioReader"
        )
        
        audio, sr = fn.decoders.audio(
            audio_bytes,
            dtype=types.FLOAT,
            downmix=True,
            device="mixed"  # GPU decode when possible
        )
        
        # Resample
        audio = fn.audio_resample(
            audio,
            in_rate=sr,
            out_rate=sample_rate,
        )
        
        # Pad/trim
        target_length = int(sample_rate * max_duration)
        audio = fn.pad(audio, axes=[0], shape=[target_length], fill_value=0.0)
        audio = fn.slice(audio, start=[0], shape=[target_length], axes=[0])
        
        # Move to GPU for spectrogram computation
        audio = audio.gpu()
        
        # Power spectrogram
        spec = fn.spectrogram(
            audio,
            nfft=n_fft,
            window_length=n_fft,
            window_step=hop_length,
            power=2,
            device="gpu"
        )
        
        # Mel filterbank
        mel_spec = fn.mel_filter_bank(
            spec,
            sample_rate=sample_rate,
            nfilter=n_mels,
            freq_low=0.0,
            freq_high=sample_rate / 2,
            device="gpu"
        )
        
        # Log compression (to decibels)
        log_mel = fn.to_decibels(
            mel_spec,
            multiplier=10.0,
            reference=1.0,
            cutoff_db=-80.0,
            device="gpu"
        )
        
        return log_mel, labels


    @pipeline_def
    def augmented_audio_pipeline(
        audio_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
    ):
        """
        Audio pipeline with data augmentation.
        
        Augmentations:
        - Random gain
        - Random noise injection
        - Time shifting
        """
        # Read and decode
        audio_bytes, labels = fn.readers.file(
            file_root=audio_dir,
            random_shuffle=True,
            name="AudioReader"
        )
        
        audio, sr = fn.decoders.audio(
            audio_bytes,
            dtype=types.FLOAT,
            downmix=True,
        )
        
        # Resample
        audio = fn.audio_resample(audio, in_rate=sr, out_rate=sample_rate)
        
        # Pad/trim
        target_length = int(sample_rate * max_duration)
        audio = fn.pad(audio, axes=[0], shape=[target_length], fill_value=0.0)
        audio = fn.slice(audio, start=[0], shape=[target_length], axes=[0])
        
        # === AUGMENTATIONS ===
        
        # Random gain (volume change)
        gain_db = fn.random.uniform(range=[-6.0, 6.0])
        gain_linear = fn.math.pow(10.0, gain_db / 20.0)
        audio = audio * gain_linear
        
        # Random noise injection
        noise = fn.random.normal(audio, mean=0.0, stddev=0.01)
        noise_weight = fn.random.uniform(range=[0.0, 0.1])
        audio = audio + noise * noise_weight
        
        # Normalize to [-1, 1]
        max_val = fn.reductions.max(fn.math.abs(audio))
        audio = audio / (max_val + 1e-8)
        
        return audio, labels


    @pipeline_def
    def streaming_audio_pipeline(
        sample_rate: int = 16000,
        chunk_size: int = 16000,  # 1 second chunks
    ):
        """
        Pipeline for streaming/real-time audio processing.
        
        Uses external_source for feeding live audio data.
        """
        # External source for live data
        audio = fn.external_source(
            name="audio_input",
            device="gpu",
            dtype=types.FLOAT,
        )
        
        # Process chunk
        # Spectrogram for the chunk
        spec = fn.spectrogram(
            audio,
            nfft=400,
            window_length=400,
            window_step=160,
            power=2,
        )
        
        mel = fn.mel_filter_bank(
            spec,
            sample_rate=sample_rate,
            nfilter=80,
        )
        
        log_mel = fn.to_decibels(mel, multiplier=10.0, reference=1.0, cutoff_db=-80.0)
        
        return log_mel


# ============================================================
# PYTORCH INTEGRATION
# ============================================================

class DALIAudioDataLoader:
    """
    Wrapper for DALI pipeline with PyTorch integration.
    
    Usage:
        loader = DALIAudioDataLoader(
            audio_dir="/path/to/audio",
            batch_size=32,
            num_threads=4,
            device_id=0
        )
        
        for batch in loader:
            audio = batch["audio"]  # Already on GPU!
            labels = batch["labels"]
            ...
    """
    
    def __init__(
        self,
        audio_dir: str,
        batch_size: int = 32,
        num_threads: int = 4,
        device_id: int = 0,
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        extract_mel: bool = False,
        n_mels: int = 80,
    ):
        if not DALI_AVAILABLE:
            raise RuntimeError("DALI not available")
        
        self.batch_size = batch_size
        
        # Choose pipeline
        if extract_mel:
            pipe = mel_spectrogram_pipeline(
                audio_dir=audio_dir,
                sample_rate=sample_rate,
                n_mels=n_mels,
                max_duration=max_duration,
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
            )
            output_map = ["mel_spec", "labels"]
        else:
            pipe = basic_audio_pipeline(
                audio_dir=audio_dir,
                sample_rate=sample_rate,
                max_duration=max_duration,
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
            )
            output_map = ["audio", "labels"]
        
        pipe.build()
        
        # Create PyTorch iterator
        self.iterator = DALIGenericIterator(
            pipelines=[pipe],
            output_map=output_map,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )
    
    def __iter__(self):
        return iter(self.iterator)
    
    def __len__(self):
        return len(self.iterator)


# ============================================================
# BENCHMARKING
# ============================================================

def create_test_audio_files(directory: str, num_files: int = 100, 
                            duration: float = 10.0, sample_rate: int = 16000):
    """Create synthetic WAV files for benchmarking."""
    os.makedirs(directory, exist_ok=True)
    
    for i in range(num_files):
        # Generate random audio
        num_samples = int(duration * sample_rate)
        audio = torch.randn(1, num_samples) * 0.5
        
        # Save as WAV
        filepath = os.path.join(directory, f"audio_{i:04d}.wav")
        torchaudio.save(filepath, audio, sample_rate)
    
    print(f"Created {num_files} test audio files in {directory}")


def benchmark_dali_vs_pytorch(audio_dir: str, batch_size: int = 32,
                               num_batches: int = 50):
    """Compare DALI vs standard PyTorch data loading."""
    
    print("\n" + "=" * 70)
    print("DALI vs PyTorch DataLoader Benchmark")
    print("=" * 70)
    
    # Count files
    audio_files = list(Path(audio_dir).glob("*.wav"))
    print(f"\nDataset: {len(audio_files)} audio files")
    print(f"Batch size: {batch_size}")
    print(f"Batches to process: {num_batches}")
    
    results = {}
    
    # === PyTorch DataLoader ===
    print("\n--- PyTorch DataLoader ---")
    
    class PyTorchAudioDataset(torch.utils.data.Dataset):
        def __init__(self, audio_dir, sample_rate=16000, max_duration=10.0):
            self.files = list(Path(audio_dir).glob("*.wav"))
            self.sample_rate = sample_rate
            self.max_samples = int(sample_rate * max_duration)
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            audio, sr = torchaudio.load(self.files[idx])
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            # Mono
            audio = audio.mean(dim=0)
            
            # Pad/trim
            if audio.shape[0] > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.max_samples - audio.shape[0]))
            
            return audio, idx
    
    dataset = PyTorchAudioDataset(audio_dir)
    pytorch_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
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
    
    # === DALI ===
    if DALI_AVAILABLE:
        print("\n--- NVIDIA DALI ---")
        
        pipe = basic_audio_pipeline(
            audio_dir=audio_dir,
            batch_size=batch_size,
            num_threads=4,
            device_id=0,
        )
        pipe.build()
        
        dali_loader = DALIGenericIterator(
            pipelines=[pipe],
            output_map=["audio", "labels"],
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )
        
        # Warmup
        for i, batch in enumerate(dali_loader):
            if i >= 3:
                break
        dali_loader.reset()
        
        # Benchmark
        start = time.perf_counter()
        batches_processed = 0
        for batch in dali_loader:
            audio = batch[0]["audio"]  # Already on GPU!
            torch.cuda.synchronize()
            batches_processed += 1
            if batches_processed >= num_batches:
                break
        
        dali_time = time.perf_counter() - start
        results["dali"] = dali_time
        print(f"Time: {dali_time:.3f}s for {batches_processed} batches")
        print(f"Throughput: {batches_processed * batch_size / dali_time:.1f} samples/sec")
        
        # Speedup
        speedup = pytorch_time / dali_time
        print(f"\nDALI Speedup: {speedup:.2f}x")
    
    return results


def benchmark_mel_extraction(audio_dir: str, batch_size: int = 32,
                             num_batches: int = 50):
    """Benchmark mel spectrogram extraction."""
    
    print("\n" + "=" * 70)
    print("Mel Spectrogram Extraction Benchmark")
    print("=" * 70)
    
    if not DALI_AVAILABLE:
        print("DALI not available, skipping")
        return
    
    # DALI mel pipeline
    pipe = mel_spectrogram_pipeline(
        audio_dir=audio_dir,
        batch_size=batch_size,
        num_threads=4,
        device_id=0,
        n_mels=80,
        max_duration=10.0,
    )
    pipe.build()
    
    loader = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["mel_spec", "labels"],
        auto_reset=True,
    )
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 3:
            break
    loader.reset()
    
    # Benchmark
    start = time.perf_counter()
    batches_processed = 0
    for batch in loader:
        mel = batch[0]["mel_spec"]
        torch.cuda.synchronize()
        batches_processed += 1
        if batches_processed >= num_batches:
            break
    
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.3f}s for {batches_processed} batches")
    print(f"Throughput: {batches_processed * batch_size / elapsed:.1f} samples/sec")
    print(f"Output shape: {mel.shape}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  NVIDIA DALI AUDIO PIPELINES".center(68) + "█")
    print("█" + "  High-throughput audio data loading".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nNo GPU available!")
        return
    
    if not DALI_AVAILABLE:
        print("DALI not available. Install with: pip install nvidia-dali-cuda120")
        return
    
    # Create test dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        
        print("\nCreating test dataset...")
        create_test_audio_files(audio_dir, num_files=200, duration=10.0)
        
        # Run benchmarks
        benchmark_dali_vs_pytorch(audio_dir, batch_size=32, num_batches=30)
        benchmark_mel_extraction(audio_dir, batch_size=32, num_batches=30)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. DALI provides significant speedup over PyTorch DataLoader
   - Especially for preprocessing-heavy pipelines
   - GPU-accelerated transforms

2. Zero-copy GPU transfers eliminate bottlenecks
   - Data stays on GPU through the pipeline
   - No CPU-GPU round trips

3. Prefetching hides I/O latency
   - DALI automatically prefetches batches
   - GPU stays saturated

4. Best for training, not inference
   - Overhead amortizes over many batches
   - For single-sample inference, use torchaudio
""")


if __name__ == "__main__":
    main()
