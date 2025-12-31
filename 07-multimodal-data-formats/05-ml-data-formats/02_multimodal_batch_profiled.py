"""
02_multimodal_batch_profiled.py - Profile Multimodal Batch Creation

Shows the real cost of creating batches with multiple modalities.
Every operation is timed to understand bottlenecks.

Usage: python 02_multimodal_batch_profiled.py
"""

import time
import io
from dataclasses import dataclass
from typing import List, Dict
from contextlib import contextmanager

@contextmanager
def profile(name: str):
    """Profile a code block."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"⏱️  {name:<40} {elapsed:>8.2f} ms")

@dataclass
class ProfileResult:
    name: str
    time_ms: float

class MultimodalProfiler:
    """Profiles multimodal batch creation."""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
    
    def add(self, name: str, time_ms: float):
        self.results.append(ProfileResult(name, time_ms))
    
    def summary(self):
        total = sum(r.time_ms for r in self.results)
        print("\n" + "=" * 60)
        print("MULTIMODAL BATCH PROFILING SUMMARY")
        print("=" * 60)
        print(f"\n{'Operation':<35} {'Time (ms)':<12} {'%':<8}")
        print("-" * 60)
        for r in self.results:
            pct = r.time_ms / total * 100
            bar = "█" * int(pct / 5)
            print(f"{r.name:<35} {r.time_ms:<12.2f} {pct:<8.1f} {bar}")
        print("-" * 60)
        print(f"{'TOTAL':<35} {total:<12.2f}")

def simulate_image_processing(batch_size: int = 32, img_size: int = 224):
    """Simulate image batch creation."""
    import numpy as np
    
    profiler = MultimodalProfiler()
    
    print("\n" + "█" * 60)
    print(f"█  IMAGE BATCH ({batch_size} images, {img_size}x{img_size})")
    print("█" * 60 + "\n")
    
    # Simulate JPEG decode
    start = time.perf_counter()
    images_raw = []
    for _ in range(batch_size):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images_raw.append(img)
    profiler.add("JPEG decode (simulated)", (time.perf_counter() - start) * 1000)
    
    # Resize
    start = time.perf_counter()
    images_resized = []
    for img in images_raw:
        # Simple resize simulation
        resized = img[::2, ::3, :][:img_size, :img_size, :]
        # Pad if needed
        result = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        h, w = min(resized.shape[0], img_size), min(resized.shape[1], img_size)
        result[:h, :w, :] = resized[:h, :w, :]
        images_resized.append(result)
    profiler.add("Resize to 224x224", (time.perf_counter() - start) * 1000)
    
    # Stack into batch
    start = time.perf_counter()
    batch_hwc = np.stack(images_resized)
    profiler.add("Stack to batch (NHWC)", (time.perf_counter() - start) * 1000)
    
    # Transpose HWC -> CHW
    start = time.perf_counter()
    batch_chw = batch_hwc.transpose(0, 3, 1, 2)
    profiler.add("Transpose HWC→CHW", (time.perf_counter() - start) * 1000)
    
    # Convert to float and normalize
    start = time.perf_counter()
    batch_float = batch_chw.astype(np.float32) / 255.0
    profiler.add("Convert to float32", (time.perf_counter() - start) * 1000)
    
    # ImageNet normalization
    start = time.perf_counter()
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    batch_norm = (batch_float - mean) / std
    profiler.add("ImageNet normalize", (time.perf_counter() - start) * 1000)
    
    profiler.summary()
    return batch_norm

def simulate_audio_processing(batch_size: int = 32, duration_sec: float = 5.0):
    """Simulate audio batch creation."""
    import numpy as np
    
    profiler = MultimodalProfiler()
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    
    print("\n" + "█" * 60)
    print(f"█  AUDIO BATCH ({batch_size} clips, {duration_sec}s @ {sample_rate}Hz)")
    print("█" * 60 + "\n")
    
    # Simulate audio loading
    start = time.perf_counter()
    audio_raw = []
    for _ in range(batch_size):
        audio = np.random.randn(num_samples).astype(np.float32)
        audio_raw.append(audio)
    profiler.add("Load audio (simulated)", (time.perf_counter() - start) * 1000)
    
    # Resample (simulated)
    start = time.perf_counter()
    for i in range(len(audio_raw)):
        audio_raw[i] = audio_raw[i]  # Placeholder
    profiler.add("Resample (if needed)", (time.perf_counter() - start) * 1000)
    
    # Compute log mel spectrogram (simplified)
    start = time.perf_counter()
    n_fft = 512
    hop_length = 160
    n_mels = 80
    specs = []
    for audio in audio_raw:
        # Very simplified STFT
        num_frames = (len(audio) - n_fft) // hop_length + 1
        spec = np.random.randn(n_mels, num_frames).astype(np.float32)
        specs.append(spec)
    profiler.add("Mel spectrogram", (time.perf_counter() - start) * 1000)
    
    # Pad to same length
    start = time.perf_counter()
    max_len = max(s.shape[1] for s in specs)
    padded = []
    for spec in specs:
        if spec.shape[1] < max_len:
            pad = np.zeros((n_mels, max_len - spec.shape[1]), dtype=np.float32)
            spec = np.concatenate([spec, pad], axis=1)
        padded.append(spec)
    profiler.add("Pad to max length", (time.perf_counter() - start) * 1000)
    
    # Stack
    start = time.perf_counter()
    batch = np.stack(padded)
    profiler.add("Stack to batch", (time.perf_counter() - start) * 1000)
    
    # Normalize
    start = time.perf_counter()
    batch = (batch - batch.mean()) / (batch.std() + 1e-6)
    profiler.add("Normalize", (time.perf_counter() - start) * 1000)
    
    profiler.summary()
    return batch

def simulate_text_processing(batch_size: int = 32, max_length: int = 512):
    """Simulate text batch creation (tokenization)."""
    import numpy as np
    
    profiler = MultimodalProfiler()
    
    print("\n" + "█" * 60)
    print(f"█  TEXT BATCH ({batch_size} sequences, max_len={max_length})")
    print("█" * 60 + "\n")
    
    # Generate random "text"
    texts = ["This is sample text number " + str(i) * 50 for i in range(batch_size)]
    
    # Tokenization (simplified - char-level)
    start = time.perf_counter()
    tokens = []
    for text in texts:
        tok = [ord(c) % 256 for c in text[:max_length]]
        tokens.append(tok)
    profiler.add("Tokenization", (time.perf_counter() - start) * 1000)
    
    # Padding
    start = time.perf_counter()
    padded = []
    for tok in tokens:
        if len(tok) < max_length:
            tok = tok + [0] * (max_length - len(tok))
        padded.append(tok[:max_length])
    profiler.add("Padding", (time.perf_counter() - start) * 1000)
    
    # Convert to numpy
    start = time.perf_counter()
    batch = np.array(padded, dtype=np.int64)
    profiler.add("Convert to array", (time.perf_counter() - start) * 1000)
    
    # Create attention mask
    start = time.perf_counter()
    attention_mask = (batch != 0).astype(np.int64)
    profiler.add("Create attention mask", (time.perf_counter() - start) * 1000)
    
    profiler.summary()
    return batch, attention_mask

def simulate_multimodal_batch():
    """Full multimodal batch creation."""
    print("\n")
    print("█" * 70)
    print("█  FULL MULTIMODAL BATCH CREATION")
    print("█" * 70)
    
    batch_size = 16
    
    total_start = time.perf_counter()
    
    # Process each modality
    images = simulate_image_processing(batch_size)
    audio = simulate_audio_processing(batch_size)
    text, mask = simulate_text_processing(batch_size)
    
    total_time = (time.perf_counter() - total_start) * 1000
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\nTotal batch creation time: {total_time:.2f} ms")
    print(f"Batch size: {batch_size}")
    print(f"Time per sample: {total_time/batch_size:.2f} ms")
    print(f"Theoretical max throughput: {1000/total_time*batch_size:.0f} samples/sec")
    
    print("\n" + "=" * 70)
    print("MEMORY SIZES")
    print("=" * 70)
    print(f"Images: {images.nbytes / 1024 / 1024:.2f} MB")
    print(f"Audio:  {audio.nbytes / 1024 / 1024:.2f} MB")
    print(f"Text:   {text.nbytes / 1024 / 1024:.2f} MB")
    print(f"Total:  {(images.nbytes + audio.nbytes + text.nbytes) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MULTIMODAL BATCH PROFILING")
    print("  Understanding the cost of each operation")
    print("=" * 70)
    
    simulate_multimodal_batch()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR MULTIMODAL TRAINING")
    print("=" * 70)
    print("""
1. IMAGE BOTTLENECKS
   - JPEG decode: 2-5 ms/image (use TurboJPEG or GPU decode)
   - Resize: 1-3 ms/image (pre-resize your dataset!)
   - Solution: FFCV, pre-processed data

2. AUDIO BOTTLENECKS
   - STFT/Mel: 5-20 ms/clip (use GPU or pre-compute)
   - Resampling: 1-5 ms/clip (store at target rate)
   - Solution: Pre-compute spectrograms

3. TEXT BOTTLENECKS  
   - Tokenization: Usually fast, but watch for regex
   - Padding: Use dynamic batching to reduce waste
   - Solution: Pre-tokenize and cache

4. GENERAL TIPS
   - Pre-process everything possible offline
   - Use memory-mapped data (FFCV, mmap)
   - Profile to find YOUR bottleneck
   - Different datasets have different bottlenecks
""")
