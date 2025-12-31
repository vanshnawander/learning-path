"""
05_profiling_multimodal_pipeline.py - Profile Every Stage of ML Pipeline

THIS FILE TEACHES YOU TO PROFILE LIKE A PROFESSIONAL

Every operation is timed. Run this to understand where time goes
in a multimodal training pipeline.

Requirements:
    pip install torch torchvision torchaudio pillow numpy

Usage:
    python 05_profiling_multimodal_pipeline.py
"""

import time
import os
import io
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any
import tracemalloc

# ============================================================
# PROFILING UTILITIES - USE THESE EVERYWHERE
# ============================================================

@dataclass
class ProfileResult:
    """Stores profiling results for analysis."""
    name: str
    time_ms: float
    memory_mb: float
    throughput: str
    
class Profiler:
    """
    Professional-grade profiler for understanding ML performance.
    
    Usage:
        profiler = Profiler()
        
        with profiler.profile("Load Image"):
            img = load_image(path)
        
        with profiler.profile("Decode"):
            tensor = decode(img)
        
        profiler.summary()
    """
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.indent_level = 0
        
    @contextmanager
    def profile(self, name: str, data_size_bytes: int = 0):
        """Profile a code block with timing and memory."""
        tracemalloc.start()
        start_time = time.perf_counter()
        
        yield
        
        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_ms = elapsed * 1000
        memory_mb = peak / 1024 / 1024
        
        if data_size_bytes > 0:
            throughput = f"{data_size_bytes / elapsed / 1e6:.1f} MB/s"
        else:
            throughput = "-"
        
        result = ProfileResult(name, time_ms, memory_mb, throughput)
        self.results.append(result)
        
        # Print immediately for real-time feedback
        indent = "  " * self.indent_level
        print(f"{indent}⏱️  {name}: {time_ms:.3f} ms | Mem: {memory_mb:.2f} MB | {throughput}")
        
    def summary(self):
        """Print summary of all profiled operations."""
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        
        total_time = sum(r.time_ms for r in self.results)
        
        print(f"\n{'Operation':<35} {'Time (ms)':<12} {'%':<8} {'Memory':<10}")
        print("-" * 70)
        
        for r in self.results:
            pct = (r.time_ms / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"{r.name:<35} {r.time_ms:<12.3f} {pct:<8.1f} {r.memory_mb:<10.2f} {bar}")
        
        print("-" * 70)
        print(f"{'TOTAL':<35} {total_time:<12.3f} {'100.0':<8}")
        print("=" * 70)
        
        # Identify bottleneck
        if self.results:
            bottleneck = max(self.results, key=lambda r: r.time_ms)
            print(f"\n🔥 BOTTLENECK: {bottleneck.name} ({bottleneck.time_ms:.1f} ms)")
            print(f"   Optimize this first for maximum impact!")


# ============================================================
# TEST 1: IMAGE LOADING PIPELINE
# ============================================================

def profile_image_pipeline():
    """Profile each stage of image loading for ML."""
    print("\n" + "█" * 70)
    print("█  PROFILING: IMAGE LOADING PIPELINE")
    print("█" * 70 + "\n")
    
    profiler = Profiler()
    
    try:
        import numpy as np
        from PIL import Image
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return
    
    # Create test image (simulates JPEG from disk)
    print("Creating test data...")
    img_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    
    # Save to bytes (simulates file)
    jpeg_buffer = io.BytesIO()
    pil_img.save(jpeg_buffer, format='JPEG', quality=85)
    jpeg_bytes = jpeg_buffer.getvalue()
    jpeg_size = len(jpeg_bytes)
    print(f"JPEG size: {jpeg_size / 1024:.1f} KB")
    print(f"Raw size: {img_array.nbytes / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {img_array.nbytes / jpeg_size:.1f}x\n")
    
    # Profile each stage
    print("Profiling pipeline stages:\n")
    
    # Stage 1: JPEG Decode
    with profiler.profile("1. JPEG Decode", jpeg_size):
        jpeg_buffer.seek(0)
        decoded = Image.open(jpeg_buffer)
        decoded.load()  # Force decode
    
    # Stage 2: Resize
    with profiler.profile("2. Resize (224x224)"):
        resized = decoded.resize((224, 224), Image.BILINEAR)
    
    # Stage 3: Convert to NumPy
    with profiler.profile("3. PIL → NumPy"):
        np_array = np.array(resized)
    
    # Stage 4: Convert to Tensor
    with profiler.profile("4. NumPy → Tensor"):
        tensor = torch.from_numpy(np_array)
    
    # Stage 5: Permute HWC → CHW
    with profiler.profile("5. HWC → CHW"):
        tensor = tensor.permute(2, 0, 1)
    
    # Stage 6: Normalize
    with profiler.profile("6. Normalize (float, /255)"):
        tensor = tensor.float() / 255.0
    
    # Stage 7: ImageNet normalization
    with profiler.profile("7. ImageNet Normalize"):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    
    profiler.summary()
    
    print("\n💡 OPTIMIZATION TIPS:")
    print("   • Use TurboJPEG for 2-3x faster decode")
    print("   • Use torchvision.transforms.v2 for fused ops")
    print("   • Pre-resize images to training size")
    print("   • Use FFCV for pre-decoded, memory-mapped data")


# ============================================================
# TEST 2: AUDIO LOADING PIPELINE
# ============================================================

def profile_audio_pipeline():
    """Profile each stage of audio loading for ML."""
    print("\n" + "█" * 70)
    print("█  PROFILING: AUDIO LOADING PIPELINE")
    print("█" * 70 + "\n")
    
    profiler = Profiler()
    
    try:
        import numpy as np
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return
    
    # Create test audio (16kHz, 10 seconds)
    sample_rate = 16000
    duration = 10  # seconds
    num_samples = sample_rate * duration
    
    print(f"Test audio: {duration}s @ {sample_rate} Hz")
    print(f"Samples: {num_samples:,}")
    print(f"Raw size: {num_samples * 2 / 1024:.1f} KB (int16)\n")
    
    # Simulate raw PCM audio
    audio_int16 = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)
    
    print("Profiling pipeline stages:\n")
    
    # Stage 1: Convert to float
    with profiler.profile("1. int16 → float32"):
        audio_float = audio_int16.astype(np.float32) / 32768.0
    
    # Stage 2: Convert to tensor
    with profiler.profile("2. NumPy → Tensor"):
        tensor = torch.from_numpy(audio_float)
    
    # Stage 3: Resample (if needed) - simulated
    with profiler.profile("3. Resample (simulated)"):
        # In real code: torchaudio.functional.resample
        resampled = tensor  # Placeholder
    
    # Stage 4: Compute STFT
    with profiler.profile("4. STFT (n_fft=512)"):
        stft = torch.stft(
            tensor, 
            n_fft=512, 
            hop_length=160, 
            win_length=400,
            window=torch.hann_window(400),
            return_complex=True
        )
    
    # Stage 5: Magnitude
    with profiler.profile("5. Complex → Magnitude"):
        magnitude = torch.abs(stft)
    
    # Stage 6: Log scale
    with profiler.profile("6. Log scale"):
        log_magnitude = torch.log(magnitude + 1e-9)
    
    # Stage 7: Simulate mel filterbank
    with profiler.profile("7. Mel filterbank (simulated)"):
        # In real code: torchaudio.transforms.MelSpectrogram
        mel = log_magnitude[:80, :]  # Take first 80 bins as placeholder
    
    profiler.summary()
    
    print("\n💡 OPTIMIZATION TIPS:")
    print("   • Use GPU for STFT when possible")
    print("   • Pre-compute mel spectrograms for training")
    print("   • Use torchaudio with sox/ffmpeg backend")
    print("   • Batch audio processing for efficiency")


# ============================================================
# TEST 3: DATA TRANSFER COSTS
# ============================================================

def profile_data_transfer():
    """Profile CPU to GPU transfer costs."""
    print("\n" + "█" * 70)
    print("█  PROFILING: CPU → GPU DATA TRANSFER")
    print("█" * 70 + "\n")
    
    profiler = Profiler()
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU tests")
            return
    except ImportError:
        print("PyTorch not installed")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Test different tensor sizes
    sizes = [
        ("Small (224x224x3)", (1, 3, 224, 224)),
        ("Medium batch (32, 224x224)", (32, 3, 224, 224)),
        ("Large batch (128, 224x224)", (128, 3, 224, 224)),
        ("Video clip (16 frames)", (1, 16, 3, 224, 224)),
    ]
    
    for name, shape in sizes:
        tensor = torch.randn(*shape)
        size_mb = tensor.numel() * 4 / 1024 / 1024
        
        print(f"\n{name} - {size_mb:.1f} MB:")
        
        # Pageable memory (default)
        torch.cuda.synchronize()
        with profiler.profile(f"  Pageable → GPU"):
            gpu_tensor = tensor.cuda()
            torch.cuda.synchronize()
        
        # Pinned memory
        pinned = tensor.pin_memory()
        torch.cuda.synchronize()
        with profiler.profile(f"  Pinned → GPU"):
            gpu_tensor = pinned.cuda()
            torch.cuda.synchronize()
        
        del gpu_tensor, pinned
        torch.cuda.empty_cache()
    
    profiler.summary()
    
    print("\n💡 OPTIMIZATION TIPS:")
    print("   • Use pin_memory=True in DataLoader")
    print("   • Use non_blocking=True for async transfer")
    print("   • Overlap transfer with compute using streams")
    print("   • Keep data on GPU when possible (minimize transfers)")


# ============================================================
# TEST 4: BATCH SIZE EFFECTS
# ============================================================

def profile_batch_effects():
    """Profile how batch size affects throughput."""
    print("\n" + "█" * 70)
    print("█  PROFILING: BATCH SIZE EFFECTS ON THROUGHPUT")
    print("█" * 70 + "\n")
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Simple model for testing
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    batch_sizes = [1, 8, 32, 64, 128, 256, 512]
    
    print(f"{'Batch':<8} {'Time/batch':<12} {'Time/sample':<12} {'Samples/sec':<12}")
    print("-" * 50)
    
    results = []
    for bs in batch_sizes:
        x = torch.randn(bs, 784).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        time_per_batch = elapsed / iterations * 1000
        time_per_sample = time_per_batch / bs
        samples_per_sec = bs * iterations / elapsed
        
        print(f"{bs:<8} {time_per_batch:<12.3f} {time_per_sample:<12.3f} {samples_per_sec:<12.0f}")
        results.append((bs, samples_per_sec))
    
    best_bs, best_throughput = max(results, key=lambda x: x[1])
    
    print(f"\n🔥 OPTIMAL BATCH SIZE: {best_bs} ({best_throughput:.0f} samples/sec)")
    print("\n💡 KEY INSIGHTS:")
    print("   • Small batches: High overhead per sample (kernel launch, etc.)")
    print("   • Large batches: Better GPU utilization, amortized overhead")
    print("   • Too large: Memory limits, diminishing returns")
    print("   • Profile YOUR model to find sweet spot!")


# ============================================================
# TEST 5: PYTORCH PROFILER DEMO
# ============================================================

def demo_pytorch_profiler():
    """Demonstrate PyTorch's built-in profiler."""
    print("\n" + "█" * 70)
    print("█  DEMO: PyTorch Built-in Profiler")
    print("█" * 70 + "\n")
    
    try:
        import torch
        import torch.nn as nn
        from torch.profiler import profile, ProfilerActivity
    except ImportError:
        print("PyTorch not installed")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and data
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    x = torch.randn(64, 1024).to(device)
    
    print("Running PyTorch profiler...\n")
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(10):
            output = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    print("Top operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if torch.cuda.is_available():
        print("\nTop operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n💡 PROFILER TIPS:")
    print("   • Use schedule() for training loop profiling")
    print("   • Export to TensorBoard: on_trace_ready=tensorboard_trace_handler")
    print("   • Look for GPU idle time (indicates CPU bottleneck)")
    print("   • Look for memory copies (indicates transfer overhead)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MULTIMODAL PIPELINE PROFILING - LEARN WHERE TIME GOES")
    print("=" * 70)
    print("\nThis script profiles every stage of ML data pipelines.")
    print("Run it to understand the cost of each operation.\n")
    
    profile_image_pipeline()
    profile_audio_pipeline()
    profile_data_transfer()
    profile_batch_effects()
    demo_pytorch_profiler()
    
    print("\n" + "=" * 70)
    print("  SUMMARY: PROFILING IS NOT OPTIONAL")
    print("=" * 70)
    print("""
    KEY LESSONS:
    
    1. MEASURE FIRST, OPTIMIZE SECOND
       - Your intuition about bottlenecks is often wrong
       - Always profile before optimizing
    
    2. DATA LOADING IS OFTEN THE BOTTLENECK
       - Image decode, augmentation, transfer
       - Use FFCV, DALI, or pre-processing
    
    3. BATCH SIZE MATTERS
       - Too small = overhead dominates
       - Too large = memory issues, diminishing returns
    
    4. CPU → GPU TRANSFER IS EXPENSIVE
       - Use pinned memory
       - Overlap transfer with compute
       - Minimize transfers
    
    5. PROFILE REGULARLY
       - Performance changes with code changes
       - Different hardware = different bottlenecks
    """)
