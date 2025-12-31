"""
01_dataloader_profiling.py - Profile PyTorch DataLoader Performance

This file teaches you to identify data loading bottlenecks.
Every operation is timed. Run this to understand your pipeline.

Usage:
    python 01_dataloader_profiling.py
"""

import time
import os
import io
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from contextlib import contextmanager
import threading
import queue

# ============================================================
# PROFILING UTILITIES
# ============================================================

class DataLoaderProfiler:
    """
    Profile DataLoader performance in detail.
    
    Tracks:
    - Time to get each batch
    - Queue wait time
    - Data transfer time
    - GPU utilization (if available)
    """
    
    def __init__(self):
        self.batch_times: List[float] = []
        self.transfer_times: List[float] = []
        self.total_samples = 0
        
    def record_batch(self, batch_time: float, transfer_time: float, batch_size: int):
        self.batch_times.append(batch_time)
        self.transfer_times.append(transfer_time)
        self.total_samples += batch_size
        
    def summary(self):
        if not self.batch_times:
            print("No batches recorded")
            return
            
        avg_batch = sum(self.batch_times) / len(self.batch_times) * 1000
        avg_transfer = sum(self.transfer_times) / len(self.transfer_times) * 1000
        total_time = sum(self.batch_times)
        throughput = self.total_samples / total_time if total_time > 0 else 0
        
        print("\n" + "=" * 60)
        print("DATALOADER PROFILING SUMMARY")
        print("=" * 60)
        print(f"  Total batches: {len(self.batch_times)}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Avg batch time: {avg_batch:.2f} ms")
        print(f"  Avg transfer time: {avg_transfer:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print("=" * 60)


@contextmanager
def time_block(name: str = ""):
    """Simple context manager for timing."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name:
        print(f"⏱️  {name}: {elapsed*1000:.2f} ms")


# ============================================================
# SIMULATED DATASET
# ============================================================

class SimulatedImageDataset:
    """
    Simulates an image dataset for profiling.
    
    Allows testing different bottleneck scenarios:
    - Fast (in-memory)
    - Slow decode (simulated JPEG decode)
    - Slow I/O (simulated disk read)
    """
    
    def __init__(self, size: int, image_size: Tuple[int, int] = (224, 224),
                 decode_time_ms: float = 0, io_time_ms: float = 0):
        self.size = size
        self.image_size = image_size
        self.decode_time_ms = decode_time_ms
        self.io_time_ms = io_time_ms
        
        # Pre-generate "compressed" data
        import numpy as np
        self.compressed_size = 50000  # ~50KB JPEG
        self.data = [np.random.bytes(self.compressed_size) for _ in range(size)]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        import numpy as np
        
        # Simulate I/O time
        if self.io_time_ms > 0:
            time.sleep(self.io_time_ms / 1000)
        
        # Simulate decode time
        if self.decode_time_ms > 0:
            time.sleep(self.decode_time_ms / 1000)
        
        # Generate "image"
        image = np.random.randn(3, *self.image_size).astype(np.float32)
        label = idx % 1000
        
        return image, label


# ============================================================
# TEST 1: SINGLE vs MULTI-WORKER
# ============================================================

def test_num_workers():
    """Compare DataLoader performance with different worker counts."""
    print("\n" + "█" * 60)
    print("█  TEST 1: NUMBER OF WORKERS")
    print("█" * 60 + "\n")
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed, skipping")
        return
    
    # Dataset with simulated decode time
    dataset = SimulatedImageDataset(
        size=1000,
        decode_time_ms=5  # 5ms decode per image
    )
    
    batch_size = 32
    num_batches = 10
    
    worker_counts = [0, 1, 2, 4, 8]
    
    print(f"Dataset: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Simulated decode time: 5ms per image")
    print(f"Testing {num_batches} batches\n")
    
    print(f"{'Workers':<10} {'Time (s)':<12} {'Batches/s':<12} {'Speedup':<10}")
    print("-" * 50)
    
    baseline_time = None
    
    for num_workers in worker_counts:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Warmup
        for i, batch in enumerate(loader):
            if i >= 2:
                break
        
        # Benchmark
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.perf_counter() - start
        
        if baseline_time is None:
            baseline_time = elapsed
        
        speedup = baseline_time / elapsed if elapsed > 0 else 0
        batches_per_sec = num_batches / elapsed if elapsed > 0 else 0
        
        print(f"{num_workers:<10} {elapsed:<12.3f} {batches_per_sec:<12.1f} {speedup:<10.2f}x")
    
    print("\n💡 More workers help when data loading is slow!")
    print("   But too many workers waste resources and memory.")


# ============================================================
# TEST 2: PIN_MEMORY EFFECT
# ============================================================

def test_pin_memory():
    """Compare pinned vs pageable memory transfer."""
    print("\n" + "█" * 60)
    print("█  TEST 2: PIN_MEMORY EFFECT")
    print("█" * 60 + "\n")
    
    try:
        import torch
        from torch.utils.data import DataLoader
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping")
            return
    except ImportError:
        print("PyTorch not installed, skipping")
        return
    
    dataset = SimulatedImageDataset(size=500)
    batch_size = 64
    num_batches = 20
    
    print(f"Testing CPU → GPU transfer for {num_batches} batches of {batch_size}\n")
    
    for pin_memory in [False, True]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=pin_memory
        )
        
        transfer_times = []
        
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            torch.cuda.synchronize()
            transfer_times.append(time.perf_counter() - start)
        
        avg_transfer = sum(transfer_times) / len(transfer_times) * 1000
        
        status = "ON" if pin_memory else "OFF"
        print(f"pin_memory={status}: {avg_transfer:.2f} ms avg transfer time")
    
    print("\n💡 Pinned memory enables faster DMA transfers to GPU!")


# ============================================================
# TEST 3: PREFETCH FACTOR
# ============================================================

def test_prefetch():
    """Compare different prefetch factors."""
    print("\n" + "█" * 60)
    print("█  TEST 3: PREFETCH FACTOR")
    print("█" * 60 + "\n")
    
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed, skipping")
        return
    
    # Dataset with slow decode
    dataset = SimulatedImageDataset(
        size=500,
        decode_time_ms=10
    )
    
    batch_size = 32
    num_batches = 15
    
    print(f"Testing prefetch factor with 10ms decode time\n")
    
    prefetch_factors = [1, 2, 4, 8]
    
    print(f"{'Prefetch':<12} {'Time (s)':<12} {'Batches/s':<12}")
    print("-" * 40)
    
    for prefetch in prefetch_factors:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            prefetch_factor=prefetch
        )
        
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            # Simulate some processing
            time.sleep(0.01)
        elapsed = time.perf_counter() - start
        
        print(f"{prefetch:<12} {elapsed:<12.3f} {num_batches/elapsed:<12.1f}")
    
    print("\n💡 Higher prefetch keeps GPU fed when decode is slow.")
    print("   But uses more memory (prefetch × batch_size × workers).")


# ============================================================
# TEST 4: BOTTLENECK IDENTIFICATION
# ============================================================

def test_bottleneck_identification():
    """Identify whether you're data-loading or compute bound."""
    print("\n" + "█" * 60)
    print("█  TEST 4: BOTTLENECK IDENTIFICATION")
    print("█" * 60 + "\n")
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed, skipping")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(224 * 224 * 3, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1000)
    ).to(device)
    
    # Two scenarios
    scenarios = [
        ("Fast data, slow model", 0, 50),   # No decode delay, 50ms model
        ("Slow data, fast model", 20, 5),   # 20ms decode, 5ms model
    ]
    
    batch_size = 32
    num_batches = 10
    
    for name, decode_ms, model_ms in scenarios:
        print(f"\n{name}:")
        print("-" * 40)
        
        dataset = SimulatedImageDataset(
            size=500,
            decode_time_ms=decode_ms
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        data_wait_times = []
        compute_times = []
        
        loader_iter = iter(loader)
        
        for i in range(num_batches):
            # Time data loading
            start = time.perf_counter()
            images, labels = next(loader_iter)
            images = images.to(device)
            data_time = time.perf_counter() - start
            data_wait_times.append(data_time)
            
            # Simulate compute
            start = time.perf_counter()
            time.sleep(model_ms / 1000)  # Simulated model time
            with torch.no_grad():
                output = model(images.view(images.size(0), -1))
            compute_time = time.perf_counter() - start
            compute_times.append(compute_time)
        
        avg_data = sum(data_wait_times) / len(data_wait_times) * 1000
        avg_compute = sum(compute_times) / len(compute_times) * 1000
        
        print(f"  Avg data wait: {avg_data:.1f} ms")
        print(f"  Avg compute: {avg_compute:.1f} ms")
        
        if avg_data > avg_compute * 1.5:
            print(f"  ⚠️  DATA LOADING BOTTLENECK!")
            print(f"      Solutions: more workers, prefetch, faster storage")
        elif avg_compute > avg_data * 1.5:
            print(f"  ✓ Compute bound (good - GPU is busy)")
        else:
            print(f"  ~ Balanced (both similar)")


# ============================================================
# TEST 5: FULL TRAINING LOOP PROFILE
# ============================================================

def test_full_training_profile():
    """Profile a complete training iteration."""
    print("\n" + "█" * 60)
    print("█  TEST 5: FULL TRAINING ITERATION BREAKDOWN")
    print("█" * 60 + "\n")
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed, skipping")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Setup
    dataset = SimulatedImageDataset(size=500, decode_time_ms=5)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 1000)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Profile 10 iterations
    num_iters = 10
    
    timings = {
        'data_load': [],
        'to_gpu': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }
    
    print("Profiling training iterations...\n")
    
    loader_iter = iter(loader)
    
    for i in range(num_iters):
        iter_start = time.perf_counter()
        
        # Data loading
        start = time.perf_counter()
        images, labels = next(loader_iter)
        timings['data_load'].append(time.perf_counter() - start)
        
        # Transfer to GPU
        start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        timings['to_gpu'].append(time.perf_counter() - start)
        
        # Forward pass
        start = time.perf_counter()
        outputs = model(images)
        if device == 'cuda':
            torch.cuda.synchronize()
        timings['forward'].append(time.perf_counter() - start)
        
        # Loss
        start = time.perf_counter()
        loss = criterion(outputs, labels)
        timings['loss'].append(time.perf_counter() - start)
        
        # Backward pass
        start = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        timings['backward'].append(time.perf_counter() - start)
        
        # Optimizer step
        start = time.perf_counter()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        timings['optimizer'].append(time.perf_counter() - start)
        
        timings['total'].append(time.perf_counter() - iter_start)
    
    # Print results
    print(f"{'Stage':<15} {'Avg (ms)':<12} {'%':<8} {'Bar':<20}")
    print("-" * 60)
    
    total_avg = sum(timings['total']) / len(timings['total']) * 1000
    
    for stage, times in timings.items():
        if stage == 'total':
            continue
        avg = sum(times) / len(times) * 1000
        pct = avg / total_avg * 100
        bar = "█" * int(pct / 5)
        print(f"{stage:<15} {avg:<12.2f} {pct:<8.1f} {bar}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_avg:<12.2f}")
    
    # Identify bottleneck
    avgs = {k: sum(v)/len(v) for k, v in timings.items() if k != 'total'}
    bottleneck = max(avgs, key=avgs.get)
    
    print(f"\n🔥 BOTTLENECK: {bottleneck}")
    
    if bottleneck == 'data_load':
        print("   → More workers, prefetch, faster storage, FFCV")
    elif bottleneck == 'to_gpu':
        print("   → pin_memory, non_blocking, keep data on GPU")
    elif bottleneck in ['forward', 'backward']:
        print("   → Smaller model, mixed precision, better GPU")
    elif bottleneck == 'optimizer':
        print("   → Fused optimizer, gradient accumulation")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DATALOADER PROFILING - FIND YOUR BOTTLENECK")
    print("=" * 60)
    
    test_num_workers()
    test_pin_memory()
    test_prefetch()
    test_bottleneck_identification()
    test_full_training_profile()
    
    print("\n" + "=" * 60)
    print("  KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. PROFILE FIRST
       - Don't guess where time goes
       - Use torch.profiler for detailed analysis
    
    2. NUM_WORKERS
       - Start with num_workers = num_cpu_cores
       - Reduce if memory issues
       - 0 workers = no parallelism!
    
    3. PIN_MEMORY
       - Always True when using GPU
       - Enables faster DMA transfers
    
    4. PREFETCH_FACTOR  
       - Higher = more batches ready
       - Uses more memory
       - Default 2 is usually good
    
    5. IDENTIFY BOTTLENECK
       - Data time > compute time? → Optimize data pipeline
       - Compute time > data time? → GPU is the limit (good!)
    
    6. SOLUTIONS FOR DATA BOTTLENECK
       - FFCV: Pre-decoded, memory-mapped
       - WebDataset: Sequential reads
       - NVIDIA DALI: GPU decode
       - Pre-process and cache
    """)
