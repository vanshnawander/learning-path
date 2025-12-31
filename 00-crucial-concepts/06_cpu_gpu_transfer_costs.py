"""
06_cpu_gpu_transfer_costs.py - Profile CPU↔GPU Data Transfer

Every ML engineer MUST understand these costs.
This is often the hidden bottleneck in training pipelines.

Usage: python 06_cpu_gpu_transfer_costs.py
"""

import time
import sys

def check_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available. Install PyTorch with CUDA support.")
            return False
        return True
    except ImportError:
        print("PyTorch not installed.")
        return False

def format_size(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"

def profile_transfer():
    """Profile CPU to GPU transfer with different settings."""
    import torch
    
    print("\n" + "█" * 60)
    print("█  CPU ↔ GPU TRANSFER PROFILING")
    print("█" * 60)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}\n")
    
    # Test sizes (in MB)
    sizes_mb = [1, 4, 16, 64, 256, 1024]
    
    print("=" * 70)
    print("TEST 1: PAGEABLE vs PINNED MEMORY")
    print("=" * 70)
    print(f"\n{'Size':<12} {'Pageable':<15} {'Pinned':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4  # Float32 elements
        
        # Pageable memory (default)
        data_pageable = torch.randn(size)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(10):
            gpu_data = data_pageable.cuda()
            torch.cuda.synchronize()
        pageable_time = (time.perf_counter() - start) / 10 * 1000
        
        # Pinned memory
        data_pinned = torch.randn(size).pin_memory()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(10):
            gpu_data = data_pinned.cuda()
            torch.cuda.synchronize()
        pinned_time = (time.perf_counter() - start) / 10 * 1000
        
        speedup = pageable_time / pinned_time
        
        print(f"{size_mb:>4} MB      {pageable_time:>8.2f} ms     {pinned_time:>8.2f} ms     {speedup:>5.2f}x")
        
        del data_pageable, data_pinned, gpu_data
        torch.cuda.empty_cache()
    
    print("\n💡 Pinned memory enables DMA (Direct Memory Access)")
    print("   Always use pin_memory=True in DataLoader!\n")
    
    # Test 2: Blocking vs Non-blocking
    print("=" * 70)
    print("TEST 2: BLOCKING vs NON-BLOCKING TRANSFER")
    print("=" * 70)
    
    size = 64 * 1024 * 1024 // 4  # 64 MB
    data = torch.randn(size).pin_memory()
    
    # Blocking
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_data = data.cuda()  # Blocking by default
        torch.cuda.synchronize()
    blocking_time = (time.perf_counter() - start) / 10 * 1000
    
    # Non-blocking
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_data = data.cuda(non_blocking=True)
        torch.cuda.synchronize()
    nonblocking_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"\n64 MB transfer:")
    print(f"  Blocking:     {blocking_time:.2f} ms")
    print(f"  Non-blocking: {nonblocking_time:.2f} ms")
    print("\n💡 Non-blocking allows overlap with CPU work!")
    print("   Use .cuda(non_blocking=True) when possible.\n")
    
    # Test 3: Transfer + Compute Overlap
    print("=" * 70)
    print("TEST 3: TRANSFER + COMPUTE OVERLAP")
    print("=" * 70)
    
    data1 = torch.randn(size).pin_memory()
    data2 = torch.randn(size).pin_memory()
    
    # Sequential
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu1 = data1.cuda()
    torch.cuda.synchronize()
    result1 = gpu1 * 2
    torch.cuda.synchronize()
    gpu2 = data2.cuda()
    torch.cuda.synchronize()
    result2 = gpu2 * 2
    torch.cuda.synchronize()
    sequential_time = (time.perf_counter() - start) * 1000
    
    # Overlapped with streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.cuda.stream(stream1):
        gpu1 = data1.cuda(non_blocking=True)
        result1 = gpu1 * 2
    with torch.cuda.stream(stream2):
        gpu2 = data2.cuda(non_blocking=True)
        result2 = gpu2 * 2
    torch.cuda.synchronize()
    overlapped_time = (time.perf_counter() - start) * 1000
    
    print(f"\nTwo 64MB transfers + compute:")
    print(f"  Sequential:  {sequential_time:.2f} ms")
    print(f"  Overlapped:  {overlapped_time:.2f} ms")
    print(f"  Speedup:     {sequential_time/overlapped_time:.2f}x")
    print("\n💡 Use CUDA streams to overlap transfer and compute!\n")
    
    # Test 4: Bandwidth calculation
    print("=" * 70)
    print("TEST 4: PCIe BANDWIDTH MEASUREMENT")
    print("=" * 70)
    
    size_bytes = 256 * 1024 * 1024  # 256 MB
    size = size_bytes // 4
    data = torch.randn(size).pin_memory()
    
    # H2D (Host to Device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_data = data.cuda()
        torch.cuda.synchronize()
    h2d_time = (time.perf_counter() - start) / 10
    h2d_bw = size_bytes / h2d_time / 1e9
    
    # D2H (Device to Host)
    gpu_data = data.cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        cpu_data = gpu_data.cpu()
        torch.cuda.synchronize()
    d2h_time = (time.perf_counter() - start) / 10
    d2h_bw = size_bytes / d2h_time / 1e9
    
    print(f"\n256 MB transfer bandwidth:")
    print(f"  Host → Device: {h2d_bw:.1f} GB/s ({h2d_time*1000:.1f} ms)")
    print(f"  Device → Host: {d2h_bw:.1f} GB/s ({d2h_time*1000:.1f} ms)")
    
    # Reference
    print("\n  PCIe theoretical max:")
    print("    PCIe 3.0 x16: 16 GB/s")
    print("    PCIe 4.0 x16: 32 GB/s")
    print("    PCIe 5.0 x16: 64 GB/s")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS FOR ML TRAINING")
    print("=" * 70)
    print("""
1. USE PINNED MEMORY
   - DataLoader: pin_memory=True
   - Manual: tensor.pin_memory()
   - 2-3x faster transfers!

2. MINIMIZE TRANSFERS
   - Keep data on GPU as long as possible
   - Avoid .cpu() in training loop
   - Use in-place operations

3. USE NON-BLOCKING
   - .cuda(non_blocking=True)
   - Overlap with CPU preprocessing

4. BATCH YOUR TRANSFERS
   - One large transfer > many small transfers
   - Transfer entire batch at once

5. PROFILE YOUR PIPELINE
   - torch.cuda.synchronize() for accurate timing
   - Check GPU utilization (nvidia-smi)
   - Low GPU util = transfer bottleneck
""")

if __name__ == "__main__":
    if check_cuda():
        profile_transfer()
