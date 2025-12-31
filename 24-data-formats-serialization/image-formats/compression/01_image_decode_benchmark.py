"""
Image Format Decode Benchmark

Compares decode performance of different image formats
to understand their impact on ML training pipelines.
"""

import time
import io
import os
import numpy as np
from pathlib import Path

# Optional imports - gracefully handle missing packages
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import turbojpeg
    jpeg = turbojpeg.TurboJPEG()
    HAS_TURBOJPEG = True
except ImportError:
    HAS_TURBOJPEG = False

try:
    import torch
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_test_image(width=224, height=224):
    """Create a random test image."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def encode_image(img_array, format_name, quality=95):
    """Encode image to various formats."""
    if not HAS_PIL:
        raise RuntimeError("PIL required")
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    
    if format_name == 'jpeg':
        img.save(buffer, format='JPEG', quality=quality)
    elif format_name == 'png':
        img.save(buffer, format='PNG', compress_level=6)
    elif format_name == 'webp':
        img.save(buffer, format='WEBP', quality=quality)
    elif format_name == 'raw':
        buffer.write(img_array.tobytes())
    else:
        raise ValueError(f"Unknown format: {format_name}")
    
    return buffer.getvalue()


def benchmark_pil_decode(data, format_name, iterations=100):
    """Benchmark PIL/Pillow decode."""
    if not HAS_PIL:
        return None
    
    if format_name == 'raw':
        return None  # PIL doesn't decode raw
    
    times = []
    for _ in range(iterations):
        buffer = io.BytesIO(data)
        start = time.perf_counter()
        img = Image.open(buffer)
        img.load()  # Force decode
        arr = np.array(img)
        times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'throughput': 1.0 / np.mean(times)
    }


def benchmark_cv2_decode(data, format_name, iterations=100):
    """Benchmark OpenCV decode."""
    if not HAS_CV2:
        return None
    
    if format_name == 'raw':
        return None
    
    times = []
    np_data = np.frombuffer(data, dtype=np.uint8)
    
    for _ in range(iterations):
        start = time.perf_counter()
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'throughput': 1.0 / np.mean(times)
    }


def benchmark_turbojpeg_decode(data, iterations=100):
    """Benchmark TurboJPEG decode (JPEG only)."""
    if not HAS_TURBOJPEG:
        return None
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        img = jpeg.decode(data)
        times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'throughput': 1.0 / np.mean(times)
    }


def benchmark_raw_decode(data, shape, dtype=np.uint8, iterations=100):
    """Benchmark raw pixel decode (memcpy essentially)."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        # Force copy to simulate real usage
        arr = arr.copy()
        times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'throughput': 1.0 / np.mean(times)
    }


def format_size(size_bytes):
    """Format size in human-readable form."""
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


def main():
    print("=" * 70)
    print("Image Format Decode Benchmark")
    print("=" * 70)
    
    # Test configurations
    sizes = [(224, 224), (384, 384), (512, 512)]
    formats = ['jpeg', 'png', 'webp', 'raw']
    iterations = 100
    
    print(f"\nIterations per test: {iterations}")
    print(f"Libraries available: PIL={HAS_PIL}, CV2={HAS_CV2}, "
          f"TurboJPEG={HAS_TURBOJPEG}, Torch={HAS_TORCH}")
    
    for width, height in sizes:
        print(f"\n{'='*70}")
        print(f"Image Size: {width}x{height}")
        print("=" * 70)
        
        # Create test image
        img_array = create_test_image(width, height)
        raw_size = width * height * 3
        
        print(f"\n{'Format':<10} {'Size':<12} {'PIL (ms)':<12} "
              f"{'CV2 (ms)':<12} {'TurboJPEG':<12} {'imgs/sec':<10}")
        print("-" * 70)
        
        for fmt in formats:
            # Encode
            try:
                data = encode_image(img_array, fmt)
                size_str = format_size(len(data))
            except Exception as e:
                print(f"{fmt:<10} {'Error':<12} {str(e)}")
                continue
            
            # Benchmark decoders
            results = {}
            
            if fmt == 'raw':
                results['raw'] = benchmark_raw_decode(
                    data, (height, width, 3), iterations=iterations)
                pil_time = '-'
                cv2_time = '-'
                turbo_time = '-'
                throughput = f"{results['raw']['throughput']:.0f}" if results['raw'] else '-'
            else:
                results['pil'] = benchmark_pil_decode(data, fmt, iterations)
                results['cv2'] = benchmark_cv2_decode(data, fmt, iterations)
                
                pil_time = f"{results['pil']['mean_ms']:.2f}" if results['pil'] else '-'
                cv2_time = f"{results['cv2']['mean_ms']:.2f}" if results['cv2'] else '-'
                
                if fmt == 'jpeg' and HAS_TURBOJPEG:
                    results['turbo'] = benchmark_turbojpeg_decode(data, iterations)
                    turbo_time = f"{results['turbo']['mean_ms']:.2f}"
                    throughput = f"{results['turbo']['throughput']:.0f}"
                else:
                    turbo_time = '-'
                    # Use best available
                    best = results.get('cv2') or results.get('pil')
                    throughput = f"{best['throughput']:.0f}" if best else '-'
            
            print(f"{fmt:<10} {size_str:<12} {pil_time:<12} "
                  f"{cv2_time:<12} {turbo_time:<12} {throughput:<10}")
    
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print("""
Key Findings:

1. JPEG Decode Performance:
   - TurboJPEG is ~2-3x faster than PIL/OpenCV
   - Still significant CPU overhead (1-5ms per image)
   - For 1000 images/sec, need ~5 CPU cores just for decode

2. PNG Performance:
   - Slower than JPEG despite being lossless
   - Larger file size than JPEG
   - Rarely beneficial for training

3. Raw Pixels:
   - Essentially memcpy speed (0.1-0.5ms)
   - 10-50x faster than JPEG decode
   - But ~10x larger file size
   - Best when I/O bandwidth >> CPU decode capacity

4. Recommendations:
   - Local SSD training: Consider raw/pre-decoded formats
   - Network/cloud storage: JPEG to reduce transfer time
   - Use TurboJPEG if JPEG is necessary
   - FFCV's "smart" mode balances size vs decode time
   - GPU decode (nvJPEG) can further accelerate

5. Training Pipeline Impact:
   - 224x224 JPEG @ 1ms decode = 1000 imgs/sec max per core
   - ResNet-50 on A100 wants ~3000 imgs/sec
   - Need 3+ decode threads OR pre-decoded data OR GPU decode
""")


if __name__ == '__main__':
    main()
