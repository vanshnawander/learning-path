"""
Memory-Mapped Data Loading for ML Training

Demonstrates how to implement efficient data loading using memory mapping,
which is the core technique behind FFCV and other fast data loaders.

Key concepts:
- mmap bypasses kernel buffer cache copies
- OS handles paging automatically
- Enables random access without loading entire file
"""

import mmap
import numpy as np
import struct
import os
import time
from typing import Iterator, Tuple, Optional

# ============================================================
# 1. Basic Memory Mapping
# ============================================================

def basic_mmap_demo():
    """Demonstrate basic memory mapping operations."""
    
    # Create a test file
    filename = "test_mmap.bin"
    data = np.arange(1000000, dtype=np.float32)
    data.tofile(filename)
    
    # Memory map the file
    with open(filename, 'r+b') as f:
        # mmap the entire file
        mm = mmap.mmap(f.fileno(), 0)  # 0 = entire file
        
        # Read as numpy array (zero-copy view!)
        arr = np.frombuffer(mm, dtype=np.float32)
        print(f"Shape: {arr.shape}, First 5: {arr[:5]}")
        
        # Random access is efficient
        print(f"Element 500000: {arr[500000]}")
        
        # Modify in place (writes to file)
        arr[0] = 999.0
        mm.flush()  # Ensure written to disk
        
        mm.close()
    
    os.remove(filename)
    print("Basic mmap demo complete\n")


# ============================================================
# 2. Custom Binary Dataset Format
# ============================================================

class BinaryDatasetWriter:
    """
    Write a simple binary dataset format:
    
    Header (64 bytes):
        - magic: 4 bytes ("BDAT")
        - version: 4 bytes (uint32)
        - num_samples: 8 bytes (uint64)
        - sample_size: 8 bytes (uint64)
        - dtype: 4 bytes (enum)
        - reserved: 36 bytes
    
    Index (num_samples * 8 bytes):
        - offsets to each sample
    
    Data:
        - concatenated sample data
    """
    
    MAGIC = b'BDAT'
    VERSION = 1
    DTYPE_MAP = {
        np.float32: 0,
        np.float64: 1,
        np.int32: 2,
        np.int64: 3,
        np.uint8: 4,
    }
    
    def __init__(self, filename: str, sample_shape: Tuple[int, ...], dtype=np.float32):
        self.filename = filename
        self.sample_shape = sample_shape
        self.dtype = dtype
        self.sample_size = int(np.prod(sample_shape)) * np.dtype(dtype).itemsize
        self.samples = []
        self.offsets = []
        
    def add_sample(self, data: np.ndarray):
        """Add a sample to the dataset."""
        assert data.shape == self.sample_shape
        assert data.dtype == self.dtype
        self.samples.append(data.tobytes())
        
    def close(self):
        """Write the dataset to disk."""
        num_samples = len(self.samples)
        header_size = 64
        index_size = num_samples * 8
        data_start = header_size + index_size
        
        with open(self.filename, 'wb') as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            f.write(struct.pack('<Q', num_samples))
            f.write(struct.pack('<Q', self.sample_size))
            f.write(struct.pack('<I', self.DTYPE_MAP[self.dtype]))
            f.write(b'\x00' * 36)  # Reserved
            
            # Calculate offsets
            offset = data_start
            offsets = []
            for sample in self.samples:
                offsets.append(offset)
                offset += len(sample)
            
            # Write index
            for off in offsets:
                f.write(struct.pack('<Q', off))
            
            # Write data
            for sample in self.samples:
                f.write(sample)
        
        print(f"Wrote {num_samples} samples to {self.filename}")


class BinaryDatasetReader:
    """Memory-mapped reader for binary dataset."""
    
    DTYPE_MAP = {
        0: np.float32,
        1: np.float64,
        2: np.int32,
        3: np.int64,
        4: np.uint8,
    }
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse header
        magic = self.mm[:4]
        assert magic == b'BDAT', f"Invalid magic: {magic}"
        
        self.version = struct.unpack('<I', self.mm[4:8])[0]
        self.num_samples = struct.unpack('<Q', self.mm[8:16])[0]
        self.sample_size = struct.unpack('<Q', self.mm[16:24])[0]
        dtype_id = struct.unpack('<I', self.mm[24:28])[0]
        self.dtype = self.DTYPE_MAP[dtype_id]
        
        # Load index
        index_start = 64
        index_end = index_start + self.num_samples * 8
        self.offsets = np.frombuffer(
            self.mm[index_start:index_end], dtype=np.uint64
        )
        
        # Calculate sample shape from size
        elem_size = np.dtype(self.dtype).itemsize
        self.num_elements = self.sample_size // elem_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a sample by index (zero-copy when possible)."""
        if idx < 0:
            idx = self.num_samples + idx
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of range")
        
        offset = self.offsets[idx]
        # Create view into mmap (zero-copy!)
        data = np.frombuffer(
            self.mm[offset:offset + self.sample_size],
            dtype=self.dtype
        )
        return data
    
    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.num_samples):
            yield self[i]
    
    def close(self):
        self.mm.close()
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def binary_dataset_demo():
    """Demonstrate custom binary dataset format."""
    print("=== Binary Dataset Demo ===")
    
    # Create dataset
    filename = "test_dataset.bdat"
    sample_shape = (3, 224, 224)  # Like ImageNet images
    num_samples = 1000
    
    writer = BinaryDatasetWriter(filename, sample_shape, np.float32)
    for i in range(num_samples):
        sample = np.random.randn(*sample_shape).astype(np.float32)
        writer.add_sample(sample)
    writer.close()
    
    # Read with mmap
    with BinaryDatasetReader(filename) as reader:
        print(f"Samples: {len(reader)}, dtype: {reader.dtype}")
        
        # Random access benchmark
        indices = np.random.permutation(num_samples)[:100]
        
        start = time.time()
        for idx in indices:
            sample = reader[idx]
        elapsed = time.time() - start
        print(f"100 random reads: {elapsed*1000:.2f} ms")
        
        # Sequential access benchmark
        start = time.time()
        for i, sample in enumerate(reader):
            if i >= 100:
                break
        elapsed = time.time() - start
        print(f"100 sequential reads: {elapsed*1000:.2f} ms")
    
    os.remove(filename)
    print()


# ============================================================
# 3. Quasi-Random Sampling (FFCV-style)
# ============================================================

class QuasiRandomSampler:
    """
    FFCV-style quasi-random sampling.
    
    Instead of truly random sampling (causes random disk seeks),
    quasi-random sampling accesses data in mostly-sequential order
    while still providing sufficient randomness for training.
    
    Strategy:
    1. Divide dataset into pages
    2. Shuffle page order
    3. Within each page, access sequentially (or with small shuffles)
    """
    
    def __init__(self, num_samples: int, page_size: int = 1000):
        self.num_samples = num_samples
        self.page_size = page_size
        self.num_pages = (num_samples + page_size - 1) // page_size
        
    def __iter__(self) -> Iterator[int]:
        # Shuffle pages
        pages = np.random.permutation(self.num_pages)
        
        for page_idx in pages:
            start = page_idx * self.page_size
            end = min(start + self.page_size, self.num_samples)
            
            # Small shuffle within page (optional)
            indices = np.arange(start, end)
            np.random.shuffle(indices)
            
            for idx in indices:
                yield idx
    
    def __len__(self):
        return self.num_samples


def quasi_random_demo():
    """Compare true random vs quasi-random sampling."""
    print("=== Quasi-Random Sampling Demo ===")
    
    num_samples = 10000
    
    # True random
    true_random = np.random.permutation(num_samples)
    
    # Quasi-random
    sampler = QuasiRandomSampler(num_samples, page_size=100)
    quasi_random = list(sampler)
    
    # Analyze access patterns
    def analyze_pattern(indices, name):
        diffs = np.abs(np.diff(indices))
        print(f"{name}:")
        print(f"  Mean jump: {np.mean(diffs):.1f}")
        print(f"  Max jump: {np.max(diffs)}")
        print(f"  Sequential (diff<=100): {np.sum(diffs <= 100) / len(diffs) * 100:.1f}%")
    
    analyze_pattern(true_random, "True random")
    analyze_pattern(quasi_random, "Quasi-random")
    print()


# ============================================================
# 4. Prefetching with Threading
# ============================================================

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

class PrefetchingDataLoader:
    """
    Data loader with background prefetching.
    
    Uses a thread pool to load data ahead of consumption,
    overlapping I/O with computation.
    """
    
    def __init__(self, reader: BinaryDatasetReader, 
                 batch_size: int = 32,
                 prefetch_factor: int = 2,
                 num_workers: int = 4):
        self.reader = reader
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        
    def __iter__(self):
        indices = list(range(len(self.reader)))
        np.random.shuffle(indices)
        
        # Split into batches
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        
        # Prefetch queue
        queue = Queue(maxsize=self.prefetch_factor)
        stop_event = threading.Event()
        
        def load_batch(batch_indices):
            samples = [self.reader[i] for i in batch_indices]
            return np.stack(samples)
        
        def producer():
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for batch in batches:
                    if stop_event.is_set():
                        break
                    future = executor.submit(load_batch, batch)
                    futures.append(future)
                    
                    # Limit prefetch
                    if len(futures) >= self.prefetch_factor:
                        result = futures.pop(0).result()
                        queue.put(result)
                
                # Drain remaining
                for future in futures:
                    if not stop_event.is_set():
                        queue.put(future.result())
                
                queue.put(None)  # Sentinel
        
        # Start producer thread
        thread = threading.Thread(target=producer)
        thread.start()
        
        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            thread.join()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    basic_mmap_demo()
    binary_dataset_demo()
    quasi_random_demo()
    
    print("All demos complete!")
