# Tensor Storage: How ML Data Lives in Memory

Understanding tensor memory layout is essential for optimization.

## NumPy/PyTorch Array Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    TENSOR OBJECT                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Metadata (small):              Data Buffer (large):        │
│  ┌─────────────────┐           ┌──────────────────────┐    │
│  │ shape: (3,4)    │           │ 1.0 2.0 3.0 4.0      │    │
│  │ dtype: float32  │   ──────▶ │ 5.0 6.0 7.0 8.0      │    │
│  │ strides: (16,4) │           │ 9.0 10. 11. 12.      │    │
│  │ data_ptr: 0x... │           └──────────────────────┘    │
│  │ device: cpu     │                                        │
│  └─────────────────┘                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Strides: The Key Concept

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)

print(a.shape)    # (2, 3)
print(a.strides)  # (12, 4)

# strides = (bytes to next row, bytes to next column)
# 12 bytes = 3 elements × 4 bytes/element
# 4 bytes = 1 element × 4 bytes/element
```

## Memory Layout

### Row-Major (C-order, NumPy/PyTorch default)
```
Logical:          Memory:
[[1, 2, 3],       [1, 2, 3, 4, 5, 6]
 [4, 5, 6]]        ───────────────▶

strides = (cols * sizeof, sizeof) = (12, 4)

a[i,j] at offset: i * stride[0] + j * stride[1]
a[1,2] at offset: 1 * 12 + 2 * 4 = 20 bytes = element 5 (value 6)
```

### Column-Major (Fortran-order)
```
Logical:          Memory:
[[1, 2, 3],       [1, 4, 2, 5, 3, 6]
 [4, 5, 6]]        ───────────────▶

strides = (sizeof, rows * sizeof) = (4, 8)

Columns are contiguous, not rows!
```

## Views vs Copies

### View (Same Data, Different Metadata)
```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)  # VIEW - no copy!

b[0, 0] = 99
print(a)  # [99, 2, 3, 4, 5, 6] - a changed!

# b shares data with a
print(np.shares_memory(a, b))  # True
```

### Copy (New Data Buffer)
```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3).copy()  # COPY - new data

b[0, 0] = 99
print(a)  # [1, 2, 3, 4, 5, 6] - a unchanged
```

## Non-Contiguous Tensors

### Transpose Creates Non-Contiguous View
```python
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)

b = a.T  # Transpose - just swaps strides!

print(a.strides)  # (12, 4) - rows are contiguous
print(b.strides)  # (4, 12) - columns are contiguous

# b is NOT contiguous in row-major sense
print(b.flags['C_CONTIGUOUS'])  # False

# Memory layout unchanged!
# b[0] accesses: 1, 4 (stride 12, not 4)
```

### Slicing Can Create Non-Contiguous
```python
a = np.arange(12).reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

b = a[:, ::2]  # Every other column
# [[0, 2],
#  [4, 6],
#  [8, 10]]

print(b.strides)  # (16, 8) - skipping elements!
print(b.flags['C_CONTIGUOUS'])  # False
```

## Why Contiguity Matters

### Performance Impact
```python
import numpy as np
import time

# Contiguous array
a_contig = np.random.randn(1000, 1000).astype(np.float32)

# Non-contiguous (transposed)
a_noncontig = a_contig.T

# Sum (memory access pattern)
start = time.time()
for _ in range(100):
    a_contig.sum()
print(f"Contiguous: {time.time() - start:.3f}s")

start = time.time()
for _ in range(100):
    a_noncontig.sum()
print(f"Non-contiguous: {time.time() - start:.3f}s")

# Non-contiguous is typically 2-10x slower!
```

### Making Contiguous
```python
# NumPy
b_contig = np.ascontiguousarray(b_noncontig)

# PyTorch
tensor_contig = tensor_noncontig.contiguous()

# Check before operations
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
```

## PyTorch Memory Formats

### Standard (NCHW)
```python
# Default for most operations
x = torch.randn(32, 3, 224, 224)  # [Batch, Channel, Height, Width]
# Memory: B0C0H0W0, B0C0H0W1, ..., B0C0H1W0, ...
```

### Channels Last (NHWC)
```python
# Better for some GPU operations (Tensor Cores)
x = x.to(memory_format=torch.channels_last)
# Memory: B0H0W0C0, B0H0W0C1, B0H0W0C2, B0H0W1C0, ...

# Up to 2x faster for convolutions on recent GPUs!
```

## Dataset File Formats

### NumPy (.npy)
```python
# Simple, single array
np.save('array.npy', arr)
arr = np.load('array.npy')

# Optionally memory-mapped
arr = np.load('array.npy', mmap_mode='r')  # No RAM copy!
```

### NumPy Archive (.npz)
```python
# Multiple arrays in one file
np.savez('data.npz', images=images, labels=labels)

# Load (lazy)
data = np.load('data.npz')
images = data['images']
```

### HDF5 (.h5)
```python
import h5py

# Hierarchical, chunked, compressed
with h5py.File('data.h5', 'w') as f:
    f.create_dataset('images', data=images, 
                     chunks=(100, 3, 224, 224),
                     compression='gzip')

# Random access without loading all
with h5py.File('data.h5', 'r') as f:
    sample = f['images'][1000]  # Load just one
```

### PyTorch (.pt)
```python
# Pickle-based, includes metadata
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 
           'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
```

### FFCV (.beton)
```
Optimized for ML training:
- Memory-mapped access
- Pre-decoded images
- Quasi-random sampling
- Zero-copy transfers
```

## Memory Alignment

```python
# Check alignment
def is_aligned(arr, alignment=64):
    return arr.ctypes.data % alignment == 0

# Aligned allocation (for SIMD)
arr = np.empty(1000, dtype=np.float32)
# NumPy typically aligns to 64 bytes

# PyTorch aligns for GPU efficiency
tensor = torch.empty(1000)  # Aligned for CUDA
```
