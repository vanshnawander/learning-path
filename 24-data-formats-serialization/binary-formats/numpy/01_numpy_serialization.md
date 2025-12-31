# NumPy Serialization: .npy and .npz Formats

## Overview

NumPy's native formats are simple, fast, and ideal for array storage.

| Format | Extension | Multiple Arrays | Compression |
|--------|-----------|-----------------|-------------|
| NPY | `.npy` | No | No |
| NPZ | `.npz` | Yes | Optional |

## .npy Format Structure

```
.npy file layout:
┌──────────────────────────────────────────────────────────────┐
│ Magic String (6 bytes): \x93NUMPY                            │
├──────────────────────────────────────────────────────────────┤
│ Version (2 bytes): Major.Minor (e.g., 1.0, 2.0, 3.0)        │
├──────────────────────────────────────────────────────────────┤
│ Header Length (2 or 4 bytes depending on version)            │
├──────────────────────────────────────────────────────────────┤
│ Header (Python dict literal):                                │
│   {'descr': '<f8', 'fortran_order': False, 'shape': (100,)} │
├──────────────────────────────────────────────────────────────┤
│ Data (raw binary, row-major or column-major)                │
└──────────────────────────────────────────────────────────────┘
```

### Header Fields

- **descr**: Data type (e.g., `'<f4'` for little-endian float32)
- **fortran_order**: True for column-major (Fortran), False for row-major (C)
- **shape**: Tuple of dimensions

## Basic Usage

### Saving and Loading

```python
import numpy as np

# Create array
data = np.random.randn(1000, 256).astype(np.float32)

# Save single array
np.save('data.npy', data)

# Load single array
loaded = np.load('data.npy')

# Save multiple arrays (npz)
np.savez('multiple.npz', 
         features=data, 
         labels=np.arange(1000))

# Save compressed
np.savez_compressed('compressed.npz', 
                    features=data, 
                    labels=np.arange(1000))

# Load npz (returns dict-like object)
with np.load('multiple.npz') as npz:
    features = npz['features']
    labels = npz['labels']
```

### Memory Mapping

```python
# Memory-map large file (doesn't load into RAM)
data_mmap = np.load('large_data.npy', mmap_mode='r')

# Access subset (only that portion loaded)
subset = data_mmap[1000:2000]

# mmap modes:
# 'r'  - Read-only
# 'r+' - Read-write (changes saved to file)
# 'w+' - Create/overwrite, read-write
# 'c'  - Copy-on-write (changes not saved)
```

## Performance Characteristics

### Benchmark: Save/Load Times

```python
import numpy as np
import time
import os

def benchmark_numpy_formats(shape=(10000, 1000)):
    data = np.random.randn(*shape).astype(np.float32)
    
    # .npy (uncompressed)
    start = time.time()
    np.save('test.npy', data)
    npy_save = time.time() - start
    
    start = time.time()
    _ = np.load('test.npy')
    npy_load = time.time() - start
    
    npy_size = os.path.getsize('test.npy')
    
    # .npz compressed
    start = time.time()
    np.savez_compressed('test.npz', data=data)
    npz_save = time.time() - start
    
    start = time.time()
    with np.load('test.npz') as f:
        _ = f['data']
    npz_load = time.time() - start
    
    npz_size = os.path.getsize('test.npz')
    
    print(f"Data: {shape}, {data.nbytes / 1e6:.1f} MB")
    print(f".npy save: {npy_save*1000:.1f} ms, load: {npy_load*1000:.1f} ms, size: {npy_size/1e6:.1f} MB")
    print(f".npz save: {npz_save*1000:.1f} ms, load: {npz_load*1000:.1f} ms, size: {npz_size/1e6:.1f} MB")

# Example output:
# Data: (10000, 1000), 40.0 MB
# .npy save: 45.2 ms, load: 12.3 ms, size: 40.0 MB
# .npz save: 1523.4 ms, load: 234.5 ms, size: 31.2 MB
```

### When to Use Which

| Scenario | Recommendation |
|----------|----------------|
| Fast I/O needed | `.npy` (no compression) |
| Storage constrained | `.npz` compressed |
| Multiple related arrays | `.npz` |
| Memory-mapped access | `.npy` |
| Cross-language | Consider HDF5 or Arrow |

## Advanced Usage

### Custom Header (Low-Level)

```python
import numpy.lib.format as fmt

# Read header without loading data
with open('data.npy', 'rb') as f:
    version = fmt.read_magic(f)
    header = fmt.read_array_header_1_0(f)  # or _2_0
    print(f"Shape: {header[0]}, Dtype: {header[2]}")

# Write array with specific format
with open('custom.npy', 'wb') as f:
    fmt.write_array_header_1_0(f, 
        {'shape': (100, 100), 
         'fortran_order': False, 
         'descr': np.dtype('<f4').descr})
    data = np.zeros((100, 100), dtype=np.float32)
    f.write(data.tobytes())
```

### Structured Arrays

```python
# Define structured dtype
dt = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('label', np.int32),
    ('name', 'U20')  # Unicode string
])

# Create and save
data = np.zeros(1000, dtype=dt)
data['x'] = np.random.randn(1000)
data['y'] = np.random.randn(1000)
data['label'] = np.random.randint(0, 10, 1000)

np.save('structured.npy', data)

# Load preserves structure
loaded = np.load('structured.npy')
print(loaded['x'][:5])
print(loaded['label'][:5])
```

## Comparison with Other Formats

| Feature | .npy/.npz | HDF5 | Parquet | Arrow |
|---------|-----------|------|---------|-------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Compression | Optional | Yes | Yes | Yes |
| Partial Read | mmap only | Yes | Yes | Yes |
| Metadata | Minimal | Rich | Schema | Schema |
| Cross-Language | Python only | Yes | Yes | Yes |
| Streaming | No | No | Yes | Yes |

## Best Practices

1. **Use `.npy` for temporary/cache files** - Fastest I/O
2. **Use `.npz` for distribution** - Smaller size
3. **Use mmap for large arrays** - Avoids loading entire file
4. **Specify dtype explicitly** - Avoid dtype surprises
5. **Consider HDF5 for complex data** - Better metadata support

## References

- NumPy documentation: https://numpy.org/doc/stable/reference/routines.io.html
- NPY format specification: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
