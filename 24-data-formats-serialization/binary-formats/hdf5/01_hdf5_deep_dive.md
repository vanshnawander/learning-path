# HDF5: Hierarchical Data Format

## What is HDF5?

HDF5 (Hierarchical Data Format version 5) is a file format and library for storing and managing large, complex data. Used extensively in:

- Scientific computing
- Climate/weather data
- Astronomy
- Machine learning datasets
- Medical imaging

## Core Concepts

### File Structure

```
my_data.h5 (HDF5 File)
├── / (Root Group)
│   ├── experiment_1/ (Group)
│   │   ├── images (Dataset: 1000x256x256 float32)
│   │   ├── labels (Dataset: 1000 int32)
│   │   └── metadata (Attributes)
│   │
│   ├── experiment_2/ (Group)
│   │   ├── images (Dataset)
│   │   └── labels (Dataset)
│   │
│   └── config (Dataset: compound type)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **File** | Container for all data |
| **Group** | Folder-like container |
| **Dataset** | N-dimensional array |
| **Attribute** | Small metadata on groups/datasets |
| **Datatype** | Data element specification |
| **Dataspace** | Dimensions and shape |

## Python h5py Usage

### Basic Operations

```python
import h5py
import numpy as np

# Create file
with h5py.File('data.h5', 'w') as f:
    # Create group
    grp = f.create_group('experiment')
    
    # Create dataset
    data = np.random.randn(1000, 256, 256).astype(np.float32)
    dset = grp.create_dataset('images', data=data)
    
    # Add attributes
    dset.attrs['description'] = 'Random images'
    dset.attrs['timestamp'] = '2024-01-01'
    grp.attrs['experiment_id'] = 42

# Read file
with h5py.File('data.h5', 'r') as f:
    # Navigate hierarchy
    print(list(f.keys()))  # ['experiment']
    print(list(f['experiment'].keys()))  # ['images']
    
    # Read data
    images = f['experiment/images'][:]  # Load all
    subset = f['experiment/images'][0:10]  # Slice
    
    # Read attributes
    print(f['experiment'].attrs['experiment_id'])
```

## Chunking: The Key to Performance

### What is Chunking?

```
Without chunking (contiguous):
┌────────────────────────────────────────┐
│ Row 0, Row 1, Row 2, ... Row 999       │
└────────────────────────────────────────┘
Reading column 0: Must scan entire file

With chunking:
┌──────────┬──────────┬──────────┬───────┐
│ Chunk 0  │ Chunk 1  │ Chunk 2  │ ...   │
│ [0:100,  │ [0:100,  │ [100:200,│       │
│  0:100]  │  100:200]│  0:100]  │       │
└──────────┴──────────┴──────────┴───────┘
Reading subset: Only load relevant chunks
```

### Chunking Strategy

```python
# Create chunked dataset
with h5py.File('chunked.h5', 'w') as f:
    # Auto chunking
    f.create_dataset('auto', shape=(10000, 1000), 
                     dtype='float32', chunks=True)
    
    # Manual chunking
    f.create_dataset('manual', shape=(10000, 1000), 
                     dtype='float32', chunks=(100, 100))
    
    # Check chunk shape
    print(f['manual'].chunks)  # (100, 100)
```

### Chunk Size Guidelines

| Access Pattern | Recommended Chunk Shape |
|----------------|------------------------|
| Row-wise access | (1, N_cols) or (small, N_cols) |
| Column-wise access | (N_rows, 1) or (N_rows, small) |
| Random access | Square-ish chunks |
| Time series | (time_window, spatial_dims) |

**Rule of thumb**: Chunk size 10KB - 1MB

## Compression

### Built-in Filters

```python
with h5py.File('compressed.h5', 'w') as f:
    data = np.random.randn(1000, 1000).astype(np.float32)
    
    # GZIP (good compression, slower)
    f.create_dataset('gzip', data=data, 
                     compression='gzip', compression_opts=4)
    
    # LZF (fast, moderate compression)
    f.create_dataset('lzf', data=data, compression='lzf')
    
    # SZIP (good for scientific data)
    f.create_dataset('szip', data=data, 
                     compression='szip', compression_opts=('nn', 8))
```

### Compression Comparison

| Filter | Speed | Ratio | Best For |
|--------|-------|-------|----------|
| gzip-1 | Fast | 2-3x | General |
| gzip-9 | Slow | 3-5x | Archival |
| lzf | Very fast | 1.5-2x | Real-time |
| szip | Medium | 2-4x | Scientific arrays |
| blosc | Very fast | 2-4x | Numeric data |

### Using Blosc (via hdf5plugin)

```python
import hdf5plugin

with h5py.File('blosc.h5', 'w') as f:
    f.create_dataset('data', data=data,
                     **hdf5plugin.Blosc(cname='lz4', clevel=5))
```

## Parallel I/O with MPI

```python
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parallel file access
with h5py.File('parallel.h5', 'w', driver='mpio', comm=comm) as f:
    dset = f.create_dataset('data', (1000, 1000), dtype='f')
    
    # Each rank writes its portion
    chunk_size = 1000 // size
    start = rank * chunk_size
    end = start + chunk_size
    
    dset[start:end, :] = np.random.randn(chunk_size, 1000)
```

## Virtual Datasets

Combine multiple files into single logical dataset:

```python
# Create source files
for i in range(4):
    with h5py.File(f'part_{i}.h5', 'w') as f:
        f.create_dataset('data', data=np.arange(100) + i*100)

# Create virtual dataset
layout = h5py.VirtualLayout(shape=(400,), dtype='i')
for i in range(4):
    vsource = h5py.VirtualSource(f'part_{i}.h5', 'data', shape=(100,))
    layout[i*100:(i+1)*100] = vsource

with h5py.File('virtual.h5', 'w', libver='latest') as f:
    f.create_virtual_dataset('combined', layout)

# Access as single dataset
with h5py.File('virtual.h5', 'r') as f:
    print(f['combined'][150])  # Transparently reads from part_1.h5
```

## Performance Tips

### 1. Use Appropriate Chunk Cache

```python
# Increase chunk cache for random access
propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
settings[2] = 100 * 1024 * 1024  # 100 MB cache
propfaid.set_cache(*settings)

fid = h5py.h5f.open(b'data.h5', flags=h5py.h5f.ACC_RDONLY, fapl=propfaid)
f = h5py.File(fid)
```

### 2. Batch Reads/Writes

```python
# BAD: Many small reads
for i in range(1000):
    row = dset[i, :]  # 1000 I/O operations

# GOOD: Single read
data = dset[0:1000, :]  # 1 I/O operation
```

### 3. Memory Mapping

```python
# Memory-map for read-only access
with h5py.File('data.h5', 'r', driver='core', backing_store=False) as f:
    # File loaded into memory
    pass

# Or use direct memory mapping (contiguous datasets only)
with h5py.File('data.h5', 'r') as f:
    if f['data'].id.get_offset() is not None:
        # Dataset is contiguous, can mmap
        pass
```

## HDF5 vs Alternatives

| Feature | HDF5 | Zarr | Parquet | Arrow |
|---------|------|------|---------|-------|
| N-dimensional | ✅ | ✅ | ❌ | ❌ |
| Hierarchical | ✅ | ✅ | ❌ | ❌ |
| Compression | ✅ | ✅ | ✅ | ✅ |
| Parallel I/O | ✅ | ✅ | ❌ | ❌ |
| Cloud-native | ❌ | ✅ | ✅ | ✅ |
| Partial read | ✅ | ✅ | ✅ | ✅ |
| Append | ✅ | ✅ | ❌ | ❌ |

## Common Issues

1. **File bloat after deletes**: HDF5 doesn't reclaim space
   - Fix: `h5repack` or recreate file

2. **Slow random access**: Wrong chunk size
   - Fix: Match chunks to access pattern

3. **Concurrent writes**: HDF5 not thread-safe for writes
   - Fix: Use SWMR (Single Writer Multiple Reader) or locks

## References

- HDF5 User Guide: https://portal.hdfgroup.org/
- h5py documentation
- "HDF5 Best Practices" - HDF Group
