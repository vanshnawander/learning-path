# 09 - Data Formats & Serialization

Understanding efficient data formats for ML training pipelines.

## 📚 Topics Covered

### Binary Data Formats
- **NumPy .npy/.npz**: Array serialization
- **HDF5**: Hierarchical data format
- **Parquet**: Columnar storage
- **Apache Arrow**: In-memory columnar format
- **MessagePack**: Binary JSON alternative

### ML-Specific Formats
- **FFCV .beton Format**: Fast training data format
  - Memory-mapped access
  - Quasi-random sampling
  - On-the-fly decoding
- **WebDataset**: TAR-based sharded format
  - Streaming from remote storage
  - Shard-based shuffling
- **TFRecord**: TensorFlow's format
- **RecordIO**: MXNet's format

### Image Formats
- **JPEG**: Lossy compression, decode overhead
- **PNG**: Lossless, larger files
- **WebP**: Modern efficient format
- **Raw Pixels**: Fastest but largest

### Format Comparison

| Format | Random Access | Compression | Memory Map | Use Case |
|--------|--------------|-------------|------------|----------|
| .beton | ✅ | ✅ | ✅ | Local fast training |
| WebDataset | ❌ | ✅ | ❌ | Cloud/streaming |
| TFRecord | ❌ | ✅ | ❌ | TensorFlow |
| Arrow | ✅ | ✅ | ✅ | Analytics |
| HDF5 | ✅ | ✅ | ✅ | Scientific |

### Design Considerations
- **Sequential vs Random Access**: Training patterns
- **Compression Trade-offs**: Decode time vs storage
- **Memory Mapping**: Avoiding copies
- **Prefetching**: Overlapping I/O and compute

## 🎯 Learning Objectives

- [ ] Understand .beton format internals
- [ ] Compare WebDataset vs FFCV
- [ ] Implement a simple binary format
- [ ] Measure decode performance

## 💻 Practical Exercises

1. Convert ImageNet to .beton format
2. Benchmark different image formats
3. Implement memory-mapped data reader
4. Compare shuffling strategies

## 📖 Resources

### Papers
- "FFCV: Accelerating Training by Removing Data Bottlenecks" (CVPR 2023)
- WebDataset documentation

### Code References
- `ffcv-main/` - FFCV implementation
- `ffcv-main/ffcv/fields/` - Field type implementations

## 📁 Structure

```
09-data-formats-serialization/
├── binary-formats/
│   ├── numpy/
│   ├── hdf5/
│   ├── arrow/
│   └── msgpack/
├── ml-formats/
│   ├── beton/
│   ├── webdataset/
│   └── tfrecord/
├── image-formats/
│   ├── jpeg-decode/
│   ├── compression/
│   └── benchmarks/
└── design/
    ├── memory-mapping/
    └── prefetching/
```

## ⏱️ Estimated Time: 2-3 weeks
