# WebDataset

TAR-based format for streaming large datasets.

## Why WebDataset?

- Works with cloud storage (S3, GCS)
- Streaming access (no full download)
- Shard-based shuffling
- Standard TAR format

## Concept

```
shard-000000.tar:
├── sample0001.jpg
├── sample0001.cls
├── sample0002.jpg
├── sample0002.cls
└── ...

shard-000001.tar:
├── sample1001.jpg
├── sample1001.cls
└── ...
```

## Usage

```python
import webdataset as wds

dataset = wds.WebDataset("s3://bucket/data-{0000..0099}.tar")
dataset = dataset.shuffle(1000)
dataset = dataset.decode("rgb8")
dataset = dataset.to_tuple("jpg", "cls")
```

## Trade-offs vs .beton

| Aspect | WebDataset | .beton |
|--------|------------|--------|
| Random Access | ❌ Sequential | ✅ Yes |
| Cloud Native | ✅ Yes | ❌ Local |
| Compression | ✅ Good | ✅ Good |
| Speed (local) | Slower | Faster |
| Speed (cloud) | Good | N/A |

## When to Use
- Large-scale distributed training
- Data on cloud storage
- Need TAR compatibility
