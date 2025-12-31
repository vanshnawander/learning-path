# WebDataset: TAR-Based Streaming Format

## What is WebDataset?

WebDataset is a PyTorch-compatible data loading library that uses **TAR files** for efficient streaming from local or remote storage. Designed for large-scale ML training.

## Key Concepts

### TAR-Based Sharding

```
dataset/
├── shard-000000.tar
│   ├── sample_00000.jpg
│   ├── sample_00000.cls    # Class label
│   ├── sample_00000.json   # Metadata
│   ├── sample_00001.jpg
│   ├── sample_00001.cls
│   └── ...
├── shard-000001.tar
├── shard-000002.tar
└── ...

Files with same base name = one sample
Extension determines content type
```

### Why TAR?

| Benefit | Explanation |
|---------|-------------|
| **Sequential I/O** | TAR is read sequentially = fast streaming |
| **Cloud-native** | HTTP range requests, S3/GCS compatible |
| **No index needed** | Files processed in order within TAR |
| **Compression** | Optional per-shard compression |
| **Atomic shards** | Easy to shuffle, distribute, resume |

## Creating WebDatasets

### Using the CLI

```bash
# Create shards from directory
tar cvf shard-000000.tar sample_00000.jpg sample_00000.cls sample_00001.jpg ...

# Or using webdataset's tools
python -m webdataset.cli create dataset.tar --maxcount 10000 files/*
```

### Using Python

```python
import webdataset as wds
import json

# Create a shard writer
with wds.TarWriter("shard-000000.tar") as sink:
    for i in range(10000):
        # Each sample is a dict
        sample = {
            "__key__": f"sample_{i:06d}",
            "jpg": open(f"images/{i}.jpg", "rb").read(),
            "cls": str(labels[i]).encode(),
            "json": json.dumps({"id": i, "metadata": "..."}).encode(),
        }
        sink.write(sample)

# Create multiple shards with ShardWriter
with wds.ShardWriter("shards/shard-%06d.tar", maxcount=10000) as sink:
    for i, (image, label) in enumerate(dataset):
        sink.write({
            "__key__": f"sample_{i:06d}",
            "jpg": image_to_bytes(image),
            "cls": str(label).encode(),
        })
```

## Loading WebDatasets

### Basic Loading

```python
import webdataset as wds

# Local files
dataset = wds.WebDataset("shards/shard-{000000..000099}.tar")

# Remote (S3)
dataset = wds.WebDataset("s3://bucket/shards/shard-{000000..000099}.tar")

# HTTP
dataset = wds.WebDataset("http://server/shards/shard-{000000..000099}.tar")

# Process samples
dataset = (
    dataset
    .decode("pil")           # Decode images as PIL
    .to_tuple("jpg", "cls")  # Extract fields as tuple
)

for image, label in dataset:
    # image is PIL.Image, label is string
    pass
```

### Full Training Pipeline

```python
import webdataset as wds
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def make_sample(sample):
    return transform(sample["jpg"]), int(sample["cls"])

dataset = (
    wds.WebDataset("shards/shard-{000000..000099}.tar")
    .shuffle(1000)                    # Shuffle buffer
    .decode("pil")                    # Decode images
    .map(make_sample)                 # Apply transform
)

# Create DataLoader
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
)

for images, labels in loader:
    # Training step
    pass
```

### Distributed Training

```python
import webdataset as wds

# Automatic sharding across workers
dataset = (
    wds.WebDataset("shards/shard-{000000..000099}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
)

# For distributed (multi-GPU)
dataset = (
    wds.WebDataset(
        "shards/shard-{000000..000099}.tar",
        shardshuffle=True,      # Shuffle shard order
        nodesplitter=wds.split_by_node,  # Split across nodes
    )
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
)
```

## Shuffling Strategies

### Shard-Level Shuffling

```python
# Shuffle shard order each epoch
dataset = wds.WebDataset(urls, shardshuffle=True)
```

### Sample-Level Shuffling

```python
# Buffer-based shuffling within shards
dataset = dataset.shuffle(buffer_size=5000)
```

### Combined Approach

```
Epoch 1:
  Shards: [3, 7, 1, 9, 2, ...] (shuffled order)
  Within each shard: buffer shuffle

Epoch 2:
  Shards: [8, 2, 5, 1, 4, ...] (different order)
  Within each shard: buffer shuffle

Result: Good randomization without full shuffle
```

## Performance Optimization

### Prefetching

```python
# Parallel shard loading
dataset = (
    wds.WebDataset(urls)
    .decode("pil")
    .to_tuple("jpg", "cls")
    .batched(64)
    .prefetch(2)  # Prefetch 2 batches
)
```

### Caching

```python
# Cache decoded samples
dataset = (
    wds.WebDataset(urls)
    .decode("pil")
    .cached(cache_dir="/tmp/wds_cache", cache_size=10000)
    .to_tuple("jpg", "cls")
)
```

### Optimal Shard Size

```
Recommendations:
- 100MB - 1GB per shard (good balance)
- 1000-10000 samples per shard
- Avoid too small (overhead) or too large (shuffle granularity)
```

## WebDataset vs FFCV

| Feature | WebDataset | FFCV |
|---------|-----------|------|
| **Access Pattern** | Sequential streaming | Random access |
| **Best For** | Cloud/streaming | Local SSD |
| **Shuffling** | Shard + buffer | True random |
| **Memory Map** | No | Yes |
| **Decode** | CPU (Python) | CPU/GPU |
| **Setup Complexity** | Low | Medium |
| **Cross-Platform** | Excellent | Linux-focused |

### When to Use WebDataset

✅ **Good for:**
- Cloud storage (S3, GCS)
- Streaming large datasets
- Multi-node training
- Variable-size samples
- Simple setup

❌ **Consider FFCV when:**
- Local SSD training
- Need true random access
- Maximum throughput critical
- GPU-accelerated decode

## Common Patterns

### Multi-Modal Data

```python
# Image + text + audio
dataset = (
    wds.WebDataset(urls)
    .decode("pil", handler=wds.warn_and_continue)
    .to_tuple("jpg", "txt", "flac")
)

for image, text, audio in dataset:
    # Process multimodal sample
    pass
```

### Validation Split

```python
# Training shards
train_urls = "shards/train-{000000..000079}.tar"
# Validation shards (different files)
val_urls = "shards/val-{000000..000019}.tar"

train_dataset = wds.WebDataset(train_urls).shuffle(1000)
val_dataset = wds.WebDataset(val_urls)  # No shuffle for validation
```

### Resume Training

```python
# WebDataset is stateless - resume by re-shuffling shards
# For exact resume, track which shards completed
completed_shards = load_checkpoint()["completed_shards"]
remaining_urls = [u for u in all_urls if u not in completed_shards]
dataset = wds.WebDataset(remaining_urls)
```

## References

- WebDataset GitHub: https://github.com/webdataset/webdataset
- Documentation: https://webdataset.github.io/webdataset/
- "Large Scale Dataset Pragmatics" - Aizman et al.
