# FFCV .beton Format Deep Dive

Understanding the fastest format for ML data loading.

## What is .beton?

```
╔═══════════════════════════════════════════════════════════════════════╗
║                        .BETON FILE FORMAT                              ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  A single binary file optimized for:                                  ║
║  • Memory mapping (mmap)                                              ║
║  • Random access                                                      ║
║  • Zero-copy loading                                                  ║
║  • Page-aligned data                                                  ║
║                                                                        ║
║  FILE STRUCTURE:                                                      ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │ HEADER (metadata, field definitions)                             │ ║
║  ├─────────────────────────────────────────────────────────────────┤ ║
║  │ INDEX (offsets to each sample)                                   │ ║
║  ├─────────────────────────────────────────────────────────────────┤ ║
║  │ SAMPLE 0: [field0_data] [field1_data] ...                       │ ║
║  │ SAMPLE 1: [field0_data] [field1_data] ...                       │ ║
║  │ SAMPLE 2: [field0_data] [field1_data] ...                       │ ║
║  │ ...                                                              │ ║
║  │ SAMPLE N: [field0_data] [field1_data] ...                       │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  KEY FEATURES:                                                        ║
║  • Page-aligned: Each sample aligns to OS page boundary              ║
║  • Pre-processed: Images already resized, decoded                    ║
║  • Indexed: O(1) access to any sample                                ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## Why .beton is Fast

### Traditional Image Loading
```
1. open("image.jpg")           ~50 µs
2. read() into buffer          ~100 µs (depends on size)
3. JPEG decode                 ~5 ms
4. Resize to 224x224           ~2 ms
5. Convert to tensor           ~0.5 ms
6. Normalize                   ~0.2 ms
─────────────────────────────────────
Total: ~8 ms per image

For batch of 64: 64 × 8ms = 512 ms
```

### .beton Loading
```
1. mmap lookup (pointer)       ~0.001 µs
2. Page fault (if not cached)  ~10 µs
3. Read pre-resized data       ~100 µs
4. (Optional) JPEG decode      ~1 ms (smaller image)
5. Normalize                   ~0.2 ms
─────────────────────────────────────
Total: ~1.3 ms per image

For batch of 64: Parallel → ~5 ms total
```

## Field Types

| Field Type | Storage | Use Case |
|------------|---------|----------|
| `RGBImageField` | JPEG/RAW | Images |
| `IntField` | int64 | Labels, indices |
| `FloatField` | float64 | Continuous values |
| `NDArrayField` | numpy | Pre-computed features |
| `BytesField` | raw bytes | Custom data |
| `JSONField` | JSON string | Metadata |

## Image Storage Options

### 1. JPEG (Compressed)
```python
from ffcv.fields import RGBImageField

# JPEG storage - small files, decode overhead
RGBImageField(
    write_mode='jpg',
    max_resolution=256,
    jpeg_quality=90,  # Quality vs size tradeoff
)

# File size: ~10 KB per 256x256 image
# Decode time: ~1 ms
```

### 2. RAW (Uncompressed)
```python
# RAW storage - large files, no decode
RGBImageField(
    write_mode='raw',
    max_resolution=256,
)

# File size: ~196 KB per 256x256 image (256×256×3)
# Decode time: ~0 ms
```

### 3. Smart Mix
```python
# Hybrid: small images RAW, large images JPEG
RGBImageField(
    write_mode='smart',
    max_resolution=256,
    smart_threshold=200 * 200 * 3,  # Below this = RAW
    jpeg_quality=90,
)
```

## Creating .beton Files

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torch.utils.data import Dataset

# Your PyTorch dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Return PIL Image and label
        from PIL import Image
        image = Image.open(self.paths[idx]).convert('RGB')
        return image, self.labels[idx]

# Create writer
writer = DatasetWriter(
    "dataset.beton",
    {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90,
        ),
        'label': IntField(),
    },
    num_workers=16,  # Parallel writing
)

# Write dataset
writer.from_indexed_dataset(dataset)
```

## Loading .beton Files

```python
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor, ToDevice, ToTorchImage, 
    RandomHorizontalFlip, NormalizeImage
)
from ffcv.fields.decoders import (
    IntDecoder, 
    SimpleRGBImageDecoder,
    RandomResizedCropRGBImageDecoder
)

# Define pipelines
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]

loader = Loader(
    "dataset.beton",
    batch_size=256,
    num_workers=8,
    order=OrderOption.QUASI_RANDOM,  # Good locality + randomness
    
    pipelines={
        'image': [
            # Decode + random crop in one step
            RandomResizedCropRGBImageDecoder(
                output_size=(224, 224),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.33),
            ),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
        ]
    }
)

# Training loop
for images, labels in loader:
    # images: already on GPU, normalized, fp16!
    loss = model(images)
```

## Order Options

```python
from ffcv.loader import OrderOption

# SEQUENTIAL: Read in order (fastest I/O)
order=OrderOption.SEQUENTIAL

# RANDOM: Fully random (worst I/O, best ML)
order=OrderOption.RANDOM

# QUASI_RANDOM: Groups nearby samples (good balance)
order=OrderOption.QUASI_RANDOM
# Shuffles within "pages" of data for locality
```

## Performance Numbers

### ImageNet Throughput (A100 GPU)

| Mode | Batch Size | Images/sec | Notes |
|------|------------|------------|-------|
| JPEG, Sequential | 256 | 25,000 | Max I/O |
| JPEG, Quasi-random | 256 | 18,000 | Training mode |
| RAW, Sequential | 256 | 40,000 | Memory limited |
| RAW, Quasi-random | 256 | 30,000 | Huge files |

### File Sizes (ImageNet)

| Format | Quality | 256px | 512px |
|--------|---------|-------|-------|
| JPEG | 50 | 9 GB | 26 GB |
| JPEG | 90 | 22 GB | 65 GB |
| RAW | - | 170 GB | 616 GB |

## Memory Mapping Details

```c
// What FFCV does internally (simplified)
int fd = open("dataset.beton", O_RDONLY);

// Memory map entire file
void* data = mmap(NULL, file_size, PROT_READ, 
                  MAP_SHARED | MAP_POPULATE, fd, 0);

// Random access hint (for shuffled access)
madvise(data, file_size, MADV_RANDOM);

// Access sample directly via pointer
Sample* sample = (Sample*)(data + sample_offsets[idx]);

// OS handles:
// - Page faults (load from disk on first access)
// - Page cache (keep hot pages in RAM)
// - Eviction (remove cold pages when RAM full)
```

## Best Practices

```
1. PRE-RESIZE IMAGES
   • Store at training resolution + small margin
   • 256px for 224px training
   • Saves decode + resize time

2. CHOOSE STORAGE FORMAT
   • JPEG 90 for most cases
   • RAW only if you have huge NVMe bandwidth
   
3. USE QUASI_RANDOM ORDER
   • Best training + I/O tradeoff
   • Groups nearby samples

4. MATCH WORKERS TO CORES
   • num_workers = physical CPU cores
   • Too many = overhead

5. PIN TO NUMA NODE
   • Especially for multi-GPU
   • numactl --cpunodebind=0 python train.py
```

## Comparison with Other Formats

| Feature | .beton | TFRecord | WebDataset | HDF5 |
|---------|--------|----------|------------|------|
| Random access | ✅ O(1) | ❌ Sequential | ❌ Sequential | ✅ O(1) |
| Memory map | ✅ | ❌ | ❌ | ⚠️ Partial |
| Zero-copy | ✅ | ❌ | ❌ | ❌ |
| Cloud native | ❌ | ⚠️ | ✅ | ❌ |
| Compression | ✅ JPEG | ✅ Any | ✅ Any | ✅ Various |
| Multi-node | ⚠️ | ✅ | ✅ | ⚠️ |
