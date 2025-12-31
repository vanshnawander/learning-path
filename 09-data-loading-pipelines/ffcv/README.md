# FFCV (Fast Forward Computer Vision)

Eliminate data loading as a training bottleneck.

## Why FFCV?

Standard PyTorch DataLoader issues:
- Random disk access for each sample
- JPEG decoding on CPU
- Python GIL limitations
- Memory copies between processes

FFCV solves these with:
- Memory-mapped .beton files
- Quasi-random sampling
- C++ backend
- GPU-side augmentations

## Quick Start

### 1. Convert Dataset

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# Your PyTorch dataset
dataset = torchvision.datasets.ImageFolder(...)

writer = DatasetWriter(
    "train.beton",
    {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90
        ),
        'label': IntField()
    },
    num_workers=16
)
writer.from_indexed_dataset(dataset)
```

### 2. Create Loader

```python
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder

loader = Loader(
    "train.beton",
    batch_size=256,
    num_workers=8,
    order=OrderOption.QUASI_RANDOM,  # Key for performance!
    pipelines={
        'image': [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToDevice(torch.device('cuda')),
            ToTorchImage(),
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            ToDevice(torch.device('cuda')),
        ]
    }
)
```

### 3. Training Loop

```python
for images, labels in loader:
    # images and labels are already on GPU!
    outputs = model(images)
    loss = criterion(outputs, labels)
    ...
```

## Performance Tips

1. **Use QUASI_RANDOM ordering** - Enables sequential disk reads
2. **Store at lower resolution** - Faster reads, resize on GPU
3. **Use GPU transforms** - Offload from CPU
4. **Match num_workers to CPU cores**

## Reference Code
- `ffcv-main/examples/` - Usage examples
- `ffcv-main/ffcv/loader.py` - Loader implementation
- `ffcv-main/ffcv/writer.py` - Writer implementation
