# FFCV Internals: How Fast Data Loading Works

## The Data Loading Bottleneck

Traditional PyTorch data loading pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Traditional Pipeline                          │
│                                                                  │
│  Disk ──► Read ──► Decode ──► Transform ──► Collate ──► GPU    │
│            │         │           │            │                  │
│           slow      CPU        CPU          copy                 │
│         (random   (JPEG)     (Python)      (sync)               │
│          seeks)                                                  │
└─────────────────────────────────────────────────────────────────┘

Problems:
1. Random I/O for shuffled access
2. JPEG decode is CPU-bound
3. Python transforms are slow
4. DataLoader workers have GIL issues
5. Memory copies between processes
```

## FFCV Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FFCV Pipeline                               │
│                                                                  │
│  .beton ──► mmap ──► C++ decode ──► GPU transforms ──► GPU     │
│    │         │          │              │                        │
│  optimized  zero-    compiled       CUDA                        │
│   format    copy     (numba/C++)   kernels                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Solutions:
1. Sequential I/O via quasi-random sampling
2. Optional pre-decoded storage
3. Compiled transforms (Numba)
4. GPU-native operations
5. Memory-mapped, zero-copy access
```

## .beton File Format

### File Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        .beton File                               │
├─────────────────────────────────────────────────────────────────┤
│ Header (metadata)                                                │
│   - Version                                                      │
│   - Number of samples                                            │
│   - Field definitions                                            │
├─────────────────────────────────────────────────────────────────┤
│ Field Metadata                                                   │
│   - Field names and types                                        │
│   - Encoding parameters                                          │
│   - Shape information                                            │
├─────────────────────────────────────────────────────────────────┤
│ Page Table / Index                                               │
│   - Offsets to each sample                                       │
│   - Page boundaries                                              │
├─────────────────────────────────────────────────────────────────┤
│ Sample Data (contiguous)                                         │
│   - Sample 0: [field0_data][field1_data]...                     │
│   - Sample 1: [field0_data][field1_data]...                     │
│   - ...                                                          │
│   - Sample N: [field0_data][field1_data]...                     │
└─────────────────────────────────────────────────────────────────┘
```

### Field Types

| Field Type | Storage | Decode | Use Case |
|------------|---------|--------|----------|
| `RGBImageField` | JPEG/Raw | CPU/GPU | Images |
| `IntField` | Raw int | None | Labels |
| `FloatField` | Raw float | None | Regression targets |
| `BytesField` | Raw bytes | None | Arbitrary data |
| `NDArrayField` | Raw numpy | None | Features |
| `JSONField` | Compressed | CPU | Metadata |

### Image Storage Options

```python
from ffcv.fields import RGBImageField

# Option 1: Store as JPEG (smaller, decode overhead)
RGBImageField(
    write_mode='jpeg',
    jpeg_quality=95
)

# Option 2: Store raw pixels (larger, no decode)
RGBImageField(
    write_mode='raw'
)

# Option 3: Smart mode (raw for small, jpeg for large)
RGBImageField(
    write_mode='smart',
    max_resolution=256  # Raw if <= 256, else JPEG
)

# Option 4: Proportion mode
RGBImageField(
    write_mode='proportion',
    proportion=0.5  # 50% raw, 50% JPEG
)
```

## Quasi-Random Sampling

### The Problem with True Random

```
True random access on disk:
Sample order: [45632, 12, 78901, 234, 99000, ...]

Disk head movement:
  ───────────────────────────────────────────►
  │        │                    │            │
  ▼        ▼                    ▼            ▼
[0]......[12]...[234].......[45632]......[78901]...

Result: ~10ms per seek, throughput collapses
```

### FFCV's Solution: Quasi-Random

```python
# Conceptual implementation
class QuasiRandomSampler:
    def __init__(self, num_samples, num_workers):
        # Divide into pages (OS typically reads 4KB-2MB chunks)
        self.page_size = calculate_optimal_page_size()
        self.pages = divide_into_pages(num_samples)
    
    def sample_order(self, epoch):
        # Shuffle pages, not individual samples
        shuffled_pages = shuffle(self.pages, seed=epoch)
        
        indices = []
        for page in shuffled_pages:
            # Within page: mostly sequential
            # Small local shuffle for randomness
            page_indices = shuffle_local(page)
            indices.extend(page_indices)
        
        return indices
```

### Randomness Analysis

```
Metric              True Random    Quasi-Random
─────────────────────────────────────────────────
Sequential runs     ~1%            ~80%
Mean seek distance  N/2            page_size/2
Disk throughput     ~50 MB/s       ~500 MB/s
Convergence         Baseline       Same (proven)
```

## Memory Mapping Implementation

```python
# Simplified FFCV reader logic
class FFCVReader:
    def __init__(self, fname):
        self.file = open(fname, 'rb')
        # Memory map entire file
        self.mmap = mmap.mmap(
            self.file.fileno(), 
            0,  # Entire file
            access=mmap.ACCESS_READ
        )
        
        # Parse header and index
        self.header = parse_header(self.mmap)
        self.index = parse_index(self.mmap, self.header)
    
    def read_sample(self, idx):
        offset, length = self.index[idx]
        # Zero-copy view into mmap
        data = self.mmap[offset:offset + length]
        return data
    
    def decode_sample(self, data, field):
        if field.needs_decode:
            # Decode JPEG, decompress, etc.
            return field.decode(data)
        else:
            # Return numpy view (zero-copy)
            return np.frombuffer(data, dtype=field.dtype)
```

## Compiled Transforms

FFCV uses Numba to compile transforms to native code:

```python
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage
from ffcv.transforms import NormalizeImage, RandomHorizontalFlip

# These get compiled to efficient machine code
image_pipeline = [
    # Decode (C++ or CUDA)
    SimpleRGBImageDecoder(),
    
    # Compiled transforms
    RandomHorizontalFlip(),  # Numba JIT
    NormalizeImage(           # Numba JIT
        mean=np.array([0.485, 0.456, 0.406]),
        std=np.array([0.229, 0.224, 0.225]),
        type=np.float16
    ),
    
    # GPU transfer
    ToTensor(),
    ToDevice('cuda:0', non_blocking=True),
    ToTorchImage()
]
```

### Custom Compiled Transform

```python
from ffcv.transforms import Operation
from dataclasses import replace
import numba as nb

class MyTransform(Operation):
    def generate_code(self):
        # This gets JIT compiled
        @nb.njit(parallel=True)
        def my_transform(images, dst):
            for i in nb.prange(images.shape[0]):
                for c in range(3):
                    for h in range(images.shape[2]):
                        for w in range(images.shape[3]):
                            # Your transform logic
                            dst[i, c, h, w] = images[i, c, h, w] * 2
            return dst
        
        return my_transform
    
    def declare_state_and_memory(self, previous_state):
        return (
            replace(previous_state, dtype=np.float32),
            AllocationQuery(previous_state.shape, previous_state.dtype)
        )
```

## Performance Numbers

### ImageNet Training (ResNet-50)

| Method | Images/sec | GPU Util | CPU Util |
|--------|-----------|----------|----------|
| PyTorch DataLoader (8 workers) | 800 | 60% | 100% |
| DALI | 1200 | 80% | 40% |
| FFCV (JPEG) | 1500 | 90% | 30% |
| FFCV (Raw) | 2000 | 95% | 10% |

### Breakdown of Speedup

```
Traditional PyTorch:
  Disk read:     30%
  JPEG decode:   40%
  Python GIL:    15%
  Transforms:    10%
  Collate/Copy:   5%
  ─────────────────
  Total:        100% of time

FFCV:
  mmap read:      5%  (10x faster)
  C++ decode:    15%  (2.5x faster)
  No GIL:         0%
  Numba xforms:   5%  (2x faster)
  Zero-copy:      2%
  ─────────────────
  Total:         27% of original time = 3.7x speedup
```

## Usage Example

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import *

# Write dataset
writer = DatasetWriter(
    'imagenet_train.beton',
    {
        'image': RGBImageField(write_mode='smart', max_resolution=256),
        'label': IntField()
    },
    num_workers=16
)
writer.from_indexed_dataset(train_dataset)

# Create loader
loader = Loader(
    'imagenet_train.beton',
    batch_size=512,
    num_workers=8,
    order=OrderOption.QUASI_RANDOM,  # Key for speed!
    pipelines={
        'image': [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice('cuda:0'),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ],
        'label': [
            IntDecoder(),
            ToTensor(),
            ToDevice('cuda:0')
        ]
    }
)

# Training loop
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(loader):
        # images already on GPU, normalized, fp16
        output = model(images)
        loss = criterion(output, labels)
        ...
```

## References

- FFCV Paper: "FFCV: Accelerating Training by Removing Data Bottlenecks"
- FFCV GitHub: https://github.com/libffcv/ffcv
- Numba documentation for custom transforms
