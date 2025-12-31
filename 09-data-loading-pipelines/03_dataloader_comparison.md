# Complete DataLoader Comparison

Comprehensive comparison of ALL major data loading solutions for ML training.

## The Data Loading Landscape (2024)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    DATA LOADING SOLUTIONS TAXONOMY                         ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌────────────────────────────────────────────────────────────────────┐   ║
║  │                     FRAMEWORK NATIVE                                │   ║
║  │  PyTorch DataLoader │ TensorFlow tf.data │ JAX data loading        │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                 │                                          ║
║              ┌──────────────────┼──────────────────┐                      ║
║              ▼                  ▼                  ▼                      ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          ║
║  │ HIGH PERFORMANCE │  │    STREAMING    │  │  GPU ACCELERATED │          ║
║  │                  │  │                 │  │                  │          ║
║  │ • FFCV          │  │ • WebDataset    │  │ • NVIDIA DALI    │          ║
║  │ • Mosaic        │  │ • HuggingFace   │  │ • RAPIDS cuDF    │          ║
║  │   StreamingData │  │   datasets      │  │ • torchdata      │          ║
║  │ • Petastorm     │  │ • AIStore       │  │                  │          ║
║  └─────────────────┘  └─────────────────┘  └─────────────────┘          ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Feature Comparison Matrix

| Feature | PyTorch DL | FFCV | WebDataset | DALI | Mosaic | HF datasets |
|---------|------------|------|------------|------|--------|-------------|
| **GPU Decode** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Zero-Copy** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Cloud Native** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Multimodal** | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| **Video** | ⚠️ | ❌ | ✅ | ✅ | ✅ | ⚠️ |
| **Audio** | ⚠️ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Distributed** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Custom Transforms** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |

## Performance Benchmarks

### ImageNet Training (images/sec, single A100)

| DataLoader | Resolution | Batch 64 | Batch 256 | GPU Util |
|------------|------------|----------|-----------|----------|
| PyTorch (8 workers) | 224×224 | 1,200 | 2,400 | 45% |
| PyTorch (optimized) | 224×224 | 2,000 | 4,000 | 65% |
| FFCV | 224×224 | 6,000 | 12,000 | 92% |
| NVIDIA DALI | 224×224 | 7,500 | 15,000 | 95% |
| WebDataset | 224×224 | 3,500 | 7,000 | 75% |

### Video Loading (clips/sec)

| DataLoader | Resolution | Frames | Throughput |
|------------|------------|--------|------------|
| OpenCV | 1080p | 16 | 5 |
| decord | 1080p | 16 | 25 |
| DALI (GPU) | 1080p | 16 | 100 |
| PyAV | 1080p | 16 | 15 |

### Audio Loading (hours/sec processed)

| DataLoader | Sample Rate | Throughput |
|------------|-------------|------------|
| librosa | 16kHz | 50x realtime |
| torchaudio | 16kHz | 100x realtime |
| DALI | 16kHz | 200x realtime |
| soundfile | 16kHz | 150x realtime |

---

## 1. PyTorch DataLoader (Baseline)

### Architecture
```python
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Each worker runs this
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Optimized settings
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,           # CPU cores for loading
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=2,       # Batches to prefetch per worker
    persistent_workers=True, # Keep workers alive
    drop_last=True,          # Consistent batch size
)
```

### Pros & Cons
```
✅ Pros:
- Simple, well-documented
- Maximum flexibility
- Works with any data format
- Easy debugging

❌ Cons:
- CPU bottleneck on preprocessing
- No GPU decode
- High memory usage (worker copies)
- Individual file I/O is slow
```

---

## 2. FFCV (Fast Forward Computer Vision)

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        FFCV ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WRITE PHASE (once):                                            │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐     │
│  │ Dataset │───▶│ FFCV Writer │───▶│ .beton file         │     │
│  │ (any)   │    │ (resize,    │    │ • Page-aligned      │     │
│  └─────────┘    │  encode)    │    │ • Pre-processed     │     │
│                 └─────────────┘    │ • Index + samples   │     │
│                                    └─────────────────────┘     │
│                                                                  │
│  READ PHASE (training):                                         │
│  ┌─────────────────────┐    ┌───────────┐    ┌───────────┐    │
│  │ Memory-mapped file  │───▶│ Quasi-    │───▶│ Transform │    │
│  │ (mmap, zero-copy)   │    │ random    │    │ Pipeline  │    │
│  └─────────────────────┘    │ sampling  │    └───────────┘    │
│                              └───────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Usage
```python
# Writing dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

writer = DatasetWriter(
    'imagenet.beton',
    {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90,
        ),
        'label': IntField()
    },
    num_workers=16
)
writer.from_indexed_dataset(pytorch_dataset)

# Reading dataset
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder

loader = Loader(
    'imagenet.beton',
    batch_size=256,
    num_workers=8,
    order=OrderOption.QUASI_RANDOM,  # Good locality + randomness
    pipelines={
        'image': [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
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
    # images already on GPU!
    loss = model(images)
```

### Pros & Cons
```
✅ Pros:
- 3-5x faster than PyTorch DataLoader
- Zero-copy with memory mapping
- Pre-resized images (no resize at train time)
- Excellent for local SSD training

❌ Cons:
- Requires preprocessing step
- Large .beton files
- Limited to images (no native video/audio)
- Not cloud-native
```

---

## 3. NVIDIA DALI

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                       NVIDIA DALI PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │     Readers     │  File, COCO, TFRecord, Numpy, Video       │
│  │     (CPU)       │  Caffe, Caffe2, LMDB, RecordIO            │
│  └────────┬────────┘                                            │
│           │                                                      │
│  ┌────────▼────────┐                                            │
│  │    Decoders     │  JPEG (nvJPEG), PNG, Video (NVDEC)        │
│  │   (GPU/CPU)     │  Audio (libsndfile)                        │
│  └────────┬────────┘                                            │
│           │                                                      │
│  ┌────────▼────────┐                                            │
│  │   Transforms    │  Resize, Crop, ColorAugment, Normalize    │
│  │   (GPU/CPU)     │  All on GPU = no CPU↔GPU transfer!        │
│  └────────┬────────┘                                            │
│           │                                                      │
│  ┌────────▼────────┐                                            │
│  │   Output        │  PyTorch tensors on GPU                    │
│  │   (GPU memory)  │  Ready for model.forward()                 │
│  └─────────────────┘                                            │
│                                                                  │
│  Key: Entire pipeline can run on GPU!                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Image Pipeline
```python
from nvidia.dali import pipeline_def, fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def image_pipeline(image_dir, batch_size):
    # Read files
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        name="Reader"
    )
    
    # GPU decode
    images = fn.decoders.image(
        jpegs,
        device="mixed",  # CPU read, GPU decode
        output_type=types.RGB
    )
    
    # GPU augmentations
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop(images, crop_h=224, crop_w=224)
    images = fn.normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )
    
    return images, labels

# Create pipeline
pipe = image_pipeline(
    image_dir="/data/imagenet/train",
    batch_size=64,
    num_threads=4,
    device_id=0
)
pipe.build()

# PyTorch iterator
train_loader = DALIGenericIterator(
    pipe,
    ["images", "labels"],
    reader_name="Reader"
)

# Training
for batch in train_loader:
    images = batch[0]["images"]  # Already on GPU!
    labels = batch[0]["labels"]
```

### Video Pipeline
```python
@pipeline_def
def video_pipeline(video_files):
    # GPU video decode (NVDEC hardware)
    video = fn.readers.video(
        device="gpu",
        filenames=video_files,
        sequence_length=16,      # Frames per clip
        stride=2,                # Frame skip
        step=-1,                 # Random start
        file_list_include_preceding_frame=False,
    )
    
    # GPU processing
    video = fn.resize(video, resize_x=224, resize_y=224)
    video = fn.normalize(video, mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    
    return video
```

### Audio Pipeline
```python
@pipeline_def  
def audio_pipeline(audio_files):
    # Read and decode audio
    audio, sr = fn.readers.file(file_root=audio_files)
    audio, sr = fn.decoders.audio(audio, dtype=types.FLOAT)
    
    # Mel spectrogram on GPU
    spec = fn.spectrogram(
        audio,
        nfft=512,
        window_length=400,
        window_step=160
    )
    mel = fn.mel_filter_bank(spec, nfilter=80, sample_rate=16000)
    mel = fn.to_decibels(mel)
    
    return mel
```

### Pros & Cons
```
✅ Pros:
- GPU decode (10x faster than CPU)
- Complete GPU pipeline
- Native video/audio support
- Best throughput possible

❌ Cons:
- NVIDIA GPUs only
- Learning curve
- Limited custom operations
- Some operators less flexible
```

---

## 4. WebDataset

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     WEBDATASET ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Storage: Sequential TAR archives                               │
│                                                                  │
│  shard-000000.tar                                               │
│  ├── sample0.jpg                                                │
│  ├── sample0.json                                               │
│  ├── sample0.cls                                                │
│  ├── sample1.jpg                                                │
│  ├── sample1.json                                               │
│  └── ...                                                         │
│                                                                  │
│  shard-000001.tar                                               │
│  └── ...                                                         │
│                                                                  │
│  Benefits:                                                       │
│  • Sequential read = maximum throughput                         │
│  • Works with S3, GCS, HTTP                                     │
│  • Shard-level shuffling for distributed                        │
│  • Streaming (don't need whole dataset)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage
```python
import webdataset as wds
import torch

# Create dataset
url = "s3://bucket/data/shard-{000000..001000}.tar"

dataset = wds.WebDataset(url)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "json")
    .map_tuple(transform_image, transform_label)
    .batched(64)

# Or with DataLoader
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,  # Batching done in pipeline
    num_workers=4,
)

# Training
for images, labels in loader:
    loss = model(images.cuda())
```

### Multimodal WebDataset
```python
# Video + Audio + Text
dataset = wds.WebDataset("s3://bucket/multimodal-{000..100}.tar")
    .decode()
    .to_tuple("mp4", "wav", "txt", "json")
    .map_tuple(
        decode_video,
        decode_audio,
        tokenize_text,
        parse_metadata
    )
```

### Pros & Cons
```
✅ Pros:
- Cloud-native (S3, GCS, HTTP)
- Sequential I/O = fast
- Streaming (no local storage needed)
- Simple format (just tar files)
- Great for multimodal

❌ Cons:
- No random access
- Must pre-shard data
- Shuffling less random than FFCV
- Creating shards takes time
```

---

## 5. Mosaic StreamingDataset

### Architecture
```python
from streaming import StreamingDataset, MDSWriter

# Write dataset
with MDSWriter(out="s3://bucket/data", columns={"image": "jpeg", "label": "int"}) as out:
    for img, label in dataset:
        out.write({"image": img, "label": label})

# Read dataset
dataset = StreamingDataset(
    remote="s3://bucket/data",
    local="/tmp/cache",
    shuffle=True,
    batch_size=64,
)

loader = DataLoader(dataset, batch_size=None)
```

### Pros & Cons
```
✅ Pros:
- Deterministic shuffling
- Resumable training
- Elastic scaling
- Cloud-native

❌ Cons:
- Less mature than alternatives
- MosaicML ecosystem focused
```

---

## 6. HuggingFace Datasets

### Usage
```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("imagenet-1k", split="train")

# Set format for PyTorch
dataset.set_format(type="torch", columns=["image", "label"])

# DataLoader
loader = DataLoader(dataset, batch_size=64, num_workers=4)
```

### Pros & Cons
```
✅ Pros:
- Huge dataset hub
- Easy to use
- Good for NLP
- Arrow format (fast)

❌ Cons:
- Not optimized for training speed
- Limited preprocessing control
```

---

## Choosing the Right DataLoader

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION FLOWCHART                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Do you need maximum speed?                                     │
│  ├── YES: Is data on local SSD?                                │
│  │   ├── YES: Use FFCV                                         │
│  │   └── NO: Use NVIDIA DALI or WebDataset                     │
│  └── NO: Use PyTorch DataLoader                                 │
│                                                                  │
│  Do you have NVIDIA GPU + need video?                           │
│  ├── YES: Use NVIDIA DALI (GPU decode)                         │
│  └── NO: Use decord or WebDataset                              │
│                                                                  │
│  Is data in cloud storage?                                      │
│  ├── YES: Use WebDataset or Mosaic                             │
│  └── NO: Use FFCV                                              │
│                                                                  │
│  Is this multimodal (video + audio + text)?                     │
│  ├── YES: Use WebDataset or DALI                               │
│  └── NO: Choose based on above                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Recommendation

| Use Case | Recommendation |
|----------|----------------|
| ImageNet on local SSD | FFCV |
| Large-scale cloud training | WebDataset + S3 |
| Video understanding | NVIDIA DALI |
| Audio/Speech | DALI or torchaudio |
| Multimodal | WebDataset |
| Quick prototyping | PyTorch DataLoader |
| NLP/Text | HuggingFace datasets |
