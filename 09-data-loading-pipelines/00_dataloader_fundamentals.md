# DataLoader Fundamentals

## Why DataLoader is Critical for ML Training

### The Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│  │   STORAGE    │────▶│  DATALOADER  │────▶│    MODEL     │               │
│  │              │     │              │     │              │               │
│  │ • Disk/SSD   │     │ • Load       │     │ • Forward    │               │
│  │ • S3/GCS     │     │ • Decode     │     │ • Backward   │               │
│  │ • Network    │     │ • Transform  │     │ • Optimizer  │               │
│  └──────────────┘     └──────────────┘     └──────────────┘               │
│         │                   │                   │                          │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│     10-100 ms          50-500 ms           10-100 ms                       │
│   (I/O latency)      (CPU bottleneck)    (GPU compute)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Problem: GPU Starvation

**Without an optimized DataLoader:**

```
Timeline for 1 training iteration (naive approach):

GPU:  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
      [Forward+Backward ~50ms]                     [Waiting for data ~450ms]

CPU:  ░░░░░░░░░█████████████████████████████████████████████████████████████
      [Idle]              [Loading+Decoding+Transforming ~450ms]

Result: GPU utilization = 50ms / 500ms = 10%
        Training speed = 2 iterations/second
        Wasted compute = 90%
```

**With an optimized DataLoader:**

```
Timeline for 1 training iteration (optimized):

GPU:  ██████████████████████████████████████████████████████████████████████
      [Forward+Backward ~50ms] [Next batch ready immediately]

CPU:  ████████████████████████████████████████████████████████████████████
      [Prefetching next batch in background]

Result: GPU utilization = 95%
        Training speed = 18 iterations/second
        9x faster training!
```

---

## 1. What is a DataLoader?

### Definition

A DataLoader is a component that:

1. **Reads** data from storage (disk, cloud, network)
2. **Decodes** data formats (JPEG, PNG, video, audio)
3. **Transforms** data (resize, crop, normalize, augment)
4. **Batches** samples for efficient GPU processing
5. **Shuffles** data for stochastic gradient descent
6. **Prefetches** batches to keep GPU fed

### Why Not Just Load Data in the Training Loop?

```python
# ❌ BAD: Loading data in training loop
for epoch in range(epochs):
    for i in range(len(dataset)):
        # This blocks the GPU!
        image = Image.open(dataset[i])      # ~10ms I/O
        image = transform(image)            # ~50ms CPU
        image = image.to(device)            # ~5ms transfer
        
        loss = model(image)                 # ~50ms GPU
        loss.backward()
        optimizer.step()
        
# Total per iteration: 115ms (GPU only busy for 50ms = 43% utilization)
```

```python
# ✅ GOOD: Using DataLoader with prefetching
loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)

for images, labels in loader:
    # Data already loaded, decoded, and ready on GPU!
    images = images.to(device, non_blocking=True)
    loss = model(images)                     # ~50ms GPU
    loss.backward()
    optimizer.step()

# Total per iteration: ~55ms (GPU busy for 50ms = 91% utilization)
# 2x faster!
```

---

## 2. DataLoader Internals

### How PyTorch DataLoader Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYTORCH DATALOADER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Main Process (Training Loop)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  for batch in loader:                                              │   │
│  │      loss = model(batch.cuda())  ← GPU gets batch immediately       │   │
│  │      loss.backward()                                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    │ Queue (prefetched batches)            │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BATCH QUEUE                                  │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │   │
│  │  │ B0  │ │ B1  │ │ B2  │ │ B3  │ │ B4  │ │ B5  │ │ B6  │ │ B7  │   │   │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │   │
│  └─────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┘   │
│        │        │        │        │        │        │        │              │
│        ▼        ▼        ▼        ▼        ▼        ▼        ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      WORKER PROCESSES                               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │Worker 0 │ │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker 4 │  ...  │   │
│  │  │         │ │         │ │         │ │         │ │         │        │   │
│  │  │• Load   │ │• Load   │ │• Load   │ │• Load   │ │• Load   │        │   │
│  │  │• Decode │ │• Decode │ │• Decode │ │• Decode │ │• Decode │        │   │
│  │  │• Transform││• Transform││• Transform││• Transform││• Transform│   │   │
│  │  │• Batch  │ │• Batch  │ │• Batch  │ │• Batch  │ │• Batch  │        │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │   │
│  └───────┼────────────┼────────────┼────────────┼────────────┼───────────┘   │
│          │            │            │            │            │               │
│          ▼            ▼            ▼            ▼            ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          DATASET                                     │   │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐          │   │
│  │  │img0│ │img1│ │img2│ │img3│ │img4│ │img5│ │img6│ │img7│  ...      │   │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Typical Time |
|-----------|---------|--------------|
| **Dataset** | Provides access to individual samples | O(1) |
| **Sampler** | Determines which samples to load (shuffling) | O(1) |
| **Worker Processes** | Parallel loading/decoding | 50-200ms per batch |
| **Batch Collator** | Combines samples into batches | 1-5ms |
| **Prefetch Queue** | Buffers batches for GPU | Configurable |

---

## 3. Common Bottlenecks

### Bottleneck 1: I/O (Disk/Network)

**Symptoms:**
- High disk read time
- Network latency for cloud storage
- Many small file reads

**Diagnosis:**
```python
import time
import os

def measure_io_time(file_paths):
    start = time.time()
    for path in file_paths[:1000]:
        with open(path, 'rb') as f:
            data = f.read()
    return time.time() - start

# If > 5 seconds for 1000 files, I/O is bottleneck
```

**Solutions:**
- Store data on SSD instead of HDD
- Use sequential formats (TAR, LMDB, .beton)
- Cache data locally for cloud storage
- Increase `num_workers` to parallelize I/O

### Bottleneck 2: CPU Decoding

**Symptoms:**
- High CPU usage during training
- GPU waiting for data
- Slow image/video decode

**Diagnosis:**
```python
import cProfile
import pstats

def profile_decode():
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        img = Image.open("test.jpg")
        img = np.array(img)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)
    # Look for PIL.Image.open or decode functions
```

**Solutions:**
- Use faster decoders (libjpeg-turbo, nvJPEG)
- Pre-decode to numpy/torch format
- Use GPU decode (NVIDIA DALI)
- Reduce image resolution

### Bottleneck 3: CPU Transforms

**Symptoms:**
- Data augmentation taking too long
- Complex transforms on CPU

**Diagnosis:**
```python
def profile_transforms():
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    start = time.time()
    for _ in range(1000):
        img = torch.rand(3, 256, 256)
        img = transforms(img)
    print(f"1000 transforms: {time.time() - start:.2f}s")
```

**Solutions:**
- Move transforms to GPU (DALI, Kornia)
- Pre-compute heavy transforms
- Reduce augmentation complexity
- Use faster libraries (albumentations)

### Bottleneck 4: CPU-GPU Transfer

**Symptoms:**
- High `cudaMemcpy` time in profiler
- `pin_memory` not enabled

**Diagnosis:**
```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

data = torch.randn(1000, 3, 224, 224)

start.record()
data_gpu = data.cuda()  # Without pin_memory
end.record()
torch.cuda.synchronize()
print(f"Transfer without pin_memory: {start.elapsed_time(end):.2f}ms")

data = data.pin_memory()
start.record()
data_gpu = data.cuda()  # With pin_memory
end.record()
torch.cuda.synchronize()
print(f"Transfer with pin_memory: {start.elapsed_time(end):.2f}ms")
```

**Solutions:**
- Enable `pin_memory=True`
- Use `non_blocking=True` for async transfer
- Minimize transfer frequency (batch larger)
- Use zero-copy when possible

### Bottleneck 5: Worker Overhead

**Symptoms:**
- Too many workers causing thrashing
- Workers starting/stopping frequently
- High memory usage

**Diagnosis:**
```python
import psutil

def check_worker_memory():
    # Run in worker process
    print(f"Memory: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
```

**Solutions:**
- Set `num_workers` based on CPU cores (usually 4-8)
- Use `persistent_workers=True`
- Adjust `prefetch_factor`
- Monitor memory usage

---

## 4. DataLoader Configuration Deep Dive

### num_workers

**What it does:**
- Number of subprocesses for data loading
- Each worker runs independently
- More workers = more parallelism

**Trade-offs:**
```
num_workers = 0:  Main process loads (slow, debugging easy)
num_workers = 4:  Good balance for most systems
num_workers = 8:  High performance, more memory
num_workers = 16: Diminishing returns, possible thrashing
```

**Rule of thumb:**
```python
import multiprocessing

# Start with number of CPU cores
num_workers = multiprocessing.cpu_count()

# Reduce if memory is limited
if psutil.virtual_memory().available < 16e9:  # < 16GB
    num_workers = min(num_workers, 4)

# Increase for I/O bound workloads
if is_io_bound:
    num_workers = min(num_workers * 2, 16)
```

### pin_memory

**What it does:**
- Allocates data in page-locked (pinned) memory
- Enables faster CPU→GPU transfer
- Uses DMA (Direct Memory Access)

**When to use:**
- Always use with CUDA training
- Not needed for CPU-only training

**Impact:**
```
Without pin_memory:  ~50ms for 256 images (224x224)
With pin_memory:     ~10ms for 256 images (224x224)
5x faster transfer!
```

### prefetch_factor

**What it does:**
- Number of batches to prefetch per worker
- Default: 2 (each worker prefetches 2 batches)
- Total prefetched = `num_workers * prefetch_factor`

**Tuning:**
```python
# Low latency, low memory
prefetch_factor = 2  # Default

# High GPU utilization, more memory
prefetch_factor = 4

# Very fast GPU, slow storage
prefetch_factor = 8
```

### persistent_workers

**What it does:**
- Keeps worker processes alive between epochs
- Avoids worker startup overhead

**When to use:**
- Always use for long training runs
- Especially important with many workers

**Impact:**
```
Without persistent_workers:
  Epoch 1: 100s (includes worker startup)
  Epoch 2: 100s (includes worker startup)
  ...

With persistent_workers:
  Epoch 1: 100s (includes worker startup)
  Epoch 2: 95s  (workers already running)
  ...
```

### drop_last

**What it does:**
- Drops the last incomplete batch
- Ensures consistent batch size

**When to use:**
- BatchNorm layers (need consistent batch size)
- Distributed training (sync across GPUs)
- Not needed for inference

---

## 5. Profiling DataLoader Performance

### Method 1: Simple Timing

```python
import time

def profile_dataloader(loader, num_batches=100):
    times = []
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        start = time.time()
        _ = batch  # Force evaluation
        times.append(time.time() - start)
    
    print(f"Average batch time: {np.mean(times)*1000:.2f}ms")
    print(f"Min batch time: {np.min(times)*1000:.2f}ms")
    print(f"Max batch time: {np.max(times)*1000:.2f}ms")
    print(f"Std dev: {np.std(times)*1000:.2f}ms")
    
    # Check for spikes (indicates I/O issues)
    spikes = [t for t in times if t > np.mean(times) + 2*np.std(times)]
    print(f"Spikes: {len(spikes)}/{len(times)}")
```

### Method 2: GPU Utilization

```python
import torch

def check_gpu_starvation():
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    compute_time = 0
    total_time = 0
    
    for batch in loader:
        start.record()
        output = model(batch.cuda())
        loss = criterion(output, labels.cuda())
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        
        compute_time += start.elapsed_time(end)
        total_time += start.elapsed_time(end)
    
    utilization = compute_time / total_time * 100
    print(f"GPU Utilization: {utilization:.1f}%")
    
    if utilization < 70:
        print("⚠️  GPU is starving! Optimize DataLoader.")
    elif utilization < 85:
        print("⚠️  GPU could be better utilized.")
    else:
        print("✅ GPU is well utilized!")
```

### Method 3: PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in loader:
        output = model(batch.cuda())
        loss = criterion(output, labels.cuda())
        loss.backward()
        if prof.step_num() > 100:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Look for:
# - DataLoaderIter.__next__ (should be fast)
# - cudaMemcpy (should be minimal with pin_memory)
# - High CPU time in transforms
```

### Method 4: System Monitoring

```python
import psutil
import time

def monitor_system_during_training(loader, num_batches=50):
    cpu_usage = []
    mem_usage = []
    disk_io = []
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        cpu_usage.append(psutil.cpu_percent())
        mem_usage.append(psutil.virtual_memory().percent)
        disk_io.append(psutil.disk_io_counters().read_bytes)
        
        # Simulate training
        time.sleep(0.05)
    
    print(f"Average CPU: {np.mean(cpu_usage):.1f}%")
    print(f"Average Memory: {np.mean(mem_usage):.1f}%")
    print(f"Disk I/O: {(disk_io[-1] - disk_io[0]) / 1e6:.1f} MB")
    
    # Interpretation
    if np.mean(cpu_usage) > 90:
        print("⚠️  CPU bound - consider more workers or GPU transforms")
    if np.mean(mem_usage) > 80:
        print("⚠️  Memory bound - reduce num_workers or batch_size")
```

---

## 6. Optimization Strategies

### Strategy 1: Increase Parallelism

```python
# Baseline
loader = DataLoader(dataset, batch_size=64, num_workers=0)

# Better
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,              # Parallel loading
    pin_memory=True,            # Faster transfer
    prefetch_factor=2,          # Prefetch batches
    persistent_workers=True,    # Keep workers alive
)
```

### Strategy 2: Optimize Transforms

```python
# Slow: Python transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# Faster: Albumentations (C++ backend)
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    A.Normalize(mean, std),
    ToTensorV2(),
])
```

### Strategy 3: Pre-process Data

```python
# Write pre-processed dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = ImageFolder("raw_images", transform=preprocess)
torch.save(dataset, "preprocessed_dataset.pt")

# Load pre-processed (instant!)
dataset = torch.load("preprocessed_dataset.pt")
```

### Strategy 4: Use Efficient Formats

```python
# Instead of individual JPEG files:
# images/
#   img0001.jpg
#   img0002.jpg
#   ...

# Use WebDataset (TAR format):
# shards/
#   shard-000000.tar
#   shard-000001.tar
#   ...

import webdataset as wds

dataset = wds.WebDataset("shards/shard-{000000..000999}.tar") \
    .decode("pil") \
    .to_tuple("jpg", "cls") \
    .batched(64)

loader = DataLoader(dataset, batch_size=None, num_workers=4)
```

### Strategy 5: GPU Acceleration

```python
# Use NVIDIA DALI for GPU decode and transforms
from nvidia.dali import pipeline_def, fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def gpu_pipeline(data_dir, batch_size):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
    )
    images = fn.decoders.image(jpegs, device="mixed")  # GPU decode!
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.normalize(images, mean=mean, std=std)
    return images, labels

pipe = gpu_pipeline(batch_size=256, num_threads=4, device_id=0)
loader = DALIGenericIterator(pipe, ["images", "labels"])
```

---

## 7. Common Mistakes

### Mistake 1: Not Using pin_memory

```python
# ❌ Wrong
loader = DataLoader(dataset, batch_size=64, num_workers=4)

# ✅ Right
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,  # Essential for CUDA!
)
```

### Mistake 2: Too Many Workers

```python
# ❌ Wrong - too many workers cause thrashing
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=32,  # Too many!
)

# ✅ Right - match to CPU cores
import multiprocessing
num_workers = multiprocessing.cpu_count()
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=num_workers,
)
```

### Mistake 3: Not Using persistent_workers

```python
# ❌ Wrong - workers restart every epoch
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=False,  # Default
)

# ✅ Right - keep workers alive
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    persistent_workers=True,  # Faster!
)
```

### Mistake 4: Heavy Transforms in __getitem__

```python
# ❌ Wrong - heavy computation in dataset
class SlowDataset(Dataset):
    def __getitem__(self, idx):
        img = self.load_image(idx)
        # Heavy computation here!
        for _ in range(100):
            img = self.complex_transform(img)
        return img

# ✅ Right - pre-compute or use efficient transforms
class FastDataset(Dataset):
    def __init__(self):
        self.precomputed = self.precompute_all()
    
    def __getitem__(self, idx):
        return self.precomputed[idx]
```

### Mistake 5: Not Monitoring Performance

```python
# ❌ Wrong - no monitoring
for epoch in range(100):
    for batch in loader:
        loss = model(batch)

# ✅ Right - monitor and optimize
from torch.utils.data import DataLoader

def train_with_monitoring():
    for epoch in range(100):
        epoch_start = time.time()
        for i, batch in enumerate(loader):
            iter_start = time.time()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                iter_time = time.time() - iter_start
                print(f"Epoch {epoch}, Iter {i}: {iter_time*1000:.1f}ms/iter")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}: {epoch_time:.1f}s")
```

---

## 8. Advanced Topics

### Custom Collate Function

```python
from torch.nn.utils.rnn import pad_sequence

def variable_length_collate(batch):
    """Handle variable-length sequences (e.g., text, audio)"""
    sequences = [item['sequence'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [len(seq) for seq in sequences]
    
    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True)
    
    return {
        'sequences': padded,
        'labels': torch.tensor(labels),
        'lengths': torch.tensor(lengths),
    }

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=variable_length_collate,
)
```

### Distributed Training

```python
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup distributed training
torch.distributed.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Create distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=torch.distributed.get_world_size(),
    rank=torch.distributed.get_rank(),
    shuffle=True,
)

# Create loader with sampler
loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,  # Use distributed sampler
    num_workers=8,
    pin_memory=True,
)

# Wrap model
model = DDP(model.cuda(), device_ids=[local_rank])
```

### WebDataset Integration

```python
import webdataset as wds

# Create WebDataset pipeline
dataset = wds.WebDataset("s3://bucket/data/shard-{000000..000999}.tar") \
    .shuffle(1000) \
    .decode("pil") \
    .to_tuple("jpg", "cls") \
    .map_tuple(transform_image, lambda x: x) \
    .batched(64)

# Use with PyTorch DataLoader
loader = DataLoader(
    dataset,
    batch_size=None,  # Batching done in pipeline
    num_workers=4,
    pin_memory=True,
)
```

---

## 9. Performance Checklist

Use this checklist to optimize your DataLoader:

### Configuration
- [ ] `num_workers` set to appropriate value (4-8)
- [ ] `pin_memory=True` for CUDA training
- [ ] `persistent_workers=True` for long runs
- [ ] `prefetch_factor` tuned (2-4)
- [ ] `drop_last=True` for BatchNorm

### Data Format
- [ ] Using efficient storage (SSD, sequential format)
- [ ] Pre-computed heavy transforms
- [ ] Cached cloud data locally
- [ ] Appropriate image resolution

### Transforms
- [ ] Using fast libraries (albumentations, DALI)
- [ ] GPU-accelerated when possible
- [ ] Minimal augmentation overhead
- [ ] Vectorized operations

### Monitoring
- [ ] GPU utilization > 80%
- [ ] CPU not bottlenecked
- [ ] Memory usage < 80%
- [ ] No I/O spikes

### Advanced
- [ ] Distributed sampler for multi-GPU
- [ ] Custom collate for variable lengths
- [ ] WebDataset for cloud storage
- [ ] DALI for GPU decode

---

## 10. Summary

### Key Takeaways

1. **DataLoader is critical** - Poor data loading can waste 90% of GPU compute
2. **Identify bottlenecks** - Profile to find I/O, CPU, or transfer issues
3. **Configure properly** - Use `num_workers`, `pin_memory`, `persistent_workers`
4. **Optimize transforms** - Move to GPU, pre-compute, use fast libraries
5. **Monitor continuously** - Track GPU utilization and iterate times

### Quick Reference

| Setting | Default | Recommended | When to Change |
|---------|---------|-------------|----------------|
| `num_workers` | 0 | 4-8 | More for I/O bound |
| `pin_memory` | False | True (CUDA) | Always for GPU |
| `prefetch_factor` | 2 | 2-4 | More for fast GPU |
| `persistent_workers` | False | True | Always for training |
| `batch_size` | 1 | 32-256 | Based on GPU memory |

### Performance Targets

- **GPU utilization**: > 80%
- **Batch time**: < 100ms (for typical models)
- **CPU usage**: 60-80% (not 100%)
- **Memory usage**: < 80% of available

### When to Use Advanced Solutions

| Situation | Solution |
|-----------|----------|
| Local SSD, images only | FFCV |
| Cloud storage | WebDataset |
| Video training | NVIDIA DALI |
| Maximum speed | DALI + FFCV |
| Multimodal | WebDataset or DALI |

---

## Further Reading

- **PyTorch DataLoader docs**: https://pytorch.org/docs/stable/data.html
- **FFCV**: https://docs.ffcv.io/
- **NVIDIA DALI**: https://docs.nvidia.com/deeplearning/dali/
- **WebDataset**: https://github.com/webdataset/webdataset
- **Albumentations**: https://albumentations.ai/
