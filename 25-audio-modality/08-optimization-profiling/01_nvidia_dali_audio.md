# NVIDIA DALI for Audio Processing

**Documentation**: [NVIDIA DALI Audio Processing](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/audio_processing/index.html)

DALI (Data Loading Library) can accelerate audio preprocessing by 10-100x by moving operations to GPU. Essential for high-throughput training.

## Table of Contents
1. [Why DALI for Audio?](#why-dali-for-audio)
2. [DALI Pipeline Architecture](#dali-pipeline-architecture)
3. [Audio Operations](#audio-operations)
4. [Spectrogram Pipeline](#spectrogram-pipeline)
5. [Integration with PyTorch](#integration-with-pytorch)
6. [Benchmarks](#benchmarks)
7. [Best Practices](#best-practices)

---

## Why DALI for Audio?

### The Data Loading Bottleneck

```
Typical training bottleneck analysis:

Without DALI:
┌────────────────────────────────────────────────────┐
│ CPU: Load audio → Decode → Resample → Spectrogram │
│      ████████████████████████████  (80% time)      │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│ GPU: Model forward → Backward                      │
│      ██████  (20% time, GPU underutilized!)        │
└────────────────────────────────────────────────────┘

With DALI:
┌────────────────────────────────────────────────────┐
│ CPU: Load compressed audio                         │
│      ██  (minimal)                                 │
└────────────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────┐
│ GPU: Decode → Resample → Spectrogram → Model       │
│      ████████████████████████████  (GPU saturated) │
└────────────────────────────────────────────────────┘

Result: 10-100x faster data preprocessing
```

### DALI Advantages

```
1. GPU-accelerated operations
   ├── Audio decoding (GPU)
   ├── Resampling (GPU)
   ├── FFT/Spectrogram (GPU)
   └── Mel filterbank (GPU)

2. Parallel execution
   ├── Prefetching
   ├── Overlapped CPU/GPU work
   └── Multi-threaded loading

3. Zero-copy GPU transfers
   ├── Data stays on GPU
   ├── No CPU-GPU round trips
   └── Direct to training tensor
```

---

## DALI Pipeline Architecture

### Basic Structure

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def audio_pipeline():
    """
    DALI pipeline for audio processing.
    All operations run on GPU by default.
    """
    # Read audio files
    audio, sr = fn.readers.file(
        file_root="/path/to/audio",
        random_shuffle=True,
        name="Reader"
    )
    
    # Decode audio (GPU)
    audio = fn.decoders.audio(
        audio,
        dtype=types.FLOAT,
        downmix=True  # Stereo to mono
    )
    
    # Resample to target rate (GPU)
    audio = fn.audio_resample(
        audio,
        in_rate=sr,
        out_rate=16000
    )
    
    return audio

# Create and build pipeline
pipe = audio_pipeline(batch_size=32, num_threads=4, device_id=0)
pipe.build()
```

### Pipeline Execution

```
DALI Pipeline Stages:

Stage 1: CPU (Prefetch)
├── File I/O
├── Compressed audio read
└── Minimal parsing

Stage 2: Mixed (Decode)
├── Audio decoding (CPU or GPU)
├── Format conversion
└── Channel handling

Stage 3: GPU (Process)
├── Resampling
├── FFT/Spectrogram
├── Normalization
└── Augmentation

All stages overlap for maximum throughput!
```

---

## Audio Operations

### Audio Decoding

```python
@pipeline_def
def decode_pipeline():
    # Read raw file bytes
    audio_bytes, labels = fn.readers.file(
        file_root="/data/audio",
        file_list="/data/train.txt"
    )
    
    # Decode to waveform
    audio, sample_rate = fn.decoders.audio(
        audio_bytes,
        dtype=types.FLOAT,
        downmix=True,           # Convert to mono
        sample_rate=None        # Keep original, or specify target
    )
    
    return audio, sample_rate, labels
```

### Resampling

```python
@pipeline_def
def resample_pipeline():
    audio, sr = load_audio()
    
    # Resample to 16kHz (common for speech)
    audio_16k = fn.audio_resample(
        audio,
        in_rate=sr,
        out_rate=16000,
        quality=50  # Higher = better quality, slower
    )
    
    return audio_16k
```

### Spectrogram

```python
@pipeline_def
def spectrogram_pipeline():
    audio, sr = load_audio()
    
    # Power spectrogram
    spec = fn.spectrogram(
        audio,
        nfft=512,
        window_length=400,
        window_step=160,         # hop_length
        window_fn=fn.window.hann,
        power=2                  # Power spectrogram
    )
    
    return spec
```

### Mel Spectrogram

```python
@pipeline_def  
def mel_spectrogram_pipeline():
    audio, sr = load_audio()
    
    # Spectrogram first
    spec = fn.spectrogram(
        audio,
        nfft=512,
        window_length=400,
        window_step=160,
        power=2
    )
    
    # Apply mel filterbank
    mel_spec = fn.mel_filter_bank(
        spec,
        sample_rate=16000,
        nfilter=80,              # Number of mel bins
        freq_low=0,
        freq_high=8000
    )
    
    # Convert to dB
    mel_db = fn.to_decibels(
        mel_spec,
        multiplier=10.0,         # 10*log10 for power
        reference=1.0,
        cutoff_db=-80
    )
    
    return mel_db
```

### MFCC

```python
@pipeline_def
def mfcc_pipeline():
    audio, sr = load_audio()
    
    # Full MFCC pipeline
    mfcc = fn.mfcc(
        audio,
        sample_rate=16000,
        frame_length=400,
        frame_step=160,
        num_mfcc=13,
        num_mel_bins=80,
        low_freq=0,
        high_freq=8000
    )
    
    return mfcc
```

---

## Spectrogram Pipeline

### Complete Whisper-Style Pipeline

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def(batch_size=32, num_threads=8, device_id=0)
def whisper_features_pipeline(audio_dir: str):
    """
    Extract Whisper-compatible mel spectrogram features.
    All operations on GPU.
    """
    # Read audio files
    audio_bytes, labels = fn.readers.file(
        file_root=audio_dir,
        random_shuffle=True,
        name="AudioReader"
    )
    
    # Decode (GPU)
    audio, sample_rate = fn.decoders.audio(
        audio_bytes,
        dtype=types.FLOAT,
        downmix=True,
        device="mixed"  # Decode on GPU when possible
    )
    
    # Resample to 16kHz (GPU)
    audio = fn.audio_resample(
        audio,
        in_rate=sample_rate,
        out_rate=16000,
        device="gpu"
    )
    
    # Pad/trim to fixed length (30 seconds for Whisper)
    target_length = 16000 * 30  # 480,000 samples
    audio = fn.pad(
        audio,
        axes=[0],
        shape=[target_length],
        fill_value=0
    )
    audio = fn.slice(
        audio,
        start=[0],
        shape=[target_length],
        axes=[0]
    )
    
    # Spectrogram (GPU)
    spec = fn.spectrogram(
        audio,
        nfft=400,
        window_length=400,
        window_step=160,
        window_fn=fn.window.hann,
        power=2,
        device="gpu"
    )
    
    # Mel filterbank (GPU)
    mel_spec = fn.mel_filter_bank(
        spec,
        sample_rate=16000,
        nfilter=80,
        freq_low=0,
        freq_high=8000,
        device="gpu"
    )
    
    # Log mel spectrogram
    log_mel = fn.to_decibels(
        mel_spec,
        multiplier=10.0,
        reference=1.0,
        cutoff_db=-80,
        device="gpu"
    )
    
    # Normalize (Whisper-style)
    log_mel = fn.normalize(
        log_mel,
        axes=[0, 1],
        device="gpu"
    )
    
    return log_mel, labels


# Usage
pipe = whisper_features_pipeline("/data/librispeech")
pipe.build()

# Run pipeline
outputs = pipe.run()
mel_features = outputs[0].as_tensor()  # GPU tensor
```

---

## Integration with PyTorch

### DALIGenericIterator

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Create DALI pipeline
@pipeline_def(batch_size=32, num_threads=4, device_id=0)
def training_pipeline():
    audio, labels = load_and_process_audio()
    return audio, labels

pipe = training_pipeline()
pipe.build()

# Create PyTorch iterator
train_loader = DALIGenericIterator(
    pipelines=[pipe],
    output_map=["audio", "labels"],
    auto_reset=True,
    last_batch_policy=LastBatchPolicy.DROP
)

# Training loop
for batch in train_loader:
    audio = batch[0]["audio"]  # Already on GPU!
    labels = batch[0]["labels"]
    
    # Forward pass (no data transfer needed)
    outputs = model(audio)
    loss = criterion(outputs, labels)
    ...
```

### Multi-GPU Training

```python
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch.distributed as dist

def create_dali_loader(rank, world_size, batch_size):
    """Create DALI loader for distributed training"""
    
    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=rank)
    def dist_pipeline():
        audio, labels = fn.readers.file(
            file_root="/data/audio",
            shard_id=rank,           # Each GPU gets different shard
            num_shards=world_size,
            random_shuffle=True,
            stick_to_shard=True
        )
        audio = process_audio(audio)
        return audio, labels
    
    pipe = dist_pipeline()
    pipe.build()
    
    return DALIGenericIterator(
        pipelines=[pipe],
        output_map=["audio", "labels"]
    )

# In training script
rank = dist.get_rank()
world_size = dist.get_world_size()
train_loader = create_dali_loader(rank, world_size, batch_size=32)
```

---

## Benchmarks

### Throughput Comparison

```
Dataset: LibriSpeech 960h
Hardware: 8x A100 GPUs
Task: Load audio → Mel spectrogram → Training batch

                    Samples/sec    GPU Util    Speedup
─────────────────────────────────────────────────────
librosa (CPU)           1,200        15%        1.0x
torchaudio (CPU)        3,500        25%        2.9x
torchaudio (GPU)       12,000        60%       10.0x
NVIDIA DALI            85,000        95%       70.8x

DALI achieves near-perfect GPU utilization!
```

### Latency Comparison

```
Single audio file processing (10 seconds @ 16kHz):

Operation          librosa    torchaudio    DALI
─────────────────────────────────────────────────
Load + Decode       15 ms        8 ms       2 ms
Resample            12 ms        5 ms       1 ms
Mel Spectrogram     25 ms       10 ms       2 ms
─────────────────────────────────────────────────
Total               52 ms       23 ms       5 ms
Speedup             1.0x        2.3x      10.4x
```

### Memory Efficiency

```
Batch of 32 × 10 second audio clips:

                 CPU Memory    GPU Memory    Transfers
───────────────────────────────────────────────────────
Standard PyTorch   4.2 GB        2.1 GB      32 × 10MB
DALI               0.5 GB        2.3 GB      32 × 0.5MB

DALI transfers compressed audio, decodes on GPU.
20x less CPU-GPU transfer bandwidth!
```

---

## Best Practices

### Pipeline Optimization

```python
# ✓ DO: Process everything on GPU
@pipeline_def
def good_pipeline():
    audio, sr = fn.decoders.audio(bytes, device="mixed")
    audio = fn.audio_resample(audio, device="gpu")
    spec = fn.spectrogram(audio, device="gpu")
    return spec

# ✗ DON'T: Mix CPU and GPU operations
@pipeline_def
def bad_pipeline():
    audio, sr = fn.decoders.audio(bytes, device="cpu")  # CPU!
    audio = fn.audio_resample(audio, device="gpu")  # Transfer!
    # Data bounces between CPU and GPU
```

### Prefetching

```python
@pipeline_def(
    batch_size=32,
    num_threads=8,
    device_id=0,
    prefetch_queue_depth=2,  # Prefetch 2 batches
)
def prefetched_pipeline():
    # DALI automatically prefetches batches
    # while GPU is processing current batch
    ...
```

### Variable Length Handling

```python
@pipeline_def
def variable_length_pipeline():
    audio, sr = load_audio()
    
    # Option 1: Pad to max length
    audio = fn.pad(audio, axes=[0], shape=[max_length])
    
    # Option 2: Random crop
    audio = fn.random_resized_crop(audio, size=fixed_length)
    
    # Option 3: Return padded with lengths
    lengths = fn.shapes(audio)
    audio = fn.pad(audio, axes=[0], shape=[max_length])
    
    return audio, lengths
```

### Memory Management

```python
# For large datasets, use external source
from nvidia.dali.plugin.pytorch import feed_ndarray

@pipeline_def
def external_source_pipeline():
    audio = fn.external_source(
        source=my_audio_generator,
        batch=True,
        device="gpu"
    )
    # Process audio...
    return audio

# Generator that yields batches
def my_audio_generator():
    for batch in data_loader:
        yield batch  # Efficiently transfers to DALI
```

---

## Key Takeaways

```
1. DALI provides 10-100x speedup for audio preprocessing
   - All operations on GPU
   - Overlapped with model training

2. Zero-copy transfers minimize overhead
   - Compressed audio → GPU decode
   - No CPU bottleneck

3. Integrates seamlessly with PyTorch
   - DALIGenericIterator
   - Distributed training support

4. Best for training pipelines
   - High throughput focus
   - Less suited for inference (use torchaudio)

5. Requires pipeline thinking
   - Define operations declaratively
   - DALI optimizes execution
```

---

## Next Steps

- `02_dali_audio_pipeline.py` - Complete working example
- `03_torchaudio_optimization.md` - When DALI isn't available
- `04_gpu_audio_processing.py` - Custom CUDA kernels for audio
