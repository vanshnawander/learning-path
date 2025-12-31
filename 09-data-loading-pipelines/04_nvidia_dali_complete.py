"""
04_nvidia_dali_complete.py - Complete NVIDIA DALI Guide

Comprehensive DALI examples for all modalities with profiling.
This is THE reference for GPU-accelerated data loading.

Requirements:
    pip install nvidia-dali-cuda120  # For CUDA 12.x
    # or nvidia-dali-cuda110 for CUDA 11.x
"""

import time
from contextlib import contextmanager

@contextmanager
def profile(name):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"⏱️  {name}: {elapsed:.2f} ms")

# ============================================================
# Check DALI availability
# ============================================================

def check_dali():
    try:
        import nvidia.dali
        print(f"DALI Version: {nvidia.dali.__version__}")
        return True
    except ImportError:
        print("DALI not installed. Install with:")
        print("  pip install nvidia-dali-cuda120")
        return False

# ============================================================
# 1. IMAGE PIPELINE - Basic
# ============================================================

def create_image_pipeline_basic():
    """Basic image pipeline with GPU decode."""
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali.types as types
    except ImportError:
        print("DALI not available")
        return None
    
    @pipeline_def
    def image_pipe(image_dir):
        # File reader
        jpegs, labels = fn.readers.file(
            file_root=image_dir,
            random_shuffle=True,
            name="Reader"
        )
        
        # GPU decode (nvJPEG hardware acceleration)
        images = fn.decoders.image(
            jpegs,
            device="mixed",  # Read on CPU, decode on GPU
            output_type=types.RGB
        )
        
        # GPU augmentations
        images = fn.resize(
            images,
            resize_x=256,
            resize_y=256,
            interp_type=types.INTERP_LINEAR
        )
        
        # Random crop
        images = fn.random_resized_crop(
            images,
            size=(224, 224),
            random_aspect_ratio=[0.75, 1.33],
            random_area=[0.08, 1.0]
        )
        
        # Random flip
        images = fn.flip(images, horizontal=fn.random.coin_flip())
        
        # Normalize
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        
        return images, labels
    
    return image_pipe

# ============================================================
# 2. IMAGE PIPELINE - Advanced with External Source
# ============================================================

def create_external_source_pipeline():
    """Pipeline with external data source (custom data)."""
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali.types as types
        import numpy as np
    except ImportError:
        return None
    
    class ExternalInputIterator:
        """Custom data source - can load from anywhere."""
        
        def __init__(self, batch_size, shape=(224, 224, 3)):
            self.batch_size = batch_size
            self.shape = shape
            self.n = 0
            
        def __iter__(self):
            return self
        
        def __next__(self):
            # Generate batch
            batch = []
            labels = []
            for _ in range(self.batch_size):
                # Replace with actual data loading
                img = np.random.randint(0, 255, self.shape, dtype=np.uint8)
                label = np.array([self.n % 1000], dtype=np.int32)
                batch.append(img)
                labels.append(label)
                self.n += 1
            return batch, labels
    
    @pipeline_def
    def external_pipe(external_data):
        images, labels = fn.external_source(
            source=external_data,
            num_outputs=2,
            batch=True,
            dtype=[types.UINT8, types.INT32]
        )
        
        # Move to GPU
        images = images.gpu()
        
        # Process
        images = fn.normalize(
            images,
            dtype=types.FLOAT,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        
        return images, labels
    
    return external_pipe, ExternalInputIterator

# ============================================================
# 3. VIDEO PIPELINE - GPU Decode
# ============================================================

def create_video_pipeline():
    """Video pipeline with GPU decoding (NVDEC)."""
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali.types as types
    except ImportError:
        return None
    
    @pipeline_def
    def video_pipe(video_files, sequence_length=16):
        # GPU video decode
        video = fn.readers.video(
            device="gpu",
            filenames=video_files,
            sequence_length=sequence_length,
            stride=1,           # Every frame
            step=-1,            # Random start position
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            initial_fill=16,
            name="VideoReader"
        )
        
        # Resize (still on GPU)
        video = fn.resize(
            video,
            resize_x=224,
            resize_y=224
        )
        
        # Normalize
        video = fn.normalize(
            video,
            dtype=types.FLOAT,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        
        return video
    
    return video_pipe

# ============================================================
# 4. AUDIO PIPELINE - Mel Spectrogram
# ============================================================

def create_audio_pipeline():
    """Audio pipeline with spectrogram computation."""
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali.types as types
    except ImportError:
        return None
    
    @pipeline_def
    def audio_pipe(audio_files, sample_rate=16000):
        # Read audio files
        audio, sr = fn.readers.file(file_root=audio_files)
        
        # Decode audio
        audio, sr = fn.decoders.audio(
            audio,
            dtype=types.FLOAT,
            downmix=True  # Convert to mono
        )
        
        # Resample if needed
        # audio = fn.audio_resample(audio, in_rate=sr, out_rate=sample_rate)
        
        # Compute spectrogram
        spectrogram = fn.spectrogram(
            audio,
            nfft=512,
            window_length=400,
            window_step=160
        )
        
        # Mel filterbank
        mel = fn.mel_filter_bank(
            spectrogram,
            nfilter=80,
            sample_rate=sample_rate,
            freq_low=0,
            freq_high=8000
        )
        
        # Convert to decibels
        mel_db = fn.to_decibels(
            mel,
            multiplier=10.0,
            reference=1.0,
            cutoff_db=-80.0
        )
        
        return mel_db
    
    return audio_pipe

# ============================================================
# 5. MULTIMODAL PIPELINE - Video + Audio
# ============================================================

def create_multimodal_pipeline():
    """Combined video + audio pipeline."""
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali.types as types
    except ImportError:
        return None
    
    @pipeline_def
    def multimodal_pipe(video_files, audio_files, seq_len=16):
        # Video branch (GPU decode)
        video = fn.readers.video(
            device="gpu",
            filenames=video_files,
            sequence_length=seq_len,
            name="VideoReader"
        )
        video = fn.resize(video, resize_x=224, resize_y=224)
        video = fn.normalize(video, dtype=types.FLOAT,
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        
        # Audio branch
        audio_raw, sr = fn.readers.file(file_root=audio_files)
        audio, sr = fn.decoders.audio(audio_raw, dtype=types.FLOAT)
        mel = fn.spectrogram(audio, nfft=512, window_length=400, window_step=160)
        mel = fn.mel_filter_bank(mel, nfilter=80, sample_rate=16000)
        mel = fn.to_decibels(mel)
        
        return video, mel
    
    return multimodal_pipe

# ============================================================
# 6. PYTORCH INTEGRATION
# ============================================================

def pytorch_integration_example():
    """How to use DALI with PyTorch training loop."""
    
    code = '''
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# Create and build pipeline
pipe = image_pipeline(
    image_dir="/data/imagenet/train",
    batch_size=64,
    num_threads=4,
    device_id=0
)
pipe.build()

# Create PyTorch iterator
train_loader = DALIGenericIterator(
    pipelines=[pipe],
    output_map=["images", "labels"],
    reader_name="Reader",
    last_batch_policy=LastBatchPolicy.PARTIAL,
    auto_reset=True
)

# Training loop
model = torchvision.models.resnet50().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch[0]["images"]  # Already on GPU!
        labels = batch[0]["labels"].squeeze()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Reset for next epoch
    train_loader.reset()
'''
    return code

# ============================================================
# 7. MULTI-GPU DALI
# ============================================================

def multi_gpu_example():
    """DALI with multiple GPUs."""
    
    code = '''
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
world_size = dist.get_world_size()

@pipeline_def
def distributed_image_pipe(image_dir, shard_id, num_shards):
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        shard_id=shard_id,      # This GPU's shard
        num_shards=num_shards,  # Total GPUs
        name="Reader"
    )
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.resize(images, resize_x=224, resize_y=224)
    return images, labels

# Create pipeline for this GPU
pipe = distributed_image_pipe(
    image_dir="/data/imagenet",
    shard_id=local_rank,
    num_shards=world_size,
    batch_size=64,
    num_threads=4,
    device_id=local_rank
)
pipe.build()
'''
    return code

# ============================================================
# 8. PERFORMANCE PROFILING
# ============================================================

def profile_dali_pipeline():
    """Profile DALI pipeline performance."""
    
    code = '''
import time
import torch

# Build pipeline
pipe = image_pipeline(batch_size=64, num_threads=4, device_id=0)
pipe.build()

# Create iterator
loader = DALIGenericIterator(pipe, ["images", "labels"], reader_name="Reader")

# Warmup
for i, batch in enumerate(loader):
    if i >= 5:
        break
loader.reset()

# Benchmark
num_batches = 100
torch.cuda.synchronize()
start = time.perf_counter()

for i, batch in enumerate(loader):
    images = batch[0]["images"]
    torch.cuda.synchronize()
    if i >= num_batches:
        break

elapsed = time.perf_counter() - start
images_per_sec = num_batches * 64 / elapsed

print(f"Throughput: {images_per_sec:.0f} images/sec")
print(f"Batch time: {elapsed/num_batches*1000:.2f} ms")
'''
    return code

# ============================================================
# MAIN - Print Examples
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  NVIDIA DALI COMPLETE GUIDE")
    print("=" * 70)
    
    has_dali = check_dali()
    
    print("\n" + "█" * 70)
    print("█  1. IMAGE PIPELINE (GPU DECODE)")
    print("█" * 70)
    print("""
@pipeline_def
def image_pipe(image_dir):
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device="mixed")  # GPU decode!
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT,
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255])
    return images, labels
""")
    
    print("\n" + "█" * 70)
    print("█  2. VIDEO PIPELINE (NVDEC HARDWARE)")
    print("█" * 70)
    print("""
@pipeline_def
def video_pipe(video_files):
    video = fn.readers.video(
        device="gpu",           # NVDEC hardware decode
        filenames=video_files,
        sequence_length=16,     # Frames per clip
        stride=1,               # Every frame
    )
    video = fn.resize(video, resize_x=224, resize_y=224)
    return video
    
# NVDEC can decode 1000+ FPS for 1080p video!
""")
    
    print("\n" + "█" * 70)
    print("█  3. AUDIO PIPELINE (MEL SPECTROGRAM)")
    print("█" * 70)
    print("""
@pipeline_def
def audio_pipe(audio_files):
    audio, sr = fn.readers.file(file_root=audio_files)
    audio, sr = fn.decoders.audio(audio, dtype=types.FLOAT)
    spec = fn.spectrogram(audio, nfft=512)
    mel = fn.mel_filter_bank(spec, nfilter=80, sample_rate=16000)
    mel_db = fn.to_decibels(mel)
    return mel_db
""")
    
    print("\n" + "█" * 70)
    print("█  4. PYTORCH INTEGRATION")
    print("█" * 70)
    print("""
from nvidia.dali.plugin.pytorch import DALIGenericIterator

pipe = image_pipe(batch_size=64, num_threads=4, device_id=0)
pipe.build()

train_loader = DALIGenericIterator(
    pipe, ["images", "labels"], reader_name="Reader"
)

for batch in train_loader:
    images = batch[0]["images"]  # Already on GPU!
    labels = batch[0]["labels"]
    loss = model(images)
""")
    
    print("\n" + "█" * 70)
    print("█  5. KEY OPERATORS REFERENCE")
    print("█" * 70)
    print("""
READERS:
  fn.readers.file()       - Read files from directory
  fn.readers.video()      - Read video files (NVDEC)
  fn.readers.numpy()      - Read numpy arrays
  fn.readers.tfrecord()   - Read TFRecord files
  fn.external_source()    - Custom data source

DECODERS:
  fn.decoders.image()     - JPEG/PNG decode (nvJPEG on GPU)
  fn.decoders.audio()     - Audio decode
  
IMAGE TRANSFORMS:
  fn.resize()             - Resize image
  fn.crop()               - Center crop
  fn.random_resized_crop() - Random crop + resize
  fn.flip()               - Horizontal/vertical flip
  fn.rotate()             - Rotation
  fn.color_twist()        - Color augmentation
  fn.crop_mirror_normalize() - Combined transform
  
AUDIO TRANSFORMS:
  fn.spectrogram()        - STFT
  fn.mel_filter_bank()    - Mel filterbank
  fn.mfcc()               - MFCCs
  fn.to_decibels()        - Power to dB
  
GENERAL:
  fn.normalize()          - Normalize values
  fn.cast()               - Change dtype
  fn.reshape()            - Reshape tensor
  fn.pad()                - Pad tensor
""")
    
    print("\n" + "█" * 70)
    print("█  PERFORMANCE COMPARISON")
    print("█" * 70)
    print("""
ImageNet Training Throughput (single A100):

| Method               | Images/sec | GPU Util |
|---------------------|------------|----------|
| PyTorch DataLoader   | 2,500      | 50%      |
| DALI (CPU decode)    | 8,000      | 85%      |
| DALI (GPU decode)    | 15,000     | 95%      |

Video Decode (1080p):

| Method               | FPS        |
|---------------------|------------|
| OpenCV              | 30         |
| decord              | 200        |
| DALI (NVDEC)        | 1,000+     |
""")
    
    print("\n" + "=" * 70)
    print("  WHEN TO USE DALI")
    print("=" * 70)
    print("""
✅ USE DALI WHEN:
  • Training is data-loading bottlenecked
  • You have NVIDIA GPUs
  • Working with images or video
  • Need maximum throughput

❌ SKIP DALI WHEN:
  • Quick prototyping
  • Small datasets
  • Non-NVIDIA hardware
  • Need very custom transforms
""")
