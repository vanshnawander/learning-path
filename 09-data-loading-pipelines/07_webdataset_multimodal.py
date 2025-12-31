"""
07_webdataset_multimodal.py - WebDataset for Multimodal Training

Comprehensive WebDataset guide for video, audio, text, and images.
This is the best solution for cloud-native multimodal training.

Requirements:
    pip install webdataset torch torchvision torchaudio
"""

import time
import io
from contextlib import contextmanager

@contextmanager  
def profile(name):
    start = time.perf_counter()
    yield
    print(f"⏱️  {name}: {(time.perf_counter()-start)*1000:.2f} ms")

# ============================================================
# WEBDATASET ARCHITECTURE
# ============================================================

ARCHITECTURE = """
╔═══════════════════════════════════════════════════════════════════════╗
║                     WEBDATASET ARCHITECTURE                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  FILE FORMAT: Standard TAR archives                                   ║
║                                                                        ║
║  shard-000000.tar                                                     ║
║  ├── sample_00001.jpg     ← Image data                               ║
║  ├── sample_00001.json    ← Metadata                                 ║
║  ├── sample_00001.cls     ← Class label                              ║
║  ├── sample_00002.jpg                                                ║
║  ├── sample_00002.json                                               ║
║  └── ...                                                              ║
║                                                                        ║
║  STREAMING BENEFITS:                                                  ║
║  • Sequential read = maximum I/O throughput                          ║
║  • Works with S3, GCS, HTTP, local files                             ║
║  • No random seeks needed                                             ║
║  • Shard-level shuffling for distributed training                    ║
║                                                                        ║
║  DATA FLOW:                                                           ║
║  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐          ║
║  │  Shards  │──▶│  Decode  │──▶│Transform │──▶│  Batch   │          ║
║  │ (stream) │   │ (lazy)   │   │(pipeline)│   │(collate) │          ║
║  └──────────┘   └──────────┘   └──────────┘   └──────────┘          ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# 1. CREATING WEBDATASET SHARDS
# ============================================================

def create_image_shards_example():
    """Example: Create image dataset shards."""
    
    code = '''
import webdataset as wds
import os
from PIL import Image
import json

def create_image_shards(image_dir, output_pattern, samples_per_shard=10000):
    """
    Convert image folder to WebDataset shards.
    
    Args:
        image_dir: Directory with images
        output_pattern: e.g., "shards/shard-%06d.tar"
        samples_per_shard: Samples per tar file
    """
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Create shard writer
    with wds.ShardWriter(output_pattern, maxcount=samples_per_shard) as sink:
        for i, img_file in enumerate(image_files):
            # Read image
            img_path = os.path.join(image_dir, img_file)
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            # Create sample
            sample = {
                "__key__": f"sample_{i:08d}",
                "jpg": img_data,  # Extension determines decoder
                "json": json.dumps({"filename": img_file, "index": i}),
                "cls": str(i % 1000),  # Class label as string
            }
            
            sink.write(sample)
            
            if i % 1000 == 0:
                print(f"Written {i} samples...")

# Usage:
create_image_shards("/data/images", "shards/imagenet-%06d.tar")
'''
    return code

def create_multimodal_shards_example():
    """Example: Create video+audio+text shards."""
    
    code = '''
import webdataset as wds
import json

def create_multimodal_shards(data_items, output_pattern):
    """
    Create shards with video, audio, and text.
    
    data_items: List of dicts with paths to each modality
    """
    
    with wds.ShardWriter(output_pattern, maxcount=1000) as sink:
        for i, item in enumerate(data_items):
            # Read each modality
            with open(item["video_path"], "rb") as f:
                video_data = f.read()
            with open(item["audio_path"], "rb") as f:
                audio_data = f.read()
            
            sample = {
                "__key__": f"sample_{i:08d}",
                "mp4": video_data,           # Video
                "flac": audio_data,          # Audio  
                "txt": item["transcript"],   # Text
                "json": json.dumps({
                    "duration": item["duration"],
                    "speaker": item["speaker"],
                }),
            }
            sink.write(sample)

# File extensions determine decoders:
# .mp4, .avi, .webm → video
# .wav, .flac, .mp3 → audio
# .txt → text (string)
# .json → parsed JSON
# .jpg, .png → image
# .npy → numpy array
# .pth, .pt → PyTorch tensor
'''
    return code

# ============================================================
# 2. LOADING IMAGE DATASETS
# ============================================================

def image_dataset_example():
    """Basic image dataset loading."""
    
    code = '''
import webdataset as wds
import torchvision.transforms as T

# Define transforms
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])

# Create dataset
url = "s3://bucket/imagenet/shard-{000000..001000}.tar"
# Or local: "/data/shards/shard-{000000..001000}.tar"
# Or HTTP: "http://server/shards/shard-{000000..001000}.tar"

dataset = (
    wds.WebDataset(url)
    .shuffle(1000)                    # Shuffle buffer
    .decode("pil")                    # Decode images to PIL
    .to_tuple("jpg", "cls")           # Select fields
    .map_tuple(transform, int)        # Apply transforms
    .batched(64)                      # Batch samples
)

# Use with DataLoader
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=None,      # Batching done in pipeline
    num_workers=4,
    pin_memory=True,
)

# Training loop
for images, labels in loader:
    images = images.cuda()
    loss = model(images)
'''
    return code

# ============================================================
# 3. LOADING VIDEO DATASETS
# ============================================================

def video_dataset_example():
    """Video dataset with frame extraction."""
    
    code = '''
import webdataset as wds
import torch
import decord
from decord import VideoReader, cpu
import numpy as np

def decode_video(video_bytes, num_frames=16, size=224):
    """Decode video bytes to tensor of frames."""
    # Use decord for fast video decode
    vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
    
    # Sample frames uniformly
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Get frames
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    
    # Resize and normalize
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = torch.nn.functional.interpolate(frames, size=(size, size))
    frames = frames / 255.0
    
    return frames

# Video dataset
url = "s3://bucket/videos/shard-{000..100}.tar"

dataset = (
    wds.WebDataset(url)
    .shuffle(500)
    .to_tuple("mp4", "json")
    .map_tuple(
        lambda x: decode_video(x, num_frames=16),
        lambda x: json.loads(x)
    )
    .batched(8)  # Smaller batches for video
)

for video_batch, metadata_batch in loader:
    # video_batch: (B, T, C, H, W)
    video_batch = video_batch.cuda()
'''
    return code

# ============================================================
# 4. LOADING AUDIO DATASETS
# ============================================================

def audio_dataset_example():
    """Audio dataset with spectrogram computation."""
    
    code = '''
import webdataset as wds
import torch
import torchaudio
import io

def decode_audio(audio_bytes, target_sr=16000, max_len=10.0):
    """Decode audio and compute mel spectrogram."""
    # Load audio
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Truncate/pad to fixed length
    max_samples = int(max_len * target_sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=512,
        hop_length=160,
        n_mels=80
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-9)
    
    return mel

# Audio dataset
url = "/data/audio/shard-{000..050}.tar"

dataset = (
    wds.WebDataset(url)
    .shuffle(1000)
    .to_tuple("flac", "txt")  # Audio + transcript
    .map_tuple(decode_audio, str)
    .batched(32)
)

for mel_batch, transcript_batch in loader:
    # mel_batch: (B, 1, n_mels, time)
    mel_batch = mel_batch.cuda()
'''
    return code

# ============================================================
# 5. MULTIMODAL DATASET (VIDEO + AUDIO + TEXT)
# ============================================================

def multimodal_dataset_example():
    """Complete multimodal dataset for video understanding."""
    
    code = '''
import webdataset as wds
import torch
import json

class MultimodalDecoder:
    """Decode all modalities for multimodal training."""
    
    def __init__(self, video_frames=16, audio_len=10.0, max_text_len=512):
        self.video_frames = video_frames
        self.audio_len = audio_len
        self.max_text_len = max_text_len
        self.tokenizer = ...  # Your tokenizer
    
    def __call__(self, sample):
        # Decode video
        video = self.decode_video(sample["mp4"])
        
        # Decode audio  
        audio = self.decode_audio(sample["flac"])
        
        # Tokenize text
        text = sample["txt"]
        tokens = self.tokenizer(
            text, 
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Parse metadata
        metadata = json.loads(sample["json"])
        
        return {
            "video": video,           # (T, C, H, W)
            "audio": audio,           # (1, n_mels, time)
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "metadata": metadata,
        }

# Multimodal dataset
url = "s3://bucket/multimodal/shard-{000000..010000}.tar"

decoder = MultimodalDecoder()

dataset = (
    wds.WebDataset(url)
    .shuffle(500)
    .map(decoder)
    .batched(8, collation_fn=multimodal_collate)
)

def multimodal_collate(samples):
    """Custom collation for multimodal batch."""
    return {
        "video": torch.stack([s["video"] for s in samples]),
        "audio": torch.stack([s["audio"] for s in samples]),
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
    }

# Training
for batch in loader:
    video = batch["video"].cuda()   # (B, T, C, H, W)
    audio = batch["audio"].cuda()   # (B, 1, M, T)
    text_ids = batch["input_ids"].cuda()
    
    # Multimodal forward pass
    output = model(video, audio, text_ids)
'''
    return code

# ============================================================
# 6. DISTRIBUTED TRAINING
# ============================================================

def distributed_example():
    """WebDataset with distributed training."""
    
    code = '''
import webdataset as wds
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Sharded URL with worker splitting
url = "s3://bucket/data/shard-{000000..010000}.tar"

dataset = (
    wds.WebDataset(url, shardshuffle=True)
    .shuffle(1000)
    # Split by node
    .slice(rank, world_size)  # Each GPU gets different shards
    .decode("pil")
    .to_tuple("jpg", "cls")
    .map_tuple(transform, int)
    .batched(64)
)

# Or use WebLoader for more control
loader = wds.WebLoader(
    dataset,
    batch_size=None,
    num_workers=4,
)

# With epoch-based resampling
dataset = (
    wds.WebDataset(url, resampled=True)  # Infinite stream
    .shuffle(1000)
    ...
)

# Manually set epoch for reproducibility
loader = wds.WebLoader(dataset, ...)
loader.epoch = current_epoch
'''
    return code

# ============================================================
# 7. CLOUD STORAGE (S3/GCS)
# ============================================================

def cloud_storage_example():
    """Using WebDataset with cloud storage."""
    
    code = '''
import webdataset as wds

# AWS S3
s3_url = "s3://my-bucket/data/shard-{000000..001000}.tar"

# Google Cloud Storage  
gcs_url = "gs://my-bucket/data/shard-{000000..001000}.tar"

# HTTP/HTTPS
http_url = "https://storage.example.com/data/shard-{000000..001000}.tar"

# Azure Blob Storage (via HTTP)
azure_url = "https://account.blob.core.windows.net/container/shard-{000..100}.tar"

# Configure credentials (environment variables)
# AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
# GCS: GOOGLE_APPLICATION_CREDENTIALS

# With authentication pipe
dataset = wds.WebDataset(
    s3_url,
    handler=wds.warn_and_continue,  # Skip bad samples
)

# Caching for repeated access
dataset = wds.WebDataset(
    s3_url,
    cache_dir="/tmp/webdataset_cache",  # Local cache
    cache_size=10 * 1024**3,  # 10 GB cache
)
'''
    return code

# ============================================================
# 8. PERFORMANCE TIPS
# ============================================================

PERFORMANCE_TIPS = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    WEBDATASET PERFORMANCE TIPS                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  1. SHARD SIZE                                                        ║
║     • 100MB - 1GB per shard is optimal                                ║
║     • Too small: overhead per shard                                   ║
║     • Too large: less shuffle randomness                              ║
║                                                                        ║
║  2. SHUFFLE BUFFER                                                    ║
║     • shuffle(N): buffer N samples for randomization                  ║
║     • Larger = more random, more memory                               ║
║     • 1000-10000 is typical                                           ║
║                                                                        ║
║  3. SHARD SHUFFLING                                                   ║
║     • shardshuffle=True: randomize shard order                        ║
║     • Essential for distributed training                              ║
║                                                                        ║
║  4. NUM_WORKERS                                                       ║
║     • More workers = more parallel decode                             ║
║     • 4-8 typically sufficient                                        ║
║                                                                        ║
║  5. CACHING                                                           ║
║     • Use cache_dir for repeated cloud access                         ║
║     • Or use local SSD for hot data                                   ║
║                                                                        ║
║  6. PREFETCHING                                                       ║
║     • Pipeline prefetches automatically                               ║
║     • Keep batched() at end for efficiency                            ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  WEBDATASET COMPLETE MULTIMODAL GUIDE")
    print("=" * 70)
    
    print(ARCHITECTURE)
    
    print("\n" + "█" * 70)
    print("█  1. CREATE SHARDS")
    print("█" * 70)
    print(create_image_shards_example())
    
    print("\n" + "█" * 70)
    print("█  2. IMAGE DATASET")
    print("█" * 70)
    print(image_dataset_example())
    
    print("\n" + "█" * 70)
    print("█  3. VIDEO DATASET")
    print("█" * 70)
    print(video_dataset_example())
    
    print("\n" + "█" * 70)
    print("█  4. AUDIO DATASET")
    print("█" * 70)
    print(audio_dataset_example())
    
    print("\n" + "█" * 70)
    print("█  5. MULTIMODAL DATASET")
    print("█" * 70)
    print(multimodal_dataset_example())
    
    print("\n" + "█" * 70)
    print("█  6. DISTRIBUTED TRAINING")
    print("█" * 70)
    print(distributed_example())
    
    print(PERFORMANCE_TIPS)
    
    print("\n" + "=" * 70)
    print("  WHEN TO USE WEBDATASET")
    print("=" * 70)
    print("""
✅ USE WEBDATASET WHEN:
  • Data is in cloud storage (S3, GCS)
  • Dataset is too large for local storage
  • Training is distributed across nodes
  • Need streaming without full download
  • Working with multimodal data

❌ CONSIDER ALTERNATIVES WHEN:
  • Need maximum single-GPU speed (use FFCV)
  • Need random access to specific samples
  • Data fits in memory
  • Need GPU decode (use DALI)
""")
