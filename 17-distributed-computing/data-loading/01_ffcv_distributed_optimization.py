"""
FFCV: Fast Data Loading for Distributed Training
=================================================

Based on the FFCV paper: "Accelerating Training by Removing Data Bottlenecks"

Key Topics:
1. The Data Loading Bottleneck
2. FFCV Architecture (.beton format)
3. Memory Management Strategies
4. JIT-Compiled Augmentations
5. Distributed Data Loading
6. Quasi-Random Sampling
"""

import torch
from typing import Optional, Sequence, Callable
import os

# =============================================================================
# SECTION 1: THE DATA BOTTLENECK PROBLEM
# =============================================================================
"""
DATA LOADING AS THE BOTTLENECK:
═══════════════════════════════

From FFCV paper experiments (ImageNet on 8×A100):

Standard PyTorch ImageFolder:
    - Data reading only: 75 seconds/epoch
    - Reading + Processing: 1200 seconds/epoch (16x slower!)
    - Full training: 1200 seconds/epoch (GPU starved!)
    - Idealized (no data loading): 40 seconds/epoch

The GPU spends most of its time WAITING for data!

BOTTLENECK BREAKDOWN:
┌───────────────────────────────────────────────────────────────┐
│ Stage              │ Time         │ % of Total               │
├───────────────────────────────────────────────────────────────┤
│ Data Reading       │ 75s          │ 6%                       │
│ Data Processing    │ 1125s        │ 94%                      │
│   - JPEG Decode    │ ~600s        │ 50%                      │
│   - Augmentations  │ ~400s        │ 33%                      │
│   - Tensor Convert │ ~125s        │ 11%                      │
│ GPU Training       │ ~0s (hidden) │ Overlapped but starved   │
└───────────────────────────────────────────────────────────────┘

WHY IS PROCESSING SO SLOW?

1. Python GIL:
   - Only one thread can execute Python at a time
   - Multi-processing overhead (IPC)

2. Inefficient File Format:
   - Individual files = many syscalls
   - Random access patterns = cache thrashing

3. CPU-bound Operations:
   - JPEG decoding is expensive
   - Augmentations run on CPU
   - Multiple memory copies

FFCV SOLUTION: Eliminate each bottleneck systematically
"""


# =============================================================================
# SECTION 2: FFCV ARCHITECTURE
# =============================================================================
"""
FFCV .BETON FILE FORMAT:
════════════════════════

Instead of individual files, FFCV uses a single optimized file.

File Structure:
┌─────────────────────────────────────────────────────────────┐
│                         HEADER                              │
│  - Number of samples                                        │
│  - Field definitions                                        │
│  - Metadata                                                 │
├─────────────────────────────────────────────────────────────┤
│                       DATA TABLE                            │
│  - Fixed-width metadata per sample                          │
│  - Pointers to heap storage                                 │
│  - Enables fast indexing/filtering                          │
├─────────────────────────────────────────────────────────────┤
│                      HEAP STORAGE                           │
│  - Variable-size data (images, audio)                       │
│  - Organized in 8MB pages                                   │
│  - Contiguous for sequential reads                          │
├─────────────────────────────────────────────────────────────┤
│                   ALLOCATION TABLE                          │
│  - Bookkeeping for heap regions                             │
└─────────────────────────────────────────────────────────────┘

ADVANTAGES:
1. Single file = one open(), no directory traversal
2. Pages = large sequential reads (good for any storage)
3. Indexable = fast subset selection
4. Pre-decoded option = skip JPEG decode at train time


CREATING A .BETON FILE:
═══════════════════════

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

# Define fields
writer = DatasetWriter(
    'imagenet_train.beton',
    {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90,      # Store as JPEG (smaller)
            # write_mode='raw',   # Or store raw pixels (faster decode)
        ),
        'label': IntField(),
    },
    num_workers=16,
)

# Write from PyTorch dataset
writer.from_indexed_dataset(imagenet_dataset)
"""


# =============================================================================
# SECTION 3: MEMORY MANAGEMENT
# =============================================================================
"""
FFCV MEMORY STRATEGIES:
═══════════════════════

1. OS CACHE (Default):
   - Let OS cache the file in RAM
   - Good when dataset fits in memory
   - Multiple processes share same cache
   
   loader = Loader('data.beton', os_cache=True, ...)

2. PROCESS CACHE:
   - FFCV manages caching in process memory
   - Better for larger-than-RAM datasets
   - Predictive prefetching
   
   loader = Loader('data.beton', os_cache=False, ...)

3. QUASI-RANDOM SAMPLING:
   - For slow storage (network drives)
   - Trades randomness for sequential reads
   - Loads pages sequentially, shuffles within pages
   
   loader = Loader(
       'data.beton',
       order=OrderOption.QUASI_RANDOM,
       os_cache=False,
   )


MEMORY LAYOUT COMPARISON:
═════════════════════════

PyTorch DataLoader (multi-process):
    Process 0: [batch_buffer] [decode_buffer] [aug_buffer]
    Process 1: [batch_buffer] [decode_buffer] [aug_buffer]
    Process 2: [batch_buffer] [decode_buffer] [aug_buffer]
    ...
    → Memory scales with num_workers!

FFCV (multi-thread):
    Thread 0 ─┐
    Thread 1 ─┼→ [shared_circular_buffer]
    Thread 2 ─┘
    → Constant memory regardless of workers!
"""


# =============================================================================
# SECTION 4: JIT-COMPILED AUGMENTATIONS
# =============================================================================
"""
FFCV AUGMENTATION PIPELINE:
═══════════════════════════

FFCV uses Numba JIT compilation for CPU augmentations.

Standard PyTorch:
    Python function → Interpreter → Slow

FFCV:
    Python function → Numba JIT → LLVM → Machine code → Fast!

PIPELINE COMPILATION:

1. User defines pipeline:
   pipeline = [
       RandomResizedCrop(224),
       RandomHorizontalFlip(),
       ToTensor(),
       ToDevice('cuda'),
       Normalize(mean, std),
   ]

2. FFCV categorizes operations:
   - JIT-able (RandomResizedCrop, Flip, Normalize)
   - Non-JIT-able (PyTorch ops)

3. Groups consecutive JIT-able ops into "stages"

4. Generates fused code for each stage

5. Compiles with Numba/LLVM

RESULT:
    10-50x faster augmentation pipeline!
    Parallel execution (no GIL with compiled code)


THREADING VS MULTIPROCESSING:
═════════════════════════════

PyTorch DataLoader:
    - Uses multiprocessing (fork)
    - Each worker = separate process
    - IPC overhead for data transfer
    - No shared memory (by default)

FFCV:
    - Uses threading
    - JIT code releases GIL
    - Shared memory, no IPC
    - Workers collaborate on same batch
    - Async GPU transfers
"""


# =============================================================================
# SECTION 5: DISTRIBUTED LOADING
# =============================================================================
"""
FFCV DISTRIBUTED DATA LOADING:
══════════════════════════════

FFCV integrates with distributed training seamlessly.

USAGE:
======

from ffcv.loader import Loader, OrderOption

loader = Loader(
    'data.beton',
    batch_size=64,
    num_workers=12,
    order=OrderOption.RANDOM,
    distributed=True,       # Enable distributed mode
    seed=42,               # Same seed for reproducibility
    os_cache=True,
    drop_last=True,
    pipelines={
        'image': image_pipeline,
        'label': label_pipeline,
    },
)

# Each rank gets different data (like DistributedSampler)
for images, labels in loader:
    ...


HOW IT WORKS:
=============

1. Each rank gets the same .beton file
2. distributed=True partitions indices by rank
3. OS cache shared across ranks on same node
4. No coordination needed between ranks

MULTI-GPU SAME-NODE:
    All 8 GPUs read from same file
    OS caches it once
    8x less disk I/O than naive approach!


QUASI-RANDOM FOR SLOW STORAGE:
==============================

For network filesystems (common in cloud):

loader = Loader(
    'data.beton',
    order=OrderOption.QUASI_RANDOM,
    os_cache=False,
    ...
)

How quasi-random works:
1. Divide dataset into pages (8MB each)
2. Shuffle page order (not sample order)
3. Within each page, shuffle samples
4. Batch from loaded pages only

Result:
- Sequential page reads (fast on NFS)
- Good-enough randomness for training
- Constant memory footprint
"""


# =============================================================================
# SECTION 6: COMPLETE EXAMPLE
# =============================================================================

def ffcv_example():
    """Complete FFCV distributed training example."""
    print("\n" + "="*60)
    print("FFCV DISTRIBUTED TRAINING EXAMPLE")
    print("="*60)
    
    print('''
STEP 1: CREATE .BETON FILE
══════════════════════════

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder

dataset = ImageFolder('/path/to/imagenet/train')

writer = DatasetWriter(
    '/path/to/imagenet_train.beton',
    {
        'image': RGBImageField(
            max_resolution=512,
            jpeg_quality=90,
        ),
        'label': IntField(),
    },
    num_workers=16,
)
writer.from_indexed_dataset(dataset)


STEP 2: DEFINE PIPELINES
════════════════════════

from ffcv.transforms import (
    RandomResizedCrop, RandomHorizontalFlip,
    ToTensor, ToDevice, ToTorchImage,
    NormalizeImage,
)
from ffcv.fields.decoders import (
    IntDecoder, RandomResizedCropRGBImageDecoder,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_pipeline = [
    RandomResizedCropRGBImageDecoder((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    ToDevice('cuda', non_blocking=True),
    ToTorchImage(),
    NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
]

label_pipeline = [
    IntDecoder(),
    ToTensor(),
    ToDevice('cuda', non_blocking=True),
]


STEP 3: CREATE DISTRIBUTED LOADER
═════════════════════════════════

from ffcv.loader import Loader, OrderOption

loader = Loader(
    '/path/to/imagenet_train.beton',
    batch_size=128,
    num_workers=12,
    order=OrderOption.RANDOM,
    distributed=True,
    seed=42,
    os_cache=True,
    drop_last=True,
    pipelines={
        'image': image_pipeline,
        'label': label_pipeline,
    },
)


STEP 4: TRAINING LOOP
═════════════════════

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
model = DDP(model.cuda())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(90):
    for images, labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()


PERFORMANCE:
════════════

ImageNet ResNet-50 on 8×A100:
    PyTorch DataLoader: ~4 hours
    FFCV: ~20 minutes
    
    Speedup: 12x!
''')


# =============================================================================
# SECTION 7: FFCV BEST PRACTICES
# =============================================================================

def ffcv_best_practices():
    """FFCV best practices."""
    print("\n" + "="*60)
    print("FFCV BEST PRACTICES")
    print("="*60)
    print("""
1. CHOOSE STORAGE FORMAT:
   - jpeg_quality=90: Smaller file, decode overhead
   - write_mode='raw': Larger file, no decode
   - Mix: 50% JPEG, 50% raw (good balance)

2. TUNE num_workers:
   - Start with number of physical cores
   - FFCV uses threads, not processes
   - Less overhead than PyTorch workers

3. USE os_cache WHEN POSSIBLE:
   - Dataset fits in RAM? Use os_cache=True
   - Multiple jobs on same machine? Use os_cache=True
   - Cloud/NFS storage? Use os_cache=False + quasi-random

4. PIN MEMORY CONSIDERATIONS:
   - FFCV handles GPU transfer internally
   - Use ToDevice('cuda', non_blocking=True)
   - Data goes directly to GPU

5. REPRODUCIBILITY:
   - Set seed= for deterministic order
   - Same seed across ranks for distributed

6. DEBUGGING:
   - Start with small batch, check outputs
   - Ensure pipeline produces correct shapes
   - Profile to find remaining bottlenecks
""")


if __name__ == "__main__":
    ffcv_example()
    ffcv_best_practices()
    print("\n" + "="*60)
    print("FFCV MODULE COMPLETE")
    print("="*60)
