# Complete Custom Format Implementation Project

## Project Overview

This project guides you through building a complete high-performance data format from scratch, incorporating all the concepts learned.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROJECT: MULTIMODAL DATA FORMAT                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Goal: Build a format supporting:                                        │
│  ✓ Images (JPEG/raw)                                                    │
│  ✓ Audio (waveform/compressed/spectrogram)                              │
│  ✓ Text (raw/tokenized)                                                 │
│  ✓ Video (frames)                                                       │
│  ✓ Arbitrary labels/metadata                                            │
│                                                                          │
│  Features:                                                               │
│  ✓ Memory-mapped reading                                                │
│  ✓ Page-aligned writing for parallel I/O                               │
│  ✓ JIT-compiled decoders                                                │
│  ✓ Configurable compression                                             │
│  ✓ Multi-worker data loading                                            │
│                                                                          │
│  Performance Target:                                                     │
│  • 10-50x faster than naive file-per-sample                             │
│  • Saturate GPU with data during training                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
fastloader/
├── __init__.py
├── types.py              # Type definitions, constants
├── fields/
│   ├── __init__.py
│   ├── base.py           # Abstract Field class
│   ├── scalar.py         # Int, Float fields
│   ├── bytes.py          # Variable bytes
│   ├── image.py          # Image fields
│   ├── audio.py          # Audio fields
│   ├── text.py           # Text fields
│   └── video.py          # Video fields
├── writer.py             # Dataset writer
├── reader.py             # Dataset reader
├── memory.py             # Memory management
├── pipeline/
│   ├── __init__.py
│   ├── operations.py     # Transform operations
│   └── compiler.py       # JIT pipeline compiler
├── loader.py             # DataLoader interface
└── utils.py              # Utilities
```

## Step 1: Type Definitions

```python
# fastloader/types.py
"""
Core type definitions for the data format.
"""

import numpy as np
from enum import IntEnum

# Magic bytes
MAGIC = b'\x89FLD\r\n\x1a\n'

# Version
VERSION_MAJOR = 1
VERSION_MINOR = 0

# Default page size (2 MB for huge page compatibility)
DEFAULT_PAGE_SIZE = 2 * 1024 * 1024


class TypeID(IntEnum):
    """Type identifiers for fields."""
    INT = 1
    FLOAT = 2
    BYTES = 3
    NDARRAY = 4
    JSON = 5
    
    # Images
    IMAGE_RAW = 10
    IMAGE_JPEG = 11
    IMAGE_PNG = 12
    
    # Audio
    AUDIO_WAVEFORM = 20
    AUDIO_COMPRESSED = 21
    AUDIO_SPECTROGRAM = 22
    AUDIO_TOKENS = 23
    
    # Video
    VIDEO_FRAMES = 30
    VIDEO_COMPRESSED = 31
    VIDEO_OPTICAL_FLOW = 32
    
    # Text
    TEXT_RAW = 40
    TEXT_TOKENIZED = 41
    TEXT_PACKED = 42


class Flags(IntEnum):
    """Header flags."""
    LITTLE_ENDIAN = 0x0001
    COMPRESSED = 0x0002
    CHECKSUMMED = 0x0004


# Header structure (64 bytes)
HeaderType = np.dtype([
    ('version_major', '<u2'),
    ('version_minor', '<u2'),
    ('flags', '<u4'),
    ('num_samples', '<u8'),
    ('num_fields', '<u2'),
    ('_pad1', '<u2'),
    ('page_size', '<u4'),
    ('field_desc_ptr', '<u8'),
    ('metadata_ptr', '<u8'),
    ('data_ptr', '<u8'),
    ('alloc_table_ptr', '<u8'),
    ('_reserved', '8u1'),
], align=True)

assert HeaderType.itemsize == 64


# Field descriptor (208 bytes)
FieldDescType = np.dtype([
    ('name', 'S64'),
    ('type_id', '<u2'),
    ('flags', '<u2'),
    ('metadata_size', '<u4'),
    ('_pad', '<u4'),
    ('arguments', '128u1'),
], align=True)


# Allocation table entry
AllocEntryType = np.dtype([
    ('ptr', '<u8'),
    ('size', '<u4'),
    ('_pad', '<u4'),
], align=True)


# Type registry
TYPE_REGISTRY = {}

def register_type(type_id):
    """Decorator to register field types."""
    def decorator(cls):
        TYPE_REGISTRY[type_id] = cls
        cls.type_id = type_id
        return cls
    return decorator
```

## Step 2: Abstract Field Base

```python
# fastloader/fields/base.py
"""
Abstract base class for all field types.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np


class Field(ABC):
    """
    Abstract base class for data fields.
    
    Each field type must implement:
    - metadata_type: dtype for per-sample metadata
    - encode(): Convert Python value to (metadata, bytes)
    - get_decoder_class(): Return decoder class
    - to_binary() / from_binary(): Serialize/deserialize field config
    """
    
    type_id: int  # Set by @register_type decorator
    
    @property
    @abstractmethod
    def metadata_type(self) -> np.dtype:
        """Return numpy dtype for per-sample metadata."""
        pass
    
    @abstractmethod
    def encode(self, value: Any) -> Tuple[np.ndarray, bytes]:
        """
        Encode a value to binary.
        
        Returns:
            metadata: Numpy structured array with field metadata
            data: Raw bytes to store
        """
        pass
    
    @abstractmethod
    def get_decoder_class(self):
        """Return the decoder class for this field."""
        pass
    
    def to_binary(self) -> bytes:
        """Serialize field configuration to bytes."""
        return b''
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'Field':
        """Deserialize field from bytes."""
        return cls()
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Decoder(ABC):
    """Abstract base class for decoders."""
    
    @abstractmethod
    def __call__(
        self,
        metadata: np.ndarray,
        read_fn
    ) -> Any:
        """Decode data from storage."""
        pass
    
    def supports_jit(self) -> bool:
        """Whether decoder can be JIT compiled."""
        return False
```

## Step 3: Image Field Implementation

```python
# fastloader/fields/image.py
"""
Image field implementations.
"""

import numpy as np
from typing import Tuple, Any
import struct

from .base import Field, Decoder
from ..types import TypeID, register_type


@register_type(TypeID.IMAGE_JPEG)
class JPEGImageField(Field):
    """
    Store images as JPEG compressed data.
    """
    
    def __init__(
        self,
        quality: int = 90,
        max_height: int = 1024,
        max_width: int = 1024
    ):
        self.quality = quality
        self.max_height = max_height
        self.max_width = max_width
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
            ('height', '<u2'),
            ('width', '<u2'),
            ('channels', '<u1'),
            ('_pad', '3u1'),
        ], align=True)
    
    def encode(self, value: np.ndarray) -> Tuple[np.ndarray, bytes]:
        """Encode image to JPEG."""
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        
        # Ensure uint8
        if value.dtype != np.uint8:
            value = (value * 255).clip(0, 255).astype(np.uint8)
        
        # Encode
        jpeg_bytes = jpeg.encode(value, quality=self.quality)
        
        # Metadata
        h, w = value.shape[:2]
        c = value.shape[2] if value.ndim == 3 else 1
        
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['data_size'] = len(jpeg_bytes)
        meta['height'] = h
        meta['width'] = w
        meta['channels'] = c
        
        return meta, jpeg_bytes
    
    def get_decoder_class(self):
        return JPEGImageDecoder
    
    def to_binary(self) -> bytes:
        return struct.pack('<HHHxx', self.quality, self.max_height, self.max_width)
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'JPEGImageField':
        quality, max_h, max_w = struct.unpack('<HHHxx', data[:8])
        return cls(quality=quality, max_height=max_h, max_width=max_w)


class JPEGImageDecoder(Decoder):
    """Decode JPEG images."""
    
    def __init__(self):
        from turbojpeg import TurboJPEG
        self.jpeg = TurboJPEG()
    
    def __call__(self, metadata, read_fn) -> np.ndarray:
        ptr = metadata['data_ptr']
        size = metadata['data_size']
        
        jpeg_bytes = bytes(read_fn(ptr, size))
        return self.jpeg.decode(jpeg_bytes)


@register_type(TypeID.IMAGE_RAW)
class RawImageField(Field):
    """
    Store images as raw pixel data.
    Faster to decode but larger storage.
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        channels: int = 3,
        dtype: str = 'uint8'
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = np.dtype(dtype)
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('offset', '<u8'),
        ], align=True)
    
    def encode(self, value: np.ndarray) -> Tuple[np.ndarray, bytes]:
        # Ensure correct shape and type
        value = value.astype(self.dtype)
        if value.shape != (self.height, self.width, self.channels):
            raise ValueError(f"Expected shape {(self.height, self.width, self.channels)}, got {value.shape}")
        
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        return meta, value.tobytes()
    
    def get_decoder_class(self):
        return RawImageDecoder
    
    def to_binary(self) -> bytes:
        return struct.pack('<HHBx', self.height, self.width, self.channels) + \
               self.dtype.str.encode('ascii').ljust(8, b'\x00')
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'RawImageField':
        h, w, c = struct.unpack('<HHBx', data[:6])
        dtype_str = data[6:14].rstrip(b'\x00').decode('ascii')
        return cls(height=h, width=w, channels=c, dtype=dtype_str)


class RawImageDecoder(Decoder):
    """Decode raw image data."""
    
    def __init__(self, height, width, channels, dtype):
        self.shape = (height, width, channels)
        self.dtype = np.dtype(dtype)
        self.size = np.prod(self.shape) * self.dtype.itemsize
    
    def __call__(self, metadata, read_fn) -> np.ndarray:
        offset = metadata['offset']
        data = read_fn(offset, self.size)
        return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)
    
    def supports_jit(self):
        return True
```

## Step 4: Audio Field Implementation

```python
# fastloader/fields/audio.py
"""
Audio field implementations.
"""

import numpy as np
from typing import Tuple
import struct

from .base import Field, Decoder
from ..types import TypeID, register_type


@register_type(TypeID.AUDIO_WAVEFORM)
class AudioWaveformField(Field):
    """Store raw audio waveforms."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        mono: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.mono = mono
        self.max_samples = int(sample_rate * max_duration)
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_samples', '<u4'),
            ('sample_rate', '<u4'),
            ('num_channels', '<u1'),
            ('_pad', '3u1'),
        ], align=True)
    
    def encode(self, audio: np.ndarray) -> Tuple[np.ndarray, bytes]:
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Convert to mono if needed
        if self.mono and audio.ndim > 1:
            audio = audio.mean(axis=-1)
        
        # Truncate if too long
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        
        # Metadata
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['num_samples'] = len(audio)
        meta['sample_rate'] = self.sample_rate
        meta['num_channels'] = 1 if self.mono else audio.shape[-1]
        
        return meta, audio.tobytes()
    
    def get_decoder_class(self):
        return AudioWaveformDecoder
    
    def to_binary(self) -> bytes:
        return struct.pack('<IfB3x', self.sample_rate, self.max_duration, self.mono)
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'AudioWaveformField':
        sr, max_dur, mono = struct.unpack('<IfB3x', data[:12])
        return cls(sample_rate=sr, max_duration=max_dur, mono=bool(mono))


class AudioWaveformDecoder(Decoder):
    """Decode audio waveforms."""
    
    def __init__(self, target_length: int = None):
        self.target_length = target_length
    
    def __call__(self, metadata, read_fn) -> np.ndarray:
        ptr = metadata['data_ptr']
        num_samples = metadata['num_samples']
        
        size = num_samples * 4  # float32
        data = read_fn(ptr, size)
        audio = np.frombuffer(data, dtype=np.float32)
        
        # Pad or truncate to target length
        if self.target_length:
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
        
        return audio


@register_type(TypeID.AUDIO_SPECTROGRAM)
class SpectrogramField(Field):
    """Store pre-computed spectrograms."""
    
    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_frames: int = 1000
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_frames', '<u4'),
            ('n_mels', '<u2'),
            ('_pad', '<u2'),
        ], align=True)
    
    def encode(self, audio_or_spec) -> Tuple[np.ndarray, bytes]:
        if audio_or_spec.ndim == 1:
            # Compute spectrogram from audio
            import librosa
            spec = librosa.feature.melspectrogram(
                y=audio_or_spec,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            spec = librosa.power_to_db(spec)
        else:
            spec = audio_or_spec
        
        # Truncate if too long
        if spec.shape[1] > self.max_frames:
            spec = spec[:, :self.max_frames]
        
        spec = spec.astype(np.float32)
        
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['num_frames'] = spec.shape[1]
        meta['n_mels'] = spec.shape[0]
        
        return meta, spec.tobytes()
    
    def get_decoder_class(self):
        return SpectrogramDecoder


class SpectrogramDecoder(Decoder):
    """Decode spectrograms."""
    
    def __call__(self, metadata, read_fn) -> np.ndarray:
        ptr = metadata['data_ptr']
        n_mels = metadata['n_mels']
        num_frames = metadata['num_frames']
        
        size = n_mels * num_frames * 4  # float32
        data = read_fn(ptr, size)
        
        return np.frombuffer(data, dtype=np.float32).reshape(n_mels, num_frames)
```

## Step 5: Dataset Writer

```python
# fastloader/writer.py
"""
Dataset writer with parallel writing support.
"""

import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import threading
import os

from .types import (
    MAGIC, HeaderType, FieldDescType, AllocEntryType,
    DEFAULT_PAGE_SIZE, VERSION_MAJOR, VERSION_MINOR, Flags
)
from .fields.base import Field


class DatasetWriter:
    """
    Write samples to the custom format.
    
    Features:
    - Parallel encoding
    - Page-aligned allocation
    - Memory-efficient writing
    """
    
    def __init__(
        self,
        output_path: str,
        fields: Dict[str, Field],
        page_size: int = DEFAULT_PAGE_SIZE,
        num_workers: int = None
    ):
        self.output_path = output_path
        self.fields = fields
        self.page_size = page_size
        self.num_workers = num_workers or os.cpu_count()
        
        # Build metadata dtype
        self.metadata_dtype = np.dtype([
            (name, field.metadata_type)
            for name, field in fields.items()
        ], align=True)
        
        # Internal state
        self._samples = []
        self._lock = threading.Lock()
    
    def write_sample(self, sample: Dict[str, Any]):
        """Add a sample to the dataset."""
        with self._lock:
            self._samples.append(sample)
    
    def write_many(self, samples: List[Dict[str, Any]]):
        """Add multiple samples."""
        with self._lock:
            self._samples.extend(samples)
    
    def close(self):
        """Finalize and write the dataset."""
        samples = self._samples
        num_samples = len(samples)
        
        if num_samples == 0:
            raise ValueError("No samples to write")
        
        # Parallel encode
        encoded = self._parallel_encode(samples)
        
        # Write to file
        self._write_file(encoded, num_samples)
    
    def _parallel_encode(self, samples):
        """Encode samples in parallel."""
        
        def encode_one(sample):
            result = {}
            for name, field in self.fields.items():
                value = sample.get(name)
                if value is not None:
                    meta, data = field.encode(value)
                    result[name] = (meta, data)
                else:
                    result[name] = (None, b'')
            return result
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            encoded = list(executor.map(encode_one, samples))
        
        return encoded
    
    def _write_file(self, encoded, num_samples):
        """Write encoded data to file."""
        
        with open(self.output_path, 'wb') as f:
            # Write magic
            f.write(MAGIC)
            
            # Reserve header space
            header_pos = f.tell()
            f.write(b'\x00' * HeaderType.itemsize)
            
            # Write field descriptors
            field_desc_ptr = f.tell()
            for name, field in self.fields.items():
                desc = np.zeros(1, dtype=FieldDescType)[0]
                desc['name'] = name.encode('utf-8')[:63]
                desc['type_id'] = field.type_id
                desc['metadata_size'] = field.metadata_type.itemsize
                
                args = field.to_binary()
                desc['arguments'][:len(args)] = np.frombuffer(args, dtype='u1')
                
                f.write(desc.tobytes())
            
            # Align to page for data region
            data_ptr = self._align_to_page(f.tell())
            f.seek(data_ptr)
            
            # Write data and collect metadata
            metadata = np.zeros(num_samples, dtype=self.metadata_dtype)
            allocations = []
            
            for i, sample_encoded in enumerate(encoded):
                for name in self.fields:
                    meta, data = sample_encoded.get(name, (None, b''))
                    
                    if meta is not None and len(data) > 0:
                        # Update pointer
                        if 'data_ptr' in meta.dtype.names:
                            meta['data_ptr'] = f.tell()
                            allocations.append((f.tell(), len(data)))
                        
                        # Write data
                        f.write(data)
                        
                        # Store metadata
                        metadata[i][name] = meta
            
            # Write metadata table
            metadata_ptr = self._align_to(f.tell(), 64)
            f.seek(metadata_ptr)
            f.write(metadata.tobytes())
            
            # Write allocation table
            alloc_table_ptr = f.tell()
            if allocations:
                alloc_array = np.zeros(len(allocations), dtype=AllocEntryType)
                for i, (ptr, size) in enumerate(allocations):
                    alloc_array[i]['ptr'] = ptr
                    alloc_array[i]['size'] = size
                f.write(alloc_array.tobytes())
            
            # Write header
            header = np.zeros(1, dtype=HeaderType)[0]
            header['version_major'] = VERSION_MAJOR
            header['version_minor'] = VERSION_MINOR
            header['flags'] = Flags.LITTLE_ENDIAN
            header['num_samples'] = num_samples
            header['num_fields'] = len(self.fields)
            header['page_size'] = self.page_size
            header['field_desc_ptr'] = field_desc_ptr
            header['metadata_ptr'] = metadata_ptr
            header['data_ptr'] = data_ptr
            header['alloc_table_ptr'] = alloc_table_ptr
            
            f.seek(header_pos)
            f.write(header.tobytes())
    
    def _align_to_page(self, offset):
        return self._align_to(offset, self.page_size)
    
    def _align_to(self, offset, alignment):
        if offset % alignment == 0:
            return offset
        return offset + (alignment - offset % alignment)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

## Step 6: Dataset Reader

```python
# fastloader/reader.py
"""
Dataset reader with memory-mapped access.
"""

import numpy as np
from typing import Dict, Any, List, Optional

from .types import (
    MAGIC, HeaderType, FieldDescType, AllocEntryType,
    TYPE_REGISTRY
)


class DatasetReader:
    """
    Read samples from the custom format.
    
    Features:
    - Memory-mapped access
    - Lazy loading
    - Field-specific decoding
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._mmap = None
        self._read_header()
    
    def _read_header(self):
        """Read and parse file header."""
        with open(self.file_path, 'rb') as f:
            # Check magic
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"Invalid file format")
            
            # Read header
            header_bytes = f.read(HeaderType.itemsize)
            self.header = np.frombuffer(header_bytes, dtype=HeaderType)[0]
            
            # Read field descriptors
            f.seek(self.header['field_desc_ptr'])
            self.field_descs = np.fromfile(
                f,
                dtype=FieldDescType,
                count=self.header['num_fields']
            )
            
            # Build field handlers
            self.fields = {}
            self.decoders = {}
            
            for desc in self.field_descs:
                name = bytes(desc['name']).rstrip(b'\x00').decode('utf-8')
                type_id = desc['type_id']
                
                if type_id in TYPE_REGISTRY:
                    field_cls = TYPE_REGISTRY[type_id]
                    args = bytes(desc['arguments'])
                    field = field_cls.from_binary(args)
                    self.fields[name] = field
                    self.decoders[name] = field.get_decoder_class()()
            
            # Read metadata table
            f.seek(self.header['metadata_ptr'])
            
            metadata_dtype = np.dtype([
                (name, field.metadata_type)
                for name, field in self.fields.items()
            ], align=True)
            
            self.metadata = np.fromfile(
                f,
                dtype=metadata_dtype,
                count=self.header['num_samples']
            )
            
            # Read allocation table
            f.seek(self.header['alloc_table_ptr'])
            file_size = f.seek(0, 2)  # Get file size
            alloc_size = file_size - self.header['alloc_table_ptr']
            num_allocs = alloc_size // AllocEntryType.itemsize
            
            f.seek(self.header['alloc_table_ptr'])
            self.alloc_table = np.fromfile(f, dtype=AllocEntryType, count=num_allocs)
    
    def __len__(self):
        return int(self.header['num_samples'])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        return self.read_sample(idx)
    
    def read_sample(
        self,
        idx: int,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Read a sample, optionally loading only specific fields."""
        
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        
        if fields is None:
            fields = list(self.fields.keys())
        
        # Ensure mmap is open
        if self._mmap is None:
            self._mmap = np.memmap(self.file_path, dtype='uint8', mode='r')
        
        sample_meta = self.metadata[idx]
        result = {}
        
        for field_name in fields:
            if field_name not in self.fields:
                continue
            
            field_meta = sample_meta[field_name]
            decoder = self.decoders[field_name]
            
            result[field_name] = decoder(field_meta, self._read_data)
        
        return result
    
    def _read_data(self, ptr: int, size: int) -> np.ndarray:
        """Read data from memory map."""
        return self._mmap[ptr:ptr + size]
    
    def close(self):
        """Close the memory map."""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def num_samples(self):
        return len(self)
    
    @property
    def field_names(self):
        return list(self.fields.keys())
```

## Step 7: DataLoader

```python
# fastloader/loader.py
"""
PyTorch-compatible DataLoader.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from .reader import DatasetReader


class DataLoader:
    """
    High-performance data loader.
    
    Features:
    - Multi-threaded prefetching
    - Configurable batching
    - Transform pipeline support
    """
    
    def __init__(
        self,
        reader: DatasetReader,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = False,
        transforms: Optional[Dict[str, Callable]] = None,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: int = 2
    ):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.transforms = transforms or {}
        self.collate_fn = collate_fn or self._default_collate
        self.prefetch_factor = prefetch_factor
        
        self._indices = np.arange(len(reader))
        self._epoch = 0
    
    def __len__(self):
        n = len(self.reader)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        self._epoch += 1
        
        # Shuffle indices
        indices = self._indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Create batches
        batches = []
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if end > len(indices) and self.drop_last:
                break
            batch_indices = indices[start:end]
            batches.append(batch_indices)
        
        # Prefetch with thread pool
        batch_queue = queue.Queue(maxsize=self.prefetch_factor * self.num_workers)
        
        def load_batch(batch_indices):
            samples = []
            for idx in batch_indices:
                sample = self.reader.read_sample(idx)
                
                # Apply transforms
                for field_name, transform in self.transforms.items():
                    if field_name in sample:
                        sample[field_name] = transform(sample[field_name])
                
                samples.append(sample)
            
            return self.collate_fn(samples)
        
        # Producer thread
        def producer():
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for batch_indices in batches:
                    future = executor.submit(load_batch, batch_indices)
                    futures.append(future)
                
                for future in futures:
                    batch = future.result()
                    batch_queue.put(batch)
                
                batch_queue.put(None)  # Sentinel
        
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        # Yield batches
        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            yield batch
        
        producer_thread.join()
    
    def _default_collate(self, samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Default collation: stack arrays."""
        if not samples:
            return {}
        
        batch = {}
        for key in samples[0]:
            values = [s[key] for s in samples]
            
            # Stack if all same shape
            try:
                batch[key] = np.stack(values)
            except ValueError:
                # Variable shapes - keep as list
                batch[key] = values
        
        return batch
```

## Step 8: Usage Example

```python
# example_usage.py
"""
Example: Create and use a multimodal dataset.
"""

from fastloader import DatasetWriter, DatasetReader, DataLoader
from fastloader.fields import JPEGImageField, AudioWaveformField, TokenizedTextField
import numpy as np

# 1. Create dataset
fields = {
    'image': JPEGImageField(quality=90),
    'audio': AudioWaveformField(sample_rate=16000, max_duration=10.0),
    'label': IntField(),
}

with DatasetWriter('train.fld', fields) as writer:
    for i in range(10000):
        writer.write_sample({
            'image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'audio': np.random.randn(16000 * 5).astype(np.float32),
            'label': i % 10,
        })

# 2. Read dataset
reader = DatasetReader('train.fld')
print(f"Dataset has {len(reader)} samples")
print(f"Fields: {reader.field_names}")

# Read single sample
sample = reader.read_sample(0)
print(f"Image shape: {sample['image'].shape}")
print(f"Audio shape: {sample['audio'].shape}")

# 3. Use DataLoader
def normalize_image(img):
    return (img.astype(np.float32) / 255.0 - 0.5) / 0.5

loader = DataLoader(
    reader,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    transforms={
        'image': normalize_image
    }
)

for batch in loader:
    images = batch['image']  # (32, 224, 224, 3)
    audio = batch['audio']   # (32, 80000)
    labels = batch['label']  # (32,)
    
    # Training step...
    break

reader.close()
```

## Benchmarking

```python
# benchmark.py
"""
Benchmark the custom format vs alternatives.
"""

import time
import numpy as np
from pathlib import Path
import tempfile

def benchmark_format(name, create_fn, read_fn, num_samples=1000, num_reads=100):
    """Benchmark a data format."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'data'
        
        # Create dataset
        start = time.perf_counter()
        create_fn(path, num_samples)
        create_time = time.perf_counter() - start
        
        # Random reads
        indices = np.random.randint(0, num_samples, num_reads)
        
        start = time.perf_counter()
        for idx in indices:
            read_fn(path, idx)
        read_time = time.perf_counter() - start
        
        print(f"{name}:")
        print(f"  Create: {create_time:.2f}s ({num_samples/create_time:.0f} samples/s)")
        print(f"  Read:   {read_time*1000/num_reads:.2f}ms/sample")
        print(f"  Size:   {path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "")
```

## Next Steps

1. **Add more field types**: Video, multimodal, embeddings
2. **Implement transforms**: Image augmentation, audio processing
3. **Add compression**: Optional zstd compression for data
4. **GPU integration**: Direct transfer to GPU memory
5. **Distributed support**: Sharding and distributed reading
