# Decoder Architecture: From Bytes to Tensors

## The Decoding Challenge

Decoding must be:
1. **Fast** - No bottleneck in training loop
2. **Memory-efficient** - Minimal allocations
3. **Parallelizable** - Scale with CPU cores
4. **Flexible** - Support transforms/augmentations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DECODER PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw Bytes          Decode          Transform        Batch             │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐        │
│  │ JPEG    │ ──►  │ Decomp  │ ──►  │ Resize  │ ──►  │ Stack   │        │
│  │ bytes   │      │ RGB     │      │ Augment │      │ Tensor  │        │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘        │
│                                                                          │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐           │              │
│  │ Audio   │ ──►  │ Decode  │ ──►  │ Resamp  │ ──────────┤              │
│  │ bytes   │      │ WAV     │      │ MelSpec │           │              │
│  └─────────┘      └─────────┘      └─────────┘           │              │
│                                                           │              │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐           ▼              │
│  │ Text    │ ──►  │ Decode  │ ──►  │ Token   │ ──►  ┌─────────┐        │
│  │ bytes   │      │ UTF-8   │      │ -ize    │      │  GPU    │        │
│  └─────────┘      └─────────┘      └─────────┘      │  Batch  │        │
│                                                      └─────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Abstract Decoder Base

```python
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Any
import numpy as np

class Decoder(ABC):
    """
    Base class for all decoders.
    
    A decoder is responsible for:
    1. Converting raw bytes/metadata to usable data
    2. Allocating output memory
    3. Optionally applying initial transforms
    """
    
    @abstractmethod
    def declare_state_and_memory(
        self,
        previous_state: Any
    ) -> Tuple[Any, Tuple[int, np.dtype]]:
        """
        Declare what memory this decoder needs.
        
        Returns:
            state: Any state to pass through pipeline
            allocation: (size, dtype) for output array
        """
        pass
    
    @abstractmethod
    def generate_code(self) -> Callable:
        """
        Generate the decoding function.
        
        Returns:
            A function that takes (metadata, storage, memory_context)
            and returns decoded data.
        """
        pass
    
    def supports_jit(self) -> bool:
        """Whether this decoder can be JIT compiled with Numba."""
        return True


class FixedSizeDecoder(Decoder):
    """Decoder for fixed-size data (scalars, fixed arrays)."""
    
    def __init__(self, dtype: np.dtype, shape: Tuple[int, ...]):
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.size = np.prod(shape) * self.dtype.itemsize
    
    def declare_state_and_memory(self, previous_state):
        return previous_state, (np.prod(self.shape), self.dtype)
    
    def generate_code(self):
        shape = self.shape
        dtype = self.dtype
        
        def decode(metadata, storage, mem_ctx):
            # metadata contains offset directly to data
            offset = metadata['offset']
            # View as target type
            return storage[offset:offset + dtype.itemsize].view(dtype).reshape(shape)
        
        return decode


class VariableSizeDecoder(Decoder):
    """Decoder for variable-size data (images, audio, text)."""
    
    def __init__(self, max_size: int, dtype: np.dtype):
        self.max_size = max_size
        self.dtype = dtype
    
    def declare_state_and_memory(self, previous_state):
        # Allocate max possible size
        return previous_state, (self.max_size, self.dtype)
    
    def generate_code(self):
        def decode(metadata, storage, mem_ctx):
            ptr = metadata['data_ptr']
            size = metadata['data_size']
            return mem_ctx.read_data(ptr)[:size]
        
        return decode
```

## Modality-Specific Decoders

### Image Decoder

```python
import numpy as np
from typing import Tuple
import numba as nb

class JPEGDecoder(Decoder):
    """
    High-performance JPEG decoder using TurboJPEG.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = None,  # (H, W)
        mode: str = 'RGB'
    ):
        self.output_size = output_size
        self.mode = mode
        self.channels = 3 if mode == 'RGB' else 1
    
    def declare_state_and_memory(self, previous_state):
        if self.output_size:
            H, W = self.output_size
        else:
            # Use max size from previous state
            H, W = previous_state.get('max_size', (256, 256))
        
        shape = (H, W, self.channels)
        return previous_state, (np.prod(shape), np.uint8)
    
    def generate_code(self):
        # Import TurboJPEG at code generation time
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        output_size = self.output_size
        
        def decode(metadata, storage, mem_ctx):
            # Read JPEG bytes
            ptr = metadata['data_ptr']
            size = metadata['data_size']
            jpeg_bytes = bytes(mem_ctx.read_data(ptr)[:size])
            
            # Decode
            rgb = jpeg.decode(jpeg_bytes)
            
            # Resize if needed
            if output_size and (rgb.shape[0] != output_size[0] or 
                               rgb.shape[1] != output_size[1]):
                import cv2
                rgb = cv2.resize(rgb, (output_size[1], output_size[0]))
            
            return rgb
        
        return decode
    
    def supports_jit(self):
        return False  # Can't JIT TurboJPEG calls


class RawImageDecoder(Decoder):
    """
    Decoder for raw pixel data (already decompressed).
    Supports JIT compilation for maximum speed.
    """
    
    def __init__(self, height: int, width: int, channels: int = 3):
        self.height = height
        self.width = width
        self.channels = channels
    
    def declare_state_and_memory(self, previous_state):
        size = self.height * self.width * self.channels
        return previous_state, (size, np.uint8)
    
    def generate_code(self):
        H, W, C = self.height, self.width, self.channels
        
        @nb.njit(cache=True)
        def decode(metadata, storage, mmap_ptr):
            offset = metadata['offset']
            size = H * W * C
            # Direct view, zero copy
            result = storage[offset:offset + size].reshape((H, W, C))
            return result
        
        return decode
```

### Audio Decoder

```python
class AudioWaveformDecoder(Decoder):
    """
    Decoder for raw audio waveforms.
    """
    
    def __init__(
        self,
        max_samples: int,
        sample_rate: int,
        target_sample_rate: int = None,
        mono: bool = True
    ):
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.mono = mono
    
    def declare_state_and_memory(self, previous_state):
        # Calculate output size
        if self.target_sample_rate:
            ratio = self.target_sample_rate / self.sample_rate
            size = int(self.max_samples * ratio)
        else:
            size = self.max_samples
        
        channels = 1 if self.mono else 2
        return previous_state, (size * channels, np.float32)
    
    def generate_code(self):
        sr = self.sample_rate
        target_sr = self.target_sample_rate
        mono = self.mono
        
        def decode(metadata, storage, mem_ctx):
            # Read raw audio
            ptr = metadata['data_ptr']
            num_samples = metadata['num_samples']
            num_channels = metadata['num_channels']
            
            audio = mem_ctx.read_data(ptr).view(np.float32)
            audio = audio[:num_samples * num_channels]
            audio = audio.reshape(-1, num_channels)
            
            # Convert to mono if needed
            if mono and num_channels > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if target_sr and target_sr != sr:
                import librosa
                audio = librosa.resample(
                    audio.flatten(), 
                    orig_sr=sr, 
                    target_sr=target_sr
                )
            
            return audio
        
        return decode


class CompressedAudioDecoder(Decoder):
    """
    Decoder for compressed audio (FLAC, AAC, OGG).
    """
    
    def __init__(
        self,
        codec: str,
        max_duration: float,
        target_sample_rate: int = 16000
    ):
        self.codec = codec
        self.max_duration = max_duration
        self.target_sr = target_sample_rate
        self.max_samples = int(max_duration * target_sample_rate)
    
    def declare_state_and_memory(self, previous_state):
        return previous_state, (self.max_samples, np.float32)
    
    def generate_code(self):
        codec = self.codec
        target_sr = self.target_sr
        
        def decode(metadata, storage, mem_ctx):
            import soundfile as sf
            import io
            
            # Read compressed bytes
            ptr = metadata['data_ptr']
            size = metadata['data_size']
            audio_bytes = bytes(mem_ctx.read_data(ptr)[:size])
            
            # Decode using soundfile
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Resample if needed
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            return audio.astype(np.float32)
        
        return decode


class SpectrogramDecoder(Decoder):
    """
    Decoder that outputs mel spectrograms directly.
    Can either decode pre-computed spectrograms or compute on-the-fly.
    """
    
    def __init__(
        self,
        precomputed: bool = True,
        n_mels: int = 80,
        max_frames: int = 1000
    ):
        self.precomputed = precomputed
        self.n_mels = n_mels
        self.max_frames = max_frames
    
    def declare_state_and_memory(self, previous_state):
        return previous_state, (self.n_mels * self.max_frames, np.float32)
    
    def generate_code(self):
        if self.precomputed:
            n_mels = self.n_mels
            
            def decode(metadata, storage, mem_ctx):
                ptr = metadata['data_ptr']
                num_frames = metadata['num_frames']
                
                spec = mem_ctx.read_data(ptr).view(np.float32)
                return spec[:n_mels * num_frames].reshape(n_mels, num_frames)
            
            return decode
        else:
            # Compute on-the-fly
            n_mels = self.n_mels
            
            def decode(metadata, storage, mem_ctx):
                import librosa
                
                # First decode audio
                ptr = metadata['data_ptr']
                num_samples = metadata['num_samples']
                sr = metadata['sample_rate']
                
                audio = mem_ctx.read_data(ptr).view(np.float32)[:num_samples]
                
                # Compute mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=n_mels
                )
                return librosa.power_to_db(mel)
            
            return decode
```

### Text Decoder

```python
class TextDecoder(Decoder):
    """
    Decoder for text data.
    """
    
    def __init__(
        self,
        max_length: int,
        output_type: str = 'string'  # 'string', 'token_ids', 'bytes'
    ):
        self.max_length = max_length
        self.output_type = output_type
    
    def declare_state_and_memory(self, previous_state):
        if self.output_type == 'token_ids':
            return previous_state, (self.max_length, np.int64)
        else:
            return previous_state, (self.max_length * 4, np.uint8)  # UTF-8
    
    def generate_code(self):
        output_type = self.output_type
        
        def decode(metadata, storage, mem_ctx):
            ptr = metadata['data_ptr']
            size = metadata['data_size']
            
            raw_bytes = mem_ctx.read_data(ptr)[:size]
            
            if output_type == 'bytes':
                return raw_bytes
            elif output_type == 'string':
                return bytes(raw_bytes).decode('utf-8')
            else:  # token_ids
                return raw_bytes.view(np.int64)
        
        return decode


class TokenizedTextDecoder(Decoder):
    """
    Decoder for pre-tokenized text.
    """
    
    def __init__(
        self,
        max_tokens: int,
        vocab_size: int = None  # For validation
    ):
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
    
    def declare_state_and_memory(self, previous_state):
        return previous_state, (self.max_tokens, np.int32)
    
    def generate_code(self):
        max_tokens = self.max_tokens
        
        @nb.njit(cache=True)
        def decode(metadata, storage, mmap_ptr):
            offset = metadata['token_offset']
            num_tokens = metadata['num_tokens']
            
            # Direct view of token IDs
            tokens = storage[offset:offset + num_tokens * 4].view(np.int32)
            
            # Pad to max length
            result = np.zeros(max_tokens, dtype=np.int32)
            result[:num_tokens] = tokens
            
            return result
        
        return decode
```

## Decoder Registry and Factory

```python
class DecoderRegistry:
    """
    Registry for decoder types.
    Maps type IDs to decoder classes.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, type_id: int):
        def decorator(decoder_class):
            cls._registry[type_id] = decoder_class
            return decoder_class
        return decorator
    
    @classmethod
    def get(cls, type_id: int):
        return cls._registry.get(type_id)
    
    @classmethod
    def create_decoder(cls, field_descriptor):
        """Create decoder from field descriptor."""
        type_id = field_descriptor['type_id']
        args = field_descriptor['arguments']
        
        decoder_class = cls.get(type_id)
        if decoder_class is None:
            raise ValueError(f"Unknown type_id: {type_id}")
        
        return decoder_class.from_descriptor(args)


# Register decoders
@DecoderRegistry.register(TYPE_ID_JPEG_IMAGE)
class RegisteredJPEGDecoder(JPEGDecoder):
    @classmethod
    def from_descriptor(cls, args):
        return cls(
            output_size=(args['height'], args['width']),
            mode=args['mode']
        )


@DecoderRegistry.register(TYPE_ID_AUDIO_WAVEFORM)
class RegisteredAudioDecoder(AudioWaveformDecoder):
    @classmethod
    def from_descriptor(cls, args):
        return cls(
            max_samples=args['max_samples'],
            sample_rate=args['sample_rate']
        )
```

## Batched Decoding

```python
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

class BatchDecoder:
    """
    Efficient batch decoding with parallel workers.
    """
    
    def __init__(
        self,
        reader,
        decoders: dict,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.reader = reader
        self.decoders = decoders
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Compile all decoders
        self._compiled = {
            name: dec.generate_code()
            for name, dec in decoders.items()
        }
    
    def decode_batch(
        self,
        sample_ids: list,
        mem_ctx
    ) -> dict:
        """Decode a batch of samples."""
        
        # Pre-allocate output arrays
        outputs = {}
        for name, decoder in self.decoders.items():
            _, (size, dtype) = decoder.declare_state_and_memory({})
            outputs[name] = np.empty(
                (len(sample_ids), size),
                dtype=dtype
            )
        
        # Decode in parallel
        def decode_one(i, sid):
            metadata = self.reader.metadata[sid]
            for name, decode_fn in self._compiled.items():
                field_meta = metadata[name]
                outputs[name][i] = decode_fn(
                    field_meta,
                    mem_ctx.mmap,
                    mem_ctx
                )
        
        with ThreadPoolExecutor(self.num_workers) as executor:
            list(executor.map(
                lambda args: decode_one(*args),
                enumerate(sample_ids)
            ))
        
        return outputs
```

## Next Steps

With decoders defined, we need to:
1. Chain decoders with transforms (see `03_transform_pipeline.md`)
2. JIT compile entire pipelines (see `04_jit_compilation.md`)
3. Manage memory efficiently (see `05_memory_management.md`)
