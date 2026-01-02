# Field Abstraction: The Foundation of Extensible Formats

## What is a "Field"?

In the context of our data format, a **Field** is a handler for a specific type of data (e.g., an image, a label, an audio clip). It encapsulates:
1.  How to **encode** a Python/NumPy object into raw bytes.
2.  How to **decode** raw bytes back into a usable object.
3.  What **metadata** to store for quick lookup.
4.  What **configuration** the field type requires (e.g., JPEG quality).

Using a field abstraction means the core writer and reader logic doesn't need to know *how* to handle images vs. audio. It just calls `field.encode()` and `field.decode()`.

## The Abstract Base Class

Every field type inherits from a common base class. Let's define it completely:

```python
from abc import ABC, abstractmethod
from typing import Type, Tuple, Any
import numpy as np

class Field(ABC):
    """
    Abstract Base Class for all data field types.
    
    A Field is responsible for:
    1. Defining the numpy dtype for its per-sample metadata.
    2. Encoding Python values into (metadata, raw_bytes).
    3. Providing a Decoder class to read data back.
    4. Serializing/deserializing its own configuration.
    
    Subclasses MUST implement all abstract methods.
    """
    
    # Each subclass must define a unique type ID (integer constant).
    # This is stored in the file's field descriptor to identify the type.
    TYPE_ID: int
    
    @property
    @abstractmethod
    def metadata_type(self) -> np.dtype:
        """
        Return the numpy dtype for this field's per-sample metadata.
        
        This dtype defines what is stored in the metadata table for each sample.
        
        For fixed-size data (like labels):
            The metadata IS the data (stored directly).
            Example: np.dtype('<i8') for a 64-bit integer.
        
        For variable-size data (like images):
            The metadata contains a POINTER to the data region + size + any other quick-access info.
            Example: np.dtype([('data_ptr', '<u8'), ('data_size', '<u4'), ('height', '<u2'), ('width', '<u2')])
        """
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, value: Any) -> Tuple[np.ndarray, bytes]:
        """
        Encode a single Python value into binary format.
        
        Args:
            value: The Python object to encode (e.g., a PIL Image, a numpy array, an integer).
        
        Returns:
            Tuple of (metadata, data_bytes):
                metadata: A numpy scalar or array matching self.metadata_type.
                          This will be written to the metadata table.
                data_bytes: Raw bytes to write to the data region.
                            For fixed-size data, this can be b'' (empty).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_decoder_class(self) -> Type:
        """
        Return the Decoder class that knows how to read this field.
        
        A Decoder is an Operation that takes metadata + storage access and produces
        the decoded value. Different decoders might exist for the same field
        (e.g., SimpleImageDecoder vs ResizedCropImageDecoder).
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_binary(self) -> bytes:
        """
        Serialize this field's configuration to bytes.
        
        This is stored in the file's field descriptor so that a reader
        can reconstruct the Field object without external information.
        
        Returns:
            bytes: Up to 128 bytes of configuration data.
        """
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def from_binary(cls, data: bytes) -> 'Field':
        """
        Reconstruct a Field instance from serialized configuration.
        
        Args:
            data: The bytes previously returned by to_binary().
        
        Returns:
            A Field instance with the same configuration.
        """
        raise NotImplementedError


class Decoder(ABC):
    """
    Abstract Base Class for field decoders.
    
    A Decoder is responsible for:
    1. Declaring its memory requirements (output shape, dtype).
    2. Generating code (a function) that performs the decoding.
    """
    
    @abstractmethod
    def declare_state_and_memory(self, previous_state):
        """
        Declare the output state (shape, dtype) and memory allocation.
        
        Args:
            previous_state: The state object from the previous operation in the pipeline.
                           For the first operation (decoder), this is a base state.
        
        Returns:
            Tuple of (new_state, allocation_query):
                new_state: Describes the output (shape, dtype, device, jit_mode).
                allocation_query: Specifies how much memory to pre-allocate.
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_code(self):
        """
        Generate the decoding function.
        
        Returns:
            A function with signature:
                decode(indices, destination, metadata, storage_state) -> result
            
            Where:
                indices: Array of sample IDs to decode this batch.
                destination: Pre-allocated output buffer.
                metadata: The full metadata table (access with metadata[sample_id][field_name]).
                storage_state: Tuple for memory access (mmap, ptrs, sizes).
        """
        raise NotImplementedError
```

## Implementing a Scalar Field: IntField

The simplest case: storing a single integer per sample (like a class label).

```python
import struct

class IntField(Field):
    """
    Field for storing scalar integer values.
    
    The integer is stored DIRECTLY in the metadata table.
    No pointer to the data region is needed because the data is small and fixed-size.
    
    Usage:
        writer = DatasetWriter('data.bin', {'label': IntField()})
        writer.write({'label': 7})
    """
    
    TYPE_ID = 1
    
    def __init__(self, dtype: str = '<i8'):
        """
        Args:
            dtype: NumPy dtype string for the integer.
                   '<i8' = 64-bit signed little-endian (default).
                   '<i4' = 32-bit signed.
                   '<u4' = 32-bit unsigned.
        """
        self._dtype = np.dtype(dtype)
    
    @property
    def metadata_type(self) -> np.dtype:
        # The metadata IS the value itself (no pointer needed)
        return self._dtype
    
    def encode(self, value: int) -> Tuple[np.ndarray, bytes]:
        # Create a numpy scalar with the correct dtype
        metadata = np.array(value, dtype=self._dtype)
        # No separate data bytes needed
        return metadata, b''
    
    def get_decoder_class(self) -> Type:
        return IntDecoder
    
    def to_binary(self) -> bytes:
        # Store the dtype string (max 8 chars, null-padded)
        return self._dtype.str.encode('ascii').ljust(8, b'\x00')
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'IntField':
        dtype_str = data[:8].rstrip(b'\x00').decode('ascii')
        return cls(dtype=dtype_str)


class IntDecoder(Decoder):
    """
    Decoder for IntField.
    
    Since the value is stored directly in metadata, decoding is trivial:
    just copy from metadata to output buffer.
    """
    
    def __init__(self, field: IntField, metadata: np.ndarray):
        """
        Args:
            field: The IntField instance.
            metadata: The full metadata table for this field (1D array of integers).
        """
        self.field = field
        self.metadata = metadata
    
    def declare_state_and_memory(self, previous_state):
        from collections import namedtuple
        State = namedtuple('State', ['shape', 'dtype', 'device', 'jit_mode'])
        AllocationQuery = namedtuple('AllocationQuery', ['shape', 'dtype'])
        
        # Output is a 1D array of integers (one per sample in the batch)
        # Shape is (batch_size,), but we specify None to let the system handle it
        new_state = State(
            shape=(),  # Scalar per sample
            dtype=self.field._dtype,
            device='cpu',
            jit_mode=True  # Can be JIT compiled
        )
        allocation = AllocationQuery(
            shape=(),
            dtype=self.field._dtype
        )
        return new_state, allocation
    
    def generate_code(self):
        """
        Generate a JIT-compatible decoding function.
        """
        metadata = self.metadata  # Capture in closure
        
        def decode(indices, destination, metadata_arg, storage_state):
            """
            Args:
                indices: np.array of sample indices to decode.
                destination: Pre-allocated output buffer, shape (batch_size,).
                metadata_arg: The full metadata table (unused, we use closure).
                storage_state: Tuple for storage access (unused for IntField).
            """
            for ix in range(len(indices)):
                sample_id = indices[ix]
                destination[ix] = metadata[sample_id]
            return destination[:len(indices)]
        
        # Mark as parallelizable (Numba prange can parallelize)
        decode.is_parallel = True
        return decode
```

## Implementing a Variable-Size Field: BytesField

For data that varies in length (images, audio, text), we need a pointer-based approach.

```python
class BytesField(Field):
    """
    Field for storing variable-length byte arrays.
    
    The data is stored in the data region, and the metadata table
    contains a pointer and size.
    
    Usage:
        writer = DatasetWriter('data.bin', {'payload': BytesField()})
        writer.write({'payload': b'Hello, World!'})
    """
    
    TYPE_ID = 3
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),   # Byte offset in the data region
            ('data_size', '<u4'),  # Number of bytes
            ('_pad', '<u4'),       # Padding for alignment
        ], align=True)
    
    def encode(self, value: bytes) -> Tuple[np.ndarray, bytes]:
        """
        Encode a byte string.
        
        Args:
            value: The bytes object to store.
        
        Returns:
            metadata: Has data_ptr=0 (will be set by writer), data_size=len(value).
            data_bytes: The raw bytes to write.
        """
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        # data_ptr will be filled by the writer when it knows the location
        metadata['data_ptr'] = 0  # Placeholder
        metadata['data_size'] = len(value)
        return metadata, value
    
    def get_decoder_class(self) -> Type:
        return BytesDecoder
    
    def to_binary(self) -> bytes:
        return b''  # No configuration needed
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'BytesField':
        return cls()


class BytesDecoder(Decoder):
    """
    Decoder for BytesField.
    
    Reads variable-length bytes from the data region using the pointer in metadata.
    """
    
    def __init__(self, field: BytesField, metadata: np.ndarray, memory_read):
        """
        Args:
            field: The BytesField instance.
            metadata: The full metadata table for this field.
            memory_read: Function(ptr, storage_state) -> bytes to read data.
        """
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        # Find max size to allocate output buffer
        self.max_size = int(metadata['data_size'].max())
    
    def declare_state_and_memory(self, previous_state):
        from collections import namedtuple
        State = namedtuple('State', ['shape', 'dtype', 'device', 'jit_mode'])
        AllocationQuery = namedtuple('AllocationQuery', ['shape', 'dtype'])
        
        new_state = State(
            shape=(self.max_size,),
            dtype=np.dtype('u1'),  # uint8 bytes
            device='cpu',
            jit_mode=True
        )
        allocation = AllocationQuery(
            shape=(self.max_size,),
            dtype=np.dtype('u1')
        )
        return new_state, allocation
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        
        def decode(indices, destination, metadata_arg, storage_state):
            """
            Decode bytes for a batch of samples.
            
            Args:
                indices: Sample IDs to decode.
                destination: Output buffer, shape (batch_size, max_size).
                metadata_arg: Full metadata table.
                storage_state: Tuple (mmap, ptrs, sizes) for memory access.
            """
            for ix in range(len(indices)):
                sample_id = indices[ix]
                ptr = metadata[sample_id]['data_ptr']
                size = metadata[sample_id]['data_size']
                
                # Read from storage (zero-copy view into mmap)
                raw_data = mem_read(ptr, storage_state)
                
                # Copy to destination (up to 'size' bytes)
                destination[ix, :size] = raw_data[:size]
                # Zero-pad the rest (optional, for cleanliness)
                destination[ix, size:] = 0
            
            return destination[:len(indices)]
        
        decode.is_parallel = True
        return decode
```

## The Encode/Decode Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FIELD DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  WRITING (Encoding)                                                              │
│  ─────────────────                                                               │
│                                                                                  │
│  Python Object (e.g., PIL Image, numpy array, bytes)                           │
│       │                                                                          │
│       │ field.encode(value)                                                      │
│       ▼                                                                          │
│  ┌─────────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │ metadata (tuple)│ +   │ data_bytes                                      │   │
│  │ (ptr, size, ... │     │ (raw bytes to store in data region)             │   │
│  └─────────────────┘     └─────────────────────────────────────────────────┘   │
│       │                        │                                                │
│       │ Writer sets ptr        │ Writer stores at current file position        │
│       ▼                        ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                          BINARY FILE                                       │ │
│  │  [...Metadata Table...] [...Data Region...]                              │ │
│  │       ↑                        ↑                                           │ │
│  │    metadata                 data_bytes                                     │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│                                                                                  │
│  READING (Decoding)                                                              │
│  ─────────────────                                                               │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                          BINARY FILE                                       │ │
│  │  [...Metadata Table...] [...Data Region...]                              │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│       │                        │                                                │
│       │ Load at startup        │ memory-mapped (mmap)                          │
│       ▼                        ▼                                                │
│  metadata[sample_id]    mem_read(ptr) -> view into mmap (zero-copy!)           │
│       │                        │                                                │
│       └──────────┬─────────────┘                                                │
│                  │                                                               │
│                  │ decoder.generate_code()                                       │
│                  ▼                                                               │
│  Python Object / Tensor (ready for training)                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## The Type Registry: Mapping IDs to Field Classes

When you read a file, you see a `type_id` in the field descriptor, not a Python class name. You need a registry to look up the class.

```python
# Global registry
FIELD_TYPE_REGISTRY = {}

def register_field_type(type_id: int):
    """
    Decorator to register a field class with a type ID.
    
    Usage:
        @register_field_type(1)
        class IntField(Field):
            ...
    """
    def decorator(cls):
        if type_id in FIELD_TYPE_REGISTRY:
            raise ValueError(f"Type ID {type_id} already registered to {FIELD_TYPE_REGISTRY[type_id]}")
        FIELD_TYPE_REGISTRY[type_id] = cls
        cls.TYPE_ID = type_id
        return cls
    return decorator


def get_field_class(type_id: int) -> Type[Field]:
    """Look up a field class by its type ID."""
    if type_id not in FIELD_TYPE_REGISTRY:
        raise KeyError(f"Unknown field type ID: {type_id}")
    return FIELD_TYPE_REGISTRY[type_id]


# Apply the decorator to our fields
@register_field_type(1)
class IntField(Field):
    # ... (same as before)
    pass

@register_field_type(2)
class FloatField(Field):
    TYPE_ID = 2
    # ... similar to IntField but for floats
    pass

@register_field_type(3)
class BytesField(Field):
    # ... (same as before)
    pass

@register_field_type(10)
class RGBImageField(Field):
    # For images
    pass

@register_field_type(20)
class AudioWaveformField(Field):
    # For audio
    pass
```

## A Complete Image Field Example

Images are complex:
- Variable dimensions.
- Multiple encoding options (raw, JPEG, PNG).
- Potentially require resizing or cropping during decode.

```python
@register_field_type(10)
class RGBImageField(Field):
    """
    Field for storing RGB images.
    
    Supports:
    - JPEG compression (lossy, small files).
    - PNG compression (lossless).
    - Raw storage (fastest decode, largest files).
    """
    
    def __init__(
        self,
        mode: str = 'jpeg',      # 'raw', 'jpeg', 'png'
        quality: int = 90,        # JPEG quality (1-100)
        max_resolution: int = None  # Max dimension (for resizing before storage)
    ):
        self.mode = mode
        self.quality = quality
        self.max_resolution = max_resolution
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
            ('height', '<u2'),
            ('width', '<u2'),
            ('channels', '<u1'),
            ('mode', '<u1'),  # 0=raw, 1=jpeg, 2=png
            ('_pad', '<u2'),
        ], align=True)
    
    def encode(self, image) -> Tuple[np.ndarray, bytes]:
        """
        Encode an image.
        
        Args:
            image: numpy array (H, W, C) with dtype uint8, or PIL Image.
        """
        import numpy as np
        
        # Convert PIL -> numpy if needed
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        # Resize if needed
        if self.max_resolution is not None:
            h, w = image.shape[:2]
            if max(h, w) > self.max_resolution:
                scale = self.max_resolution / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                import cv2
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1
        
        # Encode based on mode
        if self.mode == 'raw':
            data_bytes = image.tobytes()
            mode_code = 0
        elif self.mode == 'jpeg':
            from turbojpeg import TurboJPEG
            jpeg = TurboJPEG()
            data_bytes = jpeg.encode(image, quality=self.quality)
            mode_code = 1
        elif self.mode == 'png':
            import cv2
            _, encoded = cv2.imencode('.png', image)
            data_bytes = encoded.tobytes()
            mode_code = 2
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Create metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0  # Filled by writer
        metadata['data_size'] = len(data_bytes)
        metadata['height'] = h
        metadata['width'] = w
        metadata['channels'] = c
        metadata['mode'] = mode_code
        
        return metadata, data_bytes
    
    def get_decoder_class(self) -> Type:
        return SimpleRGBImageDecoder
    
    def to_binary(self) -> bytes:
        import struct
        # mode (1 byte) + quality (1 byte) + max_resolution (4 bytes, 0 = None)
        mode_byte = {'raw': 0, 'jpeg': 1, 'png': 2}[self.mode]
        max_res = self.max_resolution or 0
        return struct.pack('<BBxxI', mode_byte, self.quality, max_res)
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'RGBImageField':
        import struct
        mode_byte, quality, max_res = struct.unpack('<BBxxI', data[:8])
        mode = {0: 'raw', 1: 'jpeg', 2: 'png'}[mode_byte]
        return cls(mode=mode, quality=quality, max_resolution=max_res or None)


class SimpleRGBImageDecoder(Decoder):
    """Simple decoder that returns the full image."""
    
    def __init__(self, field, metadata, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        # Find max dimensions for buffer allocation
        self.max_h = int(metadata['height'].max())
        self.max_w = int(metadata['width'].max())
        self.max_c = int(metadata['channels'].max())
    
    def declare_state_and_memory(self, previous_state):
        from collections import namedtuple
        State = namedtuple('State', ['shape', 'dtype', 'device', 'jit_mode'])
        AllocationQuery = namedtuple('AllocationQuery', ['shape', 'dtype'])
        
        # Note: JIT mode is False because JPEG decoding requires calling C library
        new_state = State(
            shape=(self.max_h, self.max_w, self.max_c),
            dtype=np.dtype('u1'),
            device='cpu',
            jit_mode=False  # JPEG decode is not JIT-able
        )
        allocation = AllocationQuery(
            shape=(self.max_h, self.max_w, self.max_c),
            dtype=np.dtype('u1')
        )
        return new_state, allocation
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        
        # Import decode libraries once at code-gen time
        from turbojpeg import TurboJPEG
        import cv2
        jpeg = TurboJPEG()
        
        def decode(indices, destination, metadata_arg, storage_state):
            for ix in range(len(indices)):
                sample_id = indices[ix]
                ptr = metadata[sample_id]['data_ptr']
                size = metadata[sample_id]['data_size']
                h = metadata[sample_id]['height']
                w = metadata[sample_id]['width']
                c = metadata[sample_id]['channels']
                mode = metadata[sample_id]['mode']
                
                # Read compressed data
                raw = mem_read(ptr, storage_state)[:size]
                
                # Decode based on mode
                if mode == 0:  # raw
                    image = np.frombuffer(raw, dtype='u1').reshape(h, w, c)
                elif mode == 1:  # jpeg
                    image = jpeg.decode(bytes(raw))
                elif mode == 2:  # png
                    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Unknown image mode: {mode}")
                
                # Copy to destination
                destination[ix, :h, :w, :c] = image
            
            return destination[:len(indices)]
        
        return decode
```

## Exercises

1.  **Implement FloatField**: Similar to IntField but for 64-bit floats. Use `np.dtype('<f8')`.

2.  **Implement NdarrayField**: For storing arbitrary N-dimensional numpy arrays. Store the shape in metadata and serialize dtype as part of `to_binary`.

3.  **Implement TokenizedTextField**: Store a sequence of integer token IDs (like from a tokenizer). Metadata should include `num_tokens` and `pad_token_id`.

4.  **Implement MelSpectrogramField**: For audio ML. Store pre-computed mel spectrograms with shape `(n_mels, time_frames)`. Metadata should include `n_mels`, `time_frames`, `hop_length`, `n_fft`.
