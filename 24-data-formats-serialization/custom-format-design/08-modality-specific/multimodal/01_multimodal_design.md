# Multimodal Data Design: Combining Modalities

## The Multimodal Landscape

Modern AI systems often need to understand multiple types of data simultaneously. This presents unique challenges for data format design.

### Common Multimodal Combinations

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL APPLICATIONS                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Application           Modalities           Sample Size       Challenges      │
│  ───────────           ──────────           ───────────       ──────────      │
│                                                                                │
│  CLIP/BLIP             Image + Text         ~100 KB           Alignment       │
│  (Vision-Language)     [JPEG] + [tokens]                      (caption to     │
│                                                               image region)   │
│                                                                                │
│  Whisper               Audio + Text         ~50 KB            Synchronization │
│  (Speech-to-Text)      [waveform] + [transcript]              (word timing)   │
│                                                                                │
│  Video Understanding   Video + Audio        ~5 MB             Frame-to-audio  │
│                        + Text               + subtitles       alignment       │
│                                                                                │
│  Document AI           Image + Text         ~500 KB           Spatial layout  │
│  (LayoutLM)            + Layout             + bounding boxes  (text position) │
│                                                                                │
│  Robotics              Images + Depth       ~10 MB            Temporal sync   │
│                        + Proprioception     + sensor data     (sensor fusion) │
│                        + Actions                                              │
│                                                                                │
│  Music Generation      Audio + MIDI         ~1 MB             Beat alignment  │
│                        + Text (lyrics)                        (audio to notes)│
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Challenges

1.  **Size Heterogeneity**: A video frame is ~200KB, its caption is ~100 bytes.
2.  **Temporal Alignment**: Audio samples don't align 1:1 with video frames.
3.  **Missing Modalities**: Not all samples have all modalities.
4.  **Independent Processing**: Each modality needs different transforms.
5.  **Efficiency**: Loading one small modality shouldn't require reading the entire sample.

## Architecture 1: Unified Sample Record

Store all modalities together per sample.

```python
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import struct

class UnifiedMultimodalFormat:
    """
    Store all modalities for a sample in one contiguous record.
    
    File Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ File Header                                                             │
    │ ├─ magic: bytes[4] = "MULT"                                            │
    │ ├─ version: uint16                                                      │
    │ ├─ num_samples: uint32                                                 │
    │ ├─ num_modalities: uint8                                               │
    │ └─ modality_descriptors: ModDesc[num_modalities]                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Sample Index Table (num_samples entries)                                │
    │ └─ Each: (sample_ptr: uint64, sample_size: uint32)                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Sample 0: [Image data][Text data][Audio data]                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Sample 1: [Image data][Text data][Audio data]                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ ...                                                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Sample Internal Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Sample Header                                                           │
    │ ├─ modality_mask: uint32 (bitmask for present modalities)              │
    │ ├─ offsets: uint32[num_modalities] (offset to each modality's data)    │
    │ └─ sizes: uint32[num_modalities] (size of each modality's data)        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Modality 0 Metadata + Data                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Modality 1 Metadata + Data                                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ ...                                                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Advantages:
    ✓ Single seek to load all modalities
    ✓ Good cache locality
    ✓ Simple random access
    ✓ Modality mask handles missing data
    
    Disadvantages:
    ✗ Must read entire sample even if only need one modality
    ✗ Variable total size complicates parallel writing
    """
    
    MAGIC = b'MULT'
    VERSION = 1
    
    def __init__(self, modality_fields: Dict[str, 'Field']):
        """
        Args:
            modality_fields: Dict mapping modality name to Field instance.
                             E.g., {'image': RGBImageField(), 'text': TokenizedTextField()}
        """
        self.fields = modality_fields
        self.modality_names = list(modality_fields.keys())
        self.num_modalities = len(modality_fields)
        
        # Assign bit positions for modality mask
        self.modality_bits = {name: 1 << i for i, name in enumerate(self.modality_names)}
    
    @property
    def modality_descriptor_type(self) -> np.dtype:
        """Descriptor for each modality in the header."""
        return np.dtype([
            ('name_hash', '<u4'),     # Hash of modality name
            ('type_id', '<u2'),       # Field type ID
            ('metadata_size', '<u2'), # Size of per-sample metadata
        ], align=True)
    
    @property
    def sample_index_type(self) -> np.dtype:
        """Index entry for each sample."""
        return np.dtype([
            ('sample_ptr', '<u8'),
            ('sample_size', '<u4'),
            ('_pad', '<u4'),
        ], align=True)
    
    def _compute_sample_header_size(self) -> int:
        """Size of the header within each sample."""
        # modality_mask (4) + offsets (4 * N) + sizes (4 * N)
        return 4 + 4 * self.num_modalities + 4 * self.num_modalities


class UnifiedMultimodalWriter:
    """
    Writer for unified multimodal format.
    """
    
    def __init__(self, format_spec: UnifiedMultimodalFormat, output_path: str):
        self.format = format_spec
        self.output_path = output_path
        self.samples = []
    
    def add_sample(self, sample_data: Dict[str, Any]):
        """Add a sample to the dataset."""
        self.samples.append(sample_data)
    
    def finalize(self):
        """Write the complete dataset."""
        import io
        
        with open(self.output_path, 'wb') as f:
            # Calculate header size
            header_size = self._calculate_header_size()
            index_size = len(self.samples) * self.format.sample_index_type.itemsize
            
            # Reserve space for header and index
            f.seek(header_size + index_size)
            
            # Write samples and build index
            sample_index = np.zeros(len(self.samples), dtype=self.format.sample_index_type)
            
            for i, sample_data in enumerate(self.samples):
                sample_ptr = f.tell()
                sample_bytes = self._encode_sample(sample_data)
                f.write(sample_bytes)
                
                sample_index[i]['sample_ptr'] = sample_ptr
                sample_index[i]['sample_size'] = len(sample_bytes)
            
            # Write header
            f.seek(0)
            self._write_header(f, len(self.samples))
            
            # Write index
            f.write(sample_index.tobytes())
    
    def _encode_sample(self, sample_data: Dict[str, Any]) -> bytes:
        """Encode all modalities for a sample."""
        encoded = {}
        modality_mask = 0
        
        for name, field in self.format.fields.items():
            if name in sample_data and sample_data[name] is not None:
                meta, data = field.encode(sample_data[name])
                encoded[name] = (meta, data)
                modality_mask |= self.format.modality_bits[name]
            else:
                encoded[name] = None
        
        # Build sample header
        header_size = self.format._compute_sample_header_size()
        offsets = np.zeros(self.format.num_modalities, dtype='<u4')
        sizes = np.zeros(self.format.num_modalities, dtype='<u4')
        
        current_offset = header_size
        for i, name in enumerate(self.format.modality_names):
            if encoded[name] is not None:
                meta, data = encoded[name]
                total_size = meta.nbytes + len(data)
                offsets[i] = current_offset
                sizes[i] = total_size
                current_offset += total_size
        
        # Pack sample
        sample_header = struct.pack('<I', modality_mask) + offsets.tobytes() + sizes.tobytes()
        
        data_parts = [sample_header]
        for name in self.format.modality_names:
            if encoded[name] is not None:
                meta, data = encoded[name]
                data_parts.append(meta.tobytes())
                data_parts.append(data)
        
        return b''.join(data_parts)
    
    def _calculate_header_size(self) -> int:
        # magic(4) + version(2) + num_samples(4) + num_modalities(1) + descriptors
        return 4 + 2 + 4 + 1 + self.format.num_modalities * self.format.modality_descriptor_type.itemsize
    
    def _write_header(self, f, num_samples: int):
        f.write(self.format.MAGIC)
        f.write(struct.pack('<H', self.format.VERSION))
        f.write(struct.pack('<I', num_samples))
        f.write(struct.pack('<B', self.format.num_modalities))
        
        for name, field in self.format.fields.items():
            desc = np.zeros(1, dtype=self.format.modality_descriptor_type)[0]
            desc['name_hash'] = hash(name) & 0xFFFFFFFF
            desc['type_id'] = field.TYPE_ID
            desc['metadata_size'] = field.metadata_type.itemsize
            f.write(desc.tobytes())


class UnifiedMultimodalReader:
    """
    Reader for unified multimodal format.
    """
    
    def __init__(self, format_spec: UnifiedMultimodalFormat, file_path: str):
        self.format = format_spec
        self.file_path = file_path
        
        # Memory map the file
        import mmap
        self.file = open(file_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        self._read_header()
    
    def _read_header(self):
        magic = self.mm[:4]
        if magic != self.format.MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        
        version, = struct.unpack('<H', self.mm[4:6])
        self.num_samples, = struct.unpack('<I', self.mm[6:10])
        self.num_modalities, = struct.unpack('<B', self.mm[10:11])
        
        # Read modality descriptors
        desc_size = self.num_modalities * self.format.modality_descriptor_type.itemsize
        header_end = 11 + desc_size
        
        # Read sample index
        index_size = self.num_samples * self.format.sample_index_type.itemsize
        self.sample_index = np.frombuffer(
            self.mm[header_end:header_end + index_size],
            dtype=self.format.sample_index_type
        )
    
    def __len__(self) -> int:
        return self.num_samples
    
    def read_sample(
        self,
        sample_id: int,
        modalities: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Read a sample, optionally loading only specific modalities.
        """
        if modalities is None:
            modalities = self.format.modality_names
        
        # Get sample location
        sample_ptr = self.sample_index[sample_id]['sample_ptr']
        sample_size = self.sample_index[sample_id]['sample_size']
        
        sample_data = self.mm[sample_ptr:sample_ptr + sample_size]
        
        return self._decode_sample(sample_data, modalities)
    
    def _decode_sample(self, sample_data: bytes, modalities: List[str]) -> Dict[str, Any]:
        # Parse sample header
        modality_mask, = struct.unpack('<I', sample_data[:4])
        
        n = self.format.num_modalities
        offsets = np.frombuffer(sample_data[4:4 + 4*n], dtype='<u4')
        sizes = np.frombuffer(sample_data[4 + 4*n:4 + 8*n], dtype='<u4')
        
        result = {}
        for i, name in enumerate(self.format.modality_names):
            if name not in modalities:
                continue
            
            if not (modality_mask & self.format.modality_bits[name]):
                result[name] = None
                continue
            
            # Extract modality data
            start = offsets[i]
            end = start + sizes[i]
            modality_bytes = sample_data[start:end]
            
            # Decode using field
            field = self.format.fields[name]
            meta_size = field.metadata_type.itemsize
            meta = np.frombuffer(modality_bytes[:meta_size], dtype=field.metadata_type)[0]
            data = modality_bytes[meta_size:]
            
            result[name] = field.decode(meta, data)
        
        return result
    
    def close(self):
        self.mm.close()
        self.file.close()
```

## Architecture 2: Separate Streams per Modality

For maximum flexibility, store each modality in its own region.

```python
class SeparateStreamsFormat:
    """
    Store each modality in a separate region of the file.
    
    File Layout:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ File Header                                                             │
    │ ├─ magic: bytes[4] = "MSTR"                                            │
    │ ├─ num_samples: uint32                                                 │
    │ ├─ num_streams: uint8                                                  │
    │ └─ stream_table: StreamDescriptor[num_streams]                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Stream 0: "image"                                                       │
    │ ├─ Metadata Table (num_samples × metadata_size)                        │
    │ └─ Data Region                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Stream 1: "text"                                                        │
    │ ├─ Metadata Table                                                       │
    │ └─ Data Region                                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ ...                                                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Advantages:
    ✓ Load only needed modalities (skip entire streams)
    ✓ Each stream optimized independently
    ✓ Can add new modalities without rewriting others
    ✓ Better for heterogeneous access patterns
    
    Disadvantages:
    ✗ Multiple seeks for full sample
    ✗ More complex indexing
    ✗ Worse cache locality when loading all modalities
    """
    
    MAGIC = b'MSTR'
    
    @property
    def stream_descriptor_type(self) -> np.dtype:
        return np.dtype([
            ('name_hash', '<u4'),
            ('type_id', '<u2'),
            ('_pad', '<u2'),
            ('metadata_ptr', '<u8'),
            ('metadata_size', '<u4'),
            ('data_ptr', '<u8'),
            ('data_size', '<u8'),
        ], align=True)
    
    def __init__(self, modality_fields: Dict[str, 'Field']):
        self.fields = modality_fields
        self.modality_names = list(modality_fields.keys())


class SeparateStreamsReader:
    """
    Reader that can selectively load modalities.
    """
    
    def __init__(self, format_spec: SeparateStreamsFormat, file_path: str):
        self.format = format_spec
        self.file_path = file_path
        
        import mmap
        self.file = open(file_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        self._read_header()
        self._load_stream_metadata()
    
    def _read_header(self):
        # ... (similar to UnifiedMultimodalReader)
        pass
    
    def _load_stream_metadata(self):
        """
        Load metadata tables for all streams.
        
        This enables fast sample lookup without reading data.
        """
        self.metadata_tables = {}
        
        for name, desc in self.stream_descriptors.items():
            field = self.format.fields[name]
            
            # Memory-map the metadata table
            start = desc['metadata_ptr']
            size = self.num_samples * field.metadata_type.itemsize
            
            self.metadata_tables[name] = np.frombuffer(
                self.mm[start:start + size],
                dtype=field.metadata_type
            )
    
    def read_modality(
        self,
        modality: str,
        sample_ids: np.ndarray,
    ) -> List[Any]:
        """
        Read a single modality for multiple samples.
        
        This is efficient because we only touch one stream.
        """
        field = self.format.fields[modality]
        metadata = self.metadata_tables[modality]
        
        results = []
        for sample_id in sample_ids:
            meta = metadata[sample_id]
            ptr = meta['data_ptr']
            size = meta['data_size']
            
            data = self.mm[ptr:ptr + size]
            decoded = field.decode(meta, data)
            results.append(decoded)
        
        return results
    
    def read_sample(
        self,
        sample_id: int,
        modalities: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Read multiple modalities for a sample.
        """
        if modalities is None:
            modalities = self.format.modality_names
        
        result = {}
        for name in modalities:
            items = self.read_modality(name, np.array([sample_id]))
            result[name] = items[0]
        
        return result
```

## Video + Audio Synchronization

The most common multimodal sync challenge: aligning video frames with audio samples.

```python
class SyncedVideoAudioField:
    """
    Store video and audio with precise temporal synchronization.
    
    The challenge: video runs at 30 fps (33ms per frame), audio at 16000 Hz
    (0.0625ms per sample). They don't divide evenly!
    
    Solution: Store both at their native rates with a shared timeline.
    
    Sample Timeline:
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  Time (ms):  0    33    66   100   133   166   200   233   266   300     │
    │  Video:      F0   F1    F2   F3    F4    F5    F6    F7    F8    F9      │
    │  Audio:      [████████████████████████████████████████████████████████]  │
    │              0                         2400                        4800  │
    │              samples                                              samples │
    └───────────────────────────────────────────────────────────────────────────┘
    
    Each video frame corresponds to audio samples in the range:
    Frame i: audio[ floor(i * sr / fps) : floor((i+1) * sr / fps) ]
    """
    
    TYPE_ID = 50
    
    def __init__(
        self,
        video_fps: float = 30.0,
        audio_sample_rate: int = 16000,
        max_frames: int = 300,           # 10 seconds at 30fps
        frame_height: int = 224,
        frame_width: int = 224,
        jpeg_quality: int = 90,
    ):
        self.video_fps = video_fps
        self.audio_sr = audio_sample_rate
        self.max_frames = max_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.jpeg_quality = jpeg_quality
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('clip_id', '<u4'),
            ('duration_ms', '<u4'),
            
            # Video
            ('video_ptr', '<u8'),
            ('video_size', '<u4'),
            ('num_frames', '<u2'),
            ('frame_height', '<u2'),
            ('frame_width', '<u2'),
            ('video_fps', '<f4'),
            
            # Audio
            ('audio_ptr', '<u8'),
            ('audio_size', '<u4'),
            ('num_audio_samples', '<u4'),
            ('audio_sample_rate', '<u4'),
            
            # Sync
            ('audio_offset_samples', '<i4'),  # Audio delay (positive = audio behind)
        ], align=True)
    
    def encode(
        self,
        video_frames: np.ndarray,     # (T, H, W, 3) uint8
        audio: np.ndarray,            # (num_samples,) float32
        audio_offset_ms: float = 0.0, # Sync offset in milliseconds
    ) -> Tuple[np.ndarray, bytes]:
        """
        Encode synchronized video and audio.
        """
        from turbojpeg import TurboJPEG
        import cv2
        
        jpeg = TurboJPEG()
        num_frames = min(len(video_frames), self.max_frames)
        
        # Encode video frames
        frame_data = []
        for i in range(num_frames):
            frame = video_frames[i]
            
            # Resize if needed
            if frame.shape[0] != self.frame_height or frame.shape[1] != self.frame_width:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            jpeg_bytes = jpeg.encode(frame, quality=self.jpeg_quality)
            frame_data.append(jpeg_bytes)
        
        # Build frame index
        frame_offsets = np.zeros(num_frames, dtype='<u4')
        frame_sizes = np.zeros(num_frames, dtype='<u4')
        
        offset = num_frames * 8  # After index
        for i, data in enumerate(frame_data):
            frame_offsets[i] = offset
            frame_sizes[i] = len(data)
            offset += len(data)
        
        video_bytes = frame_offsets.tobytes() + frame_sizes.tobytes() + b''.join(frame_data)
        
        # Encode audio
        audio_bytes = audio.astype(np.float32).tobytes()
        
        # Combine
        all_data = video_bytes + audio_bytes
        
        # Calculate duration
        duration_ms = int(num_frames / self.video_fps * 1000)
        
        # Create metadata
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['duration_ms'] = duration_ms
        meta['video_ptr'] = 0  # Filled by writer
        meta['video_size'] = len(video_bytes)
        meta['num_frames'] = num_frames
        meta['frame_height'] = self.frame_height
        meta['frame_width'] = self.frame_width
        meta['video_fps'] = self.video_fps
        meta['audio_ptr'] = len(video_bytes)  # Relative offset
        meta['audio_size'] = len(audio_bytes)
        meta['num_audio_samples'] = len(audio)
        meta['audio_sample_rate'] = self.audio_sr
        meta['audio_offset_samples'] = int(audio_offset_ms * self.audio_sr / 1000)
        
        return meta, all_data
    
    def get_decoder_class(self):
        return SyncedVideoAudioDecoder


class SyncedVideoAudioDecoder:
    """
    Decoder that provides synchronized video and audio access.
    """
    
    def __init__(
        self,
        field: SyncedVideoAudioField,
        metadata: np.ndarray,
        memory_read,
        target_frames: int = 16,
    ):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        self.target_frames = target_frames
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        target_frames = self.target_frames
        
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        def decode(batch_indices, video_dest, audio_dest, metadata_arg, storage_state):
            """
            Decode synchronized video and audio.
            
            Returns:
                video_dest: (batch, T, H, W, C)
                audio_dest: (batch, num_samples)
            """
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                meta = metadata[sample_id]
                
                ptr = meta['video_ptr']  # Global pointer (set by writer)
                video_size = meta['video_size']
                audio_size = meta['audio_size']
                num_frames = meta['num_frames']
                
                # Read all data
                total_size = video_size + audio_size
                data = mem_read(ptr, storage_state)[:total_size]
                
                video_data = data[:video_size]
                audio_data = data[video_size:]
                
                # Parse video frame index
                frame_offsets = np.frombuffer(video_data[:num_frames * 4], dtype='<u4')
                frame_sizes = np.frombuffer(video_data[num_frames * 4:num_frames * 8], dtype='<u4')
                
                # Sample frames uniformly to target_frames
                if num_frames > target_frames:
                    indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
                else:
                    indices = np.arange(num_frames)
                
                # Decode selected frames
                for i, frame_idx in enumerate(indices):
                    offset = frame_offsets[frame_idx]
                    size = frame_sizes[frame_idx]
                    
                    jpeg_bytes = bytes(video_data[offset:offset + size])
                    frame = jpeg.decode(jpeg_bytes)
                    
                    video_dest[batch_idx, i] = frame
                
                # Zero-pad remaining frames
                for i in range(len(indices), target_frames):
                    video_dest[batch_idx, i] = 0
                
                # Decode audio
                audio = np.frombuffer(audio_data, dtype=np.float32)
                audio_dest[batch_idx, :len(audio)] = audio
                audio_dest[batch_idx, len(audio):] = 0
            
            return video_dest, audio_dest
        
        return decode
    
    def get_aligned_audio(
        self,
        sample_id: int,
        frame_indices: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Get audio segments aligned to specific video frames.
        
        Returns a list of audio arrays, one per frame.
        """
        meta = self.metadata[sample_id]
        
        fps = meta['video_fps']
        sr = meta['audio_sample_rate']
        offset = meta['audio_offset_samples']
        
        # Read audio
        ptr = meta['video_ptr'] + meta['video_size']
        audio_data = self.memory_read(ptr, None)[:meta['audio_size']]
        audio = np.frombuffer(audio_data, dtype=np.float32)
        
        segments = []
        for frame_idx in frame_indices:
            # Time range for this frame
            t_start = frame_idx / fps
            t_end = (frame_idx + 1) / fps
            
            # Convert to sample indices, applying offset
            sample_start = max(0, int(t_start * sr) + offset)
            sample_end = min(len(audio), int(t_end * sr) + offset)
            
            segments.append(audio[sample_start:sample_end])
        
        return segments
```

## Image + Text Pairs (CLIP-style)

```python
class ImageTextPairField:
    """
    Optimized storage for image-text pairs.
    
    Used for vision-language models like CLIP, BLIP, LLaVA.
    
    Pre-tokenize text and pre-compute optional metadata like CLIP scores.
    """
    
    TYPE_ID = 51
    
    def __init__(
        self,
        tokenizer,
        image_size: Tuple[int, int] = (224, 224),
        max_tokens: int = 77,           # CLIP default
        jpeg_quality: int = 95,
        store_clip_score: bool = False,
    ):
        self.tokenizer = tokenizer
        self.image_height, self.image_width = image_size
        self.max_tokens = max_tokens
        self.jpeg_quality = jpeg_quality
        self.store_clip_score = store_clip_score
    
    @property
    def metadata_type(self) -> np.dtype:
        fields = [
            ('pair_id', '<u4'),
            
            # Image
            ('image_ptr', '<u8'),
            ('image_size', '<u4'),
            
            # Text (pre-tokenized)
            ('token_ptr', '<u8'),
            ('num_tokens', '<u2'),
            ('_pad', '<u2'),
        ]
        
        if self.store_clip_score:
            fields.append(('clip_score', '<f4'))
        
        return np.dtype(fields, align=True)
    
    def encode(
        self,
        image: np.ndarray,
        text: str,
        clip_score: float = None,
    ) -> Tuple[np.ndarray, bytes]:
        """Encode image-text pair."""
        from turbojpeg import TurboJPEG
        import cv2
        
        jpeg = TurboJPEG()
        
        # Resize and encode image
        if image.shape[0] != self.image_height or image.shape[1] != self.image_width:
            image = cv2.resize(image, (self.image_width, self.image_height))
        
        image_bytes = jpeg.encode(image, quality=self.jpeg_quality)
        
        # Tokenize text
        tokens = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_tokens,
        )
        token_array = np.array(tokens, dtype='<u2')  # uint16 for most vocabs
        token_bytes = token_array.tobytes()
        
        # Combine
        all_data = image_bytes + token_bytes
        
        # Metadata
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['image_ptr'] = 0
        meta['image_size'] = len(image_bytes)
        meta['token_ptr'] = len(image_bytes)
        meta['num_tokens'] = len(tokens)
        
        if self.store_clip_score and clip_score is not None:
            meta['clip_score'] = clip_score
        
        return meta, all_data


class ImageTextDecoder:
    """Decoder for image-text pairs."""
    
    def __init__(
        self,
        field: ImageTextPairField,
        metadata: np.ndarray,
        memory_read,
    ):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        self.max_tokens = field.max_tokens
        self.pad_token_id = field.tokenizer.pad_token_id or 0
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        max_tokens = self.max_tokens
        pad_id = self.pad_token_id
        
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        def decode(batch_indices, image_dest, token_dest, mask_dest, metadata_arg, storage_state):
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                meta = metadata[sample_id]
                
                ptr = meta['image_ptr']
                image_size = meta['image_size']
                num_tokens = meta['num_tokens']
                
                # Read data
                total_size = image_size + num_tokens * 2
                data = mem_read(ptr, storage_state)[:total_size]
                
                # Decode image
                image_bytes = bytes(data[:image_size])
                image = jpeg.decode(image_bytes)
                image_dest[batch_idx] = image
                
                # Decode tokens
                tokens = np.frombuffer(data[image_size:], dtype='<u2')
                
                # Pad tokens
                token_dest[batch_idx, :num_tokens] = tokens
                token_dest[batch_idx, num_tokens:] = pad_id
                
                # Create attention mask
                mask_dest[batch_idx, :num_tokens] = 1
                mask_dest[batch_idx, num_tokens:] = 0
            
            return image_dest, token_dest, mask_dest
        
        return decode
```

## Multimodal Collation

```python
class MultimodalCollator:
    """
    Collate multimodal batches with modality-specific handling.
    """
    
    def __init__(self, modality_configs: Dict[str, Dict]):
        """
        Args:
            modality_configs: Dict mapping modality name to config.
                Example:
                {
                    'image': {'type': 'stack'},
                    'video': {'type': 'stack'},
                    'input_ids': {'type': 'pad', 'pad_value': 0, 'max_length': 77},
                    'audio': {'type': 'pad', 'pad_value': 0.0, 'max_length': 16000},
                }
        """
        self.configs = modality_configs
    
    def __call__(self, samples: List[Dict]) -> Dict[str, np.ndarray]:
        batch = {}
        
        for modality, config in self.configs.items():
            values = [s[modality] for s in samples if modality in s and s[modality] is not None]
            
            if not values:
                continue
            
            collate_type = config.get('type', 'stack')
            
            if collate_type == 'stack':
                batch[modality] = np.stack(values)
            
            elif collate_type == 'pad':
                pad_value = config['pad_value']
                max_len = config.get('max_length', max(len(v) for v in values))
                
                # Determine shape
                if values[0].ndim == 1:
                    shape = (len(values), max_len)
                else:
                    shape = (len(values), max_len) + values[0].shape[1:]
                
                padded = np.full(shape, pad_value, dtype=values[0].dtype)
                
                for i, v in enumerate(values):
                    length = min(len(v), max_len)
                    padded[i, :length] = v[:length]
                
                batch[modality] = padded
            
            else:
                # Return as list
                batch[modality] = values
        
        return batch
```

## Exercises

1.  **Implement Document AI Field**: Create a field for storing scanned documents with OCR text, bounding boxes, and labels.

2.  **Temporal Alignment Benchmark**: Measure the accuracy of video-to-audio alignment with different sync offset values.

3.  **Selective Loading**: Benchmark the speedup of loading only text vs. loading both image and text in a CLIP dataset.

4.  **Missing Modality Handling**: Extend the decoder to handle samples with missing modalities (e.g., video without audio).
