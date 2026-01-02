# Multimodal Data Design: Combining Modalities

## The Multimodal Challenge

Real-world AI applications often combine multiple modalities:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL APPLICATIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Application              Modalities                                     │
│  ───────────              ──────────                                     │
│                                                                          │
│  Image Captioning         Image + Text                                   │
│  Visual QA                Image + Text (Q) + Text (A)                   │
│  Video Understanding      Video + Audio + Text (subtitles)              │
│  Speech Recognition       Audio + Text (transcript)                     │
│  Document AI              Image (scan) + Text (OCR) + Layout            │
│  Robotics                 Image + Depth + Proprioception + Text         │
│  Music Generation         Audio + MIDI + Text (lyrics)                  │
│                                                                          │
│  Key Challenges:                                                         │
│  ───────────────                                                         │
│                                                                          │
│  1. SYNCHRONIZATION: Video frames must align with audio samples         │
│  2. VARIABLE SIZES: Each modality has different size characteristics   │
│  3. MIXED TYPES: Combine fixed (labels) with variable (images, text)   │
│  4. ALIGNMENT: Maintain correspondence across modalities                │
│  5. EFFICIENCY: Don't let one modality bottleneck the pipeline         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Multimodal Sample Architecture

### Strategy 1: Unified Sample Record

All modalities stored together per sample:

```python
import numpy as np
from typing import Dict, Any, Tuple

class MultimodalSampleFormat:
    """
    Store all modalities for a sample in one contiguous block.
    
    Layout:
    ┌─────────────────────────────────────────────┐
    │ Sample Header                               │
    │ ├─ sample_id: uint32                       │
    │ ├─ num_modalities: uint8                   │
    │ ├─ modality_offsets: uint32[]              │
    │ └─ modality_sizes: uint32[]                │
    ├─────────────────────────────────────────────┤
    │ Modality 0 Data (e.g., Image)              │
    ├─────────────────────────────────────────────┤
    │ Modality 1 Data (e.g., Text)               │
    ├─────────────────────────────────────────────┤
    │ Modality 2 Data (e.g., Audio)              │
    └─────────────────────────────────────────────┘
    
    Pros:
    - Single seek to load all modalities
    - Good cache locality
    - Simple random access
    
    Cons:
    - Must load entire sample even if only need one modality
    - Variable total size
    """
    
    def __init__(self, modality_fields: Dict[str, 'Field']):
        self.fields = modality_fields
        self.modality_names = list(modality_fields.keys())
        self.num_modalities = len(modality_fields)
    
    @property
    def sample_header_type(self) -> np.dtype:
        return np.dtype([
            ('sample_id', '<u4'),
            ('num_modalities', '<u1'),
            ('total_size', '<u4'),
            # Per-modality offsets and sizes stored after header
        ], align=True)
    
    def encode_sample(
        self,
        sample_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, bytes]:
        """Encode all modalities for one sample."""
        
        encoded_modalities = []
        modality_metadata = []
        
        # Encode each modality
        for name in self.modality_names:
            field = self.fields[name]
            data = sample_data.get(name)
            
            if data is not None:
                meta, raw_bytes = field.encode(data)
                encoded_modalities.append(raw_bytes)
                modality_metadata.append((meta, len(raw_bytes)))
            else:
                # Missing modality
                encoded_modalities.append(b'')
                modality_metadata.append((None, 0))
        
        # Build offset table
        offsets = np.zeros(self.num_modalities, dtype='<u4')
        sizes = np.zeros(self.num_modalities, dtype='<u4')
        
        current_offset = 0
        for i, (_, size) in enumerate(modality_metadata):
            offsets[i] = current_offset
            sizes[i] = size
            current_offset += size
        
        # Pack: offsets + sizes + modality data
        header_extension = offsets.tobytes() + sizes.tobytes()
        all_data = header_extension + b''.join(encoded_modalities)
        
        # Sample header
        header = np.zeros(1, dtype=self.sample_header_type)[0]
        header['num_modalities'] = self.num_modalities
        header['total_size'] = len(all_data)
        
        return header, all_data
    
    def decode_sample(
        self,
        header,
        raw_data: bytes,
        modalities_to_load: list = None
    ) -> Dict[str, Any]:
        """Decode sample, optionally loading only specific modalities."""
        
        if modalities_to_load is None:
            modalities_to_load = self.modality_names
        
        # Parse offset table
        offset_size = self.num_modalities * 4
        offsets = np.frombuffer(raw_data[:offset_size], dtype='<u4')
        sizes = np.frombuffer(raw_data[offset_size:offset_size*2], dtype='<u4')
        
        data_start = offset_size * 2
        
        result = {}
        for i, name in enumerate(self.modality_names):
            if name not in modalities_to_load:
                continue
            
            if sizes[i] == 0:
                result[name] = None
                continue
            
            # Extract modality data
            start = data_start + offsets[i]
            end = start + sizes[i]
            modality_bytes = raw_data[start:end]
            
            # Decode
            field = self.fields[name]
            result[name] = field.decode(modality_bytes)
        
        return result
```

### Strategy 2: Separate Streams per Modality

Each modality in its own region of the file:

```python
class SeparateStreamFormat:
    """
    Store each modality in its own region of the file.
    
    File Layout:
    ┌─────────────────────────────────────────────┐
    │ Global Header                               │
    │ ├─ num_samples                             │
    │ ├─ num_modalities                          │
    │ └─ stream_descriptors[]                    │
    ├─────────────────────────────────────────────┤
    │ Stream 0: Images                            │
    │ ├─ Image metadata table                    │
    │ └─ Image data region                       │
    ├─────────────────────────────────────────────┤
    │ Stream 1: Text                              │
    │ ├─ Text metadata table                     │
    │ └─ Text data region                        │
    ├─────────────────────────────────────────────┤
    │ Stream 2: Audio                             │
    │ ├─ Audio metadata table                    │
    │ └─ Audio data region                       │
    └─────────────────────────────────────────────┘
    
    Pros:
    - Load only needed modalities
    - Each stream can be optimized independently
    - Better for heterogeneous access patterns
    
    Cons:
    - Multiple seeks for full sample
    - More complex indexing
    """
    
    @property
    def stream_descriptor_type(self) -> np.dtype:
        return np.dtype([
            ('modality_id', '<u1'),
            ('modality_type', '<u1'),
            ('metadata_ptr', '<u8'),
            ('metadata_size', '<u4'),
            ('data_ptr', '<u8'),
            ('data_size', '<u8'),
        ], align=True)
    
    def __init__(self, modality_fields: Dict[str, 'Field']):
        self.fields = modality_fields
        self.streams = {}
    
    def write(self, samples: list, output_path: str):
        """Write multimodal dataset."""
        
        # Collect data by modality
        modality_data = {name: [] for name in self.fields}
        
        for sample in samples:
            for name in self.fields:
                modality_data[name].append(sample.get(name))
        
        # Write each modality stream
        with open(output_path, 'wb') as f:
            # Reserve space for header
            header_size = self._calculate_header_size()
            f.seek(header_size)
            
            stream_info = []
            
            for name, field in self.fields.items():
                data_list = modality_data[name]
                
                # Write stream
                metadata_ptr = f.tell()
                metadata, data_ptr, data_size = self._write_stream(
                    f, field, data_list
                )
                
                stream_info.append({
                    'name': name,
                    'metadata_ptr': metadata_ptr,
                    'metadata_size': len(metadata),
                    'data_ptr': data_ptr,
                    'data_size': data_size
                })
            
            # Write header
            f.seek(0)
            self._write_header(f, len(samples), stream_info)
    
    def read_stream(
        self,
        file_path: str,
        modality: str,
        sample_indices: list = None
    ):
        """Read specific modality for samples."""
        
        with open(file_path, 'rb') as f:
            # Read header
            header = self._read_header(f)
            
            # Find stream
            stream = header['streams'][modality]
            
            # Read metadata table
            f.seek(stream['metadata_ptr'])
            metadata = np.fromfile(
                f,
                dtype=self.fields[modality].metadata_type,
                count=header['num_samples']
            )
            
            # Read requested samples
            if sample_indices is None:
                sample_indices = range(header['num_samples'])
            
            results = []
            for idx in sample_indices:
                meta = metadata[idx]
                data = self._read_sample_data(f, stream, meta)
                decoded = self.fields[modality].decode(data)
                results.append(decoded)
            
            return results
```

## Video + Audio Synchronization

Critical for video understanding:

```python
class SyncedVideoAudioField:
    """
    Store video and audio with precise synchronization.
    
    Synchronization strategy:
    - Store audio at original sample rate
    - Store video at original frame rate
    - Record precise timing for both
    - Provide aligned access APIs
    """
    
    type_id = 30
    
    def __init__(
        self,
        video_fps: float = 30.0,
        audio_sample_rate: int = 16000,
        max_duration: float = 10.0
    ):
        self.video_fps = video_fps
        self.audio_sr = audio_sample_rate
        self.max_duration = max_duration
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('clip_id', '<u4'),
            ('duration_ms', '<u4'),
            
            # Video info
            ('video_ptr', '<u8'),
            ('video_size', '<u4'),
            ('num_frames', '<u2'),
            ('video_fps', '<f4'),
            ('frame_width', '<u2'),
            ('frame_height', '<u2'),
            
            # Audio info
            ('audio_ptr', '<u8'),
            ('audio_size', '<u4'),
            ('num_audio_samples', '<u4'),
            ('audio_sample_rate', '<u4'),
            ('num_audio_channels', '<u1'),
            
            # Sync info
            ('audio_offset_ms', '<i4'),  # Audio delay relative to video
        ], align=True)
    
    def encode(
        self,
        video_path: str,
        audio_path: str = None,
        start_time: float = 0
    ) -> Tuple[np.ndarray, bytes]:
        """Encode synchronized video and audio."""
        import cv2
        import soundfile as sf
        
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices for target fps
        duration = min(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / original_fps - start_time,
            self.max_duration
        )
        
        num_frames = int(duration * self.video_fps)
        frame_times = np.linspace(0, duration, num_frames)
        
        # Read frames
        frames_data = []
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        for t in frame_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start_time + t) * 1000)
            ret, frame = cap.read()
            if ret:
                # Encode as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frames_data.append(jpeg.tobytes())
        
        cap.release()
        
        # Load audio
        if audio_path is None:
            # Extract from video
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-ar', str(self.audio_sr),
                    '-ac', '1',
                    '-y', tmp.name
                ], check=True, capture_output=True)
                audio, _ = sf.read(tmp.name)
        else:
            audio, orig_sr = sf.read(audio_path)
            # Resample if needed
            if orig_sr != self.audio_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.audio_sr)
        
        # Encode audio
        audio_bytes = audio.astype(np.float32).tobytes()
        
        # Build frame index
        frame_sizes = np.array([len(f) for f in frames_data], dtype='<u4')
        frame_offsets = np.cumsum([0] + list(frame_sizes[:-1])).astype('<u4')
        
        # Pack video data: [frame_offsets | frame_sizes | frame_data]
        video_header = frame_offsets.tobytes() + frame_sizes.tobytes()
        video_data = video_header + b''.join(frames_data)
        
        # Combine: video_data + audio_data
        all_data = video_data + audio_bytes
        
        # Metadata
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['duration_ms'] = int(duration * 1000)
        meta['num_frames'] = len(frames_data)
        meta['video_fps'] = self.video_fps
        meta['video_size'] = len(video_data)
        meta['num_audio_samples'] = len(audio)
        meta['audio_sample_rate'] = self.audio_sr
        meta['audio_size'] = len(audio_bytes)
        meta['audio_offset_ms'] = 0  # Assume sync'd
        
        return meta, all_data


class SyncedVideoAudioDecoder:
    """
    Decoder that returns aligned video and audio.
    """
    
    def __init__(self, target_frames: int = 16, target_audio_len: int = None):
        self.target_frames = target_frames
        self.target_audio_len = target_audio_len
    
    def decode(self, metadata, read_fn) -> Dict[str, np.ndarray]:
        """Decode with temporal alignment."""
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        # Read all data
        total_size = metadata['video_size'] + metadata['audio_size']
        data = read_fn(metadata['data_ptr'], total_size)
        
        # Split video and audio
        video_data = data[:metadata['video_size']]
        audio_data = data[metadata['video_size']:]
        
        # Parse video frames
        num_frames = metadata['num_frames']
        header_size = num_frames * 8  # offsets + sizes
        
        frame_offsets = np.frombuffer(video_data[:num_frames*4], dtype='<u4')
        frame_sizes = np.frombuffer(video_data[num_frames*4:header_size], dtype='<u4')
        frame_data = video_data[header_size:]
        
        # Sample frames uniformly
        if num_frames > self.target_frames:
            indices = np.linspace(0, num_frames - 1, self.target_frames).astype(int)
        else:
            indices = np.arange(num_frames)
        
        # Decode selected frames
        frames = []
        for idx in indices:
            start = frame_offsets[idx]
            size = frame_sizes[idx]
            jpeg_bytes = bytes(frame_data[start:start + size])
            frame = jpeg.decode(jpeg_bytes)
            frames.append(frame)
        
        video_tensor = np.stack(frames)  # (T, H, W, C)
        
        # Decode audio
        audio = np.frombuffer(audio_data, dtype=np.float32)
        
        # Align audio to selected frames
        duration = metadata['duration_ms'] / 1000.0
        sr = metadata['audio_sample_rate']
        
        # Calculate audio segments for each frame
        audio_segments = []
        for i, idx in enumerate(indices):
            frame_time = idx / metadata['video_fps']
            next_frame_time = (indices[i+1] / metadata['video_fps']) if i+1 < len(indices) else duration
            
            start_sample = int(frame_time * sr)
            end_sample = int(next_frame_time * sr)
            
            segment = audio[start_sample:end_sample]
            audio_segments.append(segment)
        
        return {
            'video': video_tensor,          # (T, H, W, C)
            'audio': audio,                  # (num_samples,)
            'audio_segments': audio_segments,# List of per-frame audio
            'frame_times': indices / metadata['video_fps'],
            'duration': duration
        }
```

## Image + Text Pairs

For vision-language models:

```python
class ImageTextPairField:
    """
    Store image-text pairs (for CLIP, BLIP, etc.).
    
    Optimizations:
    - Pre-tokenize text
    - Store image in optimal format for training resolution
    - Include alignment score if available
    """
    
    type_id = 31
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_text_length: int = 77,  # CLIP default
        tokenizer = None,
        store_raw_text: bool = False
    ):
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.tokenizer = tokenizer
        self.store_raw_text = store_raw_text
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('pair_id', '<u4'),
            
            # Image
            ('image_ptr', '<u8'),
            ('image_size', '<u4'),
            ('image_height', '<u2'),
            ('image_width', '<u2'),
            
            # Text (tokenized)
            ('token_ptr', '<u8'),
            ('num_tokens', '<u2'),
            
            # Optional raw text
            ('raw_text_ptr', '<u8'),
            ('raw_text_size', '<u4'),
            
            # Alignment
            ('clip_score', '<f4'),  # Pre-computed CLIP similarity
        ], align=True)
    
    def encode(
        self,
        image: np.ndarray,
        text: str,
        clip_score: float = 0.0
    ) -> Tuple[np.ndarray, bytes]:
        """Encode image-text pair."""
        from turbojpeg import TurboJPEG
        import cv2
        
        jpeg = TurboJPEG()
        
        # Resize and encode image
        h, w = self.image_size
        image_resized = cv2.resize(image, (w, h))
        image_bytes = jpeg.encode(image_resized, quality=95)
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_length,
            padding=False
        )['input_ids']
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()
        
        # Optional raw text
        raw_text_bytes = text.encode('utf-8') if self.store_raw_text else b''
        
        # Combine
        all_data = image_bytes + token_bytes + raw_text_bytes
        
        # Metadata
        meta = np.zeros(1, dtype=self.metadata_type)[0]
        meta['image_size'] = len(image_bytes)
        meta['image_height'] = h
        meta['image_width'] = w
        meta['num_tokens'] = len(tokens)
        meta['raw_text_size'] = len(raw_text_bytes)
        meta['clip_score'] = clip_score
        
        return meta, all_data


class ImageTextDecoder:
    """Decoder for image-text pairs."""
    
    def __init__(
        self,
        max_text_length: int = 77,
        return_raw_text: bool = False
    ):
        self.max_text_length = max_text_length
        self.return_raw_text = return_raw_text
    
    def decode(self, metadata, read_fn) -> dict:
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        # Calculate total size
        total_size = (
            metadata['image_size'] +
            metadata['num_tokens'] * 4 +
            metadata['raw_text_size']
        )
        
        data = read_fn(metadata['data_ptr'], total_size)
        
        # Parse image
        image_end = metadata['image_size']
        image_bytes = bytes(data[:image_end])
        image = jpeg.decode(image_bytes)
        
        # Parse tokens
        token_end = image_end + metadata['num_tokens'] * 4
        tokens = np.frombuffer(data[image_end:token_end], dtype=np.int32)
        
        # Pad tokens
        input_ids = np.zeros(self.max_text_length, dtype=np.int64)
        attention_mask = np.zeros(self.max_text_length, dtype=np.int64)
        
        num_tokens = min(len(tokens), self.max_text_length)
        input_ids[:num_tokens] = tokens[:num_tokens]
        attention_mask[:num_tokens] = 1
        
        result = {
            'image': image,  # (H, W, C)
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'clip_score': metadata['clip_score']
        }
        
        if self.return_raw_text and metadata['raw_text_size'] > 0:
            raw_text = bytes(data[token_end:]).decode('utf-8')
            result['text'] = raw_text
        
        return result
```

## Multimodal Collator

```python
class MultimodalCollator:
    """
    Collate multimodal samples into batches.
    Handles different padding/stacking for each modality.
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, dict],
        return_tensors: str = 'np'
    ):
        """
        modality_configs example:
        {
            'image': {'stack': True},
            'input_ids': {'pad_value': 0, 'max_length': 77},
            'audio': {'pad_value': 0.0, 'max_length': 16000}
        }
        """
        self.configs = modality_configs
        self.return_tensors = return_tensors
    
    def __call__(self, samples: list) -> dict:
        batch = {}
        
        for modality, config in self.configs.items():
            values = [s[modality] for s in samples if modality in s]
            
            if not values:
                continue
            
            if config.get('stack', False):
                # Stack arrays directly
                batch[modality] = np.stack(values)
            
            elif 'pad_value' in config:
                # Pad sequences
                max_len = config.get('max_length', max(len(v) for v in values))
                pad_val = config['pad_value']
                
                padded = np.full(
                    (len(values), max_len),
                    pad_val,
                    dtype=values[0].dtype
                )
                
                for i, v in enumerate(values):
                    length = min(len(v), max_len)
                    padded[i, :length] = v[:length]
                
                batch[modality] = padded
            
            else:
                # Return as list
                batch[modality] = values
        
        if self.return_tensors == 'pt':
            import torch
            batch = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in batch.items()
            }
        
        return batch
```

## Complete Multimodal Loader

```python
class MultimodalLoader:
    """
    DataLoader for multimodal datasets.
    """
    
    def __init__(
        self,
        reader,
        modalities: list,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        collator = None
    ):
        self.reader = reader
        self.modalities = modalities
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collator = collator or MultimodalCollator({})
    
    def __iter__(self):
        indices = np.arange(len(self.reader))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Create batches
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            
            # Load samples
            samples = []
            for idx in batch_indices:
                sample = self.reader.read_sample(
                    idx,
                    modalities=self.modalities
                )
                samples.append(sample)
            
            # Collate
            batch = self.collator(samples)
            
            yield batch
    
    def __len__(self):
        return (len(self.reader) + self.batch_size - 1) // self.batch_size
```

## Next Steps

- See [02_modality_alignment.md](02_modality_alignment.md) for temporal alignment
- See [03_cross_modal_retrieval.md](03_cross_modal_retrieval.md) for retrieval indices
