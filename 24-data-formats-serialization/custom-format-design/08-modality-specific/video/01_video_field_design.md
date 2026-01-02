# Video Field Design: Temporal Data at Scale

## Video Data Challenges

Video presents unique challenges compared to images or audio:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VIDEO DATA CHARACTERISTICS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Challenge 1: MASSIVE SIZE                                               │
│  ─────────────────────────                                               │
│                                                                          │
│  1 second of raw video:                                                  │
│  • 1080p @ 30fps: 1920 × 1080 × 3 × 30 = 186 MB/sec                    │
│  • 4K @ 60fps:    3840 × 2160 × 3 × 60 = 1.49 GB/sec                   │
│                                                                          │
│  → Must use compression (H.264/H.265/VP9/AV1)                           │
│  → Typical compression: 100-1000x                                       │
│                                                                          │
│  Challenge 2: TEMPORAL STRUCTURE                                         │
│  ───────────────────────────────                                         │
│                                                                          │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐             │
│  │  I   │  P   │  P   │  P   │  I   │  P   │  P   │  P   │             │
│  │frame │frame │frame │frame │frame │frame │frame │frame │             │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘             │
│     ▲                          ▲                                        │
│   Keyframe                   Keyframe                                   │
│  (standalone)            (standalone)                                   │
│                                                                          │
│  P-frames depend on I-frames: Can't decode P without I!                 │
│  Random access = seek to nearest keyframe first                         │
│                                                                          │
│  Challenge 3: MULTI-STREAM                                               │
│  ─────────────────────────                                               │
│                                                                          │
│  Video files contain multiple streams:                                   │
│  • Video track(s)                                                        │
│  • Audio track(s)                                                        │
│  • Subtitles                                                             │
│  • Metadata                                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Video Storage Strategies

### Strategy 1: Pre-Extracted Frames

Store individual frames, not video files:

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class VideoClipMetadata:
    """Metadata for pre-extracted video clip."""
    video_id: int
    start_frame: int
    num_frames: int
    height: int
    width: int
    fps: float
    # Per-frame pointers (variable due to JPEG)
    frame_ptrs: np.ndarray  # (num_frames,) int64
    frame_sizes: np.ndarray  # (num_frames,) int32


class VideoFramesField:
    """
    Store video as sequence of JPEG frames.
    
    Pros:
    - Simple random access to any frame
    - Same pipeline as images
    - No video decoder dependency
    
    Cons:
    - Larger storage (no inter-frame compression)
    - Must pre-extract all frames
    """
    
    type_id = 10
    
    def __init__(
        self,
        max_frames: int = 16,
        frame_height: int = 224,
        frame_width: int = 224,
        jpeg_quality: int = 95
    ):
        self.max_frames = max_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.jpeg_quality = jpeg_quality
    
    @property
    def metadata_type(self) -> np.dtype:
        """Per-clip metadata."""
        return np.dtype([
            ('video_id', '<u4'),
            ('num_frames', '<u2'),
            ('fps', '<f4'),
            ('height', '<u2'),
            ('width', '<u2'),
            ('frame_ptrs_offset', '<u8'),  # Pointer to frame pointers
            ('total_size', '<u4'),
        ], align=True)
    
    def encode(self, video_path: str, writer) -> Tuple[np.ndarray, bytes]:
        """Extract and encode frames from video."""
        import cv2
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        if total_frames > self.max_frames:
            indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
        else:
            indices = np.arange(total_frames)
        
        # Encode each frame
        frame_data = []
        frame_sizes = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # JPEG compress
            jpeg_bytes = jpeg.encode(frame, quality=self.jpeg_quality)
            frame_data.append(jpeg_bytes)
            frame_sizes.append(len(jpeg_bytes))
        
        cap.release()
        
        # Pack all frames
        num_frames = len(frame_data)
        all_bytes = b''.join(frame_data)
        
        # Create frame pointer table
        frame_ptrs = np.zeros(num_frames, dtype='<u4')
        offset = 0
        for i, size in enumerate(frame_sizes):
            frame_ptrs[i] = offset
            offset += size
        
        # Combine: [frame_ptrs | frame_data]
        ptr_bytes = frame_ptrs.tobytes()
        total_data = ptr_bytes + all_bytes
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['num_frames'] = num_frames
        metadata['fps'] = fps
        metadata['height'] = self.frame_height
        metadata['width'] = self.frame_width
        metadata['total_size'] = len(total_data)
        
        return metadata, total_data
    
    def get_decoder(self):
        """Return decoder for this field."""
        return VideoFramesDecoder(
            self.max_frames,
            self.frame_height,
            self.frame_width
        )


class VideoFramesDecoder:
    """Decoder for pre-extracted video frames."""
    
    def __init__(self, max_frames, height, width):
        self.max_frames = max_frames
        self.height = height
        self.width = width
    
    def decode(self, metadata, read_fn) -> np.ndarray:
        """Decode video clip to numpy array."""
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        
        # Read all data
        ptr = metadata['data_ptr']
        total_size = metadata['total_size']
        data = read_fn(ptr, total_size)
        
        # Parse frame pointers
        num_frames = metadata['num_frames']
        ptr_size = num_frames * 4
        frame_ptrs = np.frombuffer(data[:ptr_size], dtype='<u4')
        frame_data = data[ptr_size:]
        
        # Decode frames
        frames = np.empty(
            (num_frames, self.height, self.width, 3),
            dtype=np.uint8
        )
        
        for i in range(num_frames):
            start = frame_ptrs[i]
            end = frame_ptrs[i + 1] if i + 1 < num_frames else len(frame_data)
            jpeg_bytes = bytes(frame_data[start:end])
            frames[i] = jpeg.decode(jpeg_bytes)
        
        return frames  # (T, H, W, C)
```

### Strategy 2: Compressed Video Segments

Store actual video data with keyframes:

```python
class CompressedVideoField:
    """
    Store video as compressed segments with keyframe index.
    
    Format:
    ┌─────────────────────────────────────────────┐
    │ Segment Header                              │
    │ ├─ codec: uint8                             │
    │ ├─ num_keyframes: uint16                    │
    │ ├─ duration_ms: uint32                      │
    │ └─ keyframe_index: (num_keyframes,) uint32  │
    ├─────────────────────────────────────────────┤
    │ Compressed Video Data                       │
    │ (H.264/H.265/VP9 bitstream)                │
    └─────────────────────────────────────────────┘
    
    Pros:
    - Much smaller storage (inter-frame compression)
    - Standard codecs (hardware decode)
    
    Cons:
    - Random access only at keyframes
    - Requires video decoder (ffmpeg/pyav)
    """
    
    type_id = 11
    CODEC_H264 = 0
    CODEC_H265 = 1
    CODEC_VP9 = 2
    CODEC_AV1 = 3
    
    def __init__(
        self,
        codec: str = 'h264',
        target_fps: float = None,
        max_duration: float = 10.0,
        crf: int = 23,  # Quality (lower = better)
        keyframe_interval: int = 30
    ):
        self.codec = codec
        self.target_fps = target_fps
        self.max_duration = max_duration
        self.crf = crf
        self.keyframe_interval = keyframe_interval
        self.codec_id = {
            'h264': self.CODEC_H264,
            'h265': self.CODEC_H265,
            'vp9': self.CODEC_VP9,
            'av1': self.CODEC_AV1
        }[codec]
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('video_id', '<u4'),
            ('codec', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('fps', '<f4'),
            ('duration_ms', '<u4'),
            ('num_keyframes', '<u2'),
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
        ], align=True)
    
    def encode(self, video_path: str, start_time: float = 0) -> Tuple[np.ndarray, bytes]:
        """Encode video segment."""
        import av
        import io
        
        # Open source
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        # Create output buffer
        output_buffer = io.BytesIO()
        output = av.open(output_buffer, 'w', format='mp4')
        
        # Configure output stream
        codec_name = ['libx264', 'libx265', 'libvpx-vp9', 'libaom-av1'][self.codec_id]
        out_stream = output.add_stream(codec_name, rate=self.target_fps or video_stream.rate)
        out_stream.width = video_stream.width
        out_stream.height = video_stream.height
        out_stream.options = {'crf': str(self.crf)}
        
        # Track keyframes
        keyframe_positions = []
        frame_count = 0
        
        # Seek to start
        container.seek(int(start_time * 1_000_000))
        
        for frame in container.decode(video_stream):
            # Check duration
            if frame.time > start_time + self.max_duration:
                break
            
            # Force keyframe at interval
            if frame_count % self.keyframe_interval == 0:
                keyframe_positions.append(output_buffer.tell())
            
            # Encode
            for packet in out_stream.encode(frame):
                output.mux(packet)
            
            frame_count += 1
        
        # Flush
        for packet in out_stream.encode():
            output.mux(packet)
        
        output.close()
        container.close()
        
        # Get video data
        video_bytes = output_buffer.getvalue()
        
        # Build keyframe index
        keyframe_array = np.array(keyframe_positions, dtype='<u4')
        
        # Combine: keyframe_index + video_data
        combined = keyframe_array.tobytes() + video_bytes
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['codec'] = self.codec_id
        metadata['width'] = out_stream.width
        metadata['height'] = out_stream.height
        metadata['fps'] = float(out_stream.rate)
        metadata['duration_ms'] = int(self.max_duration * 1000)
        metadata['num_keyframes'] = len(keyframe_positions)
        metadata['data_size'] = len(combined)
        
        return metadata, combined
```

### Strategy 3: Optical Flow + Keyframes

For action recognition, store pre-computed optical flow:

```python
class OpticalFlowField:
    """
    Store sparse keyframes + pre-computed optical flow.
    
    Format:
    ┌─────────────────────────────────────────────┐
    │ Keyframe 0 (JPEG)                           │
    ├─────────────────────────────────────────────┤
    │ Flow 0→1 (quantized)                        │
    ├─────────────────────────────────────────────┤
    │ Flow 1→2 (quantized)                        │
    ├─────────────────────────────────────────────┤
    │ ... (every N frames: another keyframe)      │
    └─────────────────────────────────────────────┘
    
    Optical flow: 2 channels (dx, dy) per pixel
    Quantized to int8 for storage (-128 to 127 pixel displacement)
    """
    
    type_id = 12
    
    def __init__(
        self,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
        keyframe_interval: int = 8,
        flow_quality: int = 1  # 0=fast, 1=balanced, 2=accurate
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.keyframe_interval = keyframe_interval
        self.flow_quality = flow_quality
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('video_id', '<u4'),
            ('num_frames', '<u2'),
            ('num_keyframes', '<u2'),
            ('height', '<u2'),
            ('width', '<u2'),
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
        ], align=True)
    
    def encode(self, video_path: str) -> Tuple[np.ndarray, bytes]:
        """Compute and store keyframes + optical flow."""
        import cv2
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        cap = cv2.VideoCapture(video_path)
        
        # Read and resize frames
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.width, self.height))
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Compute optical flow
        data_parts = []
        num_keyframes = 0
        
        prev_gray = None
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store keyframe
            if i % self.keyframe_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                jpeg_bytes = jpeg.encode(rgb, quality=90)
                data_parts.append(np.array([len(jpeg_bytes)], dtype='<u4').tobytes())
                data_parts.append(jpeg_bytes)
                num_keyframes += 1
            
            # Compute and store flow
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                
                # Quantize to int8
                flow_quantized = np.clip(flow * 4, -128, 127).astype(np.int8)
                data_parts.append(flow_quantized.tobytes())
            
            prev_gray = gray
        
        # Combine all data
        all_data = b''.join(data_parts)
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['num_frames'] = len(frames)
        metadata['num_keyframes'] = num_keyframes
        metadata['height'] = self.height
        metadata['width'] = self.width
        metadata['data_size'] = len(all_data)
        
        return metadata, all_data
    
    def decode(self, metadata, read_fn) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keyframes and flow."""
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        
        # Read data
        data = read_fn(metadata['data_ptr'], metadata['data_size'])
        
        num_frames = metadata['num_frames']
        num_keyframes = metadata['num_keyframes']
        h, w = metadata['height'], metadata['width']
        
        keyframes = []
        flows = []
        offset = 0
        
        for i in range(num_frames):
            # Keyframe?
            if i % (num_frames // num_keyframes) == 0 and len(keyframes) < num_keyframes:
                # Read size
                size = np.frombuffer(data[offset:offset+4], dtype='<u4')[0]
                offset += 4
                
                # Decode JPEG
                jpeg_bytes = bytes(data[offset:offset+size])
                keyframe = jpeg.decode(jpeg_bytes)
                keyframes.append(keyframe)
                offset += size
            
            # Flow (except for last frame)
            if i < num_frames - 1:
                flow_size = h * w * 2
                flow = np.frombuffer(
                    data[offset:offset+flow_size],
                    dtype=np.int8
                ).reshape(h, w, 2)
                flows.append(flow.astype(np.float32) / 4.0)
                offset += flow_size
        
        return np.array(keyframes), np.array(flows)
```

## Video-Specific Augmentations

```python
import numba as nb
import numpy as np

@nb.njit(parallel=True)
def temporal_crop(
    frames: np.ndarray,  # (T, H, W, C)
    output: np.ndarray,  # (T', H, W, C)
    start_frame: int
):
    """Crop temporal window."""
    out_t = output.shape[0]
    for t in nb.prange(out_t):
        output[t] = frames[start_frame + t]
    return output


@nb.njit(parallel=True)
def temporal_subsample(
    frames: np.ndarray,  # (T, H, W, C)
    output: np.ndarray,  # (T', H, W, C)
    stride: int
):
    """Subsample frames with stride."""
    out_t = output.shape[0]
    for t in nb.prange(out_t):
        output[t] = frames[t * stride]
    return output


@nb.njit(parallel=True)
def spatial_temporal_crop(
    frames: np.ndarray,  # (T, H, W, C)
    output: np.ndarray,  # (T, H', W', C)
    y_start: int,
    x_start: int
):
    """Random spatial crop applied consistently across frames."""
    t, h_out, w_out, c = output.shape
    
    for frame_idx in nb.prange(t):
        for y in range(h_out):
            for x in range(w_out):
                for ch in range(c):
                    output[frame_idx, y, x, ch] = \
                        frames[frame_idx, y_start + y, x_start + x, ch]
    
    return output


class VideoAugmentation:
    """Augmentations for video data."""
    
    def __init__(
        self,
        temporal_crop_size: int = 8,
        spatial_crop_size: Tuple[int, int] = (224, 224),
        horizontal_flip_prob: float = 0.5
    ):
        self.temporal_crop_size = temporal_crop_size
        self.spatial_crop_size = spatial_crop_size
        self.flip_prob = horizontal_flip_prob
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentations."""
        T, H, W, C = frames.shape
        
        # Temporal crop
        if T > self.temporal_crop_size:
            start = np.random.randint(0, T - self.temporal_crop_size)
            frames = frames[start:start + self.temporal_crop_size]
        
        # Spatial crop
        h_crop, w_crop = self.spatial_crop_size
        y_start = np.random.randint(0, H - h_crop + 1)
        x_start = np.random.randint(0, W - w_crop + 1)
        
        output = np.empty((self.temporal_crop_size, h_crop, w_crop, C), dtype=frames.dtype)
        spatial_temporal_crop(frames, output, y_start, x_start)
        
        # Horizontal flip
        if np.random.random() < self.flip_prob:
            output = output[:, :, ::-1, :]
        
        return output
```

## Performance Considerations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  VIDEO LOADING PERFORMANCE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Strategy            Storage    Decode Speed    Random Access           │
│  ──────────          ───────    ────────────    ─────────────           │
│  JPEG Frames         Large      Fast (TurboJPEG)    ✓ Any frame        │
│  Compressed Video    Small      Slower (FFmpeg)     △ Keyframes        │
│  Optical Flow        Medium     Fast               ✓ Precomputed       │
│                                                                          │
│  Recommendations:                                                        │
│  ────────────────                                                        │
│                                                                          │
│  Training:                                                               │
│  • Use JPEG frames if storage allows                                    │
│  • Pre-extract at target resolution                                     │
│  • Use optical flow for action recognition                              │
│                                                                          │
│  Large datasets:                                                         │
│  • Compressed segments with dense keyframes                             │
│  • GPU video decode (NVIDIA NVDEC)                                      │
│                                                                          │
│  Real-time:                                                              │
│  • Hardware video decoder                                               │
│  • Decode on GPU directly                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Next Steps

- See [02_video_transforms.md](02_video_transforms.md) for video-specific transforms
- See [03_video_batch_loading.md](03_video_batch_loading.md) for efficient batch loading
- See [../multimodal/01_video_audio_sync.md](../multimodal/01_video_audio_sync.md) for A/V synchronization
