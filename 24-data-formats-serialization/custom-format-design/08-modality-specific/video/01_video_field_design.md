# Video Field Design: Temporal Data at Scale

## Understanding Video Data

Video is fundamentally different from images or audio. It combines spatial information (each frame is an image) with temporal information (frames change over time).

### The Scale Challenge

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        VIDEO DATA SIZES                                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  RAW VIDEO (uncompressed):                                                     │
│  ─────────────────────────                                                     │
│                                                                                │
│  Resolution     FPS    Bytes/Frame     Bytes/Second     Per Minute            │
│  ──────────     ───    ───────────     ────────────     ──────────            │
│  480p           24     921,600         22 MB            1.3 GB                 │
│  720p           30     2,764,800       83 MB            5.0 GB                 │
│  1080p          30     6,220,800       187 MB           11.2 GB                │
│  1080p          60     6,220,800       373 MB           22.4 GB                │
│  4K             30     24,883,200      746 MB           44.8 GB                │
│  4K             60     24,883,200      1.5 GB           89.6 GB                │
│                                                                                │
│  COMPRESSED VIDEO (H.264, typical settings):                                   │
│  ───────────────────────────────────────────                                   │
│                                                                                │
│  Resolution     FPS    Bitrate         Per Minute       Compression            │
│  ──────────     ───    ───────         ──────────       ───────────            │
│  480p           24     1 Mbps          ~8 MB            ~160x                  │
│  720p           30     3 Mbps          ~23 MB           ~220x                  │
│  1080p          30     5 Mbps          ~38 MB           ~290x                  │
│  1080p          60     8 Mbps          ~60 MB           ~370x                  │
│  4K             30     15 Mbps         ~113 MB          ~400x                  │
│  4K             60     25 Mbps         ~188 MB          ~480x                  │
│                                                                                │
│  DATASET SCALE (example: Kinetics-400):                                        │
│  ───────────────────────────────────────                                       │
│  • 400 action classes                                                          │
│  • ~300,000 clips                                                              │
│  • ~10 seconds per clip                                                        │
│  • At 720p, 30fps: ~300,000 × 23 MB × (10/60) ≈ 1.1 TB                       │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Video Compression: I-Frames and P-Frames

Understanding video codecs is essential for efficient random access.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    VIDEO CODEC STRUCTURE                                       │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Video Stream:                                                                 │
│  ─────────────                                                                 │
│                                                                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐            │
│  │  I  │ │  P  │ │  P  │ │  B  │ │  P  │ │  P  │ │  I  │ │  P  │            │
│  │frame│ │frame│ │frame│ │frame│ │frame│ │frame│ │frame│ │frame│            │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘            │
│     │       │       │       │       │       │       │       │               │
│  [Full]  [Diff]  [Diff]  [Interpolated]    ...   [Full]  [Diff]            │
│  [Image] [From I] [From P]                       [Image] [From I]           │
│                                                                                │
│  FRAME TYPES:                                                                  │
│  ────────────                                                                  │
│                                                                                │
│  I-Frame (Intra-coded, Keyframe):                                             │
│  • Complete image (like a JPEG)                                                │
│  • Large size (~100-500 KB typical)                                           │
│  • Can be decoded independently                                                │
│  • Random access point                                                         │
│                                                                                │
│  P-Frame (Predicted):                                                          │
│  • Stores difference from previous frame                                       │
│  • Small size (~10-50 KB typical)                                             │
│  • REQUIRES previous frame to decode                                          │
│                                                                                │
│  B-Frame (Bi-directional):                                                     │
│  • Uses both past and future frames                                           │
│  • Smallest size (~5-20 KB typical)                                           │
│  • REQUIRES both adjacent frames                                              │
│                                                                                │
│  CONSEQUENCE FOR RANDOM ACCESS:                                                │
│  ───────────────────────────────                                               │
│                                                                                │
│  To decode frame N, you must:                                                  │
│  1. Seek to nearest I-frame before N                                          │
│  2. Decode all frames from I-frame to N                                       │
│                                                                                │
│  If I-frames are every 30 frames (1 second at 30fps):                         │
│  • Worst case: decode 29 extra frames                                          │
│  • Average case: decode ~15 extra frames                                       │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Storage Strategy 1: Pre-Extracted JPEG Frames

The simplest approach: extract all frames as individual JPEGs before training.

```python
import numpy as np
from typing import Tuple, List, Type
import struct

class VideoFramesField:
    """
    Store video as a sequence of JPEG-compressed frames.
    
    Each clip is stored as:
    ┌─────────────────────────────────────────────────────────────┐
    │ Frame Index (num_frames × 8 bytes)                          │
    │ ├─ frame_offset[0]: uint32  (byte offset within data block) │
    │ ├─ frame_size[0]: uint32    (compressed size in bytes)      │
    │ ├─ frame_offset[1]: uint32                                  │
    │ ├─ frame_size[1]: uint32                                    │
    │ └─ ...                                                      │
    ├─────────────────────────────────────────────────────────────┤
    │ JPEG Data (concatenated frames)                             │
    │ [frame 0 JPEG data][frame 1 JPEG data][frame 2 JPEG data]...│
    └─────────────────────────────────────────────────────────────┘
    
    Advantages:
    ✓ Simple random access to any frame
    ✓ Uses existing FFCV image infrastructure
    ✓ No video decoder dependency
    ✓ Parallel decode of all frames
    
    Disadvantages:
    ✗ 3-5x larger than H.264 (no inter-frame compression)
    ✗ Must pre-extract frames (preprocessing step)
    """
    
    TYPE_ID = 30
    
    def __init__(
        self,
        max_frames: int = 32,
        height: int = 224,
        width: int = 224,
        fps: float = 8.0,           # Target frame rate for sampling
        jpeg_quality: int = 90,
    ):
        """
        Args:
            max_frames: Maximum frames to store per clip.
            height, width: Frame dimensions (resized before storage).
            fps: Target frame rate. Videos are temporally subsampled to this rate.
            jpeg_quality: JPEG compression quality (1-100).
        """
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.jpeg_quality = jpeg_quality
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),       # Pointer to frame index + data
            ('data_size', '<u4'),      # Total size (index + all frames)
            ('num_frames', '<u2'),     # Number of frames stored
            ('height', '<u2'),
            ('width', '<u2'),
            ('original_fps', '<f4'),   # FPS of source video
            ('_pad', '<u2'),
        ], align=True)
    
    def encode(self, video_path: str, start_sec: float = 0, end_sec: float = None) -> Tuple[np.ndarray, bytes]:
        """
        Extract and compress frames from a video file.
        
        Args:
            video_path: Path to video file.
            start_sec: Start time in seconds.
            end_sec: End time in seconds (None = to end of video).
        
        Returns:
            (metadata, data_bytes)
        """
        import cv2
        from turbojpeg import TurboJPEG
        
        jpeg = TurboJPEG()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / source_fps
        
        if end_sec is None:
            end_sec = duration
        
        # Calculate which frames to extract
        # We want self.fps frames per second, up to self.max_frames total
        clip_duration = end_sec - start_sec
        total_frames_to_sample = min(
            self.max_frames,
            int(clip_duration * self.fps)
        )
        
        if total_frames_to_sample < 1:
            total_frames_to_sample = 1
        
        # Calculate frame indices to extract
        start_frame = int(start_sec * source_fps)
        end_frame = int(end_sec * source_fps)
        frame_indices = np.linspace(
            start_frame, end_frame - 1, total_frames_to_sample
        ).astype(int)
        
        # Extract and compress frames
        frame_data = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Compress to JPEG
            jpeg_bytes = jpeg.encode(frame, quality=self.jpeg_quality)
            frame_data.append(jpeg_bytes)
        
        cap.release()
        
        num_frames = len(frame_data)
        if num_frames == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Build frame index
        # Each entry: (offset, size) as uint32
        frame_index = np.zeros((num_frames, 2), dtype='<u4')
        current_offset = num_frames * 8  # After the index
        
        for i, data in enumerate(frame_data):
            frame_index[i, 0] = current_offset
            frame_index[i, 1] = len(data)
            current_offset += len(data)
        
        # Combine index + all frame data
        combined_data = frame_index.tobytes() + b''.join(frame_data)
        
        # Create metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0  # Filled by writer
        metadata['data_size'] = len(combined_data)
        metadata['num_frames'] = num_frames
        metadata['height'] = self.height
        metadata['width'] = self.width
        metadata['original_fps'] = source_fps
        
        return metadata, combined_data
    
    def to_binary(self) -> bytes:
        return struct.pack('<IHHHHBI',
            self.max_frames,
            self.height,
            self.width,
            int(self.fps * 100),
            self.jpeg_quality,
            0,  # reserved
            0,  # reserved
        )
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'VideoFramesField':
        max_frames, height, width, fps_100, quality, _, _ = struct.unpack('<IHHHHBI', data[:16])
        return cls(
            max_frames=max_frames,
            height=height,
            width=width,
            fps=fps_100 / 100.0,
            jpeg_quality=quality,
        )
    
    def get_decoder_class(self) -> Type:
        return VideoFramesDecoder


class VideoFramesDecoder:
    """
    Decoder for pre-extracted video frames.
    
    Decodes JPEG frames in parallel using TurboJPEG.
    """
    
    def __init__(self, field: VideoFramesField, metadata: np.ndarray, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        # For allocation
        self.max_frames = int(metadata['num_frames'].max())
        self.height = int(metadata['height'].max())
        self.width = int(metadata['width'].max())
    
    def declare_state_and_memory(self, previous_state):
        from dataclasses import dataclass
        
        @dataclass
        class State:
            shape: tuple
            dtype: np.dtype
            jit_mode: bool
        
        @dataclass
        class AllocationQuery:
            shape: tuple
            dtype: np.dtype
        
        new_state = State(
            shape=(self.max_frames, self.height, self.width, 3),
            dtype=np.uint8,
            jit_mode=False,  # JPEG decode requires C library, not JIT-able
        )
        allocation = AllocationQuery(
            shape=(self.max_frames, self.height, self.width, 3),
            dtype=np.uint8,
        )
        return new_state, allocation
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        
        # Import at code-gen time
        from turbojpeg import TurboJPEG
        jpeg = TurboJPEG()
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            """
            Decode video frames for a batch.
            
            Args:
                batch_indices: Array of sample IDs.
                destination: Output buffer (batch, max_frames, H, W, C).
                metadata_arg: Unused (we use closure).
                storage_state: Memory access tuple.
            
            Returns:
                destination (populated)
            """
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                
                ptr = metadata[sample_id]['data_ptr']
                data_size = metadata[sample_id]['data_size']
                num_frames = metadata[sample_id]['num_frames']
                
                # Read all data
                raw_data = mem_read(ptr, storage_state)[:data_size]
                
                # Parse frame index
                index_size = num_frames * 8
                frame_index = np.frombuffer(
                    raw_data[:index_size], dtype='<u4'
                ).reshape(num_frames, 2)
                
                # Decode each frame
                for frame_idx in range(num_frames):
                    offset = frame_index[frame_idx, 0]
                    size = frame_index[frame_idx, 1]
                    
                    jpeg_bytes = bytes(raw_data[offset:offset + size])
                    frame = jpeg.decode(jpeg_bytes)
                    
                    destination[batch_idx, frame_idx] = frame
                
                # Zero-pad remaining frames
                if num_frames < destination.shape[1]:
                    destination[batch_idx, num_frames:] = 0
            
            return destination[:len(batch_indices)]
        
        return decode
```

## Storage Strategy 2: Compressed Video Segments

For large datasets, store actual compressed video with a keyframe index for efficient seeking.

```python
class CompressedVideoField:
    """
    Store video as compressed segments (H.264/H.265/VP9).
    
    This approach keeps the original inter-frame compression,
    resulting in much smaller storage at the cost of decode complexity.
    
    Storage format:
    ┌─────────────────────────────────────────────────────────────┐
    │ Keyframe Index (num_keyframes × 16 bytes)                   │
    │ ├─ keyframe_pts[0]: int64   (presentation timestamp)        │
    │ ├─ keyframe_offset[0]: uint64 (byte offset in video data)   │
    │ └─ ...                                                      │
    ├─────────────────────────────────────────────────────────────┤
    │ Video Bitstream (raw H.264/H.265/VP9 elementary stream)     │
    └─────────────────────────────────────────────────────────────┘
    
    Advantages:
    ✓ 3-5x smaller than JPEG frames
    ✓ Standard codecs (hardware decode possible)
    
    Disadvantages:
    ✗ Random access only at keyframes
    ✗ Requires video decoder (PyAV, FFmpeg)
    ✗ More complex decode pipeline
    """
    
    TYPE_ID = 31
    
    # Codec identifiers
    CODEC_H264 = 0
    CODEC_H265 = 1
    CODEC_VP9 = 2
    CODEC_AV1 = 3
    
    def __init__(
        self,
        codec: str = 'h264',
        max_duration: float = 10.0,
        target_fps: float = None,    # None = keep original FPS
        target_height: int = None,   # None = keep original
        crf: int = 23,               # Quality (lower = better, 0-51)
        keyframe_interval: int = 30, # Force keyframe every N frames
    ):
        self.codec = codec
        self.max_duration = max_duration
        self.target_fps = target_fps
        self.target_height = target_height
        self.crf = crf
        self.keyframe_interval = keyframe_interval
        
        self.codec_id = {
            'h264': self.CODEC_H264,
            'h265': self.CODEC_H265,
            'vp9': self.CODEC_VP9,
            'av1': self.CODEC_AV1,
        }[codec]
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
            ('codec', '<u1'),
            ('_pad1', '<u1', 3),
            ('num_keyframes', '<u2'),
            ('num_frames', '<u2'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('fps', '<f4'),
            ('duration_ms', '<u4'),
        ], align=True)
    
    def encode(self, video_path: str, start_sec: float = 0, end_sec: float = None) -> Tuple[np.ndarray, bytes]:
        """
        Re-encode video segment with forced keyframes.
        """
        import av
        import io
        
        # Open source video
        input_container = av.open(video_path)
        input_stream = input_container.streams.video[0]
        
        source_fps = float(input_stream.average_rate)
        source_duration = float(input_stream.duration * input_stream.time_base)
        
        if end_sec is None:
            end_sec = min(source_duration, start_sec + self.max_duration)
        else:
            end_sec = min(end_sec, start_sec + self.max_duration)
        
        # Determine output parameters
        output_fps = self.target_fps or source_fps
        output_height = self.target_height or input_stream.height
        output_width = int(output_height * input_stream.width / input_stream.height)
        output_width = (output_width // 2) * 2  # Ensure even
        
        # Create output buffer
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format='mp4')
        
        # Configure output codec
        codec_names = {
            self.CODEC_H264: 'libx264',
            self.CODEC_H265: 'libx265',
            self.CODEC_VP9: 'libvpx-vp9',
            self.CODEC_AV1: 'libaom-av1',
        }
        
        output_stream = output_container.add_stream(
            codec_names[self.codec_id],
            rate=output_fps
        )
        output_stream.width = output_width
        output_stream.height = output_height
        output_stream.pix_fmt = 'yuv420p'
        output_stream.options = {
            'crf': str(self.crf),
            'preset': 'fast',
        }
        
        # Track keyframes
        keyframe_info = []  # (pts, byte_offset)
        frame_count = 0
        
        # Seek to start
        input_container.seek(int(start_sec * av.time_base))
        
        resampler = av.video.reformatter.VideoReformatter()
        
        for frame in input_container.decode(input_stream):
            # Check time bounds
            frame_time = float(frame.pts * input_stream.time_base)
            if frame_time < start_sec:
                continue
            if frame_time >= end_sec:
                break
            
            # Resize if needed
            if frame.width != output_width or frame.height != output_height:
                frame = frame.reformat(width=output_width, height=output_height)
            
            # Force keyframe at interval
            if frame_count % self.keyframe_interval == 0:
                frame.pict_type = 'I'
                keyframe_info.append((frame.pts, output_buffer.tell()))
            
            # Encode
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
            
            frame_count += 1
        
        # Flush
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        output_container.close()
        input_container.close()
        
        # Get video data
        video_bytes = output_buffer.getvalue()
        
        # Build keyframe index
        keyframe_index = np.zeros((len(keyframe_info), 2), dtype='<u8')
        for i, (pts, offset) in enumerate(keyframe_info):
            keyframe_index[i, 0] = pts
            keyframe_index[i, 1] = offset
        
        # Combine: keyframe_index + video_data
        index_bytes = keyframe_index.tobytes()
        combined = index_bytes + video_bytes
        
        # Create metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0
        metadata['data_size'] = len(combined)
        metadata['codec'] = self.codec_id
        metadata['num_keyframes'] = len(keyframe_info)
        metadata['num_frames'] = frame_count
        metadata['width'] = output_width
        metadata['height'] = output_height
        metadata['fps'] = output_fps
        metadata['duration_ms'] = int((end_sec - start_sec) * 1000)
        
        return metadata, combined
    
    def get_decoder_class(self) -> Type:
        return CompressedVideoDecoder


class CompressedVideoDecoder:
    """
    Decoder for compressed video segments.
    
    Uses PyAV for decoding. Can optionally use hardware acceleration.
    """
    
    def __init__(self, field, metadata, memory_read, use_hw_decode=False):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        self.use_hw_decode = use_hw_decode
        
        self.max_frames = int(metadata['num_frames'].max())
        self.height = int(metadata['height'].max())
        self.width = int(metadata['width'].max())
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        
        import av
        import io
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                
                ptr = metadata[sample_id]['data_ptr']
                data_size = metadata[sample_id]['data_size']
                num_keyframes = metadata[sample_id]['num_keyframes']
                num_frames = metadata[sample_id]['num_frames']
                
                # Read data
                raw_data = mem_read(ptr, storage_state)[:data_size]
                
                # Parse keyframe index
                index_size = num_keyframes * 16
                keyframe_index = np.frombuffer(
                    raw_data[:index_size], dtype='<u8'
                ).reshape(num_keyframes, 2)
                
                video_data = bytes(raw_data[index_size:])
                
                # Decode using PyAV
                container = av.open(io.BytesIO(video_data))
                stream = container.streams.video[0]
                
                frame_idx = 0
                for frame in container.decode(stream):
                    if frame_idx >= destination.shape[1]:
                        break
                    
                    # Convert to numpy
                    img = frame.to_ndarray(format='rgb24')
                    destination[batch_idx, frame_idx] = img
                    frame_idx += 1
                
                container.close()
                
                # Zero-pad
                if frame_idx < destination.shape[1]:
                    destination[batch_idx, frame_idx:] = 0
            
            return destination[:len(batch_indices)]
        
        return decode
```

## Video-Specific Transforms

Video transforms must maintain temporal consistency.

```python
import numba as nb
import numpy as np

@nb.njit(parallel=True, nogil=True)
def temporal_subsample(
    input_frames: np.ndarray,   # (T_in, H, W, C)
    output_frames: np.ndarray,  # (T_out, H, W, C)
    target_frames: int,
):
    """
    Uniformly subsample frames to target count.
    """
    t_in = input_frames.shape[0]
    t_out = target_frames
    
    for t in nb.prange(t_out):
        src_t = int(t * t_in / t_out)
        output_frames[t] = input_frames[src_t]
    
    return output_frames


@nb.njit(parallel=True, nogil=True)
def spatial_crop_video(
    input_frames: np.ndarray,   # (T, H_in, W_in, C)
    output_frames: np.ndarray,  # (T, H_out, W_out, C)
    y0: int,
    x0: int,
):
    """
    Apply the SAME spatial crop to all frames.
    
    This is crucial: random cropping must be consistent across frames!
    """
    t, h_out, w_out, c = output_frames.shape
    
    for frame_idx in nb.prange(t):
        for y in range(h_out):
            for x in range(w_out):
                for ch in range(c):
                    output_frames[frame_idx, y, x, ch] = \
                        input_frames[frame_idx, y0 + y, x0 + x, ch]
    
    return output_frames


@nb.njit(parallel=True, nogil=True)
def horizontal_flip_video(
    input_frames: np.ndarray,   # (T, H, W, C)
    output_frames: np.ndarray,  # (T, H, W, C)
):
    """
    Flip all frames horizontally.
    """
    t, h, w, c = input_frames.shape
    
    for frame_idx in nb.prange(t):
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    output_frames[frame_idx, y, x, ch] = \
                        input_frames[frame_idx, y, w - 1 - x, ch]
    
    return output_frames


class VideoTransformPipeline:
    """
    Complete video augmentation pipeline.
    
    All random decisions are made ONCE per clip and applied consistently to all frames.
    """
    
    def __init__(
        self,
        output_frames: int = 16,
        output_size: Tuple[int, int] = (224, 224),
        random_crop: bool = True,
        horizontal_flip: bool = True,
        flip_prob: float = 0.5,
    ):
        self.output_frames = output_frames
        self.output_h, self.output_w = output_size
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.flip_prob = flip_prob
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: Input video (T, H, W, C)
        
        Returns:
            Transformed video (output_frames, output_h, output_w, C)
        """
        t_in, h_in, w_in, c = frames.shape
        
        # Temporal subsample
        if t_in != self.output_frames:
            temp_out = np.empty((self.output_frames, h_in, w_in, c), dtype=frames.dtype)
            temporal_subsample(frames, temp_out, self.output_frames)
            frames = temp_out
        
        # Spatial crop
        if self.random_crop:
            y0 = np.random.randint(0, h_in - self.output_h + 1)
            x0 = np.random.randint(0, w_in - self.output_w + 1)
        else:
            y0 = (h_in - self.output_h) // 2
            x0 = (w_in - self.output_w) // 2
        
        crop_out = np.empty((self.output_frames, self.output_h, self.output_w, c), dtype=frames.dtype)
        spatial_crop_video(frames, crop_out, y0, x0)
        
        # Horizontal flip (with probability)
        if self.horizontal_flip and np.random.random() < self.flip_prob:
            flip_out = np.empty_like(crop_out)
            horizontal_flip_video(crop_out, flip_out)
            return flip_out
        
        return crop_out
```

## Performance Comparison

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    VIDEO FIELD PERFORMANCE COMPARISON                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  For a 10-second clip at 30fps, 720p:                                         │
│                                                                                │
│  Strategy           Storage     Decode Time    Random Access    Use Case      │
│  ────────           ───────     ───────────    ─────────────    ────────      │
│                                                                                │
│  JPEG Frames        50 MB       20 ms          Any frame        Research      │
│  (Q=90, 300 frames)             (parallel)                      Small data    │
│                                                                                │
│  JPEG Frames        12 MB       10 ms          Any frame        Training      │
│  (Q=90, 32 frames)              (parallel)                      Subsampled    │
│                                                                                │
│  H.264 Compressed   5 MB        50 ms          Keyframes        Large data    │
│  (CRF=23)                       (sequential)   (every 30 frames)              │
│                                                                                │
│  H.265 Compressed   3 MB        80 ms          Keyframes        Archive       │
│  (CRF=23)                       (sequential)                                  │
│                                                                                │
│  THROUGHPUT (clips/sec, single thread):                                       │
│                                                                                │
│  JPEG Frames (32f):    50 clips/sec    (bottleneck: TurboJPEG decode)        │
│  H.264 Compressed:     20 clips/sec    (bottleneck: video decode)            │
│                                                                                │
│  WITH HARDWARE DECODE (NVIDIA NVDEC):                                         │
│  H.264 Compressed:     100+ clips/sec  (GPU does the work)                   │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Exercises

1.  **Implement Optical Flow Field**: Store pre-computed optical flow along with keyframes for action recognition.

2.  **Hardware Decode**: Integrate NVIDIA NVDEC for GPU-accelerated video decoding using `decord` library.

3.  **Variable Frame-Rate Sampling**: Implement a decoder that samples frames based on scene changes rather than uniform intervals.

4.  **Benchmark**: Compare storage size and decode speed for JPEG frames vs. H.264 on your specific dataset.
