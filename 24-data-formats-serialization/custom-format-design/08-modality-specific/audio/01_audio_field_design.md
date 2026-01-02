# Audio Field Design: A Comprehensive Guide

## Audio Data Fundamentals

Before designing audio fields, we must understand audio data at a fundamental level.

### What is Digital Audio?

Digital audio is a sequence of **samples**: measurements of sound pressure (amplitude) taken at regular intervals.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         DIGITAL AUDIO FUNDAMENTALS                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ANALOG (continuous)                     DIGITAL (discrete)                   │
│                                                                                │
│  Sound Wave:                             Samples:                              │
│       │    /\                                 │  ●   ●                        │
│       │   /  \    /\                          │ ● \ / ●  ●                    │
│  ─────┼──/────\──/──\──────              ─────┼●───●───●──●──                │
│       │ /      \/    \                        │      ●     ●                  │
│       │/                                      │                                │
│                                                                                │
│  Each ● is a SAMPLE: a number representing amplitude at that moment.         │
│                                                                                │
│  KEY PARAMETERS:                                                               │
│  ───────────────                                                               │
│                                                                                │
│  Sample Rate (Hz)      Samples per second           Typical values            │
│  ─────────────────     ──────────────────           ─────────────             │
│  8,000 Hz              8,000 samples/sec            Telephone                  │
│  16,000 Hz             16,000 samples/sec           Speech recognition         │
│  44,100 Hz             44,100 samples/sec           CD audio                   │
│  48,000 Hz             48,000 samples/sec           Professional audio         │
│                                                                                │
│  Bit Depth             Precision of each sample     Storage per sample        │
│  ─────────             ────────────────────────     ─────────────────         │
│  8-bit                 256 levels                   1 byte                     │
│  16-bit                65,536 levels                2 bytes (standard)         │
│  24-bit                16.7 million levels          3 bytes (pro audio)        │
│  32-bit float          Infinite (floating point)    4 bytes (processing)       │
│                                                                                │
│  Channels              Number of independent streams                           │
│  ────────              ─────────────────────────────                          │
│  Mono                  1 channel                                               │
│  Stereo                2 channels (left + right)                               │
│  Surround              5.1, 7.1 channels                                       │
│                                                                                │
│  STORAGE CALCULATION:                                                          │
│  ────────────────────                                                          │
│  Bytes = sample_rate × bit_depth/8 × channels × duration_seconds              │
│                                                                                │
│  Example: 10 seconds, 16kHz, 16-bit mono                                       │
│  = 16000 × 2 × 1 × 10 = 320,000 bytes = 320 KB                                │
│                                                                                │
│  Example: 1 hour, 48kHz, 16-bit stereo                                        │
│  = 48000 × 2 × 2 × 3600 = 691,200,000 bytes = 691 MB                          │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Frequency Domain Representations

Many audio models work in the **frequency domain** rather than the time domain.

```
TIME DOMAIN (Waveform):
    Amplitude
       │  /\    /\    /\    /\
    ───┼─/──\──/──\──/──\──/──\───▶ Time
       │/    \/    \/    \/

           FFT (Fast Fourier Transform)
                    │
                    ▼

FREQUENCY DOMAIN (Spectrum):
    Magnitude
       │  ▄
       │  █      ▄
       │ ▄█▄   ▄▄█▄
    ───┼█████▄▄█████▄────────────▶ Frequency
       0    1kHz    5kHz   10kHz
```

**Common Representations:**

| Name | Shape | Description | Use Case |
|------|-------|-------------|----------|
| Waveform | `(samples,)` | Raw audio | WaveNet, raw audio models |
| Spectrogram | `(freq_bins, time_frames)` | Magnitude of STFT | Visualization, some models |
| Mel Spectrogram | `(n_mels, time_frames)` | Log-scaled frequency | ASR, TTS, audio classification |
| MFCC | `(n_mfcc, time_frames)` | Cepstral coefficients | Classic speech recognition |
| Codec Tokens | `(n_codebooks, frames)` | Discrete tokens | Audio language models |

## Storage Strategy Decision

Choose your storage strategy based on your use case:

```python
def choose_audio_storage_strategy(
    use_case: str,
    avg_duration_seconds: float,
    num_samples: int,
    disk_budget_gb: float,
    training_model: str,
) -> str:
    """
    Decide which audio storage strategy to use.
    
    Returns one of: 'raw', 'compressed', 'spectrogram', 'codec_tokens'
    """
    raw_size_gb = (
        16000 * 2  # 16kHz, 16-bit
        * avg_duration_seconds 
        * num_samples 
        / 1e9
    )
    
    # If model needs waveforms and storage allows
    if training_model in ['wavenet', 'hifi-gan', 'voicebox'] and raw_size_gb < disk_budget_gb:
        return 'raw'
    
    # If model uses spectrograms (most common for ASR/TTS)
    if training_model in ['whisper', 'tacotron', 'vits', 'conformer']:
        return 'spectrogram'  # ~5-10x smaller than raw
    
    # If model is audio language model
    if training_model in ['audioflamingo', 'musiclm', 'audiolm']:
        return 'codec_tokens'  # ~20x smaller than raw
    
    # If storage is tight, use compression
    if raw_size_gb > disk_budget_gb:
        if use_case == 'research':
            return 'compressed_lossless'  # FLAC: ~2x compression
        else:
            return 'compressed_lossy'  # Opus: ~10x compression
    
    return 'raw'
```

## Field Implementation 1: Raw Waveform

The simplest case: store the raw PCM samples.

```python
import numpy as np
from typing import Type, Tuple, Optional
import struct

class AudioWaveformField:
    """
    Field for storing raw PCM audio waveforms.
    
    The waveform is stored as a contiguous array of samples.
    Metadata includes sample count, sample rate, and format.
    
    Advantages:
    - Zero decode latency
    - Direct access for time-domain models
    - No quality loss
    
    Disadvantages:
    - Largest storage (2 bytes/sample at 16-bit)
    - I/O bound for longer audio
    """
    
    TYPE_ID = 20  # Unique identifier for this field type
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        dtype: str = 'int16',
        normalize: bool = False,
    ):
        """
        Args:
            sample_rate: Target sample rate. Audio will be resampled if needed.
            mono: If True, convert stereo to mono.
            dtype: Storage dtype ('int16' for compact, 'float32' for precision).
            normalize: If True, normalize audio to [-1, 1] before storing.
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.dtype = np.dtype(dtype)
        self.normalize = normalize
    
    @property
    def metadata_type(self) -> np.dtype:
        """
        Define the per-sample metadata structure.
        """
        return np.dtype([
            ('data_ptr', '<u8'),       # Byte offset to waveform data
            ('num_samples', '<u4'),    # Number of audio samples
            ('duration_ms', '<u4'),    # Duration in milliseconds (for quick filtering)
            ('sample_rate', '<u4'),    # Sample rate (Hz)
            ('channels', '<u1'),       # Number of channels
            ('dtype_code', '<u1'),     # 0=int16, 1=float32, 2=int32
            ('_pad', '<u2'),           # Padding for alignment
        ], align=True)
    
    def encode(self, audio: np.ndarray, sample_rate: int = None) -> Tuple[np.ndarray, bytes]:
        """
        Encode an audio waveform.
        
        Args:
            audio: Audio array, shape (samples,) or (samples, channels)
            sample_rate: Original sample rate (if resampling needed)
        
        Returns:
            (metadata, data_bytes)
        """
        # Handle sample rate
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)
        
        # Ensure 2D: (samples, channels)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        # Convert to mono if needed
        if self.mono and audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True).astype(audio.dtype)
        
        # Normalize if requested
        if self.normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        # Convert to storage dtype
        if self.dtype == np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        else:
            audio = audio.astype(self.dtype)
        
        # Create metadata
        num_samples, channels = audio.shape
        duration_ms = int(num_samples / self.sample_rate * 1000)
        
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0  # Will be filled by writer
        metadata['num_samples'] = num_samples
        metadata['duration_ms'] = duration_ms
        metadata['sample_rate'] = self.sample_rate
        metadata['channels'] = channels
        metadata['dtype_code'] = {np.int16: 0, np.float32: 1, np.int32: 2}.get(self.dtype.type, 0)
        
        # Convert to bytes
        data_bytes = audio.tobytes()
        
        return metadata, data_bytes
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        import scipy.signal
        
        if orig_sr == target_sr:
            return audio
        
        # Calculate new length
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        
        # Resample
        resampled = scipy.signal.resample(audio, new_length)
        
        return resampled.astype(audio.dtype)
    
    def to_binary(self) -> bytes:
        """Serialize field configuration."""
        return struct.pack('<IIBB',
            self.sample_rate,
            1 if self.mono else 0,
            {np.int16: 0, np.float32: 1, np.int32: 2}.get(self.dtype.type, 0),
            1 if self.normalize else 0,
        )
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'AudioWaveformField':
        """Deserialize field configuration."""
        sr, mono, dtype_code, normalize = struct.unpack('<IIBB', data[:10])
        dtype = {0: 'int16', 1: 'float32', 2: 'int32'}[dtype_code]
        return cls(sample_rate=sr, mono=bool(mono), dtype=dtype, normalize=bool(normalize))
    
    def get_decoder_class(self) -> Type:
        return AudioWaveformDecoder


class AudioWaveformDecoder:
    """Decoder for raw audio waveforms."""
    
    def __init__(self, field: AudioWaveformField, metadata: np.ndarray, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        # Find max dimensions for buffer allocation
        self.max_samples = int(metadata['num_samples'].max())
        self.max_channels = int(metadata['channels'].max())
        self.dtype = {0: np.int16, 1: np.float32, 2: np.int32}[
            int(metadata['dtype_code'][0])
        ]
    
    def declare_state_and_memory(self, previous_state):
        from dataclasses import dataclass
        
        @dataclass
        class State:
            shape: tuple
            dtype: np.dtype
            jit_mode: bool
            sample_rate: int
        
        @dataclass
        class AllocationQuery:
            shape: tuple
            dtype: np.dtype
        
        new_state = State(
            shape=(self.max_samples, self.max_channels),
            dtype=self.dtype,
            jit_mode=True,
            sample_rate=self.field.sample_rate,
        )
        allocation = AllocationQuery(
            shape=(self.max_samples, self.max_channels),
            dtype=self.dtype,
        )
        return new_state, allocation
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        dtype = self.dtype
        bytes_per_sample = np.dtype(dtype).itemsize
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            """
            Decode a batch of audio waveforms.
            
            Args:
                batch_indices: Array of sample IDs to decode.
                destination: Pre-allocated output buffer (batch, max_samples, max_channels).
                metadata_arg: The metadata table (unused, we use closure).
                storage_state: Tuple for memory access.
            
            Returns:
                destination (populated with decoded audio)
            """
            for ix in range(len(batch_indices)):
                sample_id = batch_indices[ix]
                
                # Get metadata
                ptr = metadata[sample_id]['data_ptr']
                num_samples = metadata[sample_id]['num_samples']
                channels = metadata[sample_id]['channels']
                
                # Calculate byte size
                byte_size = num_samples * channels * bytes_per_sample
                
                # Read raw bytes
                raw_bytes = mem_read(ptr, storage_state)[:byte_size]
                
                # View as audio array
                audio = raw_bytes.view(dtype).reshape(num_samples, channels)
                
                # Copy to destination
                destination[ix, :num_samples, :channels] = audio
                
                # Zero-pad the rest
                destination[ix, num_samples:, :] = 0
                destination[ix, :, channels:] = 0
            
            return destination[:len(batch_indices)]
        
        decode.is_parallel = True
        return decode
```

## Field Implementation 2: Pre-computed Mel Spectrogram

For models that use spectrograms, pre-compute them to eliminate FFT overhead during training.

```python
class MelSpectrogramField:
    """
    Field for storing pre-computed mel spectrograms.
    
    The spectrogram is computed once during dataset creation and stored.
    This is the most common approach for ASR and TTS training.
    
    Advantages:
    - No FFT during training (significant speedup)
    - Compact storage (especially with float16)
    - Consistent parameters across training
    
    Disadvantages:
    - Inflexible (can't change n_mels, hop_length, etc.)
    - Must recompute dataset to change parameters
    """
    
    TYPE_ID = 21
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        power: float = 2.0,
        log_scale: bool = True,
        store_dtype: str = 'float16',
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.log_scale = log_scale
        self.store_dtype = np.dtype(store_dtype)
        
        # Pre-compute mel filterbank
        self._mel_filters = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create the mel filterbank matrix."""
        import numpy as np
        
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        n_freqs = self.n_fft // 2 + 1
        
        # Mel points
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bins
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create triangular filters
        filters = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(self.n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            
            # Rising edge
            for k in range(left, center):
                if center > left:
                    filters[m, k] = (k - left) / (center - left)
            
            # Falling edge
            for k in range(center, right):
                if right > center:
                    filters[m, k] = (right - k) / (right - center)
        
        return filters
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('time_frames', '<u4'),
            ('n_mels', '<u2'),
            ('dtype_code', '<u1'),  # 0=float16, 1=float32
            ('_pad', '<u1'),
        ], align=True)
    
    def encode(self, audio: np.ndarray, sample_rate: int = None) -> Tuple[np.ndarray, bytes]:
        """
        Compute and encode mel spectrogram.
        
        Args:
            audio: Audio waveform (1D array)
            sample_rate: Original sample rate
        
        Returns:
            (metadata, data_bytes)
        """
        import scipy.signal
        
        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            duration = len(audio) / sample_rate
            new_length = int(duration * self.sample_rate)
            audio = scipy.signal.resample(audio, new_length)
        
        # Ensure float
        audio = audio.astype(np.float32)
        
        # Compute STFT
        _, _, stft = scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
        )
        
        # Compute power spectrum
        power_spec = np.abs(stft) ** self.power
        
        # Apply mel filterbank
        mel_spec = np.dot(self._mel_filters, power_spec)
        
        # Log scale
        if self.log_scale:
            mel_spec = np.log(mel_spec + 1e-10)
        
        # Convert to storage dtype
        mel_spec = mel_spec.astype(self.store_dtype)
        
        # Create metadata
        n_mels, time_frames = mel_spec.shape
        
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0
        metadata['time_frames'] = time_frames
        metadata['n_mels'] = n_mels
        metadata['dtype_code'] = 0 if self.store_dtype == np.float16 else 1
        
        return metadata, mel_spec.tobytes()
    
    def to_binary(self) -> bytes:
        return struct.pack('<IIIHHffBB',
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            0,  # reserved
            self.f_min,
            self.f_max,
            1 if self.log_scale else 0,
            0 if self.store_dtype == np.float16 else 1,
        )
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'MelSpectrogramField':
        sr, n_fft, hop, n_mels, _, f_min, f_max, log_scale, dtype_code = struct.unpack(
            '<IIIHHffBB', data[:24]
        )
        return cls(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            log_scale=bool(log_scale),
            store_dtype='float16' if dtype_code == 0 else 'float32',
        )
    
    def get_decoder_class(self) -> Type:
        return MelSpectrogramDecoder


class MelSpectrogramDecoder:
    """Decoder for pre-computed mel spectrograms."""
    
    def __init__(self, field: MelSpectrogramField, metadata: np.ndarray, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        self.max_time = int(metadata['time_frames'].max())
        self.n_mels = int(metadata['n_mels'].max())
        self.dtype = np.float16 if metadata['dtype_code'][0] == 0 else np.float32
    
    def declare_state_and_memory(self, previous_state):
        # ... similar to AudioWaveformDecoder
        pass
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        dtype = self.dtype
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            for ix in range(len(batch_indices)):
                sample_id = batch_indices[ix]
                
                ptr = metadata[sample_id]['data_ptr']
                time_frames = metadata[sample_id]['time_frames']
                n_mels = metadata[sample_id]['n_mels']
                
                byte_size = time_frames * n_mels * np.dtype(dtype).itemsize
                raw = mem_read(ptr, storage_state)[:byte_size]
                
                spec = raw.view(dtype).reshape(n_mels, time_frames)
                destination[ix, :n_mels, :time_frames] = spec
                
                # Zero-pad
                destination[ix, :, time_frames:] = 0
            
            return destination[:len(batch_indices)]
        
        decode.is_parallel = True
        return decode
```

## Field Implementation 3: Neural Codec Tokens

For audio language models (AudioLM, MusicLM, etc.), store discrete tokens from neural codecs.

```python
class AudioCodecTokenField:
    """
    Field for storing discrete audio tokens from neural codecs like EnCodec.
    
    Neural codecs (EnCodec, SoundStream, DAC) encode audio into sequences
    of discrete tokens from learned codebooks. This is similar to how
    BPE tokenizes text.
    
    Advantages:
    - Extremely compact (~10x smaller than waveform)
    - Native format for audio language models
    - Self-contained (no external decoder needed at inference)
    
    Disadvantages:
    - Lossy (codec determines quality)
    - Requires codec model for reconstruction
    """
    
    TYPE_ID = 22
    
    def __init__(
        self,
        n_codebooks: int = 8,
        vocab_size: int = 1024,
        frame_rate: int = 75,  # EnCodec @ 24kHz uses 75 Hz
    ):
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.frame_rate = frame_rate
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_frames', '<u4'),
            ('n_codebooks', '<u1'),
            ('_pad', '<u1', 3),
        ], align=True)
    
    def encode(self, tokens: np.ndarray) -> Tuple[np.ndarray, bytes]:
        """
        Store pre-computed codec tokens.
        
        Args:
            tokens: Array of shape (n_codebooks, num_frames), dtype uint16
        
        Returns:
            (metadata, data_bytes)
        """
        assert tokens.ndim == 2
        assert tokens.max() < self.vocab_size
        
        n_codebooks, num_frames = tokens.shape
        tokens = tokens.astype('<u2')  # uint16 little-endian
        
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0
        metadata['num_frames'] = num_frames
        metadata['n_codebooks'] = n_codebooks
        
        return metadata, tokens.tobytes()
    
    @classmethod
    def encode_from_audio(cls, audio: np.ndarray, sample_rate: int, codec) -> np.ndarray:
        """
        Encode audio to codec tokens using a pre-loaded codec model.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate of audio
            codec: Loaded EnCodec/SoundStream model
        
        Returns:
            tokens: Array of shape (n_codebooks, num_frames)
        """
        import torch
        
        # Prepare audio tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape: (batch, channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        # Resample if needed (EnCodec expects 24kHz)
        if sample_rate != 24000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            audio = resampler(audio)
        
        # Encode
        with torch.no_grad():
            encoded = codec.encode(audio)
            # EnCodec returns list of (codes, scales) tuples
            if isinstance(encoded, list):
                codes = encoded[0][0]  # First codebook
            else:
                codes = encoded
        
        return codes.squeeze().cpu().numpy()
    
    def to_binary(self) -> bytes:
        return struct.pack('<HHI', self.n_codebooks, self.vocab_size, self.frame_rate)
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'AudioCodecTokenField':
        n_cb, vocab, fr = struct.unpack('<HHI', data[:8])
        return cls(n_codebooks=n_cb, vocab_size=vocab, frame_rate=fr)
    
    def get_decoder_class(self) -> Type:
        return AudioCodecTokenDecoder


class AudioCodecTokenDecoder:
    """Decoder for audio codec tokens."""
    
    def __init__(self, field, metadata, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        self.max_frames = int(metadata['num_frames'].max())
        self.n_codebooks = int(metadata['n_codebooks'].max())
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            for ix in range(len(batch_indices)):
                sample_id = batch_indices[ix]
                
                ptr = metadata[sample_id]['data_ptr']
                num_frames = metadata[sample_id]['num_frames']
                n_codebooks = metadata[sample_id]['n_codebooks']
                
                byte_size = num_frames * n_codebooks * 2  # uint16
                raw = mem_read(ptr, storage_state)[:byte_size]
                
                tokens = raw.view('<u2').reshape(n_codebooks, num_frames)
                destination[ix, :n_codebooks, :num_frames] = tokens
            
            return destination[:len(batch_indices)]
        
        decode.is_parallel = True
        return decode
```

## Performance Comparison

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      AUDIO FIELD PERFORMANCE COMPARISON                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  For 10 seconds of audio at 16kHz:                                            │
│                                                                                │
│  Field Type          Storage Size     Decode Time    Best Use Case            │
│  ──────────          ────────────     ───────────    ─────────────            │
│  Raw Waveform        320 KB           0 ms           WaveNet, HiFi-GAN        │
│  (16-bit mono)                        (memcpy)       Real-time synthesis      │
│                                                                                │
│  FLAC Compressed     160 KB           5-10 ms        Large archives           │
│  (~50% of raw)                        (decompress)   Bandwidth limited        │
│                                                                                │
│  Opus Compressed     50 KB            2-5 ms         Very large datasets      │
│  (~15% of raw)                        (decompress)   Acceptable loss          │
│                                                                                │
│  Mel Spectrogram     64 KB            0 ms           ASR (Whisper)            │
│  (80 mels, float16)                   (memcpy)       TTS (VITS, Tacotron)     │
│                                                                                │
│  Codec Tokens        12 KB            0 ms           AudioLM, MusicLM         │
│  (8 codebooks)                        (memcpy)       VALL-E, Voicebox         │
│                                                                                │
│                                                                                │
│  THROUGHPUT (samples/sec on NVMe SSD, single thread):                         │
│                                                                                │
│  Raw Waveform:       50,000+   (I/O bound)                                    │
│  FLAC Compressed:    5,000     (CPU bound on decompress)                      │
│  Mel Spectrogram:    100,000+  (I/O bound, small files)                       │
│  Codec Tokens:       200,000+  (I/O bound, tiny files)                        │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Exercises

1.  **Implement Compressed Audio Field**: Use `soundfile` to store FLAC-compressed audio. Benchmark decode time vs. raw.

2.  **Chunked Audio**: For very long audio (podcasts, audiobooks), implement a field that stores audio in fixed-length chunks (e.g., 30 seconds each).

3.  **Multi-Track Field**: Design a field for storing stems (vocals, drums, bass, etc.) separately for music production datasets.

4.  **Streaming Decode**: Implement a decoder that can decode a subset of the audio (e.g., a 2-second window) without loading the entire clip.
