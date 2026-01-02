# Audio Field Design: Waveforms, Spectrograms, and Codecs

## Audio Data Characteristics

Audio data presents unique challenges for data loading:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUDIO DATA CHARACTERISTICS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PROPERTY             TYPICAL VALUES         STORAGE IMPACT             │
│  ──────────────────────────────────────────────────────────────────────│
│  Sample Rate          8-48 kHz               1.5-7 MB/min (16-bit)      │
│  Bit Depth            16-32 bits             2-4 bytes/sample           │
│  Channels             1-2 (mono/stereo)      1-2x size multiplier       │
│  Duration             1-30+ seconds          Variable length            │
│  Format               WAV, MP3, FLAC, etc    Decode overhead            │
│                                                                          │
│  DERIVED REPRESENTATIONS                                                 │
│  ──────────────────────────────────────────────────────────────────────│
│  Mel Spectrogram      (n_mels, time_steps)   80-128 × 100-1000         │
│  MFCC                 (n_mfcc, time_steps)   13-40 × time              │
│  EnCodec tokens       (codebooks, frames)    8-32 × frames             │
│                                                                          │
│  CHALLENGE: 1 hour of audio = 2.6 GB (16-bit, 48kHz, stereo)           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Storage Strategy Decision Tree

```
                    ┌─────────────────────┐
                    │   Audio Storage     │
                    │      Decision       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │   Raw PCM   │  │ Compressed  │  │Pre-computed │
       │  Waveform   │  │   (FLAC/    │  │  Features   │
       │             │  │   Opus)     │  │             │
       └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
              │                │                │
   ┌──────────┴──────────┐    │         ┌──────┴──────┐
   │ + No decode latency │    │         │ + Smallest  │
   │ + Direct processing │    │         │ + Fastest   │
   │ - Largest storage   │    │         │ - Inflexible│
   │ - I/O bound         │    │         │ - Must pre- │
   └─────────────────────┘    │         │   compute   │
                              │         └─────────────┘
              ┌───────────────┴───────────────┐
              │ + Good compression (2-5x)     │
              │ + Lossless (FLAC) or lossy   │
              │ - Decode overhead            │
              │ - More complex pipeline      │
              └───────────────────────────────┘
```

## 1. Raw Waveform Field

### Design
```python
import numpy as np
from typing import Type

class AudioWaveformField:
    """
    Store raw audio waveforms (PCM).
    
    Best for:
    - Short audio clips (<10 sec)
    - When decode latency matters
    - When full waveform needed (e.g., WaveNet training)
    
    Storage: 2 bytes/sample × sample_rate × duration × channels
    Example: 5 sec @ 16kHz mono = 160 KB
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 mono: bool = True,
                 dtype: str = 'int16'):
        self.sample_rate = sample_rate
        self.mono = mono
        self.dtype = np.dtype(dtype)
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),       # Pointer to waveform data
            ('num_samples', '<u4'),    # Number of audio samples
            ('sample_rate', '<u4'),    # Sample rate (Hz)
            ('channels', '<u1'),       # Number of channels
            ('dtype_code', '<u1'),     # 0=int16, 1=float32
        ], align=True)
    
    def encode(self, destination, audio, malloc):
        """
        audio: numpy array, shape (num_samples,) or (num_samples, channels)
        """
        # Normalize shape
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        # Resample if needed
        if hasattr(audio, 'sample_rate') and audio.sample_rate != self.sample_rate:
            audio = self._resample(audio, self.sample_rate)
        
        # Convert to mono if needed
        if self.mono and audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        
        # Convert dtype
        audio = audio.astype(self.dtype)
        
        # Allocate and write
        num_samples, channels = audio.shape
        byte_size = audio.nbytes
        
        ptr, buffer = malloc(byte_size)
        buffer[:] = audio.tobytes()
        
        # Store metadata
        destination['data_ptr'] = ptr
        destination['num_samples'] = num_samples
        destination['sample_rate'] = self.sample_rate
        destination['channels'] = channels
        destination['dtype_code'] = {'int16': 0, 'float32': 1}[self.dtype.name]
    
    def get_decoder_class(self) -> Type:
        return AudioWaveformDecoder
```

### Decoder
```python
class AudioWaveformDecoder:
    """Decode raw waveform from storage."""
    
    def declare_state_and_memory(self, previous_state):
        # Max length in dataset
        max_samples = int(self.metadata['num_samples'].max())
        max_channels = int(self.metadata['channels'].max())
        
        dtype = np.int16 if self.metadata['dtype_code'][0] == 0 else np.float32
        
        return (
            State(shape=(max_samples, max_channels), dtype=dtype, jit_mode=True),
            AllocationQuery((max_samples, max_channels), dtype=dtype)
        )
    
    def generate_code(self):
        mem_read = self.memory_read
        
        def decode(indices, destination, metadata, storage_state):
            for ix in range(len(indices)):
                sample_id = indices[ix]
                ptr = metadata[sample_id]['data_ptr']
                num_samples = metadata[sample_id]['num_samples']
                channels = metadata[sample_id]['channels']
                
                raw = mem_read(ptr, storage_state)
                audio = raw.view(np.int16).reshape(num_samples, channels)
                destination[ix, :num_samples, :channels] = audio
            
            return destination[:len(indices)]
        
        decode.is_parallel = True
        return decode
```

## 2. Compressed Audio Field

### Design with FLAC/Opus
```python
import subprocess
import io

class CompressedAudioField:
    """
    Store audio with lossless (FLAC) or lossy (Opus) compression.
    
    Best for:
    - Large datasets
    - When storage/bandwidth is limited
    - Longer audio clips
    
    Compression ratios:
    - FLAC: ~50-60% of original
    - Opus: ~5-10% of original (lossy)
    """
    
    CODECS = {
        'flac': {'ext': 'flac', 'lossless': True},
        'opus': {'ext': 'opus', 'lossless': False},
        'mp3': {'ext': 'mp3', 'lossless': False},
    }
    
    def __init__(self, 
                 codec: str = 'flac',
                 sample_rate: int = 16000,
                 bitrate: int = 64000):  # For lossy codecs
        self.codec = codec
        self.sample_rate = sample_rate
        self.bitrate = bitrate
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),      # Compressed size
            ('num_samples', '<u4'),    # Original sample count
            ('sample_rate', '<u4'),
            ('channels', '<u1'),
            ('codec', '<u1'),          # Codec ID
        ], align=True)
    
    def encode(self, destination, audio, malloc):
        import soundfile as sf
        
        # Compress to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format=self.codec.upper())
        compressed = buffer.getvalue()
        
        # Write to storage
        ptr, storage_buffer = malloc(len(compressed))
        storage_buffer[:] = np.frombuffer(compressed, dtype='u1')
        
        # Metadata
        destination['data_ptr'] = ptr
        destination['data_size'] = len(compressed)
        destination['num_samples'] = len(audio)
        destination['sample_rate'] = self.sample_rate
        destination['channels'] = 1 if audio.ndim == 1 else audio.shape[1]
        destination['codec'] = list(self.CODECS.keys()).index(self.codec)


class CompressedAudioDecoder:
    """Decode compressed audio on-the-fly."""
    
    def generate_code(self):
        import soundfile as sf
        mem_read = self.memory_read
        
        def decode(indices, destination, metadata, storage_state):
            for ix, sample_id in enumerate(indices):
                ptr = metadata[sample_id]['data_ptr']
                size = metadata[sample_id]['data_size']
                
                # Read compressed bytes
                compressed = mem_read(ptr, storage_state)[:size]
                
                # Decode (this is the overhead!)
                buffer = io.BytesIO(compressed.tobytes())
                audio, sr = sf.read(buffer)
                
                destination[ix, :len(audio)] = audio
            
            return destination[:len(indices)]
        
        return decode
```

## 3. Spectrogram Field

### Pre-computed Mel Spectrograms
```python
class MelSpectrogramField:
    """
    Store pre-computed mel spectrograms.
    
    Best for:
    - Models that use spectrograms (most ASR, TTS)
    - When spectrogram params are fixed
    - Fastest loading (no FFT during training)
    
    Storage: n_mels × time_frames × 4 bytes (float32)
    Example: 80 mels × 100 frames = 32 KB
    """
    
    def __init__(self,
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 sample_rate: int = 16000,
                 store_dtype: str = 'float16'):  # Save space
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.store_dtype = np.dtype(store_dtype)
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('time_frames', '<u4'),    # Number of time frames
            ('n_mels', '<u2'),         # Number of mel bins
        ], align=True)
    
    def encode(self, destination, audio, malloc):
        import librosa
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale and store dtype
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = mel_spec.astype(self.store_dtype)
        
        # Shape: (n_mels, time_frames)
        n_mels, time_frames = mel_spec.shape
        
        # Store
        ptr, buffer = malloc(mel_spec.nbytes)
        buffer[:] = mel_spec.tobytes()
        
        destination['data_ptr'] = ptr
        destination['time_frames'] = time_frames
        destination['n_mels'] = n_mels


class MelSpectrogramDecoder:
    """Decode pre-computed mel spectrogram."""
    
    def declare_state_and_memory(self, previous_state):
        max_time = int(self.metadata['time_frames'].max())
        n_mels = int(self.metadata['n_mels'].max())
        
        return (
            State(shape=(n_mels, max_time), dtype=np.float16, jit_mode=True),
            AllocationQuery((n_mels, max_time), dtype=np.float16)
        )
    
    def generate_code(self):
        mem_read = self.memory_read
        
        def decode(indices, destination, metadata, storage_state):
            for ix, sample_id in enumerate(indices):
                ptr = metadata[sample_id]['data_ptr']
                time_frames = metadata[sample_id]['time_frames']
                n_mels = metadata[sample_id]['n_mels']
                
                raw = mem_read(ptr, storage_state)
                spec = raw.view(np.float16).reshape(n_mels, time_frames)
                destination[ix, :n_mels, :time_frames] = spec
            
            return destination[:len(indices)]
        
        decode.is_parallel = True
        return decode
```

## 4. Neural Codec Field (EnCodec/SoundStream Tokens)

### Storing Discrete Audio Tokens
```python
class AudioCodecTokenField:
    """
    Store discrete tokens from neural audio codecs.
    
    Best for:
    - Audio language models (AudioLM, MusicLM)
    - Extremely compact storage
    - When using codec-based models
    
    Storage: n_codebooks × time_frames × 2 bytes (uint16)
    Example: 8 codebooks × 75 frames/sec × 10 sec = 12 KB
    """
    
    def __init__(self,
                 n_codebooks: int = 8,
                 codec_sr: int = 75,  # EnCodec: 75 Hz at 24kHz
                 vocab_size: int = 1024):
        self.n_codebooks = n_codebooks
        self.codec_sr = codec_sr
        self.vocab_size = vocab_size
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_frames', '<u4'),     # Number of codec frames
            ('n_codebooks', '<u1'),    # Number of codebooks used
        ], align=True)
    
    def encode(self, destination, tokens, malloc):
        """
        tokens: numpy array of shape (n_codebooks, num_frames)
                or pre-encoded from EnCodec/SoundStream
        """
        # Validate
        assert tokens.max() < self.vocab_size
        tokens = tokens.astype('<u2')  # uint16
        
        n_codebooks, num_frames = tokens.shape
        
        # Store
        ptr, buffer = malloc(tokens.nbytes)
        buffer[:] = tokens.tobytes()
        
        destination['data_ptr'] = ptr
        destination['num_frames'] = num_frames
        destination['n_codebooks'] = n_codebooks
    
    @staticmethod
    def from_waveform(audio, sample_rate, codec_model):
        """
        Helper to encode waveform to codec tokens.
        
        codec_model: Pre-loaded EnCodec or SoundStream model
        """
        import torch
        
        # Ensure correct sample rate (EnCodec uses 24kHz)
        if sample_rate != 24000:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, 24000)
            audio = resampler(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)
        
        # Encode
        with torch.no_grad():
            codes = codec_model.encode(audio.unsqueeze(0).unsqueeze(0))
        
        return codes.squeeze().cpu().numpy()
```

## 5. Hybrid Field: Waveform + Spectrogram

```python
class HybridAudioField:
    """
    Store both waveform AND pre-computed features.
    
    Best for:
    - Research/experimentation
    - When you need both representations
    - Flexibility over storage efficiency
    """
    
    def __init__(self,
                 waveform_config: dict,
                 spectrogram_config: dict):
        self.waveform_field = AudioWaveformField(**waveform_config)
        self.spectrogram_field = MelSpectrogramField(**spectrogram_config)
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('waveform', self.waveform_field.metadata_type),
            ('spectrogram', self.spectrogram_field.metadata_type),
        ], align=True)
    
    def encode(self, destination, audio, malloc):
        # Store waveform
        self.waveform_field.encode(
            destination['waveform'], 
            audio, 
            malloc
        )
        
        # Store spectrogram
        self.spectrogram_field.encode(
            destination['spectrogram'],
            audio,
            malloc
        )
```

## Performance Comparison

| Field Type | Storage/min | Decode Time | Best Use Case |
|------------|-------------|-------------|---------------|
| Raw PCM (16kHz, 16-bit) | 1.9 MB | 0 ms | Short clips, low latency |
| FLAC | 0.9 MB | 5-10 ms | Large datasets, lossless |
| Opus (64kbps) | 0.5 MB | 2-5 ms | Very large datasets |
| Mel Spectrogram | 0.3 MB | 0 ms | ASR/TTS training |
| Codec Tokens | 0.09 MB | 0 ms | Audio LMs |

## Complete Example: Audio Dataset Writer

```python
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class AudioDataset:
    """Example audio dataset."""
    audio_paths: list
    labels: list
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        import soundfile as sf
        audio, sr = sf.read(self.audio_paths[idx])
        return (audio, self.labels[idx])


def write_audio_dataset(
    dataset,
    output_path: str,
    field_type: str = 'spectrogram',
    **field_kwargs
):
    """Write an audio dataset to custom format."""
    
    # Choose field type
    if field_type == 'waveform':
        audio_field = AudioWaveformField(**field_kwargs)
    elif field_type == 'spectrogram':
        audio_field = MelSpectrogramField(**field_kwargs)
    elif field_type == 'compressed':
        audio_field = CompressedAudioField(**field_kwargs)
    elif field_type == 'codec':
        audio_field = AudioCodecTokenField(**field_kwargs)
    
    # Create writer
    from your_format import DatasetWriter, IntField
    
    writer = DatasetWriter(output_path, {
        'audio': audio_field,
        'label': IntField(),
    })
    
    writer.from_indexed_dataset(dataset)
    
    return output_path
```

## Exercises

1. Implement a `ChunkedAudioField` that stores long audio files in fixed-size chunks for better memory access patterns.

2. Create a benchmark comparing decode times for different audio field types on your hardware.

3. Design a `MultiTrackAudioField` for storing audio with separate stems (vocals, drums, bass, etc.).
