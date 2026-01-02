# Transform Operations: The Building Blocks of Data Pipelines

## What is a Transform?

A **transform** (or **operation**) is a processing step in a data pipeline. It takes data from the previous step and produces output for the next step.

Examples:
- **Decode**: JPEG bytes → pixel array
- **Resize**: 1024×768 → 224×224
- **Normalize**: uint8 [0-255] → float32 [0, 1] then subtract mean
- **Augment**: Random crop, flip, color jitter

In FFCV (and our format), transforms are special because:
1.  They **declare** their memory requirements upfront.
2.  They **generate code** that can be JIT-compiled.
3.  They're **composable**—chained into efficient pipelines.

## The `Operation` Abstract Base Class

Every transform inherits from a common interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Any, Callable, Optional
import numpy as np

@dataclass
class State:
    """
    Tracks the "shape" of data flowing through the pipeline.
    
    Each operation receives the previous state and returns the new state.
    """
    jit_mode: bool           # Can this be JIT compiled?
    dtype: np.dtype          # Data type
    shape: Optional[Tuple]   # Shape (None if unknown/variable)
    device: str = 'cpu'      # 'cpu' or 'cuda:N'
    
    # Modality-specific fields
    channels: int = 0
    sample_rate: int = 0
    num_tokens: int = 0


@dataclass
class AllocationQuery:
    """
    Specifies memory to pre-allocate for this operation's output.
    """
    shape: Tuple[int, ...]
    dtype: np.dtype


class Operation(ABC):
    """
    Abstract Base Class for all pipeline operations.
    
    Lifecycle:
    1. __init__: Store configuration (e.g., crop size, normalization params).
    2. accept_field: Receive information about the field being processed.
    3. accept_globals: Receive dataset-wide info (metadata table, memory reader).
    4. declare_state_and_memory: Declare output shape/dtype and memory needs.
    5. generate_code: Return the function that does the work.
    6. (optional) compile: JIT compile the generated code.
    """
    
    def accept_field(self, field):
        """
        Called by the pipeline builder to give the operation access to the field.
        
        Args:
            field: The Field object (e.g., RGBImageField) being decoded.
        """
        self.field = field
    
    def accept_globals(self, metadata, memory_read):
        """
        Called by the pipeline builder with dataset-wide information.
        
        Args:
            metadata: The per-sample metadata for this field (numpy structured array).
            memory_read: Function to read data from storage: read(ptr, storage_state) -> bytes.
        """
        self.metadata = metadata
        self.memory_read = memory_read
    
    @abstractmethod
    def declare_state_and_memory(
        self,
        previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        """
        Declare output state and memory requirements.
        
        This is called at "compile time" (before training starts) to figure out
        how much memory each pipeline step needs.
        
        Args:
            previous_state: The state of data coming into this operation.
        
        Returns:
            new_state: The state of data after this operation.
            allocation: How much memory to pre-allocate for the output.
                       None if the operation is in-place or produces no new buffer.
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_code(self) -> Callable:
        """
        Generate the function that performs this operation.
        
        Returns:
            A function with signature:
                func(input_buffer, output_buffer) -> output_buffer
            
            Or for decoders:
                func(sample_indices, output_buffer, metadata, storage_state) -> output_buffer
        
        The function should be JIT-compatible if possible (all types known, no Python objects).
        """
        raise NotImplementedError
    
    def generate_code_for_shared_state(self) -> Optional[Callable]:
        """
        Generate code to initialize shared state (e.g., RNG for augmentation).
        
        This is called once per batch to set up state that's shared across samples.
        
        Returns:
            None if no shared state, or a function: init(shared_memory) -> None
        """
        return None
    
    def declare_shared_memory(self, previous_state: State) -> Optional[AllocationQuery]:
        """
        Declare memory for shared state (e.g., per-batch random seeds).
        """
        return None
```

## A Simple Example: Normalize

Let's implement normalization step-by-step.

```python
import numba as nb

class Normalize(Operation):
    """
    Normalize image pixels to zero mean and unit variance.
    
    Operation: output = (input / 255.0 - mean) / std
    
    This is one of the most common transforms in image processing.
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet mean
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),   # ImageNet std
    ):
        """
        Args:
            mean: Per-channel mean values to subtract.
            std: Per-channel standard deviation to divide by.
        """
        # Convert to numpy arrays for JIT compatibility
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        """
        Normalization changes dtype from uint8 to float32.
        Shape remains the same.
        """
        # Input: (H, W, C) uint8
        # Output: (H, W, C) float32
        
        new_state = State(
            jit_mode=True,  # Can be JIT compiled
            dtype=np.dtype('float32'),
            shape=previous_state.shape,  # Same shape
            device=previous_state.device,
            channels=previous_state.channels,
        )
        
        allocation = AllocationQuery(
            shape=previous_state.shape,
            dtype=np.dtype('float32'),
        )
        
        return new_state, allocation
    
    def generate_code(self) -> Callable:
        """
        Generate the JIT-compilable normalization function.
        """
        # Capture mean and std in closure (these are numpy arrays, JIT-safe)
        mean = self.mean
        std = self.std
        
        # The function we return will be JIT compiled
        @nb.njit(parallel=True, nogil=True, fastmath=True)
        def normalize(input_arr, output_arr):
            """
            Normalize a batch of images.
            
            Args:
                input_arr: uint8 array of shape (batch, H, W, C)
                output_arr: float32 array of shape (batch, H, W, C), pre-allocated
            
            Returns:
                output_arr (same reference, now populated)
            """
            batch, h, w, c = input_arr.shape
            
            # Parallel over batch
            for b in nb.prange(batch):
                for y in range(h):
                    for x in range(w):
                        for ch in range(c):
                            # Scale to [0, 1], subtract mean, divide by std
                            val = input_arr[b, y, x, ch] / 255.0
                            output_arr[b, y, x, ch] = (val - mean[ch]) / std[ch]
            
            return output_arr
        
        return normalize


# Usage example (simplified)
if __name__ == '__main__':
    op = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    # Simulate previous state
    prev_state = State(
        jit_mode=True,
        dtype=np.dtype('uint8'),
        shape=(224, 224, 3),
        channels=3,
    )
    
    # Declare state and memory
    new_state, allocation = op.declare_state_and_memory(prev_state)
    print(f"Input:  {prev_state.dtype}, {prev_state.shape}")
    print(f"Output: {new_state.dtype}, {allocation.shape}")
    
    # Generate and compile the function
    normalize_fn = op.generate_code()
    
    # Allocate memory
    batch_size = 32
    input_buffer = np.random.randint(0, 256, (batch_size,) + prev_state.shape, dtype=np.uint8)
    output_buffer = np.empty((batch_size,) + allocation.shape, dtype=allocation.dtype)
    
    # Run
    result = normalize_fn(input_buffer, output_buffer)
    print(f"Result mean: {result.mean():.4f}, std: {result.std():.4f}")
```

## Image Transforms

### RandomResizedCrop

The most common ImageNet augmentation: random crop with random aspect ratio, then resize.

```python
class RandomResizedCrop(Operation):
    """
    Random crop + resize (standard ImageNet training augmentation).
    
    1. Sample a random region with random scale (8%-100% of area) and aspect ratio.
    2. Resize the crop to the target size.
    
    This is more effective than just random crop for preventing overfitting.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3/4, 4/3),
    ):
        self.output_h, self.output_w = output_size
        self.scale_min, self.scale_max = scale
        self.log_ratio_min = np.log(ratio[0])
        self.log_ratio_max = np.log(ratio[1])
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        new_state = State(
            jit_mode=True,
            dtype=previous_state.dtype,
            shape=(self.output_h, self.output_w, previous_state.channels),
            device=previous_state.device,
            channels=previous_state.channels,
        )
        
        allocation = AllocationQuery(
            shape=(self.output_h, self.output_w, previous_state.channels),
            dtype=previous_state.dtype,
        )
        
        return new_state, allocation
    
    def generate_code(self) -> Callable:
        output_h = self.output_h
        output_w = self.output_w
        scale_min = self.scale_min
        scale_max = self.scale_max
        log_ratio_min = self.log_ratio_min
        log_ratio_max = self.log_ratio_max
        
        @nb.njit(parallel=True, nogil=True)
        def random_resized_crop(input_arr, output_arr):
            """
            Apply random resized crop to a batch of images.
            """
            batch, in_h, in_w, channels = input_arr.shape
            area = in_h * in_w
            
            for b in nb.prange(batch):
                # Try to find valid crop parameters
                found = False
                
                for attempt in range(10):
                    # Random scale and aspect ratio
                    target_area = np.random.uniform(scale_min, scale_max) * area
                    aspect_ratio = np.exp(np.random.uniform(log_ratio_min, log_ratio_max))
                    
                    # Calculate crop dimensions
                    crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
                    crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
                    
                    # Check if crop fits in image
                    if crop_w <= in_w and crop_h <= in_h and crop_w > 0 and crop_h > 0:
                        # Random position
                        x0 = np.random.randint(0, in_w - crop_w + 1)
                        y0 = np.random.randint(0, in_h - crop_h + 1)
                        
                        # Bilinear resize from crop to output
                        _bilinear_resize(
                            input_arr[b, y0:y0+crop_h, x0:x0+crop_w, :],
                            output_arr[b, :, :, :]
                        )
                        found = True
                        break
                
                if not found:
                    # Fallback: center crop to match output aspect ratio
                    target_ratio = output_w / output_h
                    if in_w / in_h > target_ratio:
                        # Image is wider, crop width
                        crop_h = in_h
                        crop_w = int(in_h * target_ratio)
                    else:
                        # Image is taller, crop height
                        crop_w = in_w
                        crop_h = int(in_w / target_ratio)
                    
                    x0 = (in_w - crop_w) // 2
                    y0 = (in_h - crop_h) // 2
                    
                    _bilinear_resize(
                        input_arr[b, y0:y0+crop_h, x0:x0+crop_w, :],
                        output_arr[b, :, :, :]
                    )
            
            return output_arr
        
        return random_resized_crop


@nb.njit
def _bilinear_resize(src, dst):
    """
    Bilinear interpolation resize.
    
    Args:
        src: Input array (H_in, W_in, C)
        dst: Output array (H_out, W_out, C), pre-allocated
    """
    src_h, src_w, c = src.shape
    dst_h, dst_w = dst.shape[:2]
    
    # Scale factors (inverse: how far apart are dst pixels in src space)
    scale_y = src_h / dst_h
    scale_x = src_w / dst_w
    
    for y in range(dst_h):
        # Where does this output pixel map in input space?
        src_y = (y + 0.5) * scale_y - 0.5
        y0 = int(np.floor(src_y))
        y1 = min(y0 + 1, src_h - 1)
        y0 = max(y0, 0)
        fy = src_y - y0  # Fractional part
        
        for x in range(dst_w):
            src_x = (x + 0.5) * scale_x - 0.5
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, src_w - 1)
            x0 = max(x0, 0)
            fx = src_x - x0
            
            for ch in range(c):
                # Bilinear interpolation
                v00 = src[y0, x0, ch]
                v01 = src[y0, x1, ch]
                v10 = src[y1, x0, ch]
                v11 = src[y1, x1, ch]
                
                v0 = v00 * (1 - fx) + v01 * fx
                v1 = v10 * (1 - fx) + v11 * fx
                
                dst[y, x, ch] = np.uint8(v0 * (1 - fy) + v1 * fy)
```

### RandomHorizontalFlip

```python
class RandomHorizontalFlip(Operation):
    """
    Randomly flip images horizontally.
    
    With probability p, flip the image left-to-right.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        # Shape unchanged
        return previous_state, AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype)
    
    def generate_code(self) -> Callable:
        p = self.p
        
        @nb.njit(parallel=True, nogil=True)
        def random_hflip(input_arr, output_arr):
            batch, h, w, c = input_arr.shape
            
            for b in nb.prange(batch):
                # Each sample gets its own random decision
                if np.random.random() < p:
                    # Flip
                    for y in range(h):
                        for x in range(w):
                            for ch in range(c):
                                output_arr[b, y, x, ch] = input_arr[b, y, w - 1 - x, ch]
                else:
                    # Copy
                    for y in range(h):
                        for x in range(w):
                            for ch in range(c):
                                output_arr[b, y, x, ch] = input_arr[b, y, x, ch]
            
            return output_arr
        
        return random_hflip
```

### ColorJitter

```python
class ColorJitter(Operation):
    """
    Randomly change brightness, contrast, saturation, and hue.
    
    Each is randomly sampled from a uniform distribution.
    """
    
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return previous_state, AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype)
    
    def generate_code(self) -> Callable:
        brightness_range = self.brightness
        contrast_range = self.contrast
        saturation_range = self.saturation
        hue_range = self.hue
        
        @nb.njit(parallel=True, nogil=True)
        def color_jitter(input_arr, output_arr):
            batch, h, w, c = input_arr.shape
            
            for b in nb.prange(batch):
                # Sample random factors
                b_factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
                c_factor = 1.0 + np.random.uniform(-contrast_range, contrast_range)
                s_factor = 1.0 + np.random.uniform(-saturation_range, saturation_range)
                h_offset = np.random.uniform(-hue_range, hue_range)
                
                for y in range(h):
                    for x in range(w):
                        # Get RGB
                        r = float(input_arr[b, y, x, 0]) / 255.0
                        g = float(input_arr[b, y, x, 1]) / 255.0
                        b_val = float(input_arr[b, y, x, 2]) / 255.0
                        
                        # Brightness
                        r *= b_factor
                        g *= b_factor
                        b_val *= b_factor
                        
                        # Convert to grayscale for contrast
                        gray = 0.299 * r + 0.587 * g + 0.114 * b_val
                        
                        # Contrast
                        r = (r - gray) * c_factor + gray
                        g = (g - gray) * c_factor + gray
                        b_val = (b_val - gray) * c_factor + gray
                        
                        # Saturation
                        r = (r - gray) * s_factor + gray
                        g = (g - gray) * s_factor + gray
                        b_val = (b_val - gray) * s_factor + gray
                        
                        # Clamp and store
                        output_arr[b, y, x, 0] = np.uint8(max(0.0, min(255.0, r * 255.0)))
                        output_arr[b, y, x, 1] = np.uint8(max(0.0, min(255.0, g * 255.0)))
                        output_arr[b, y, x, 2] = np.uint8(max(0.0, min(255.0, b_val * 255.0)))
            
            return output_arr
        
        return color_jitter
```

### ToTensor (HWC → CHW)

```python
class ToTensor(Operation):
    """
    Convert HWC (Height, Width, Channels) to CHW (Channels, Height, Width).
    
    PyTorch expects CHW format for images.
    Also converts to float32 if not already.
    """
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        h, w, c = previous_state.shape
        
        new_state = State(
            jit_mode=True,
            dtype=np.dtype('float32'),
            shape=(c, h, w),  # CHW order
            device=previous_state.device,
            channels=c,
        )
        
        allocation = AllocationQuery(
            shape=(c, h, w),
            dtype=np.dtype('float32'),
        )
        
        return new_state, allocation
    
    def generate_code(self) -> Callable:
        @nb.njit(parallel=True, nogil=True)
        def to_tensor(input_arr, output_arr):
            """
            Transpose from (B, H, W, C) to (B, C, H, W) and convert to float.
            """
            batch, h, w, c = input_arr.shape
            
            for b in nb.prange(batch):
                for ch in range(c):
                    for y in range(h):
                        for x in range(w):
                            output_arr[b, ch, y, x] = float(input_arr[b, y, x, ch])
            
            return output_arr
        
        return to_tensor
```

## Audio Transforms

### MelSpectrogram

```python
class MelSpectrogram(Operation):
    """
    Compute mel spectrogram from waveform.
    
    This is the standard preprocessor for speech/audio models.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
        # Pre-compute mel filterbank (this is costly, do it once)
        self._mel_filters = self._create_mel_filterbank()
        self._window = np.hanning(win_length).astype(np.float32)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create the mel filterbank matrix."""
        # This is a simplified version; use librosa.filters.mel for production
        n_freqs = self.n_fft // 2 + 1
        
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filters = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(self.n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            
            for k in range(left, center):
                if center != left:
                    filters[m, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right != center:
                    filters[m, k] = (right - k) / (right - center)
        
        return filters
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        # Compute number of frames from input length
        # This assumes we know the max input length
        max_samples = previous_state.shape[0] if previous_state.shape else 16000 * 10
        n_frames = (max_samples - self.n_fft) // self.hop_length + 1
        
        new_state = State(
            jit_mode=False,  # FFT not JIT-able in pure Numba
            dtype=np.dtype('float32'),
            shape=(self.n_mels, n_frames),
            device=previous_state.device,
        )
        
        allocation = AllocationQuery(
            shape=(self.n_mels, n_frames),
            dtype=np.dtype('float32'),
        )
        
        return new_state, allocation
    
    def generate_code(self) -> Callable:
        mel_filters = self._mel_filters
        window = self._window
        n_fft = self.n_fft
        hop = self.hop_length
        
        def mel_spectrogram(audio, output):
            """
            Compute mel spectrogram.
            Uses scipy for FFT (not JIT-able).
            """
            from scipy import fft
            
            batch = audio.shape[0]
            n_samples = audio.shape[1]
            n_frames = (n_samples - n_fft) // hop + 1
            
            for b in range(batch):
                for frame in range(n_frames):
                    start = frame * hop
                    windowed = audio[b, start:start + n_fft] * window
                    
                    # FFT
                    spectrum = np.abs(fft.rfft(windowed)) ** 2
                    
                    # Apply mel filterbank
                    mel_spectrum = mel_filters @ spectrum
                    
                    # Log scale
                    output[b, :, frame] = np.log(mel_spectrum + 1e-10)
            
            return output
        
        return mel_spectrogram
```

### SpecAugment

```python
class SpecAugment(Operation):
    """
    SpecAugment: Frequency and time masking for spectrograms.
    
    Paper: https://arxiv.org/abs/1904.08779
    
    This is the standard augmentation for speech recognition.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,      # Max frequency bands to mask
        time_mask_param: int = 100,     # Max time frames to mask
        num_freq_masks: int = 2,        # Number of frequency masks
        num_time_masks: int = 2,        # Number of time masks
    ):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.num_freq = num_freq_masks
        self.num_time = num_time_masks
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return previous_state, AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype)
    
    def generate_code(self) -> Callable:
        F = self.F
        T = self.T
        num_freq = self.num_freq
        num_time = self.num_time
        
        @nb.njit(parallel=True, nogil=True)
        def spec_augment(input_spec, output_spec):
            batch, n_mels, n_frames = input_spec.shape
            
            for b in nb.prange(batch):
                # Copy input
                for m in range(n_mels):
                    for t in range(n_frames):
                        output_spec[b, m, t] = input_spec[b, m, t]
                
                # Apply frequency masks
                for _ in range(num_freq):
                    f = np.random.randint(0, F + 1)
                    f0 = np.random.randint(0, max(1, n_mels - f))
                    for m in range(f0, min(f0 + f, n_mels)):
                        for t in range(n_frames):
                            output_spec[b, m, t] = 0.0
                
                # Apply time masks
                for _ in range(num_time):
                    t = np.random.randint(0, T + 1)
                    t0 = np.random.randint(0, max(1, n_frames - t))
                    for m in range(n_mels):
                        for tr in range(t0, min(t0 + t, n_frames)):
                            output_spec[b, m, tr] = 0.0
            
            return output_spec
        
        return spec_augment
```

## Text Transforms

### Tokenize

```python
class Tokenize(Operation):
    """
    Tokenize text using a pre-trained tokenizer.
    
    Note: Tokenization is typically NOT JIT-able because tokenizers
    involve complex string processing and hash lookups.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        new_state = State(
            jit_mode=False,  # Cannot JIT tokenization
            dtype=np.dtype('int64'),
            shape=(self.max_length,),
            device=previous_state.device,
            num_tokens=self.max_length,
        )
        
        allocation = AllocationQuery(
            shape=(self.max_length,),
            dtype=np.dtype('int64'),
        )
        
        return new_state, allocation
    
    def generate_code(self) -> Callable:
        tokenizer = self.tokenizer
        max_len = self.max_length
        padding = self.padding
        truncation = self.truncation
        
        def tokenize(texts, output):
            """
            Tokenize a batch of texts.
            
            Args:
                texts: List of strings
                output: Pre-allocated int64 array (batch, max_length)
            """
            batch_size = len(texts)
            
            for i, text in enumerate(texts):
                encoded = tokenizer(
                    text,
                    max_length=max_len,
                    padding=padding,
                    truncation=truncation,
                    return_tensors='np',
                )
                output[i, :] = encoded['input_ids'][0]
            
            return output
        
        return tokenize
```

## Composing Operations

### The Compose Pattern

```python
class Compose(Operation):
    """
    Compose multiple operations into a single pipeline.
    """
    
    def __init__(self, operations: list):
        self.operations = operations
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, list]:
        states = [previous_state]
        allocations = []
        
        for op in self.operations:
            new_state, alloc = op.declare_state_and_memory(states[-1])
            states.append(new_state)
            allocations.append(alloc)
        
        return states[-1], allocations
    
    def generate_code(self) -> Callable:
        funcs = [op.generate_code() for op in self.operations]
        
        def composed(input_data, memories):
            current = input_data
            for func, memory in zip(funcs, memories):
                current = func(current, memory)
            return current
        
        return composed
```

## Exercises

1.  **Implement GaussianBlur**: Apply Gaussian blur with configurable sigma.

2.  **Implement MixUp**: Combine two samples with a random weight from Beta distribution.

3.  **Implement CutMix**: Cut and paste patches between images.

4.  **Profile a Pipeline**: Use Python's `timeit` to compare the speed of individual transforms vs. a fused pipeline.
