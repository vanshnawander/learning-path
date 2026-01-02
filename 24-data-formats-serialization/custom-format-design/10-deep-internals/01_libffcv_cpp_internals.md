# LibFFCV: The C++ Core Internals

## Why C++ is Necessary

No matter how optimized your Python code is, some operations require native code:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    WHEN C++ IS NECESSARY                                       │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Operation               Python/Numba      C/C++          Difference          │
│  ─────────               ────────────      ─────          ──────────          │
│                                                                                │
│  JPEG Decode             Not possible      3-8 ms/image   C++ required        │
│  (libjpeg-turbo)         (no bindings)                    (library binding)   │
│                                                                                │
│  Image Resize            ~5 ms (Numba)     0.5 ms         10x faster          │
│  (bilinear, 224→512)     (pure Python)     (OpenCV C++)   (SIMD, cache opt.)  │
│                                                                                │
│  Memory Copy             ~100 μs           ~20 μs         5x faster           │
│  (10 MB buffer)          (Python memoryview) (memcpy)     (no GIL overhead)   │
│                                                                                │
│  Video Decode            Python overhead   GPU hardware   100x+ faster        │
│  (h.264 with NVDEC)      per frame         (direct)       (hardware accel.)   │
│                                                                                │
│  WebP Decode             Python bindings   0.5-2 ms       Direct C++ faster   │
│  (libwebp)               overhead adds ms                                     │
│                                                                                │
│  KEY INSIGHT:                                                                 │
│  ─────────────                                                                │
│  FFCV's speed comes from identifying these bottlenecks and writing           │
│  minimal C++ wrappers that:                                                   │
│  1. Accept raw memory pointers (no Python object overhead)                    │
│  2. Use thread-local storage (no lock contention)                             │
│  3. Release the GIL (parallel execution)                                      │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## The libffcv.cpp Architecture

FFCV's C++ code is a single file (`libffcv.cpp`) compiled into a shared library. It's designed for:
- **Minimal surface area**: Only export the functions that Python needs.
- **No allocations**: All memory is managed by Python; C++ just uses pointers.
- **Thread safety via TLS**: Each thread has its own codec instances.

```cpp
// File: libffcv.cpp
// Compiled: g++ -shared -fPIC -O3 -o libffcv.so libffcv.cpp -lturbojpeg -lopencv_imgproc

#include <pthread.h>
#include <stdint.h>
#include <turbojpeg.h>
#include <opencv2/imgproc.hpp>

// C-compatible exports for Python ctypes
#define EXPORT extern "C" __attribute__((visibility("default")))

// ============================================================================
// SECTION 1: Thread-Local Storage for TurboJPEG
// ============================================================================

/*
 * WHY THREAD-LOCAL STORAGE?
 * 
 * TurboJPEG handles (tjhandle) are NOT thread-safe. You cannot share
 * a single decompressor across threads.
 * 
 * Options:
 * 1. Global lock: Serializes all JPEG decodes. TERRIBLE for throughput.
 * 2. Create new handle per call: tjInitDecompress() is expensive (~100μs).
 * 3. Thread-local storage: Each thread has its own handle. Best of both worlds.
 * 
 * We use POSIX pthread keys. On first use in a thread, the handle is created
 * and stored. Subsequent calls in that thread reuse the existing handle.
 */

// Keys for thread-local storage
static pthread_key_t key_tj_decompressor;
static pthread_key_t key_tj_transformer;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

// Called once (per process) to create the TLS keys
static void make_keys() {
    pthread_key_create(&key_tj_decompressor, NULL);  // No destructor (handle managed manually)
    pthread_key_create(&key_tj_transformer, NULL);
}

/*
 * Get the current thread's TurboJPEG decompressor.
 * Creates one if it doesn't exist.
 */
static tjhandle get_decompressor() {
    // Ensure keys are initialized (thread-safe, runs only once)
    pthread_once(&key_once, make_keys);
    
    // Try to get existing handle
    tjhandle handle = (tjhandle)pthread_getspecific(key_tj_decompressor);
    
    if (handle == NULL) {
        // First call in this thread: create new handle
        handle = tjInitDecompress();
        pthread_setspecific(key_tj_decompressor, handle);
    }
    
    return handle;
}

static tjhandle get_transformer() {
    pthread_once(&key_once, make_keys);
    
    tjhandle handle = (tjhandle)pthread_getspecific(key_tj_transformer);
    
    if (handle == NULL) {
        handle = tjInitTransform();
        pthread_setspecific(key_tj_transformer, handle);
    }
    
    return handle;
}

// ============================================================================
// SECTION 2: JPEG Decode with Crop and Flip
// ============================================================================

/*
 * imdecode: Decode JPEG with optional crop and horizontal flip.
 * 
 * KEY OPTIMIZATION: We use TurboJPEG's "lossless transform" feature to
 * crop the JPEG in the compressed domain BEFORE decompression. This means
 * we only decompress the pixels we actually need.
 * 
 * For a 1000x1000 image with a 224x224 center crop:
 * - Naive: Decode full image (1M pixels), then crop. ~8ms
 * - Optimized: Transform compressed data, decode 224x224. ~2ms
 * 
 * Parameters:
 *   input_buffer: Pointer to compressed JPEG data
 *   input_size: Size of compressed data in bytes
 *   source_height, source_width: Original image dimensions (from JPEG header)
 *   crop_y, crop_x, crop_height, crop_width: Crop rectangle
 *   hflip: 1 for horizontal flip, 0 for no flip
 *   output_buffer: Pre-allocated buffer for RGB output (H x W x 3)
 *   output_height, output_width: Final output dimensions (may differ from crop for scaling)
 * 
 * Returns: 0 on success, error code otherwise
 */
EXPORT int imdecode(
    uint8_t* input_buffer, uint64_t input_size,
    uint32_t source_height, uint32_t source_width,
    uint32_t crop_y, uint32_t crop_x,
    uint32_t crop_height, uint32_t crop_width,
    int hflip,
    uint8_t* output_buffer,
    uint32_t output_height, uint32_t output_width
) {
    tjhandle decompressor = get_decompressor();
    tjhandle transformer = get_transformer();
    
    // Flags for faster decoding (slight quality reduction, usually imperceptible)
    unsigned long flags = TJFLAG_FASTDCT | TJFLAG_NOREALLOC;
    
    // Buffer for transformed JPEG (compressed, cropped)
    unsigned char* transformed_buf = NULL;
    unsigned long transformed_size = 0;
    
    // ========================================
    // STEP 1: Lossless Transform (Crop + Flip)
    // ========================================
    
    // TurboJPEG can crop and flip in the compressed domain
    tjtransform xform;
    memset(&xform, 0, sizeof(xform));
    
    // Set crop region
    // NOTE: JPEG crops must align to 8 or 16 pixel boundaries (MCU size)
    // TurboJPEG handles this alignment automatically
    xform.r.x = crop_x;
    xform.r.y = crop_y;
    xform.r.w = crop_width;
    xform.r.h = crop_height;
    xform.options = TJXOPT_CROP;
    
    // Set flip
    if (hflip) {
        xform.op = TJXOP_HFLIP;
    }
    
    // Perform the transform
    int result = tjTransform(
        transformer,
        input_buffer, input_size,
        1,                    // Number of transforms (always 1)
        &transformed_buf,     // Output buffer (TurboJPEG will allocate)
        &transformed_size,    // Output size
        &xform,
        TJFLAG_FASTDCT
    );
    
    if (result != 0) {
        return -1;  // Transform failed
    }
    
    // ========================================
    // STEP 2: Decompress (to Pre-Allocated Buffer)
    // ========================================
    
    // Calculate scaling factor
    // TurboJPEG supports scaling to 1/2, 1/4, 1/8, etc.
    // For arbitrary scaling, we decode at nearest supported size then resize
    
    result = tjDecompress2(
        decompressor,
        transformed_buf, transformed_size,
        output_buffer,
        output_width,    // Pitch (row stride) - 0 means tightly packed
        0,               // Pitch
        output_height,
        TJPF_RGB,        // Output pixel format (RGB, not BGR)
        flags
    );
    
    // Free the transformed buffer (allocated by TurboJPEG)
    tjFree(transformed_buf);
    
    return result;
}

// ============================================================================
// SECTION 3: Memory Copy (Bypassing Python)
// ============================================================================

/*
 * memcpy_wrapper: Copy memory without Python overhead.
 * 
 * WHY THIS EXISTS:
 * Python's buffer protocol and numpy's copy operations involve:
 * - Reference counting updates
 * - GIL acquisition (sometimes)
 * - Type checking
 * 
 * For high-throughput pipelines, a direct memcpy is faster.
 * 
 * The GIL is NOT released here because memcpy is already fast enough
 * that the overhead of releasing/reacquiring would dominate.
 */
EXPORT void memcpy_wrapper(
    void* dest, void* src, size_t count
) {
    memcpy(dest, src, count);
}

// ============================================================================
// SECTION 4: OpenCV Resize Wrapper
// ============================================================================

/*
 * resize: Resize image using OpenCV, bypassing Python bindings.
 * 
 * The cv2.resize() Python function has overhead:
 * 1. Creates numpy array wrapper for input/output
 * 2. Performs type checking
 * 3. May copy data to ensure contiguity
 * 
 * By passing raw pointers and dimensions, we skip all of that.
 */
EXPORT void resize(
    int64_t source_ptr,
    int32_t source_height, int32_t source_width,
    int64_t dest_ptr,
    int32_t dest_height, int32_t dest_width,
    int32_t interpolation  // INTER_LINEAR, INTER_AREA, etc.
) {
    // Create cv::Mat headers that wrap existing memory (no copy!)
    cv::Mat source_mat(
        source_height, source_width, CV_8UC3,
        (void*)source_ptr
    );
    cv::Mat dest_mat(
        dest_height, dest_width, CV_8UC3,
        (void*)dest_ptr
    );
    
    // Perform resize
    cv::resize(source_mat, dest_mat, dest_mat.size(), 0, 0, interpolation);
}

// ============================================================================
// SECTION 5: Batch Operations (For Numba Integration)
// ============================================================================

/*
 * resize_batch: Resize multiple images in a single C++ call.
 * 
 * This is useful for Numba: instead of calling C++ N times in a prange loop,
 * we call once with N images. Reduces call overhead.
 */
EXPORT void resize_batch(
    int64_t* source_ptrs,   // Array of N source pointers
    int64_t* dest_ptrs,     // Array of N destination pointers
    int32_t N,
    int32_t source_height, int32_t source_width,
    int32_t dest_height, int32_t dest_width,
    int32_t interpolation
) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        cv::Mat source_mat(
            source_height, source_width, CV_8UC3,
            (void*)source_ptrs[i]
        );
        cv::Mat dest_mat(
            dest_height, dest_width, CV_8UC3,
            (void*)dest_ptrs[i]
        );
        
        cv::resize(source_mat, dest_mat, dest_mat.size(), 0, 0, interpolation);
    }
}
```

## Python Integration: ctypes + Numba

The C++ library is loaded and called from Python using `ctypes`:

```python
# File: libffcv.py

import ctypes
import numpy as np
from pathlib import Path
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

# ============================================================================
# Section 1: Loading the Library
# ============================================================================

def _find_library():
    """Find libffcv.so in the package directory."""
    import ffcv
    lib_dir = Path(ffcv.__file__).parent
    
    # Platform-specific extension
    import platform
    if platform.system() == 'Linux':
        lib_name = 'libffcv.so'
    elif platform.system() == 'Darwin':
        lib_name = 'libffcv.dylib'
    elif platform.system() == 'Windows':
        lib_name = 'ffcv.dll'
    else:
        raise OSError(f"Unsupported platform: {platform.system()}")
    
    lib_path = lib_dir / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(f"Cannot find {lib_name} in {lib_dir}")
    
    return ctypes.CDLL(str(lib_path))

# Load library at module import
_lib = _find_library()

# ============================================================================
# Section 2: Define C Function Signatures
# ============================================================================

# imdecode: JPEG decode with crop and flip
_lib.imdecode.argtypes = [
    ctypes.c_void_p,  # input_buffer (pointer)
    ctypes.c_uint64,  # input_size
    ctypes.c_uint32,  # source_height
    ctypes.c_uint32,  # source_width
    ctypes.c_uint32,  # crop_y
    ctypes.c_uint32,  # crop_x
    ctypes.c_uint32,  # crop_height
    ctypes.c_uint32,  # crop_width
    ctypes.c_int,     # hflip
    ctypes.c_void_p,  # output_buffer (pointer)
    ctypes.c_uint32,  # output_height
    ctypes.c_uint32,  # output_width
]
_lib.imdecode.restype = ctypes.c_int

# resize: OpenCV resize
_lib.resize.argtypes = [
    ctypes.c_int64,   # source_ptr
    ctypes.c_int32,   # source_height
    ctypes.c_int32,   # source_width
    ctypes.c_int64,   # dest_ptr
    ctypes.c_int32,   # dest_height
    ctypes.c_int32,   # dest_width
    ctypes.c_int32,   # interpolation
]
_lib.resize.restype = None

# memcpy_wrapper
_lib.memcpy_wrapper.argtypes = [
    ctypes.c_void_p,  # dest
    ctypes.c_void_p,  # src
    ctypes.c_size_t,  # count
]
_lib.memcpy_wrapper.restype = None

# ============================================================================
# Section 3: High-Level Python Wrappers
# ============================================================================

def decode_jpeg(
    jpeg_bytes: bytes,
    source_height: int,
    source_width: int,
    crop: tuple = None,  # (y, x, h, w) or None for full image
    hflip: bool = False,
    output_size: tuple = None,  # (h, w) or None for original size
) -> np.ndarray:
    """
    Decode JPEG with optional crop, flip, and resize.
    
    This is the Python-friendly wrapper. For maximum performance in loops,
    use the Numba-compatible version below.
    """
    # Determine output dimensions
    if crop is not None:
        crop_y, crop_x, crop_h, crop_w = crop
    else:
        crop_y, crop_x, crop_h, crop_w = 0, 0, source_height, source_width
    
    if output_size is not None:
        out_h, out_w = output_size
    else:
        out_h, out_w = crop_h, crop_w
    
    # Allocate output buffer
    output = np.empty((out_h, out_w, 3), dtype=np.uint8)
    
    # Get pointers
    input_ptr = ctypes.c_char_p(jpeg_bytes)
    output_ptr = output.ctypes.data_as(ctypes.c_void_p)
    
    # Call C++
    result = _lib.imdecode(
        input_ptr, len(jpeg_bytes),
        source_height, source_width,
        crop_y, crop_x, crop_h, crop_w,
        1 if hflip else 0,
        output_ptr,
        out_h, out_w
    )
    
    if result != 0:
        raise RuntimeError(f"JPEG decode failed with error {result}")
    
    return output

# ============================================================================
# Section 4: Numba-Compatible Wrappers
# ============================================================================

# To call C++ from Numba's nopython mode, we need to:
# 1. Get the function pointer address
# 2. Use ctypes.cfunc_type to create a callable

# Get function pointer addresses
_imdecode_addr = ctypes.cast(_lib.imdecode, ctypes.c_void_p).value
_resize_addr = ctypes.cast(_lib.resize, ctypes.c_void_p).value
_memcpy_addr = ctypes.cast(_lib.memcpy_wrapper, ctypes.c_void_p).value

# Create Numba-compatible signatures
from numba import cfunc
from numba.core.typing import cffi_utils

# For Numba to call these, we use intrinsics or the cfunc approach
# This is advanced Numba usage - see FFCV source for full implementation

def get_imdecode_cfunc():
    """
    Returns a function that can be called from Numba nopython mode.
    """
    # This requires careful setup of Numba's cfunc infrastructure
    # Simplified version shown here
    pass


# ============================================================================
# Section 5: Utility Functions
# ============================================================================

def resize(
    source: np.ndarray,
    dest_size: tuple,
    interpolation: int = 1,  # INTER_LINEAR
) -> np.ndarray:
    """
    Resize image using OpenCV via C++.
    """
    out_h, out_w = dest_size
    dest = np.empty((out_h, out_w, 3), dtype=np.uint8)
    
    _lib.resize(
        source.ctypes.data,
        source.shape[0], source.shape[1],
        dest.ctypes.data,
        out_h, out_w,
        interpolation
    )
    
    return dest
```

## Creating Your Own C++ Extensions

Follow this pattern to add new native functionality:

### Step 1: Write the C++ Code

```cpp
// my_extension.cpp
#include <pthread.h>
#include <stdint.h>

// My library's header (e.g., for WebP decoding)
#include <webp/decode.h>

#define EXPORT extern "C" __attribute__((visibility("default")))

// Thread-local storage for any non-thread-safe handles
static pthread_key_t key_my_handle;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

static void make_keys() {
    pthread_key_create(&key_my_handle, NULL);
}

EXPORT int my_decode(
    uint8_t* input_buffer, uint64_t input_size,
    uint8_t* output_buffer,
    int32_t output_height, int32_t output_width
) {
    // Get thread-local handle if needed
    pthread_once(&key_once, make_keys);
    
    // Your decoding logic here
    int result = WebPDecodeRGBInto(
        input_buffer, input_size,
        output_buffer,
        output_height * output_width * 3,
        output_width * 3
    );
    
    return result ? 0 : -1;
}
```

### Step 2: Compile

```bash
# Linux
g++ -shared -fPIC -O3 -o libmyext.so my_extension.cpp -lwebp

# macOS
clang++ -shared -fPIC -O3 -o libmyext.dylib my_extension.cpp -lwebp

# Windows (MSVC)
cl /LD /O2 my_extension.cpp webp.lib /out:myext.dll
```

### Step 3: Bind to Python

```python
import ctypes
import numpy as np

lib = ctypes.CDLL('./libmyext.so')

lib.my_decode.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_int32, ctypes.c_int32
]
lib.my_decode.restype = ctypes.c_int

def decode_webp(webp_bytes: bytes, height: int, width: int) -> np.ndarray:
    output = np.empty((height, width, 3), dtype=np.uint8)
    
    result = lib.my_decode(
        ctypes.c_char_p(webp_bytes), len(webp_bytes),
        output.ctypes.data_as(ctypes.c_void_p),
        height, width
    )
    
    if result != 0:
        raise RuntimeError("WebP decode failed")
    
    return output
```

## Exercises

1.  **Add WebP Support**: Implement a WebP decoder following the pattern above.

2.  **Benchmark TLS vs Global Lock**: Compare throughput of thread-local TurboJPEG handles vs. a single global handle with mutex.

3.  **GPU Decode**: Write a C++ wrapper for NVIDIA NVDEC video decoding and integrate with Python.

4.  **Memory Pool**: Implement a thread-local memory pool in C++ to avoid per-decode allocations.
