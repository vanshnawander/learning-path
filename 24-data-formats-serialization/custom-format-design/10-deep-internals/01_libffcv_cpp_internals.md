# LibFFCV: The C++ Core Internals

## Why C++ is Mandatory for FFCV

FFCV achieves its speed by identifying the bottlenecks that **cannot** be solved in Python, even with Numba. These are:
1.  **JPEG Decoding**: The most compute-intensive part of computer vision loading.
2.  **Memory Copying**: Python's `memcpy` often involves overhead or GIL contention.
3.  **Resizing**: OpenCV's Python bindings have overhead; calling C++ `cv::resize` directly in a loop is faster.

## The Architecture of `libffcv.cpp`

The C++ library is a shared object (`.so` or `.dll`) loaded via `ctypes`. It exposes "flat" C functions that take raw pointers (int64 identifiers) to avoid any Python object overhead.

### 1. Thread-Local Storage for TurboJPEG

TurboJPEG handles are **not thread-safe**. Since FFCV is heavily multi-threaded, every worker thread needs its own decompressor instance. FFCV solves this using `pthread` thread-local storage (TLS).

```cpp
// Global keys to access thread-specific instances
static pthread_key_t key_tj_transformer;
static pthread_key_t key_tj_decompressor;

// Called once generally to initialize keys
static void make_keys()
{
    pthread_key_create(&key_tj_decompressor, NULL);
    pthread_key_create(&key_tj_transformer, NULL);
}

// Inside the decode function:
EXPORT int imdecode(...) {
    // Ensure keys exist
    pthread_once(&key_once, make_keys);

    // Get THIS thread's transformer
    if ((tj_transformer = pthread_getspecific(key_tj_transformer)) == NULL)
    {
        // If not exists, malloc text, and set it
        tj_transformer = tjInitTransform();
        pthread_setspecific(key_tj_transformer, tj_transformer);
    }
    // ... same for decompressor
}
```

**Why this matters**: If you used a global static `tjhandle`, it would segfault under load. If you created a new handle every call (`tjInitDecompress`), you'd thrash memory. TLS is the only generic, high-performance solution.

### 2. The `imdecode` Implementation

This function is a masterpiece of optimization. It combines cropping, flipping, and decoding into a minimal number of passes.

#### The Transform Step (Crop/Flip)
TurboJPEG can perform "lossless" transforms on the compressed bitstream *before* full decompression. This is faster because it works on DCT coefficients directly.

```cpp
tjtransform xform;
if (hflip) xform.op = TJXOP_HFLIP;
// ... set crop rect ...
xform.options |= TJXOPT_CROP;

// Transform compressed -> compressed (subset)
tjTransform(tj_transformer, input_buffer, ..., &dstBuf, &dstSize, &xform, TJFLAG_FASTDCT);
```

#### The Decode Step
Now we decompress only the relevant part (or the transformed buffer).

```cpp
tjDecompress2(tj_decompressor, dstBuf, dstSize, output_buffer,
    TJSCALED(crop_width, scaling), 
    0, // pitch
    TJSCALED(crop_height, scaling),
    TJPF_RGB, 
    TJFLAG_FASTDCT | TJFLAG_NOREALLOC
);
```

*   `TJFLAG_FASTDCT`: Uses a faster, slightly less accurate IDCT algorithm. Essential for high throughput.
*   `TJFLAG_NOREALLOC`: Tells TurboJPEG "I have already allocated `output_buffer` to the correct size, do not try to help me." This allows FFCV to manage memory in Python (pre-allocated buffers) and just pass pointers to C++.

### 3. The OpenCV Resize Wrapper

FFCV bypasses the Python `cv2` bindings to call `cv::resize` on raw memory pointers.

```cpp
EXPORT void resize(int64_t source_p, ..., int64_t dest_p, ...) {
    // Cast raw int64 pointers to OpenCV Mat wrappers
    // This does NOT copy data; it just creates a view header
    cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
    cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
    
    // The actual resize
    cv::resize(source, dest, ...);
}
```

This avoids the overhead of creating Python `numpy.ndarray` objects for every single image in a batch, which would trigger reference counting and GC updates.

## Python Integration: `ctypes` + Numba

The `ctypes` glue is handled in `ffcv/libffcv.py`.

```python
import ctypes
from numba import njit

# Load library
lib = ctypes.CDLL(lib_path)

# 1. Define C signature
lib.imdecode.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, # input buffer, size
    ctypes.c_uint32, ctypes.c_uint32, # source h, w
    ctypes.c_void_p,                  # output buffer
    # ... args ...
]

# 2. Expose to Numba
# Numba needs to know about this external C function to call it 
# from nopython mode without holding the GIL
@njit
def my_fast_decoder(ptr, ...):
    lib.imdecode(ptr, ...) 
```

## Creating Your Own C++ Extensions

To support new modalities (e.g., Video Hardware Decode), you would follow this pattern:

1.  **Write C++**:
    *   Include headers (`libavcodec`, `nvdec`).
    *   Use `pthread` keys for any context structures.
    *   Export a simple C-compatible function (`extern "C"`).
    *   Accept `void*` or `int64_t` for memory addresses.
2.  **Compile**:
    *   Standard `g++ -shared -fPIC -o libmyformat.so code.cpp`.
3.  **Bind**:
    *   Use `ctypes` to load the `.so`.
    *   Pass pointers from `numpy.ndarray.ctypes.data`.

This architecture is the "nuclear option" for optimization: use it only when Numba fails to generate optimal assembly or when you need libraries (JPEG, Video) that are native by nature.
