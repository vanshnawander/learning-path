# JIT Compilation with Numba: A Complete Deep Dive

## What is JIT Compilation?

**JIT (Just-In-Time) Compilation** is a technique where code is compiled to machine code at runtime, just before it's executed, rather than ahead of time (like C/C++) or interpreted line-by-line (like Python).

### The Python Execution Model

```python
# When you run this Python code:
for i in range(1000000):
    x = i * 2 + 1
```

The Python interpreter does the following **for each iteration**:
1.  Fetch the bytecode instruction for `i * 2`.
2.  Look up `i` in the local namespace dictionary.
3.  Look up `2` (a constant, but still checked).
4.  Dispatch to the `PyNumber_Multiply` function.
5.  Check the types of `i` and `2`.
6.  Call the appropriate multiplication function.
7.  Create a new Python `int` object for the result.
8.  Repeat for `+ 1`.
9.  Assign to `x` (another dictionary update).

**This overhead exists even for trivial operations.** For 1 million iterations, you're paying this overhead 1 million times.

### The Numba Approach

Numba analyzes your Python function, infers types, and compiles it directly to LLVM IR (Intermediate Representation), which is then compiled to native machine code.

```python
from numba import njit

@njit
def fast_loop():
    total = 0
    for i in range(1000000):
        total += i * 2 + 1
    return total
```

After compilation, this becomes roughly equivalent to:

```c
int64_t fast_loop() {
    int64_t total = 0;
    for (int64_t i = 0; i < 1000000; i++) {
        total += i * 2 + 1;
    }
    return total;
}
```

No type checks. No dictionary lookups. No object creation. Just raw register operations.

## How Numba Works Under the Hood

### Step 1: Type Inference

Numba traces through your function and infers the type of every variable.

```python
@njit
def example(a, b):
    c = a + b   # c is inferred to be float64 if a and b are float64
    return c
```

If Numba can't infer types (e.g., because you're using unsupported Python features), compilation fails.

### Step 2: IR Generation

Numba converts your function to an internal Intermediate Representation (IR) that represents operations abstractly.

### Step 3: LLVM Compilation

The IR is passed to LLVM, the same compiler infrastructure used by Clang (Apple's C/C++ compiler). LLVM performs:
- Dead code elimination
- Loop unrolling
- SIMD vectorization (using AVX2/AVX-512 on x86)
- Register allocation
- Machine code generation

### Step 4: Caching

The compiled machine code is cached (if `cache=True`) so subsequent runs skip compilation.

## The `@njit` Decorator

`@njit` is shorthand for `@jit(nopython=True)`. It means: *compile this function in "nopython mode" — no Python objects, no Python API calls, pure machine code.*

### Basic Usage

```python
import numba as nb
import numpy as np

@nb.njit
def dot_product(a, b):
    """
    Compute dot product of two 1D arrays.
    """
    n = len(a)
    result = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result

# First call: compiles, then runs
result = dot_product(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
# Subsequent calls: runs cached machine code
```

### Type Signatures (Optional but Recommended)

Specifying types explicitly avoids recompilation when types change and provides documentation.

```python
# Signature: output_type(input_type1, input_type2, ...)
@nb.njit('float64(float64[:], float64[:])')
def dot_product_typed(a, b):
    n = len(a)
    result = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result
```

### Type Notation

| Signature | Meaning |
|-----------|---------|
| `float64` | Scalar double-precision float |
| `float64[:]` | 1D array of float64 (contiguous not guaranteed) |
| `float64[::1]` | 1D C-contiguous array of float64 |
| `float64[:, ::1]` | 2D array, contiguous in last dimension (C order) |
| `nb.types.Tuple((nb.float64, nb.int64))` | Tuple of (float64, int64) |

## Parallel Execution with `prange`

By default, Numba functions run on a single thread. For embarrassingly parallel loops, use `prange`:

```python
@nb.njit(parallel=True)
def parallel_sum(arr):
    n = len(arr)
    total = 0.0
    for i in nb.prange(n):  # This loop runs in parallel
        total += arr[i]
    return total
```

### How `prange` Works

1.  Numba detects the `prange` loop.
2.  It partitions the iteration space across threads (by default, one per CPU core).
3.  Each thread computes a partial result.
4.  Numba combines the partials (reduction) automatically.

### When to Use `parallel=True`

| Use Case | Recommended |
|----------|-------------|
| Independent iterations (no data dependencies) | ✅ Yes |
| Reduction operations (sum, min, max) | ✅ Yes (automatic) |
| Nested loops (parallelize outer) | ✅ Yes |
| Small loops (< 1000 iterations) | ❌ No (overhead > benefit) |
| Complex data dependencies | ❌ No |

## The GIL and `nogil=True`

Python's **Global Interpreter Lock (GIL)** prevents true parallelism in multi-threaded Python code. But Numba can release the GIL!

### The Problem

```python
import threading

def python_work():
    # This holds the GIL
    for i in range(10000000):
        pass

threads = [threading.Thread(target=python_work) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Despite 4 threads, only one runs at a time due to GIL
```

### The Solution: `nogil=True`

```python
@nb.njit(nogil=True)
def numba_work(arr, result, idx):
    # This releases the GIL
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    result[idx] = total

# Now multiple threads can run simultaneously
import threading
import numpy as np

arr = np.random.random(10000000)
result = np.zeros(4)

threads = [
    threading.Thread(target=numba_work, args=(arr, result, i))
    for i in range(4)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
# True parallelism!
```

This is why FFCV can use threads (not processes) for parallel decoding: Numba functions release the GIL, allowing true parallelism without the overhead of multiprocessing.

## Supported NumPy Operations

Numba supports a substantial subset of NumPy. Here's a reference:

### ✅ Fully Supported

```python
@nb.njit
def supported_ops():
    # Array creation
    a = np.zeros((10, 10), dtype=np.float64)
    b = np.ones((10, 10), dtype=np.float32)
    c = np.empty((10,), dtype=np.int32)
    d = np.arange(10)
    e = np.linspace(0, 1, 100)
    f = np.zeros_like(a)
    g = np.empty_like(b)
    
    # Indexing and slicing
    x = a[0, :]
    y = a[:, 0]
    z = a[2:5, 3:7]
    w = a[np.array([0, 2, 4])]  # Integer array indexing
    
    # Math operations
    s = np.sum(a)
    m = np.mean(a)
    mx = np.max(a)
    mn = np.min(a)
    st = np.std(a)
    
    # Element-wise
    sq = np.sqrt(a)
    ex = np.exp(a)
    lg = np.log(a + 1)
    sn = np.sin(a)
    
    # Linear algebra
    dot = np.dot(a, b)
    
    # Sorting
    sorted_arr = np.sort(a, axis=0)
    indices = np.argsort(a, axis=0)
    
    # Searching
    idx = np.searchsorted(np.sort(a.ravel()), 0.5)
    wh = np.where(a > 0.5)
    
    return s
```

### ❌ Not Supported

```python
# These will cause compilation errors in @njit:

# np.array() with Python lists
# arr = np.array([1, 2, 3])  # ❌

# String operations
# s = np.array(['a', 'b', 'c'])  # ❌

# Object dtype
# o = np.array([{}, {}])  # ❌

# Most scipy functions
# from scipy.stats import norm
# norm.pdf(0)  # ❌

# Python's random module
# import random
# random.random()  # ❌  (use np.random instead)
```

### Workaround: Object Mode Fallback

If you need unsupported features, you can use `forceobj=True`, but you lose all performance benefits:

```python
@nb.jit(forceobj=True)
def slow_but_flexible():
    # This runs in Python, not compiled
    return [x * 2 for x in range(10)]
```

## Memory Allocation Inside JIT Functions

A key rule: **Minimize allocations inside hot loops.**

### ❌ Bad: Allocating on Every Iteration

```python
@nb.njit
def bad_allocation(n):
    results = []
    for i in range(n):
        results.append(i * 2)  # ❌ Python list operations (slow)
    return results
```

### ✅ Good: Pre-Allocated Output

```python
@nb.njit
def good_allocation(n):
    result = np.empty(n, dtype=np.int64)  # One allocation
    for i in range(n):
        result[i] = i * 2  # Direct memory write
    return result
```

### Best: Caller Provides Output Buffer

```python
@nb.njit
def best_allocation(n, output):
    # No allocation at all inside the function
    for i in range(n):
        output[i] = i * 2

# Caller manages memory
buffer = np.empty(1000000, dtype=np.int64)
best_allocation(1000000, buffer)
```

This is exactly how FFCV works: all output buffers are pre-allocated, and JIT functions write directly into them.

## Calling C/C++ from Numba

For operations that require external libraries (like TurboJPEG for JPEG decoding), Numba can call C functions via `ctypes` or `cffi`.

### Using `ctypes`

```python
import ctypes
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from numba.core import cgutils

# Load a C library
libc = ctypes.CDLL("libc.so.6")

# Define the C function signature
libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
libc.memcpy.restype = ctypes.c_void_p

# Get the function pointer
memcpy_ptr = ctypes.cast(libc.memcpy, ctypes.c_void_p).value

@njit
def fast_copy(src, dst):
    """
    Call C memcpy from Numba.
    """
    n = len(src)
    # Get raw pointers
    src_ptr = src.ctypes.data
    dst_ptr = dst.ctypes.data
    
    # This doesn't work directly in njit, need intrinsic...
```

For complex C interop, FFCV uses a different pattern: call the C function *outside* the Numba function, or use `cffi` with Numba's `cffi` support.

## FFCV's Code Generation Pattern

FFCV doesn't just use `@njit` directly. It **generates code at runtime**, then compiles that generated code. This allows for dynamic pipelines.

### The Pattern

```python
class Operation:
    """Base class for all pipeline operations."""
    
    def generate_code(self):
        """
        Return a function that will be JIT-compiled.
        
        The returned function has no dependency on `self` (no closures to Python objects).
        All necessary data is passed as arguments.
        """
        raise NotImplementedError
    
    def declare_state_and_memory(self, previous_state):
        """
        Declare outputs: shape, dtype, memory requirements.
        """
        raise NotImplementedError


class NormalizeOperation(Operation):
    def __init__(self, mean, std):
        # Convert to numpy arrays for JIT compatibility
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def generate_code(self):
        # Capture values (not self) in closure
        mean = self.mean
        std = self.std
        
        @nb.njit(parallel=True, nogil=True)
        def normalize(input_arr, output_arr):
            """
            Normalize images: output = (input - mean) / std
            Input shape: (batch, C, H, W)
            """
            batch, c, h, w = input_arr.shape
            for b in nb.prange(batch):
                for ch in range(c):
                    for y in range(h):
                        for x in range(w):
                            output_arr[b, ch, y, x] = (
                                (input_arr[b, ch, y, x] / 255.0 - mean[ch]) / std[ch]
                            )
        
        return normalize
    
    def declare_state_and_memory(self, previous_state):
        # Output has same shape as input, but dtype float32
        return {
            'shape': previous_state['shape'],
            'dtype': np.float32,
        }, {
            'shape': previous_state['shape'],
            'dtype': np.float32,
        }


class Compiler:
    """Compile multiple operations into a single pipeline."""
    
    @staticmethod
    def compile(code_func, signature=None):
        """
        Compile a code function with Numba.
        """
        return nb.njit(signature, nogil=True, parallel=True)(code_func)
```

### Why Code Generation?

1.  **Dynamic configuration**: The `mean` and `std` values are baked into the compiled code, not looked up at runtime.
2.  **No Python overhead**: The compiled function has no reference to `self` or Python objects.
3.  **Fusion**: Multiple operations can be combined into a single function.

## Caching Compiled Functions

Compilation takes time (100ms-1s depending on complexity). Numba can cache compiled code:

```python
@nb.njit(cache=True)
def cached_function(x):
    return x * 2

# First run: compiles and saves to __pycache__
# Subsequent runs: loads from cache (milliseconds)
```

Cache is invalidated when:
- The function's source code changes.
- The Numba version changes.
- The function's dependencies change.

## Debugging Numba Functions

### Check Generated LLVM IR

```python
@nb.njit
def my_func(x):
    return x * 2

# Force compilation
my_func(np.array([1.0]))

# Print LLVM IR
print(my_func.inspect_llvm(my_func.signatures[0]))
```

### Check Assembly

```python
print(my_func.inspect_asm(my_func.signatures[0]))
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `TypingError` | Numba can't infer types | Add type hints or remove unsupported operations |
| `UnsupportedError` | Using unsupported feature | Refactor to use supported NumPy operations |
| `LoweringError` | Internal compilation error | Simplify code, check for edge cases |

## Performance Tips

1.  **Minimize transitions**: Don't call Python from Numba or Numba from Python in a loop.
2.  **Pre-allocate**: Allocate all memory before the JIT function.
3.  **Use contiguous arrays**: `np.ascontiguousarray()` before passing to Numba.
4.  **Batch operations**: Process batches, not single samples.
5.  **Profile first**: Use `%timeit` to find real bottlenecks.
6.  **Avoid small functions**: Combine operations to reduce call overhead.

## Exercises

1.  **Implement bilinear interpolation**: Write a `@njit` function that resizes a 2D image using bilinear interpolation.

2.  **Benchmark**: Compare `@njit` vs. pure Python for computing a moving average of 1 million elements.

3.  **Parallel histogram**: Implement a parallel histogram computation using `prange`.

4.  **Call C memcpy**: Using `ctypes`, call C's `memcpy` from a Numba function (advanced).
