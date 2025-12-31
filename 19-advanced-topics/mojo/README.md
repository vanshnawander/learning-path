# Mojo Programming

Python syntax with systems-level performance.

## What is Mojo?

Created by Modular (Chris Lattner, creator of LLVM/Swift):
- Python-compatible syntax
- Compiled, not interpreted
- Manual memory control
- SIMD and GPU support (coming)

## Why Mojo?

```
Python:     1x (baseline)
NumPy:      ~100x
PyTorch:    ~1000x
Mojo:       ~68000x (claimed)
```

## Key Features

### 1. `fn` vs `def`
```mojo
# def: Python-like, dynamic
def flexible_function(x):
    return x + 1

# fn: Strict, optimized
fn fast_function(x: Int) -> Int:
    return x + 1
```

### 2. Ownership System
```mojo
fn take_ownership(owned s: String):
    print(s)  # s is consumed

fn borrow(borrowed s: String):
    print(s)  # s is borrowed

fn mutate(inout s: String):
    s += "!"  # s is modified
```

### 3. SIMD Types
```mojo
from math import sqrt

fn vectorized_sqrt():
    var vec = SIMD[DType.float32, 8](1, 2, 3, 4, 5, 6, 7, 8)
    var result = sqrt(vec)  # 8 sqrts in parallel!
```

### 4. Compile-Time Parameters
```mojo
struct Matrix[rows: Int, cols: Int]:
    var data: StaticTuple[Float32, rows * cols]
    
    fn __matmul__(self, other: Matrix[cols, N]) -> Matrix[rows, N]:
        ...
```

## Current Status
- CPU support: ✅
- GPU support: In development
- Max integration: Available
- Open source: Partial

## Resources
- docs.modular.com
- Mojo playground
