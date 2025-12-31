# Pointers Deep Dive

Pointers are THE fundamental concept in systems programming. 
Every ML framework, every CUDA kernel, every data loader uses them.

## Why Pointers Matter for ML

1. **Zero-copy data sharing** - Pass pointers, not data
2. **Memory-mapped files** - FFCV .beton access
3. **GPU memory** - CUDA pointers to device memory
4. **Custom allocators** - PyTorch caching allocator
5. **C extensions** - PyTorch/NumPy C API

## Files in This Directory

| File | Description |
|------|-------------|
| `01_pointer_basics.c` | Fundamentals and mental model |
| `02_pointer_arithmetic.c` | Traversing memory |
| `03_pointers_and_arrays.c` | The array-pointer relationship |
| `04_function_pointers.c` | Callbacks and dispatch |
| `05_void_pointers.c` | Generic programming in C |
| `06_double_pointers.c` | Modifying pointers, 2D arrays |

## Mental Model

```
┌─────────────────────┐
│ Variable x          │
│ Value: 42           │
│ Address: 0x7fff1234 │
└─────────────────────┘
         ↑
         │
┌─────────────────────┐
│ Pointer p           │
│ Value: 0x7fff1234   │  ← p stores the ADDRESS of x
│ Address: 0x7fff5678 │
└─────────────────────┘

int x = 42;
int* p = &x;    // p points to x
*p = 100;       // x is now 100 (dereferencing)
```

## Common Patterns

### Zero-copy function
```c
void process(float* data, int n) {
    // No copy! Works directly on caller's data
    for (int i = 0; i < n; i++) {
        data[i] *= 2;
    }
}
```

### Memory-mapped access
```c
float* weights = mmap(...);  // Pointer to file contents
float val = weights[1000];   // Direct access, no read()
```
