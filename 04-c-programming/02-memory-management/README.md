# Memory Management in C

Understanding how memory works - essential for ML systems.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_stack_vs_heap.c` | Stack vs heap allocation comparison |
| `02_custom_allocator.c` | Build a pool allocator |

## Memory Regions

```
┌─────────────────┐ High Address
│   Stack         │ ← Function locals, grows down
│       ↓         │
│                 │
│       ↑         │
│   Heap          │ ← malloc(), grows up
├─────────────────┤
│   BSS           │ ← Uninitialized globals
├─────────────────┤
│   Data          │ ← Initialized globals
├─────────────────┤
│   Text          │ ← Code
└─────────────────┘ Low Address
```

## Allocation Comparison

| Type | Speed | Size | Lifetime |
|------|-------|------|----------|
| Stack | ~1 ns | ~MB | Function scope |
| Heap | ~100 ns | ~GB | Manual |
| Static | 0 ns | Fixed | Program |

## PyTorch CUDA Allocator

PyTorch doesn't call `cudaMalloc` for every tensor:

```python
# This would be SLOW:
for batch in data:
    x = torch.randn(1000, device='cuda')  # cudaMalloc ~1ms each!

# PyTorch does:
# 1. Allocate large pool from CUDA
# 2. Subdivide into blocks
# 3. Reuse freed blocks
# 4. Only release on empty_cache()
```

## Key Functions

```c
void* malloc(size_t size);       // Allocate
void* calloc(size_t n, size_t s); // Allocate + zero
void* realloc(void* p, size_t s); // Resize
void free(void* ptr);             // Deallocate
void* aligned_alloc(size_t a, size_t s); // Aligned
```

## Exercises

1. Measure malloc vs stack allocation speed
2. Build a simple pool allocator
3. Track memory usage in a program
