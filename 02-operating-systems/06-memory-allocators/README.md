# Memory Allocators

How malloc works - and why PyTorch has a caching allocator.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_malloc_internals.c` | Allocator behavior and pools |

## How malloc Works

### Size Classes
Allocators group allocations by size:
```
8, 16, 32, 64, 128, 256, 512, 1024, 2048...
```
Request 20 bytes → get 32-byte block

### Free Lists
Freed blocks go to a free list, not back to OS.
Next malloc of same size reuses freed block.

### Arena/Pool
Large allocations get their own pages.
Small allocations share pages.

## Popular Allocators

| Allocator | Used By | Features |
|-----------|---------|----------|
| glibc malloc | Default Linux | General purpose |
| jemalloc | Facebook, Redis | Low fragmentation |
| tcmalloc | Google | Thread caching |
| mimalloc | Microsoft | Fast, compact |

## PyTorch CUDA Allocator

```python
# See current memory
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

# Clear cache
torch.cuda.empty_cache()
```

### Why Caching?
- `cudaMalloc`: ~1 ms (SLOW!)
- Pool reuse: ~1 µs (1000x faster)

### How It Works
1. Allocate large chunk from CUDA
2. Split into blocks
3. Track free/used blocks
4. Reuse freed blocks
5. Only release to CUDA on `empty_cache()`

## Memory Fragmentation

```
[Used][Free][Used][Free][Used]
         ↑
Cannot allocate large contiguous block!
```

This causes OOM even with "enough" free memory.
