# Memory Alignment

Critical for performance on both CPU and GPU.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_alignment_basics.c` | Alignment fundamentals |

## What is Alignment?

A memory address is **N-byte aligned** if it's divisible by N.

```
0x1000: 16-aligned, 8-aligned, 4-aligned ✓
0x1004: 4-aligned ✓, NOT 8-aligned ✗
0x1001: NOT aligned to anything > 1 ✗
```

## Why Alignment Matters

### CPU
- Unaligned access may need 2 memory operations
- SIMD requires alignment (or slower unaligned loads)
- Cache lines are 64-byte aligned

### GPU (Critical!)
- Memory coalescing requires alignment
- 128-byte aligned for best performance
- Tensor Cores need specific alignment

## Alignment in C

```c
// Check alignment
#include <stdalign.h>
printf("%zu\n", alignof(double));  // Usually 8

// Aligned allocation
void* ptr = aligned_alloc(64, size);  // 64-byte aligned

// Struct alignment
struct __attribute__((aligned(64))) CacheLine {
    int data[16];
};
```

## PyTorch Tensor Alignment

```python
# Tensors are typically 64-byte aligned
x = torch.randn(1000)
print(x.data_ptr() % 64)  # Usually 0

# pin_memory for aligned DMA transfers
loader = DataLoader(..., pin_memory=True)
```

## Common Alignment Requirements

| Use Case | Alignment |
|----------|-----------|
| Cache line | 64 bytes |
| AVX | 32 bytes |
| AVX-512 | 64 bytes |
| CUDA coalescing | 128 bytes |
| Tensor Core | 16 bytes |
