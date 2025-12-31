# Struct Patterns and Data-Oriented Design

How you structure data determines performance.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_data_oriented.c` | AoS vs SoA comparison |

## Array of Structures (AoS)

```c
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
struct Particle particles[1000];

// Memory: [x0,y0,z0,vx0,vy0,vz0][x1,y1,z1,vx1,vy1,vz1]...
```

**Pros**: Natural OOP style, good for accessing all fields of one item
**Cons**: Bad cache usage when only accessing one field

## Structure of Arrays (SoA)

```c
struct Particles {
    float x[1000], y[1000], z[1000];
    float vx[1000], vy[1000], vz[1000];
};

// Memory: [x0,x1,x2,...][y0,y1,y2,...][z0,z1,z2,...]
```

**Pros**: Excellent cache usage, perfect for SIMD
**Cons**: Less intuitive, multiple arrays to manage

## PyTorch Tensors ARE SoA

```python
# A tensor is contiguous array of one type
weights = torch.randn(1000, 1000)  # 1M floats in a row

# NOT:
# [(w0, grad0, mom0), (w1, grad1, mom1), ...]

# This is why tensor operations are fast!
```

## Tensor Memory Formats

### NCHW (default)
```
Batch × Channels × Height × Width
# Channels are contiguous in memory
```

### NHWC (channels_last)
```
Batch × Height × Width × Channels
# Spatial positions contiguous - better for Tensor Cores!
```

```python
x = x.to(memory_format=torch.channels_last)
```

## Performance Impact

| Pattern | Cache Efficiency | SIMD Friendly |
|---------|------------------|---------------|
| AoS (one field) | Poor | No |
| AoS (all fields) | OK | No |
| SoA (one field) | Excellent | Yes |
| SoA (all fields) | Good | Yes |
