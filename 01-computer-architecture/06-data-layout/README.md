# Data Layout and Memory Organization

How you organize data determines performance.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_soa_vs_aos.c` | Array of Structs vs Struct of Arrays |

## AoS vs SoA

### Array of Structures (AoS)
```c
struct Point { float x, y, z; };
Point points[N];
// Memory: [x0,y0,z0][x1,y1,z1][x2,y2,z2]...
```

### Structure of Arrays (SoA)
```c
struct Points {
    float x[N], y[N], z[N];
};
// Memory: [x0,x1,x2...][y0,y1,y2...][z0,z1,z2...]
```

### When to Use

| Access Pattern | Best Layout |
|----------------|-------------|
| All fields of one item | AoS |
| One field of all items | SoA |
| SIMD processing | SoA |

## Tensor Layouts in PyTorch

### NCHW (default)
```
Batch × Channels × Height × Width
Channels are contiguous
```

### NHWC (channels_last)
```
Batch × Height × Width × Channels
Spatial positions contiguous
Better for Tensor Cores!
```

```python
x = x.to(memory_format=torch.channels_last)
```

## Row-Major vs Column-Major

```
Row-major (C, PyTorch): matrix[row][col]
  Memory: [row0_col0, row0_col1, row0_col2, ...]

Column-major (Fortran, MATLAB): matrix[col][row]
  Memory: [row0_col0, row1_col0, row2_col0, ...]
```

## Strided Access

```python
x = torch.randn(100, 100)
x[:, 0]  # Column access - strided, slower
x[0, :]  # Row access - contiguous, faster
```

`x.contiguous()` copies data to match logical layout.
