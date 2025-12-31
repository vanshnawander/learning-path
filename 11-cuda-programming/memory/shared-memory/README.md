# CUDA Shared Memory

Fast, programmer-managed memory within each SM.

## Characteristics

- Per-block (shared among threads in block)
- Very fast (~100x faster than global)
- Limited size (48-164 KB per SM)
- Bank-organized (32 banks)

## Declaration

```cuda
// Static allocation
__shared__ float smem[1024];

// Dynamic allocation
extern __shared__ float dynamic_smem[];

// Kernel launch
kernel<<<grid, block, shared_size>>>();
```

## Bank Conflicts

Shared memory is divided into 32 banks (4 bytes each).

**No conflict**: Each thread accesses different bank
**Conflict**: Multiple threads access same bank
**Broadcast**: All threads access same address (OK)

```
Bank 0:  [0] [32] [64] ...
Bank 1:  [1] [33] [65] ...
Bank 2:  [2] [34] [66] ...
...
Bank 31: [31] [63] [95] ...
```

## Avoiding Conflicts

```cuda
// Conflict-free access (stride = 1)
smem[threadIdx.x] = data;

// Conflict (stride = 32)
smem[threadIdx.x * 32] = data;  // BAD!

// Padding trick
__shared__ float smem[32][33];  // Extra column
smem[threadIdx.x][threadIdx.y] = data;
```

## Use Cases
1. Data reuse (matrix multiply tiling)
2. Communication between threads
3. Reduction operations
4. Transpose operations
