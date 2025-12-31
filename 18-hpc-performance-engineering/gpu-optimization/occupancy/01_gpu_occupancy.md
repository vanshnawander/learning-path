# GPU Occupancy: Understanding SM Utilization

## What is Occupancy?

**Occupancy** = (Active Warps per SM) / (Maximum Warps per SM)

It measures how well you're utilizing the GPU's parallel execution resources.

```
SM (Streaming Multiprocessor):
┌─────────────────────────────────────────────────────────────┐
│ Warp Schedulers (4)                                         │
├─────────────────────────────────────────────────────────────┤
│ ┌──────┐┌──────┐┌──────┐┌──────┐    Max 64 warps           │
│ │Warp 0││Warp 1││Warp 2││Warp 3│    (2048 threads)         │
│ └──────┘└──────┘└──────┘└──────┘                            │
│ ┌──────┐┌──────┐┌──────┐┌──────┐                            │
│ │Warp 4││Warp 5││Warp 6││Warp 7│    Active: 32 warps       │
│ └──────┘└──────┘└──────┘└──────┘    Occupancy: 50%         │
│    ...      ...     ...     ...                             │
├─────────────────────────────────────────────────────────────┤
│ Registers: 65536 per SM                                     │
│ Shared Memory: 164 KB (A100)                               │
└─────────────────────────────────────────────────────────────┘
```

## Occupancy Limiters

Three resources limit occupancy:

### 1. Registers per Thread

```
Example (A100):
- Max registers per SM: 65536
- Max threads per SM: 2048
- If kernel uses 64 registers/thread:
  Max threads = 65536 / 64 = 1024
  Occupancy = 1024 / 2048 = 50%

- If kernel uses 32 registers/thread:
  Max threads = 65536 / 32 = 2048
  Occupancy = 2048 / 2048 = 100%
```

### 2. Shared Memory per Block

```
Example (A100, 164KB shared per SM):
- If block uses 64KB shared memory:
  Max blocks per SM = 164 / 64 = 2 blocks
  If block has 256 threads: 2 × 256 = 512 threads
  Occupancy = 512 / 2048 = 25%

- If block uses 16KB shared memory:
  Max blocks per SM = 164 / 16 = 10 blocks
  But limited by other factors...
```

### 3. Threads per Block

```
Example:
- Max blocks per SM: 32 (A100)
- Max threads per SM: 2048
- If using 64 threads per block:
  Max blocks = min(32, 2048/64) = 32
  Threads = 32 × 64 = 2048
  Occupancy = 100%

- If using 1024 threads per block:
  Max blocks = min(32, 2048/1024) = 2
  Threads = 2 × 1024 = 2048
  Occupancy = 100%
```

## Occupancy Calculator

### CUDA Runtime API

```cpp
#include <cuda_runtime.h>

__global__ void myKernel(float* data) {
    // kernel code
}

int main() {
    int blockSize = 256;
    int minGridSize, gridSize;
    
    // Automatic occupancy-based launch config
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, 
        myKernel, 0, 0);
    
    // Query occupancy for specific config
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, myKernel, blockSize, 0);
    
    // Calculate occupancy
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    int activeWarps = numBlocks * (blockSize / 32);
    int maxWarps = props.maxThreadsPerMultiProcessor / 32;
    float occupancy = (float)activeWarps / maxWarps;
    
    printf("Block size: %d\n", blockSize);
    printf("Blocks per SM: %d\n", numBlocks);
    printf("Occupancy: %.1f%%\n", occupancy * 100);
}
```

### Nsight Compute Analysis

```bash
# Get occupancy metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name myKernel ./myprogram
```

Key metrics:
- `sm__warps_active.avg.pct_of_peak_sustained_active` - Achieved occupancy
- `launch__occupancy_limit_registers` - Limited by registers?
- `launch__occupancy_limit_shared_mem` - Limited by shared memory?
- `launch__occupancy_limit_blocks` - Limited by blocks?

## Does Higher Occupancy = Better Performance?

**Not always!** High occupancy helps with:
- Hiding memory latency (more warps to schedule)
- Hiding instruction latency

But lower occupancy might be OK if:
- Kernel is compute-bound (not waiting on memory)
- Using more registers enables better optimization
- Shared memory enables data reuse

### Example: Register Pressure Trade-off

```cpp
// Version A: Low register usage, high occupancy
__global__ void kernelA(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Recompute intermediate values
        float a = data[idx];
        float b = sinf(a);
        float c = cosf(a);  // Recalculate a
        data[idx] = b + c;
    }
}

// Version B: More registers, lower occupancy, but faster
__global__ void kernelB(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Keep intermediate values in registers
        float a = data[idx];
        float b = sinf(a);
        float c = cosf(a);
        float d = b * c;
        float e = b + c;
        float f = d / e;
        data[idx] = f + a;
    }
}
```

## Controlling Register Usage

```cpp
// Limit registers per thread (may reduce performance)
__global__ __launch_bounds__(256, 4) void myKernel() {
    // 256 threads/block, 4 blocks/SM minimum
    // Compiler will try to fit in (65536 / 256 / 4) = 64 registers
}

// Or via compiler flag
// nvcc -maxrregcount=32 kernel.cu
```

## Shared Memory Configuration

```cpp
// Set shared memory preference
cudaFuncSetCacheConfig(myKernel, cudaFuncCachePreferShared);
// Options:
// - cudaFuncCachePreferNone (default)
// - cudaFuncCachePreferShared (max shared memory)
// - cudaFuncCachePreferL1 (max L1 cache)
// - cudaFuncCachePreferEqual (balance)

// A100: Can configure shared memory carveout
cudaFuncSetAttribute(myKernel, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 
    100 * 1024);  // 100KB dynamic shared
```

## Occupancy Tuning Workflow

### Step 1: Profile Baseline

```bash
ncu --set full -o baseline.ncu-rep ./myprogram
```

Check:
- Current occupancy
- Which resource is limiting
- Memory vs. compute bound

### Step 2: Identify Limiter

```
If register-limited:
  → Reduce register usage
  → Use --maxrregcount
  → Simplify kernel

If shared-memory-limited:
  → Reduce shared memory per block
  → Use smaller tiles
  → Change cache config

If block-limited:
  → Use smaller blocks
  → Merge work into fewer threads
```

### Step 3: Test Different Configurations

```cpp
// Test different block sizes
for (int blockSize = 64; blockSize <= 1024; blockSize *= 2) {
    dim3 block(blockSize);
    dim3 grid((n + blockSize - 1) / blockSize);
    
    // Benchmark
    myKernel<<<grid, block>>>(data, n);
    cudaDeviceSynchronize();
}
```

### Step 4: Verify Improvement

Compare:
- Execution time
- Memory throughput
- Compute throughput
- Not just occupancy!

## GPU Architecture Limits Reference

| Parameter | V100 | A100 | H100 |
|-----------|------|------|------|
| Max threads/SM | 2048 | 2048 | 2048 |
| Max warps/SM | 64 | 64 | 64 |
| Max blocks/SM | 32 | 32 | 32 |
| Registers/SM | 65536 | 65536 | 65536 |
| Max registers/thread | 255 | 255 | 255 |
| Shared memory/SM | 96KB | 164KB | 228KB |
| Max shared/block | 96KB | 164KB | 228KB |
| Warp size | 32 | 32 | 32 |

## Summary

1. **Occupancy matters for latency-bound kernels**
2. **Profile before optimizing** - use Nsight Compute
3. **Higher isn't always better** - measure actual performance
4. **Trade-offs exist** - registers vs. occupancy
5. **Use tools**: cudaOccupancyMaxPotentialBlockSize

## References

- CUDA Occupancy Calculator (spreadsheet)
- CUDA C++ Programming Guide - Thread Hierarchy
- Nsight Compute documentation
