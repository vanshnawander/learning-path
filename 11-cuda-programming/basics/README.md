# CUDA Basics

Getting started with GPU programming.

## Hello World

```cuda
#include <stdio.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", 
           blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();  // 2 blocks, 4 threads each
    cudaDeviceSynchronize();
    return 0;
}
```

Compile: `nvcc hello.cu -o hello`

## Thread Hierarchy

```
Grid
├── Block (0,0)
│   ├── Thread (0,0,0)
│   ├── Thread (1,0,0)
│   └── ...
├── Block (1,0)
│   └── ...
└── ...
```

### Built-in Variables
- `threadIdx.x/y/z` - Thread index within block
- `blockIdx.x/y/z` - Block index within grid
- `blockDim.x/y/z` - Block dimensions
- `gridDim.x/y/z` - Grid dimensions

### Global Thread ID (1D)
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

## Vector Addition

```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

## Error Checking

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", \
                   __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&ptr, size));
```

## Exercises
1. Implement vector subtraction
2. Add error checking to vector add
3. Experiment with different block sizes
