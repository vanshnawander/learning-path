/*
 * Memory Coalescing Demonstration
 * 
 * Shows the dramatic performance difference between coalesced 
 * and non-coalesced memory access patterns on GPUs.
 * 
 * Compile: nvcc -O3 01_memory_coalescing.cu -o coalescing
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (32 * 1024 * 1024)  // 32M elements
#define BLOCK_SIZE 256
#define ITERATIONS 100

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/*
 * COALESCED ACCESS
 * 
 * Memory layout:
 * Address:  0    4    8    12   16   20   24   28   ...
 * Data:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  ...
 * 
 * Thread access pattern:
 * Thread 0 → Address 0
 * Thread 1 → Address 4
 * Thread 2 → Address 8
 * ...
 * 
 * All 32 threads in a warp access consecutive memory locations
 * → Single memory transaction (128 bytes)
 */
__global__ void coalesced_read(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

/*
 * STRIDED ACCESS (Non-coalesced)
 * 
 * Thread access pattern with stride=32:
 * Thread 0 → Address 0
 * Thread 1 → Address 128 (32 * 4 bytes)
 * Thread 2 → Address 256
 * ...
 * 
 * Each thread accesses different cache lines
 * → 32 separate memory transactions!
 */
__global__ void strided_read(float* out, const float* in, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (idx < n) {
        out[idx] = in[strided_idx] * 2.0f;
    }
}

/*
 * STRUCTURE OF ARRAYS vs ARRAY OF STRUCTURES
 * 
 * AoS: [x0,y0,z0,w0][x1,y1,z1,w1][x2,y2,z2,w2]...
 * SoA: [x0,x1,x2,...][y0,y1,y2,...][z0,z1,z2,...][w0,w1,w2,...]
 */

struct Particle_AoS {
    float x, y, z, w;
};

// AoS access (partially coalesced, but wastes bandwidth if only need x)
__global__ void aos_read(float* out, const Particle_AoS* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Reading x loads entire struct (16 bytes), but we only need 4
        out[idx] = particles[idx].x * 2.0f;
    }
}

// SoA access (fully coalesced for single field)
__global__ void soa_read(float* out, const float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] * 2.0f;
    }
}

/*
 * MISALIGNED ACCESS
 * 
 * Starting from non-aligned address causes extra transactions
 */
__global__ void misaligned_read(float* out, const float* in, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - offset) {
        out[idx] = in[idx + offset] * 2.0f;
    }
}

// Benchmark helper
float benchmark_kernel(void (*kernel)(float*, const float*, int), 
                       float* d_out, float* d_in, int n,
                       const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Warmup
    kernel<<<grid, block>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        kernel<<<grid, block>>>(d_out, d_in, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float bytes = 2.0f * n * sizeof(float) * ITERATIONS;  // read + write
    float bandwidth = bytes / (ms / 1000.0f) / 1e9;
    
    printf("%-25s: %8.2f ms, %8.2f GB/s\n", name, ms, bandwidth);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return bandwidth;
}

// Wrapper for strided kernel
__global__ void strided_wrapper(float* out, const float* in, int n) {
    strided_read<<<gridDim, blockDim>>>(out, in, n, 32);
}

int main() {
    printf("=== Memory Coalescing Benchmark ===\n");
    printf("Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Iterations: %d\n\n", ITERATIONS);
    
    // Allocate memory
    float *h_in, *d_in, *d_out;
    h_in = (float*)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Bandwidth: %.1f GB/s\n\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    printf("%-25s  %10s  %10s\n", "Access Pattern", "Time", "Bandwidth");
    printf("%-25s  %10s  %10s\n", "--------------", "----", "---------");
    
    // Benchmark coalesced
    float bw_coalesced = benchmark_kernel(coalesced_read, d_out, d_in, N, "Coalesced");
    
    // Benchmark strided (stride=32)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    strided_read<<<grid, block>>>(d_out, d_in, N, 32);  // warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        strided_read<<<grid, block>>>(d_out, d_in, N, 32);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float bytes = 2.0f * N * sizeof(float) * ITERATIONS;
    float bw_strided = bytes / (ms / 1000.0f) / 1e9;
    printf("%-25s: %8.2f ms, %8.2f GB/s\n", "Strided (stride=32)", ms, bw_strided);
    
    // Benchmark AoS vs SoA
    Particle_AoS* d_aos;
    float *d_soa_x;
    CUDA_CHECK(cudaMalloc(&d_aos, N * sizeof(Particle_AoS)));
    CUDA_CHECK(cudaMalloc(&d_soa_x, N * sizeof(float)));
    
    // AoS benchmark
    aos_read<<<grid, block>>>(d_out, d_aos, N);  // warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        aos_read<<<grid, block>>>(d_out, d_aos, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float bw_aos = (N * sizeof(float) + N * sizeof(float)) * ITERATIONS / (ms / 1000.0f) / 1e9;
    printf("%-25s: %8.2f ms, %8.2f GB/s (effective)\n", "AoS (read x only)", ms, bw_aos);
    
    // SoA benchmark  
    soa_read<<<grid, block>>>(d_out, d_soa_x, N);  // warmup
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        soa_read<<<grid, block>>>(d_out, d_soa_x, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float bw_soa = bytes / (ms / 1000.0f) / 1e9;
    printf("%-25s: %8.2f ms, %8.2f GB/s\n", "SoA (read x only)", ms, bw_soa);
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Coalesced is %.1fx faster than strided\n", bw_coalesced / bw_strided);
    printf("SoA is %.1fx faster than AoS for single-field access\n", bw_soa / bw_aos);
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. Ensure consecutive threads access consecutive memory\n");
    printf("2. Use Structure of Arrays (SoA) when accessing single fields\n");
    printf("3. Align data to 128-byte boundaries\n");
    printf("4. Avoid strided access patterns\n");
    
    // Cleanup
    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_aos);
    cudaFree(d_soa_x);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
