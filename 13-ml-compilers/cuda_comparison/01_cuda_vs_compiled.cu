/**
 * 01_cuda_vs_compiled.cu - Plain CUDA for Comparison with ML Compilers
 * 
 * This file contains plain CUDA C++ implementations of common operations
 * so you can compare with:
 * - PyTorch eager mode
 * - torch.compile (Triton)
 * - TensorRT
 * - Other ML compilers
 * 
 * Understanding raw CUDA helps you appreciate what compilers do for you!
 * 
 * Compile: nvcc -O3 -o cuda_comparison 01_cuda_vs_compiled.cu
 * Run: ./cuda_comparison
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// PROFILING UTILITIES
// ============================================================================

class CudaTimer {
public:
    cudaEvent_t start, stop;
    
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void tic() {
        cudaEventRecord(start);
    }
    
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// KERNEL 1: VECTOR ADDITION (Simplest kernel)
// ============================================================================

/**
 * Vector addition: C = A + B
 * 
 * This is the "Hello World" of CUDA.
 * Each thread computes one element.
 * 
 * What compilers do:
 * - Same basic structure
 * - May fuse with other operations
 * - Optimize block/grid sizes
 */
__global__ void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void benchmark_vector_add(int N) {
    printf("\n=== VECTOR ADDITION (N=%d) ===\n", N);
    
    // Allocate
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    size_t bytes = N * sizeof(float);
    
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch config
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Warmup
    vector_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CudaTimer timer;
    int iterations = 100;
    
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        vector_add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    }
    float total_ms = timer.toc();
    
    float avg_ms = total_ms / iterations;
    float bandwidth = (3 * bytes) / (avg_ms / 1000) / 1e9;  // GB/s
    
    printf("Time: %.4f ms\n", avg_ms);
    printf("Bandwidth: %.1f GB/s\n", bandwidth);
    printf("Block size: %d, Grid size: %d\n", blockSize, numBlocks);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ============================================================================
// KERNEL 2: FUSED ADD-MUL-RELU (What compilers fuse)
// ============================================================================

/**
 * Unfused version: 3 separate kernels
 * 
 * This is what happens in eager mode:
 * - Each operation is a separate kernel
 * - Data goes to global memory between ops
 * - Memory bandwidth wasted!
 */
__global__ void add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

__global__ void mul_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] * B[idx];
}

__global__ void relu_kernel(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) B[idx] = fmaxf(A[idx], 0.0f);
}

/**
 * Fused version: Single kernel does all operations
 * 
 * This is what torch.compile/Triton generates!
 * - Read inputs ONCE
 * - Compute all ops in registers
 * - Write output ONCE
 */
__global__ void fused_add_mul_relu_kernel(
    const float* A,
    const float* B,
    const float* C,
    float* out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // All computation in registers - no intermediate global memory!
        float temp = A[idx] + B[idx];  // add
        temp = temp * C[idx];           // mul
        out[idx] = fmaxf(temp, 0.0f);  // relu
    }
}

void benchmark_fusion(int N) {
    printf("\n=== KERNEL FUSION COMPARISON (N=%d) ===\n", N);
    
    size_t bytes = N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_temp1, *d_temp2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp1, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp2, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    // Initialize with random data
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_A, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_data, bytes, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    CudaTimer timer;
    int iterations = 100;
    
    // UNFUSED: 3 separate kernels
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        add_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_temp1, N);
        mul_kernel<<<numBlocks, blockSize>>>(d_temp1, d_C, d_temp2, N);
        relu_kernel<<<numBlocks, blockSize>>>(d_temp2, d_out, N);
    }
    float unfused_ms = timer.toc() / iterations;
    
    // FUSED: Single kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        fused_add_mul_relu_kernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, d_out, N);
    }
    float fused_ms = timer.toc() / iterations;
    
    printf("Unfused (3 kernels): %.4f ms\n", unfused_ms);
    printf("Fused (1 kernel):    %.4f ms\n", fused_ms);
    printf("Speedup: %.2fx\n", unfused_ms / fused_ms);
    printf("\nWhy? Memory traffic:\n");
    printf("  Unfused: Read A,B → Write temp1 → Read temp1,C → Write temp2 → Read temp2 → Write out\n");
    printf("           = 8 memory passes\n");
    printf("  Fused:   Read A,B,C → Write out\n");
    printf("           = 4 memory passes (2x less!)\n");
    
    free(h_data);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_temp1); cudaFree(d_temp2); cudaFree(d_out);
}

// ============================================================================
// KERNEL 3: SOFTMAX (Reduction + Elementwise)
// ============================================================================

/**
 * Naive softmax: Multiple passes
 * 
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * Naive approach:
 * 1. Find max (reduction)
 * 2. Subtract max and exp (elementwise)
 * 3. Sum (reduction)
 * 4. Divide (elementwise)
 * = 4 kernel launches!
 */

// Find max in a row
__global__ void row_max_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, input[row * cols + i]);
    }
    output[row] = max_val;
}

// Subtract max and exp
__global__ void sub_exp_kernel(const float* input, const float* max_vals, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    if (idx < rows * cols) {
        output[idx] = expf(input[idx] - max_vals[row]);
    }
}

// Sum rows
__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float sum = 0;
    for (int i = 0; i < cols; i++) {
        sum += input[row * cols + i];
    }
    output[row] = sum;
}

// Divide by sum
__global__ void div_kernel(float* data, const float* sums, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    if (idx < rows * cols) {
        data[idx] /= sums[row];
    }
}

/**
 * Fused softmax: Single kernel per row
 * 
 * What torch.compile generates:
 * - Compute max, exp, sum, divide all in one kernel
 * - Use shared memory for intermediate values
 */
__global__ void fused_softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float* row_data = data + row * cols;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, row_data[i]);
    }
    
    // Compute exp and sum
    float sum = 0;
    for (int i = 0; i < cols; i++) {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    
    // Normalize
    for (int i = 0; i < cols; i++) {
        row_data[i] /= sum;
    }
}

void benchmark_softmax(int rows, int cols) {
    printf("\n=== SOFTMAX COMPARISON (%dx%d) ===\n", rows, cols);
    
    int N = rows * cols;
    size_t bytes = N * sizeof(float);
    size_t row_bytes = rows * sizeof(float);
    
    float *d_input, *d_temp, *d_max, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMalloc(&d_max, row_bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, row_bytes));
    
    // Initialize
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)rand() / RAND_MAX * 10 - 5;
    CUDA_CHECK(cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    CudaTimer timer;
    int iterations = 100;
    
    // UNFUSED: 4 kernels
    CUDA_CHECK(cudaMemcpy(d_temp, d_input, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        row_max_kernel<<<rows, 1>>>(d_temp, d_max, rows, cols);
        sub_exp_kernel<<<numBlocks, blockSize>>>(d_temp, d_max, d_temp, rows, cols);
        row_sum_kernel<<<rows, 1>>>(d_temp, d_sum, rows, cols);
        div_kernel<<<numBlocks, blockSize>>>(d_temp, d_sum, rows, cols);
        CUDA_CHECK(cudaMemcpy(d_temp, d_input, bytes, cudaMemcpyDeviceToDevice));
    }
    float unfused_ms = timer.toc() / iterations;
    
    // FUSED: 1 kernel
    CUDA_CHECK(cudaMemcpy(d_temp, d_input, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        fused_softmax_kernel<<<rows, 1>>>(d_temp, rows, cols);
        CUDA_CHECK(cudaMemcpy(d_temp, d_input, bytes, cudaMemcpyDeviceToDevice));
    }
    float fused_ms = timer.toc() / iterations;
    
    printf("Unfused (4 kernels): %.4f ms\n", unfused_ms);
    printf("Fused (1 kernel):    %.4f ms\n", fused_ms);
    printf("Speedup: %.2fx\n", unfused_ms / fused_ms);
    
    free(h_data);
    cudaFree(d_input); cudaFree(d_temp);
    cudaFree(d_max); cudaFree(d_sum);
}

// ============================================================================
// KERNEL 4: MATRIX MULTIPLY (Tiled)
// ============================================================================

#define TILE_SIZE 32

/**
 * Naive matrix multiply: No tiling
 * 
 * Each thread computes one element of C.
 * Reads A and B from global memory repeatedly.
 * TERRIBLE memory access pattern!
 */
__global__ void matmul_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Tiled matrix multiply: Use shared memory
 * 
 * This is closer to what cuBLAS and Triton generate.
 * Load tiles into shared memory, reuse them.
 * Much better memory efficiency!
 */
__global__ void matmul_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void benchmark_matmul(int M, int N, int K) {
    printf("\n=== MATRIX MULTIPLY COMPARISON (%dx%dx%d) ===\n", M, N, K);
    
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    // Initialize
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    CudaTimer timer;
    int iterations = 50;
    
    // Naive
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    float naive_ms = timer.toc() / iterations;
    
    // Tiled
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < iterations; i++) {
        matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    float tiled_ms = timer.toc() / iterations;
    
    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops_naive = flops / (naive_ms / 1000) / 1e12;
    double tflops_tiled = flops / (tiled_ms / 1000) / 1e12;
    
    printf("Naive:  %.4f ms (%.2f TFLOPS)\n", naive_ms, tflops_naive);
    printf("Tiled:  %.4f ms (%.2f TFLOPS)\n", tiled_ms, tflops_tiled);
    printf("Speedup: %.2fx\n", naive_ms / tiled_ms);
    printf("\nNote: cuBLAS would be ~10x faster than tiled due to more optimizations!\n");
    
    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║           PLAIN CUDA vs ML COMPILERS COMPARISON                  ║\n");
    printf("║     Understanding what compilers optimize for you                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    
    // Run benchmarks
    benchmark_vector_add(16 * 1024 * 1024);  // 16M elements
    benchmark_fusion(16 * 1024 * 1024);       // 16M elements
    benchmark_softmax(1024, 1024);            // 1024x1024
    benchmark_matmul(1024, 1024, 1024);       // 1024x1024x1024
    
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("KEY TAKEAWAYS:\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("1. FUSION reduces memory traffic significantly\n");
    printf("2. TILING improves memory reuse (shared memory)\n");
    printf("3. ML compilers (torch.compile, Triton) do these automatically!\n");
    printf("4. cuBLAS/cuDNN are even more optimized than our examples\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    
    return 0;
}
