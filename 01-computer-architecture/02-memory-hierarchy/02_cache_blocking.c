/**
 * 02_cache_blocking.c - Cache Blocking/Tiling for Matrix Operations
 * 
 * This is THE key optimization technique used in:
 * - Flash Attention (tiling Q,K,V matrices)
 * - cuBLAS/CUTLASS GEMM
 * - Any high-performance matrix code
 * 
 * Compile: gcc -O3 -o 02_cache_blocking 02_cache_blocking.c
 * Run: ./02_cache_blocking
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Naive matrix multiply - terrible cache behavior
void matmul_naive(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];  // B access is strided!
            }
            C[i * N + j] = sum;
        }
    }
}

// Blocked/Tiled matrix multiply - much better cache behavior
void matmul_blocked(float* A, float* B, float* C, int N, int BLOCK) {
    memset(C, 0, N * N * sizeof(float));
    
    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            for (int kk = 0; kk < N; kk += BLOCK) {
                // Process one block
                int i_max = (ii + BLOCK < N) ? ii + BLOCK : N;
                int j_max = (jj + BLOCK < N) ? jj + BLOCK : N;
                int k_max = (kk + BLOCK < N) ? kk + BLOCK : N;
                
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    printf("=== CACHE BLOCKING: THE KEY TO PERFORMANCE ===\n\n");
    
    int N = 1024;
    printf("Matrix size: %d x %d\n\n", N, N);
    
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // Naive version
    double start = get_time();
    matmul_naive(A, B, C, N);
    double naive_time = get_time() - start;
    printf("Naive:        %.2f ms\n", naive_time * 1000);
    
    // Blocked versions with different block sizes
    int blocks[] = {16, 32, 64, 128};
    for (int b = 0; b < 4; b++) {
        start = get_time();
        matmul_blocked(A, B, C, N, blocks[b]);
        double blocked_time = get_time() - start;
        printf("Blocked (%3d): %.2f ms (%.1fx faster)\n", 
               blocks[b], blocked_time * 1000, naive_time / blocked_time);
    }
    
    printf("\nWHY BLOCKING WORKS:\n");
    printf("- Naive: B[k][j] access has stride N (cache miss every access)\n");
    printf("- Blocked: Keep BLOCK x BLOCK tiles in L1/L2 cache\n");
    printf("- Reuse each loaded element BLOCK times before eviction\n");
    
    printf("\nFLASH ATTENTION CONNECTION:\n");
    printf("- Q, K, V matrices are tiled into blocks\n");
    printf("- Each block fits in GPU shared memory (SRAM)\n");
    printf("- Avoids loading full N x N attention matrix to HBM\n");
    
    free(A); free(B); free(C);
    return 0;
}
