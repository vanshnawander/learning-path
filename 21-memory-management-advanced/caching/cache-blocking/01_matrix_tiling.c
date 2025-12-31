/*
 * Cache Blocking / Tiling for Matrix Multiplication
 * 
 * Demonstrates how cache blocking dramatically improves performance
 * by keeping data in cache during computation.
 * 
 * Compile: gcc -O3 -march=native 01_matrix_tiling.c -o matrix_tile -lm
 * Run: ./matrix_tile
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 1024  // Matrix dimension (NxN)

// Tile sizes to test (should fit in L1/L2 cache)
// L1: 32KB = 32*1024/8 = 4096 doubles = 64x64 matrix
// L2: 256KB = 32768 doubles = ~180x180 matrix
#define TILE_SIZE 64

double A[N][N], B[N][N], C[N][N];

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrices(void) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / N;
            B[i][j] = (double)(i - j) / N;
            C[i][j] = 0.0;
        }
    }
}

void clear_C(void) {
    memset(C, 0, sizeof(C));
}

// Naive matrix multiplication (cache-unfriendly)
void matmul_naive(void) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];  // B[k][j] causes cache misses
            }
        }
    }
}

// Loop interchange (better cache access for B)
void matmul_interchange(void) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += a_ik * B[k][j];  // Sequential access for both C and B
            }
        }
    }
}

// Cache-blocked (tiled) matrix multiplication
void matmul_tiled(void) {
    int T = TILE_SIZE;
    
    for (int ii = 0; ii < N; ii += T) {
        for (int kk = 0; kk < N; kk += T) {
            for (int jj = 0; jj < N; jj += T) {
                // Mini matrix multiply within tiles
                int i_end = (ii + T < N) ? ii + T : N;
                int k_end = (kk + T < N) ? kk + T : N;
                int j_end = (jj + T < N) ? jj + T : N;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = A[i][k];
                        for (int j = jj; j < j_end; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// Multi-level tiled (L1 + L2 blocking)
void matmul_multilevel(void) {
    int T1 = 32;   // L1 tile size
    int T2 = 128;  // L2 tile size
    
    // L2 blocking
    for (int ii2 = 0; ii2 < N; ii2 += T2) {
        for (int kk2 = 0; kk2 < N; kk2 += T2) {
            for (int jj2 = 0; jj2 < N; jj2 += T2) {
                // L1 blocking within L2 tile
                int ii2_end = (ii2 + T2 < N) ? ii2 + T2 : N;
                int kk2_end = (kk2 + T2 < N) ? kk2 + T2 : N;
                int jj2_end = (jj2 + T2 < N) ? jj2 + T2 : N;
                
                for (int ii = ii2; ii < ii2_end; ii += T1) {
                    for (int kk = kk2; kk < kk2_end; kk += T1) {
                        for (int jj = jj2; jj < jj2_end; jj += T1) {
                            int i_end = (ii + T1 < ii2_end) ? ii + T1 : ii2_end;
                            int k_end = (kk + T1 < kk2_end) ? kk + T1 : kk2_end;
                            int j_end = (jj + T1 < jj2_end) ? jj + T1 : jj2_end;
                            
                            for (int i = ii; i < i_end; i++) {
                                for (int k = kk; k < k_end; k++) {
                                    double a_ik = A[i][k];
                                    for (int j = jj; j < j_end; j++) {
                                        C[i][j] += a_ik * B[k][j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

double verify_result(void) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += C[i][j];
        }
    }
    return sum;
}

int main() {
    printf("=== Cache Blocking / Tiling Demo ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Data size: %.1f MB per matrix\n\n", (double)(N*N*sizeof(double))/(1024*1024));
    
    init_matrices();
    double start, elapsed, gflops;
    double expected_sum;
    double ops = 2.0 * N * N * N;  // 2N^3 floating point operations
    
    // Naive
    clear_C();
    start = get_time_sec();
    matmul_naive();
    elapsed = get_time_sec() - start;
    gflops = ops / elapsed / 1e9;
    expected_sum = verify_result();
    printf("Naive:        %.3f sec, %.2f GFLOPS, sum=%.6e\n", elapsed, gflops, expected_sum);
    
    // Loop interchange
    clear_C();
    start = get_time_sec();
    matmul_interchange();
    elapsed = get_time_sec() - start;
    gflops = ops / elapsed / 1e9;
    printf("Interchange:  %.3f sec, %.2f GFLOPS, sum=%.6e\n", elapsed, gflops, verify_result());
    
    // Tiled
    clear_C();
    start = get_time_sec();
    matmul_tiled();
    elapsed = get_time_sec() - start;
    gflops = ops / elapsed / 1e9;
    printf("Tiled (%dx%d): %.3f sec, %.2f GFLOPS, sum=%.6e\n", TILE_SIZE, TILE_SIZE, elapsed, gflops, verify_result());
    
    // Multi-level
    clear_C();
    start = get_time_sec();
    matmul_multilevel();
    elapsed = get_time_sec() - start;
    gflops = ops / elapsed / 1e9;
    printf("Multi-level:  %.3f sec, %.2f GFLOPS, sum=%.6e\n", elapsed, gflops, verify_result());
    
    printf("\n=== Why Tiling Works ===\n");
    printf("Without tiling: Each element of B accessed N times,\n");
    printf("  but B doesn't fit in cache, so each access is a miss.\n");
    printf("With tiling: Small tiles of A, B, C fit in cache,\n");
    printf("  reused many times before eviction.\n");
    printf("\nOptimal tile size: sqrt(cache_size / 3 / sizeof(double))\n");
    printf("For 32KB L1: sqrt(32*1024 / 3 / 8) ≈ 37 → use 32 or 64\n");
    
    return 0;
}
