/**
 * 02_dotprod_main.c - Test wrapper for assembly dot product
 * 
 * Compile: gcc -mavx2 -mfma -O3 -o dotprod 02_avx_dotproduct.s 02_dotprod_main.c
 * Run: ./dotprod
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N (10 * 1000 * 1000)

// Assembly function
extern float avx_dot(float* a, float* b, int n);

// Reference implementation
float scalar_dot(float* a, float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== HAND-WRITTEN AVX DOT PRODUCT ===\n\n");
    
    float* a = aligned_alloc(32, N * sizeof(float));
    float* b = aligned_alloc(32, N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = (float)i / N;
        b[i] = (float)(N - i) / N;
    }
    
    double start;
    float result;
    
    // Warmup
    result = avx_dot(a, b, N);
    
    // Benchmark scalar
    start = get_time();
    for (int r = 0; r < 10; r++) {
        result = scalar_dot(a, b, N);
    }
    double scalar_time = (get_time() - start) / 10;
    float scalar_result = result;
    
    // Benchmark AVX assembly
    start = get_time();
    for (int r = 0; r < 10; r++) {
        result = avx_dot(a, b, N);
    }
    double avx_time = (get_time() - start) / 10;
    float avx_result = result;
    
    printf("N = %d elements\n\n", N);
    printf("Scalar:   result=%.6f, time=%.2f ms\n", scalar_result, scalar_time * 1000);
    printf("AVX asm:  result=%.6f, time=%.2f ms\n", avx_result, avx_time * 1000);
    printf("\nSpeedup: %.1fx\n", scalar_time / avx_time);
    printf("Error: %.2e\n", fabs(scalar_result - avx_result));
    
    // Calculate throughput
    double flops = 2.0 * N;  // multiply + add per element
    printf("\nThroughput:\n");
    printf("  Scalar: %.2f GFLOPS\n", flops / scalar_time / 1e9);
    printf("  AVX:    %.2f GFLOPS\n", flops / avx_time / 1e9);
    
    free(a);
    free(b);
    
    return 0;
}
