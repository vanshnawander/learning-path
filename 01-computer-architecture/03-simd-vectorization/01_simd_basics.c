/**
 * 01_simd_basics.c - SIMD: Single Instruction Multiple Data
 * 
 * SIMD processes multiple data elements with one instruction.
 * This is what makes CPUs fast for ML preprocessing.
 * 
 * Compile: gcc -O3 -mavx2 -o 01_simd_basics 01_simd_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // AVX intrinsics

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define N (16 * 1024 * 1024)  // 16M elements

// Scalar addition
void add_scalar(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// AVX vectorized addition (8 floats at once)
void add_avx(float* a, float* b, float* c, int n) {
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    // Handle remainder
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// AVX dot product
float dot_avx(float* a, float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);  // Fused multiply-add
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

int main() {
    printf("=== SIMD: PROCESSING 8 FLOATS AT ONCE ===\n\n");
    
    float* a = aligned_alloc(32, N * sizeof(float));
    float* b = aligned_alloc(32, N * sizeof(float));
    float* c = aligned_alloc(32, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        a[i] = (float)i / N;
        b[i] = (float)(N - i) / N;
    }
    
    double start;
    
    // Vector addition
    start = get_time();
    add_scalar(a, b, c, N);
    double scalar_time = get_time() - start;
    
    start = get_time();
    add_avx(a, b, c, N);
    double avx_time = get_time() - start;
    
    printf("Vector Addition (%d M elements):\n", N / 1000000);
    printf("  Scalar: %.2f ms\n", scalar_time * 1000);
    printf("  AVX:    %.2f ms (%.1fx faster)\n", avx_time * 1000, scalar_time / avx_time);
    
    // Dot product
    start = get_time();
    float dot_s = 0;
    for (int i = 0; i < N; i++) dot_s += a[i] * b[i];
    scalar_time = get_time() - start;
    
    start = get_time();
    float dot_v = dot_avx(a, b, N);
    avx_time = get_time() - start;
    
    printf("\nDot Product:\n");
    printf("  Scalar: %.2f ms (result: %.2f)\n", scalar_time * 1000, dot_s);
    printf("  AVX:    %.2f ms (result: %.2f, %.1fx faster)\n", 
           avx_time * 1000, dot_v, scalar_time / avx_time);
    
    printf("\nSIMD LEVELS:\n");
    printf("  SSE:     128-bit = 4 floats\n");
    printf("  AVX:     256-bit = 8 floats\n");
    printf("  AVX-512: 512-bit = 16 floats\n");
    
    printf("\nML CONNECTION:\n");
    printf("- NumPy uses SIMD under the hood\n");
    printf("- Data augmentation in FFCV uses SIMD\n");
    printf("- Image decoding libraries (libjpeg-turbo) use SIMD\n");
    
    free(a); free(b); free(c);
    return 0;
}
