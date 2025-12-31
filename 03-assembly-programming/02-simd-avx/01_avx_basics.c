/**
 * 01_avx_basics.c - AVX Intrinsics for ML Operations
 * 
 * This is the foundation of CPU-based LLM inference (llama.cpp).
 * AVX processes 8 floats simultaneously!
 * 
 * Compile: gcc -mavx2 -mfma -O3 -o avx_basics 01_avx_basics.c
 * Run: ./avx_basics
 */

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>
#include <string.h>

#define N (8 * 1024 * 1024)  // 8M elements

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
// SECTION 1: Basic AVX Operations
// ============================================================

void demonstrate_avx_basics() {
    printf("=== AVX BASICS ===\n\n");
    
    // AVX uses 256-bit registers = 8 floats
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    float c[8];
    
    // Load 8 floats into AVX registers
    __m256 va = _mm256_loadu_ps(a);  // unaligned load
    __m256 vb = _mm256_loadu_ps(b);
    
    // Add 8 pairs simultaneously
    __m256 vc = _mm256_add_ps(va, vb);
    
    // Store result
    _mm256_storeu_ps(c, vc);
    
    printf("a:      ");
    for (int i = 0; i < 8; i++) printf("%.0f ", a[i]);
    printf("\n");
    printf("b:      ");
    for (int i = 0; i < 8; i++) printf("%.0f ", b[i]);
    printf("\n");
    printf("a + b:  ");
    for (int i = 0; i < 8; i++) printf("%.0f ", c[i]);
    printf("\n\n");
}

// ============================================================
// SECTION 2: FMA - Fused Multiply-Add (Critical for GEMM!)
// ============================================================

void demonstrate_fma() {
    printf("=== FMA: THE HEART OF MATRIX MULTIPLY ===\n\n");
    
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    float c[8] = {10, 10, 10, 10, 10, 10, 10, 10};
    float result[8];
    
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 vc = _mm256_loadu_ps(c);
    
    // FMA: result = a * b + c (ONE instruction!)
    __m256 vr = _mm256_fmadd_ps(va, vb, vc);
    
    _mm256_storeu_ps(result, vr);
    
    printf("a * b + c = ");
    for (int i = 0; i < 8; i++) printf("%.0f ", result[i]);
    printf("\n");
    printf("FMA does multiply and add in ONE cycle!\n");
    printf("This is 2 FLOPs per element = 16 FLOPs per instruction\n\n");
}

// ============================================================
// SECTION 3: Dot Product (Used in Attention!)
// ============================================================

float dot_product_scalar(float* a, float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float dot_product_avx(float* a, float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);  // sum += a * b
    }
    
    // Horizontal sum: add all 8 elements
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remainder
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void benchmark_dot_product() {
    printf("=== DOT PRODUCT BENCHMARK ===\n\n");
    
    float* a = aligned_alloc(32, N * sizeof(float));
    float* b = aligned_alloc(32, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        a[i] = (float)i / N;
        b[i] = (float)(N - i) / N;
    }
    
    double start;
    float result;
    
    // Warmup
    result = dot_product_avx(a, b, N);
    
    // Scalar
    start = get_time();
    result = dot_product_scalar(a, b, N);
    double scalar_time = get_time() - start;
    printf("Scalar: %.6f (%.2f ms)\n", result, scalar_time * 1000);
    
    // AVX
    start = get_time();
    result = dot_product_avx(a, b, N);
    double avx_time = get_time() - start;
    printf("AVX:    %.6f (%.2f ms)\n", result, avx_time * 1000);
    
    printf("\nSpeedup: %.1fx\n", scalar_time / avx_time);
    printf("Theoretical max: 8x (8 floats per instruction)\n\n");
    
    free(a);
    free(b);
}

// ============================================================
// SECTION 4: Broadcast (Used in Weight Application)
// ============================================================

void demonstrate_broadcast() {
    printf("=== BROADCAST: APPLY SCALAR TO VECTOR ===\n\n");
    
    float scalar = 3.14159f;
    float vec[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float result[8];
    
    // Broadcast scalar to all 8 lanes
    __m256 vs = _mm256_set1_ps(scalar);  // [s, s, s, s, s, s, s, s]
    __m256 vv = _mm256_loadu_ps(vec);
    
    __m256 vr = _mm256_mul_ps(vs, vv);
    _mm256_storeu_ps(result, vr);
    
    printf("scalar * vec = ");
    for (int i = 0; i < 8; i++) printf("%.2f ", result[i]);
    printf("\n");
    printf("Use case: Applying attention scores to values\n\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║       AVX INTRINSICS FOR LLM INFERENCE                 ║\n");
    printf("║       8 operations per instruction!                    ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    
    demonstrate_avx_basics();
    demonstrate_fma();
    benchmark_dot_product();
    demonstrate_broadcast();
    
    printf("=== KEY TAKEAWAYS ===\n");
    printf("1. AVX processes 8 floats per instruction\n");
    printf("2. FMA (fused multiply-add) is key for GEMM\n");
    printf("3. llama.cpp uses these for CPU inference\n");
    printf("4. AVX-512 doubles this to 16 floats!\n");
    
    return 0;
}
