/**
 * 01_quantized_dot.c - Quantized Dot Product for LLM Inference
 * 
 * This is the core operation in quantized LLM inference.
 * INT8 and INT4 operations are much faster than FP32.
 * 
 * Compile: gcc -mavx2 -O3 -o quantized_dot 01_quantized_dot.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>
#include <math.h>

#define N (4096)  // Typical hidden dimension

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
// SECTION 1: FP32 baseline
// ============================================================

float dot_fp32(float* a, float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float dot_fp32_avx(float* a, float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    return _mm_cvtss_f32(sum128);
}

// ============================================================
// SECTION 2: INT8 quantized dot product
// ============================================================

// Quantize float to int8
void quantize_fp32_to_int8(float* src, int8_t* dst, float* scale, int n) {
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(src[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    
    *scale = max_val / 127.0f;
    float inv_scale = (*scale != 0) ? 127.0f / max_val : 0;
    
    for (int i = 0; i < n; i++) {
        dst[i] = (int8_t)roundf(src[i] * inv_scale);
    }
}

// INT8 dot product (scalar)
float dot_int8_scalar(int8_t* a, int8_t* b, float scale_a, float scale_b, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return (float)sum * scale_a * scale_b;
}

// INT8 dot product with AVX2
float dot_int8_avx(int8_t* a, int8_t* b, float scale_a, float scale_b, int n) {
    __m256i sum = _mm256_setzero_si256();
    
    for (int i = 0; i < n; i += 32) {
        // Load 32 int8 values
        __m256i va = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        
        // Split into low and high 16-byte halves
        __m128i va_lo = _mm256_castsi256_si128(va);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_castsi256_si128(vb);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);
        
        // Sign-extend to 16-bit and multiply
        __m256i va_lo_16 = _mm256_cvtepi8_epi16(va_lo);
        __m256i vb_lo_16 = _mm256_cvtepi8_epi16(vb_lo);
        __m256i va_hi_16 = _mm256_cvtepi8_epi16(va_hi);
        __m256i vb_hi_16 = _mm256_cvtepi8_epi16(vb_hi);
        
        // Multiply and add
        __m256i prod_lo = _mm256_madd_epi16(va_lo_16, vb_lo_16);
        __m256i prod_hi = _mm256_madd_epi16(va_hi_16, vb_hi_16);
        
        sum = _mm256_add_epi32(sum, prod_lo);
        sum = _mm256_add_epi32(sum, prod_hi);
    }
    
    // Horizontal sum
    __m128i sum_lo = _mm256_castsi256_si128(sum);
    __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    
    int32_t result = _mm_cvtsi128_si32(sum128);
    return (float)result * scale_a * scale_b;
}

// ============================================================
// SECTION 3: Simulate INT4 (packed)
// ============================================================

// Pack two int4 values into one byte
void pack_int4(int8_t* src, uint8_t* dst, int n) {
    for (int i = 0; i < n; i += 2) {
        int8_t lo = src[i] & 0x0F;
        int8_t hi = src[i + 1] & 0x0F;
        dst[i / 2] = (hi << 4) | lo;
    }
}

// Unpack int4 to int8
void unpack_int4(uint8_t* src, int8_t* dst, int n) {
    for (int i = 0; i < n / 2; i++) {
        dst[i * 2] = (int8_t)((src[i] & 0x0F) - 8);      // Low nibble
        dst[i * 2 + 1] = (int8_t)((src[i] >> 4) - 8);    // High nibble
    }
}

int main() {
    printf("=== QUANTIZED DOT PRODUCT FOR LLM INFERENCE ===\n\n");
    
    // Allocate
    float* a_fp32 = aligned_alloc(32, N * sizeof(float));
    float* b_fp32 = aligned_alloc(32, N * sizeof(float));
    int8_t* a_int8 = aligned_alloc(32, N);
    int8_t* b_int8 = aligned_alloc(32, N);
    float scale_a, scale_b;
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < N; i++) {
        a_fp32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        b_fp32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Quantize
    quantize_fp32_to_int8(a_fp32, a_int8, &scale_a, N);
    quantize_fp32_to_int8(b_fp32, b_int8, &scale_b, N);
    
    printf("Vector size: %d elements\n", N);
    printf("Quantization scales: a=%.6f, b=%.6f\n\n", scale_a, scale_b);
    
    double start;
    float result;
    int iterations = 10000;
    
    // Warmup
    result = dot_fp32_avx(a_fp32, b_fp32, N);
    result = dot_int8_avx(a_int8, b_int8, scale_a, scale_b, N);
    
    // Benchmark FP32 scalar
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        result = dot_fp32(a_fp32, b_fp32, N);
    }
    double fp32_scalar_time = (get_time() - start) / iterations;
    float fp32_result = result;
    
    // Benchmark FP32 AVX
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        result = dot_fp32_avx(a_fp32, b_fp32, N);
    }
    double fp32_avx_time = (get_time() - start) / iterations;
    
    // Benchmark INT8 scalar
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        result = dot_int8_scalar(a_int8, b_int8, scale_a, scale_b, N);
    }
    double int8_scalar_time = (get_time() - start) / iterations;
    float int8_result = result;
    
    // Benchmark INT8 AVX
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        result = dot_int8_avx(a_int8, b_int8, scale_a, scale_b, N);
    }
    double int8_avx_time = (get_time() - start) / iterations;
    
    printf("Results:\n");
    printf("  FP32:  %.4f\n", fp32_result);
    printf("  INT8:  %.4f (error: %.2f%%)\n\n", 
           int8_result, 100 * fabsf(fp32_result - int8_result) / fabsf(fp32_result));
    
    printf("Timing (per dot product):\n");
    printf("  FP32 scalar:  %.2f µs\n", fp32_scalar_time * 1e6);
    printf("  FP32 AVX:     %.2f µs (%.1fx faster)\n", 
           fp32_avx_time * 1e6, fp32_scalar_time / fp32_avx_time);
    printf("  INT8 scalar:  %.2f µs\n", int8_scalar_time * 1e6);
    printf("  INT8 AVX:     %.2f µs (%.1fx faster than FP32 AVX)\n",
           int8_avx_time * 1e6, fp32_avx_time / int8_avx_time);
    
    printf("\n=== KEY INSIGHTS ===\n\n");
    printf("1. INT8 uses 4x less memory than FP32\n");
    printf("2. INT8 operations are faster (more elements per register)\n");
    printf("3. Quantization error is small for LLM weights\n");
    printf("4. llama.cpp uses Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 formats\n");
    printf("5. AVX-VNNI (vpdpbusd) is even faster on newer CPUs\n");
    
    free(a_fp32);
    free(b_fp32);
    free(a_int8);
    free(b_int8);
    
    return 0;
}
