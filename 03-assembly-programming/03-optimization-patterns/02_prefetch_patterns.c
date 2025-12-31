/**
 * 02_prefetch_patterns.c - Software Prefetching for LLM Inference
 * 
 * Prefetching hides memory latency by loading data before it's needed.
 * Critical for memory-bound operations like attention.
 * 
 * Compile: gcc -O2 -o prefetch 02_prefetch_patterns.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __GNUC__
#define PREFETCH_T0(addr) __builtin_prefetch(addr, 0, 3)  // L1 cache
#define PREFETCH_T1(addr) __builtin_prefetch(addr, 0, 2)  // L2 cache
#define PREFETCH_T2(addr) __builtin_prefetch(addr, 0, 1)  // L3 cache
#define PREFETCH_NTA(addr) __builtin_prefetch(addr, 0, 0) // Non-temporal
#else
#define PREFETCH_T0(addr)
#define PREFETCH_T1(addr)
#define PREFETCH_T2(addr)
#define PREFETCH_NTA(addr)
#endif

#define SIZE (64 * 1024 * 1024)  // 64 MB
#define PREFETCH_DISTANCE 512     // Bytes ahead to prefetch

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Escape to prevent optimization
void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

int main() {
    printf("=== SOFTWARE PREFETCHING ===\n\n");
    
    float* data = aligned_alloc(64, SIZE);
    int n = SIZE / sizeof(float);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
    
    double start;
    volatile float sum;
    
    // ========================================================
    // Test 1: Sequential access (no explicit prefetch)
    // ========================================================
    printf("--- TEST 1: SEQUENTIAL (HARDWARE PREFETCH) ---\n");
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    double seq_time = get_time() - start;
    escape((void*)&sum);
    
    printf("Time: %.2f ms, Bandwidth: %.2f GB/s\n\n", 
           seq_time * 1000, SIZE / seq_time / 1e9);
    
    // ========================================================
    // Test 2: Strided access (defeats hardware prefetch)
    // ========================================================
    printf("--- TEST 2: STRIDED WITHOUT PREFETCH ---\n");
    
    int stride = 16;  // Access every 16th element (64 bytes apart)
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < n; i += stride) {
        sum += data[i];
    }
    double strided_time = get_time() - start;
    escape((void*)&sum);
    
    printf("Stride: %d, Time: %.2f ms\n\n", stride, strided_time * 1000);
    
    // ========================================================
    // Test 3: Strided with software prefetch
    // ========================================================
    printf("--- TEST 3: STRIDED WITH PREFETCH ---\n");
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < n; i += stride) {
        PREFETCH_T0(&data[i + stride * 8]);  // Prefetch 8 iterations ahead
        sum += data[i];
    }
    double prefetch_time = get_time() - start;
    escape((void*)&sum);
    
    printf("With prefetch: %.2f ms (%.1fx faster)\n\n", 
           prefetch_time * 1000, strided_time / prefetch_time);
    
    // ========================================================
    // Test 4: Random access (worst case)
    // ========================================================
    printf("--- TEST 4: RANDOM ACCESS ---\n");
    
    // Create random permutation
    int* indices = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
    
    // Limit to first 1M accesses
    int random_n = 1000000;
    
    // Without prefetch
    sum = 0;
    start = get_time();
    for (int i = 0; i < random_n; i++) {
        sum += data[indices[i]];
    }
    double random_time = get_time() - start;
    escape((void*)&sum);
    
    printf("Random (no prefetch): %.2f ms\n", random_time * 1000);
    
    // With prefetch (limited benefit for truly random)
    sum = 0;
    start = get_time();
    for (int i = 0; i < random_n; i++) {
        if (i + 8 < random_n) {
            PREFETCH_T0(&data[indices[i + 8]]);
        }
        sum += data[indices[i]];
    }
    double random_pf_time = get_time() - start;
    escape((void*)&sum);
    
    printf("Random (with prefetch): %.2f ms (%.1fx improvement)\n\n", 
           random_pf_time * 1000, random_time / random_pf_time);
    
    // ========================================================
    // LLM Inference Patterns
    // ========================================================
    printf("=== LLM INFERENCE PREFETCH PATTERNS ===\n\n");
    
    printf("1. ATTENTION:\n");
    printf("   Prefetch next K,V cache blocks\n");
    printf("   Distance = num_heads * head_dim\n\n");
    
    printf("2. FFN:\n");
    printf("   Prefetch next weight rows\n");
    printf("   Distance depends on hidden_dim\n\n");
    
    printf("3. KV CACHE:\n");
    printf("   Prefetch next sequence positions\n");
    printf("   Can use PREFETCH_NTA (non-temporal)\n\n");
    
    printf("4. BATCH PROCESSING:\n");
    printf("   Prefetch next batch's weights\n");
    printf("   Overlap compute with memory load\n");
    
    free(data);
    free(indices);
    
    return 0;
}
