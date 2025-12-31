/**
 * 01_cache_line_effects.c - Understanding Cache Lines
 * 
 * The cache line is the fundamental unit of memory transfer.
 * This explains why sequential access is so much faster than random.
 * 
 * Compile: gcc -O2 -o cache_line 01_cache_line_effects.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#define ARRAY_SIZE (64 * 1024 * 1024)  // 64 MB
#define CACHE_LINE 64

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║              CACHE LINE EFFECTS                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    char* data = aligned_alloc(CACHE_LINE, ARRAY_SIZE);
    memset(data, 1, ARRAY_SIZE);
    
    volatile long sum = 0;
    double start, elapsed;
    
    // ================================================================
    // Test 1: Sequential access (stride = 1)
    // ================================================================
    printf("=== TEST 1: SEQUENTIAL ACCESS (stride=1) ===\n");
    
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < ARRAY_SIZE; i++) {
        sum += data[i];
    }
    elapsed = get_time() - start;
    
    printf("Time: %.3f ms\n", elapsed * 1000);
    printf("Bandwidth: %.2f GB/s\n", ARRAY_SIZE / elapsed / 1e9);
    printf("Every access is a cache line HIT (after first)\n\n");
    
    // ================================================================
    // Test 2: Stride = 16 bytes (4 ints per cache line access)
    // ================================================================
    printf("=== TEST 2: STRIDE=16 (4 per cache line) ===\n");
    
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < ARRAY_SIZE; i += 16) {
        sum += data[i];
    }
    elapsed = get_time() - start;
    
    size_t accesses = ARRAY_SIZE / 16;
    printf("Accesses: %zu\n", accesses);
    printf("Time: %.3f ms\n", elapsed * 1000);
    printf("Effective bandwidth: %.2f GB/s\n", accesses / elapsed / 1e9);
    printf("Still good - using 1/4 of each cache line\n\n");
    
    // ================================================================
    // Test 3: Stride = 64 bytes (1 per cache line - worst case)
    // ================================================================
    printf("=== TEST 3: STRIDE=64 (1 per cache line) ===\n");
    
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < ARRAY_SIZE; i += 64) {
        sum += data[i];
    }
    elapsed = get_time() - start;
    
    accesses = ARRAY_SIZE / 64;
    printf("Accesses: %zu\n", accesses);
    printf("Time: %.3f ms\n", elapsed * 1000);
    printf("Effective bandwidth: %.2f GB/s\n", accesses / elapsed / 1e9);
    printf("BAD! Loading 64 bytes but using only 1 byte\n\n");
    
    // ================================================================
    // Test 4: Stride = 128 bytes (miss every other cache line)
    // ================================================================
    printf("=== TEST 4: STRIDE=128 (skip cache lines) ===\n");
    
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < ARRAY_SIZE; i += 128) {
        sum += data[i];
    }
    elapsed = get_time() - start;
    
    accesses = ARRAY_SIZE / 128;
    printf("Accesses: %zu\n", accesses);
    printf("Time: %.3f ms\n", elapsed * 1000);
    printf("Skipping every other cache line\n\n");
    
    // ================================================================
    // Test 5: Random access (worst case for cache)
    // ================================================================
    printf("=== TEST 5: RANDOM ACCESS ===\n");
    
    // Create random indices
    size_t* indices = malloc((ARRAY_SIZE / 64) * sizeof(size_t));
    for (size_t i = 0; i < ARRAY_SIZE / 64; i++) {
        indices[i] = (rand() % (ARRAY_SIZE / 64)) * 64;
    }
    
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < ARRAY_SIZE / 64; i++) {
        sum += data[indices[i]];
    }
    elapsed = get_time() - start;
    
    accesses = ARRAY_SIZE / 64;
    printf("Accesses: %zu\n", accesses);
    printf("Time: %.3f ms\n", elapsed * 1000);
    printf("TERRIBLE! Cache lines constantly evicted\n\n");
    
    free(indices);
    
    // ================================================================
    // Summary
    // ================================================================
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       KEY INSIGHTS                             ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ 1. Cache line = 64 bytes on most CPUs                          ║\n");
    printf("║ 2. Accessing 1 byte loads entire 64-byte line                  ║\n");
    printf("║ 3. Sequential access is FREE after first load                  ║\n");
    printf("║ 4. Stride >= 64 wastes bandwidth (load but don't use)         ║\n");
    printf("║ 5. Random access destroys cache efficiency                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n=== ML IMPLICATIONS ===\n\n");
    printf("TENSOR LAYOUT:\n");
    printf("  Contiguous tensors = sequential access = fast\n");
    printf("  Strided views = skipped data = slower\n\n");
    
    printf("ATTENTION MECHANISM:\n");
    printf("  Q @ K^T accesses K in strided pattern\n");
    printf("  Flash Attention tiles to improve locality\n\n");
    
    printf("DATA LOADING:\n");
    printf("  Sequential file read = prefetcher helps\n");
    printf("  Random sample access = cache misses\n");
    printf("  FFCV uses quasi-random for locality\n");
    
    free(data);
    return 0;
}
