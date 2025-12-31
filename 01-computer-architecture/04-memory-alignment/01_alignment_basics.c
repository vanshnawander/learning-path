/**
 * 01_alignment_basics.c - Memory Alignment Deep Dive
 * 
 * Alignment is CRITICAL for:
 * - GPU memory coalescing (128-byte aligned)
 * - SIMD operations (16/32/64-byte aligned)
 * - Avoiding hardware penalties
 * 
 * Compile: gcc -O2 -o 01_alignment_basics 01_alignment_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Check if pointer is aligned to N bytes
int is_aligned(void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

// Align pointer up
void* align_up(void* ptr, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    return (void*)((addr + alignment - 1) & ~(alignment - 1));
}

int main() {
    printf("=== MEMORY ALIGNMENT: PERFORMANCE FOUNDATION ===\n\n");
    
    // Natural alignment of types
    printf("--- NATURAL ALIGNMENT OF TYPES ---\n");
    printf("Type        Size   Alignment\n");
    printf("char        %zu      %zu\n", sizeof(char), alignof(char));
    printf("short       %zu      %zu\n", sizeof(short), alignof(short));
    printf("int         %zu      %zu\n", sizeof(int), alignof(int));
    printf("long        %zu      %zu\n", sizeof(long), alignof(long));
    printf("float       %zu      %zu\n", sizeof(float), alignof(float));
    printf("double      %zu      %zu\n", sizeof(double), alignof(double));
    printf("void*       %zu      %zu\n", sizeof(void*), alignof(void*));
    
    // Struct padding demonstration
    printf("\n--- STRUCT PADDING ---\n");
    
    struct Unoptimized {
        char a;     // 1 byte + 7 padding
        double b;   // 8 bytes
        char c;     // 1 byte + 7 padding
    };
    
    struct Optimized {
        double b;   // 8 bytes
        char a;     // 1 byte
        char c;     // 1 byte + 6 padding
    };
    
    printf("Unoptimized struct: %zu bytes\n", sizeof(struct Unoptimized));
    printf("Optimized struct:   %zu bytes\n", sizeof(struct Optimized));
    printf("Savings: %zu bytes per struct!\n", 
           sizeof(struct Unoptimized) - sizeof(struct Optimized));
    
    // Aligned allocation
    printf("\n--- ALIGNED ALLOCATION ---\n");
    
    void* unaligned = malloc(1024);
    void* aligned_32 = aligned_alloc(32, 1024);
    void* aligned_64 = aligned_alloc(64, 1024);
    void* aligned_128 = aligned_alloc(128, 1024);
    
    printf("malloc():            %p (64-aligned: %s)\n", 
           unaligned, is_aligned(unaligned, 64) ? "yes" : "no");
    printf("aligned_alloc(32):   %p (32-aligned: %s)\n", 
           aligned_32, is_aligned(aligned_32, 32) ? "yes" : "no");
    printf("aligned_alloc(64):   %p (64-aligned: %s)\n", 
           aligned_64, is_aligned(aligned_64, 64) ? "yes" : "no");
    printf("aligned_alloc(128):  %p (128-aligned: %s)\n", 
           aligned_128, is_aligned(aligned_128, 128) ? "yes" : "no");
    
    printf("\n--- ML/GPU ALIGNMENT REQUIREMENTS ---\n");
    printf("CUDA coalescing:     128 bytes (32 threads × 4 bytes)\n");
    printf("AVX-256:             32 bytes\n");
    printf("AVX-512:             64 bytes\n");
    printf("Cache line:          64 bytes\n");
    printf("Tensor Core:         Matrices padded to multiples of 8/16\n");
    
    printf("\n--- WHY THIS MATTERS ---\n");
    printf("1. Unaligned access may require 2 memory transactions\n");
    printf("2. SIMD loads require alignment (or use slower unaligned loads)\n");
    printf("3. GPU coalescing fails with misaligned addresses\n");
    printf("4. False sharing occurs when threads access same cache line\n");
    
    free(unaligned);
    free(aligned_32);
    free(aligned_64);
    free(aligned_128);
    
    return 0;
}
