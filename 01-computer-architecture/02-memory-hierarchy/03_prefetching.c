/**
 * 03_prefetching.c - Hardware and Software Prefetching
 * 
 * Prefetching hides memory latency by loading data before it's needed.
 * Critical for: data loading pipelines, sequential processing
 * 
 * Compile: gcc -O2 -o 03_prefetching 03_prefetching.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __GNUC__
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#else
#define PREFETCH(addr)
#endif

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define SIZE (64 * 1024 * 1024)  // 64 MB

int main() {
    printf("=== PREFETCHING: HIDING MEMORY LATENCY ===\n\n");
    
    int* data = malloc(SIZE);
    int n = SIZE / sizeof(int);
    
    for (int i = 0; i < n; i++) data[i] = i;
    
    volatile long sum;
    double start;
    
    // Without prefetch
    sum = 0;
    start = get_time();
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    double no_prefetch = get_time() - start;
    
    // With software prefetch
    sum = 0;
    start = get_time();
    for (int i = 0; i < n; i++) {
        PREFETCH(&data[i + 64]);  // Prefetch 64 elements ahead
        sum += data[i];
    }
    double with_prefetch = get_time() - start;
    
    printf("Without explicit prefetch: %.2f ms\n", no_prefetch * 1000);
    printf("With software prefetch:    %.2f ms\n", with_prefetch * 1000);
    
    printf("\nNOTE: Modern CPUs have excellent hardware prefetchers.\n");
    printf("Software prefetch helps most with irregular access patterns.\n");
    
    printf("\nML DATA LOADING CONNECTION:\n");
    printf("- FFCV prefetches next batch while GPU processes current\n");
    printf("- Double buffering: load batch N+1 while training on batch N\n");
    printf("- PyTorch DataLoader's prefetch_factor parameter\n");
    
    free(data);
    return 0;
}
