/**
 * 02_bandwidth_profiled.c - Profile Memory Bandwidth
 * 
 * Compile: gcc -O3 -o bandwidth 02_bandwidth_profiled.c -lrt -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define MB (1024 * 1024)
volatile uint64_t sink;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void test_read_bandwidth(size_t size_mb) {
    size_t size = size_mb * MB;
    uint64_t* data = aligned_alloc(64, size);
    memset(data, 1, size);
    
    // Warmup
    uint64_t sum = 0;
    for (size_t i = 0; i < size / 8; i++) sum += data[i];
    
    // Measure
    double start = get_time();
    sum = 0;
    for (int pass = 0; pass < 3; pass++) {
        for (size_t i = 0; i < size / 8; i++) {
            sum += data[i];
        }
    }
    double elapsed = get_time() - start;
    sink = sum;
    
    double bw = (size * 3.0) / elapsed / 1e9;
    printf("  Read %4zu MB: %6.2f GB/s\n", size_mb, bw);
    free(data);
}

void test_write_bandwidth(size_t size_mb) {
    size_t size = size_mb * MB;
    uint64_t* data = aligned_alloc(64, size);
    
    double start = get_time();
    for (int pass = 0; pass < 3; pass++) {
        for (size_t i = 0; i < size / 8; i++) {
            data[i] = i;
        }
    }
    double elapsed = get_time() - start;
    sink = data[0];
    
    double bw = (size * 3.0) / elapsed / 1e9;
    printf("  Write %4zu MB: %6.2f GB/s\n", size_mb, bw);
    free(data);
}

void test_copy_bandwidth(size_t size_mb) {
    size_t size = size_mb * MB;
    uint8_t* src = aligned_alloc(64, size);
    uint8_t* dst = aligned_alloc(64, size);
    memset(src, 1, size);
    
    double start = get_time();
    for (int pass = 0; pass < 3; pass++) {
        memcpy(dst, src, size);
    }
    double elapsed = get_time() - start;
    sink = dst[0];
    
    double bw = (size * 2.0 * 3) / elapsed / 1e9;  // Read + Write
    printf("  Copy %4zu MB: %6.2f GB/s\n", size_mb, bw);
    free(src); free(dst);
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█  MEMORY BANDWIDTH PROFILING                                  █\n");
    printf("████████████████████████████████████████████████████████████████\n\n");
    
    size_t sizes[] = {1, 4, 16, 64, 256};
    
    printf("─── Read Bandwidth ───\n");
    for (int i = 0; i < 5; i++) test_read_bandwidth(sizes[i]);
    
    printf("\n─── Write Bandwidth ───\n");
    for (int i = 0; i < 5; i++) test_write_bandwidth(sizes[i]);
    
    printf("\n─── Copy Bandwidth ───\n");
    for (int i = 0; i < 5; i++) test_copy_bandwidth(sizes[i]);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  ML IMPLICATIONS                                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  • Smaller working sets = higher bandwidth (cache)           ║\n");
    printf("║  • Large tensor ops are memory-bound, not compute-bound      ║\n");
    printf("║  • This is why Flash Attention recomputes instead of reload  ║\n");
    printf("║  • Batch size affects whether data fits in cache             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
