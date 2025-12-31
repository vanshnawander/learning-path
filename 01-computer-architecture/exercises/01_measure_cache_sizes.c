/**
 * EXERCISE: Measure Your CPU's Cache Sizes
 * 
 * Instructions:
 * 1. Compile and run this program
 * 2. Observe the bandwidth drops at L1, L2, L3 boundaries
 * 3. Compare with your CPU's spec sheet
 * 
 * Compile: gcc -O2 -o measure_cache measure_cache_sizes.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KB (1024)
#define MB (1024 * KB)

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== CACHE SIZE MEASUREMENT ===\n\n");
    printf("Measuring bandwidth at different working set sizes...\n\n");
    
    size_t sizes[] = {
        4*KB, 8*KB, 16*KB, 32*KB, 48*KB, 64*KB,  // Around L1
        128*KB, 256*KB, 384*KB, 512*KB,           // Around L2
        1*MB, 2*MB, 4*MB, 8*MB, 12*MB, 16*MB,    // Around L3
        24*MB, 32*MB, 48*MB, 64*MB               // RAM
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Size       Bandwidth    Cache Level (likely)\n");
    printf("---------  ----------   -------------------\n");
    
    double prev_bandwidth = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        int* array = malloc(size);
        size_t n = size / sizeof(int);
        
        // Initialize
        for (size_t i = 0; i < n; i++) array[i] = i;
        
        // Warmup
        volatile long sum = 0;
        for (size_t i = 0; i < n; i++) sum += array[i];
        
        // Measure
        int passes = (size < 1*MB) ? 10000 : (size < 16*MB) ? 1000 : 100;
        double start = get_time();
        for (int p = 0; p < passes; p++) {
            for (size_t i = 0; i < n; i++) {
                sum += array[i];
            }
        }
        double elapsed = get_time() - start;
        
        double bandwidth = (size * passes) / (elapsed * 1e9);
        
        // Detect cache level transitions
        const char* level = "";
        if (prev_bandwidth > 0 && bandwidth < prev_bandwidth * 0.7) {
            level = "<-- Cache level transition!";
        }
        
        if (size >= MB) {
            printf("%4zu MB    %6.2f GB/s   %s\n", size/MB, bandwidth, level);
        } else {
            printf("%4zu KB    %6.2f GB/s   %s\n", size/KB, bandwidth, level);
        }
        
        prev_bandwidth = bandwidth;
        free(array);
    }
    
    printf("\n=== YOUR TASK ===\n");
    printf("1. Run this on your machine\n");
    printf("2. Identify L1, L2, L3 cache sizes from bandwidth drops\n");
    printf("3. Compare with: lscpu | grep cache\n");
    printf("4. Explain why bandwidth drops at each level\n");
    
    return 0;
}
