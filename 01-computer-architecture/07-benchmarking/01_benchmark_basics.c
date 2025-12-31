/**
 * 01_benchmark_basics.c - How to Benchmark Correctly
 * 
 * Accurate benchmarking is harder than it looks.
 * Bad benchmarks lead to wrong conclusions!
 * 
 * Compile: gcc -O2 -o 01_benchmark_basics 01_benchmark_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Prevent compiler from optimizing away
void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

void clobber() {
    asm volatile("" : : : "memory");
}

double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main() {
    printf("=== BENCHMARKING BEST PRACTICES ===\n\n");
    
    // Common mistake 1: Not warming up
    printf("--- WARMUP IS ESSENTIAL ---\n");
    
    int N = 10000000;
    int* data = malloc(N * sizeof(int));
    
    // Cold run (data not in cache)
    for (int i = 0; i < N; i++) data[i] = i;
    
    volatile long sum = 0;
    double start = get_time_ns();
    for (int i = 0; i < N; i++) sum += data[i];
    double cold_time = get_time_ns() - start;
    
    // Warm run (data in cache)
    sum = 0;
    start = get_time_ns();
    for (int i = 0; i < N; i++) sum += data[i];
    double warm_time = get_time_ns() - start;
    
    printf("Cold run: %.2f ms\n", cold_time / 1e6);
    printf("Warm run: %.2f ms\n", warm_time / 1e6);
    printf("Always discard first iteration!\n\n");
    
    // Common mistake 2: Compiler optimizing away work
    printf("--- PREVENT DEAD CODE ELIMINATION ---\n");
    
    // Bad: compiler might optimize this away
    start = get_time_ns();
    for (int i = 0; i < N; i++) {
        int x = data[i] * 2;  // Result unused!
    }
    double optimized_time = get_time_ns() - start;
    
    // Good: use escape() to keep result
    sum = 0;
    start = get_time_ns();
    for (int i = 0; i < N; i++) {
        sum += data[i] * 2;
    }
    escape((void*)&sum);
    double real_time = get_time_ns() - start;
    
    printf("Maybe optimized away: %.2f ns\n", optimized_time);
    printf("With escape():        %.2f ms\n", real_time / 1e6);
    
    // Common mistake 3: Not enough iterations
    printf("\n--- STATISTICAL SIGNIFICANCE ---\n");
    
    int runs = 10;
    double times[10];
    
    for (int r = 0; r < runs; r++) {
        sum = 0;
        start = get_time_ns();
        for (int i = 0; i < N; i++) sum += data[i];
        times[r] = get_time_ns() - start;
    }
    escape((void*)&sum);
    
    // Calculate mean and std dev
    double mean = 0, variance = 0;
    for (int r = 0; r < runs; r++) mean += times[r];
    mean /= runs;
    for (int r = 0; r < runs; r++) 
        variance += (times[r] - mean) * (times[r] - mean);
    double std_dev = sqrt(variance / runs);
    
    printf("Mean: %.2f ms, StdDev: %.2f ms (%.1f%%)\n",
           mean / 1e6, std_dev / 1e6, 100 * std_dev / mean);
    printf("Report: mean ± std_dev\n");
    
    printf("\n=== BENCHMARKING CHECKLIST ===\n");
    printf("1. Warmup: Discard first iteration\n");
    printf("2. Prevent optimization: Use escape()\n");
    printf("3. Multiple runs: Report mean ± std\n");
    printf("4. Consistent environment: Same load, power\n");
    printf("5. Measure what matters: End-to-end time\n");
    
    free(data);
    return 0;
}
