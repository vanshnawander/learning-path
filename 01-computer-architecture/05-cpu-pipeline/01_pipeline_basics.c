/**
 * 01_pipeline_basics.c - CPU Pipeline and Instruction-Level Parallelism
 * 
 * Modern CPUs don't execute instructions one at a time.
 * Understanding pipelines explains:
 * - Why branches are expensive
 * - Why dependency chains limit speed
 * - Instruction-level parallelism
 * 
 * Compile: gcc -O2 -o 01_pipeline_basics 01_pipeline_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (100 * 1000 * 1000)

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== CPU PIPELINE EFFECTS ===\n\n");
    
    double start, elapsed;
    volatile long result;
    
    // Dependency chain (serial execution)
    printf("--- DEPENDENCY CHAIN ---\n");
    result = 0;
    start = get_time();
    for (long i = 0; i < N; i++) {
        result = result + 1;  // Each add depends on previous!
    }
    elapsed = get_time() - start;
    printf("Serial adds:   %.2f ms (%.2f ns/op)\n", 
           elapsed * 1000, (elapsed / N) * 1e9);
    
    // Independent operations (parallel execution)
    printf("\n--- INDEPENDENT OPERATIONS ---\n");
    volatile long r1 = 0, r2 = 0, r3 = 0, r4 = 0;
    start = get_time();
    for (long i = 0; i < N; i += 4) {
        r1++;  // These can execute in parallel!
        r2++;
        r3++;
        r4++;
    }
    result = r1 + r2 + r3 + r4;
    elapsed = get_time() - start;
    printf("4-way parallel: %.2f ms (%.2f ns/op)\n",
           elapsed * 1000, (elapsed / N) * 1e9);
    
    // Branch prediction
    printf("\n--- BRANCH PREDICTION ---\n");
    
    int* data = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 100;
    }
    
    // Predictable branch (always true)
    result = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        if (data[i] >= 0) result++;  // Always true!
    }
    elapsed = get_time() - start;
    printf("Predictable branch:   %.2f ms\n", elapsed * 1000);
    
    // Unpredictable branch (50/50)
    result = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        if (data[i] >= 50) result++;  // 50% true, unpredictable
    }
    elapsed = get_time() - start;
    printf("Unpredictable branch: %.2f ms\n", elapsed * 1000);
    
    printf("\nMispredicted branch: ~15 cycle penalty!\n");
    
    printf("\n=== ML IMPLICATIONS ===\n");
    printf("1. GPU kernels avoid branches when possible\n");
    printf("2. Masking is often faster than branching\n");
    printf("3. Fused operations reduce dependencies\n");
    printf("4. This is why Triton uses block-based computation\n");
    
    free(data);
    return 0;
}
