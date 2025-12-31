/**
 * 04_false_sharing.c - False Sharing: A Hidden Performance Killer
 * 
 * False sharing occurs when threads modify different variables
 * that happen to be on the same cache line.
 * 
 * Compile: gcc -O2 -pthread -o 04_false_sharing 04_false_sharing.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4
#define ITERATIONS 100000000

// Bad: counters on same cache line
struct BadCounters {
    long c0, c1, c2, c3;  // All in same 64-byte cache line!
} bad_counters = {0};

// Good: counters on separate cache lines
struct __attribute__((aligned(64))) GoodCounter {
    long value;
    char padding[56];  // Pad to 64 bytes
};
struct GoodCounter good_counters[NUM_THREADS];

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void* bad_increment(void* arg) {
    int id = *(int*)arg;
    long* counter = &bad_counters.c0 + id;
    for (long i = 0; i < ITERATIONS; i++) {
        (*counter)++;
    }
    return NULL;
}

void* good_increment(void* arg) {
    int id = *(int*)arg;
    for (long i = 0; i < ITERATIONS; i++) {
        good_counters[id].value++;
    }
    return NULL;
}

int main() {
    printf("=== FALSE SHARING DEMONSTRATION ===\n\n");
    
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS] = {0, 1, 2, 3};
    double start;
    
    // Bad case: false sharing
    printf("--- WITH FALSE SHARING ---\n");
    start = get_time();
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, bad_increment, &ids[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    double bad_time = get_time() - start;
    printf("Time: %.2f ms\n", bad_time * 1000);
    
    // Good case: no false sharing
    printf("\n--- WITHOUT FALSE SHARING ---\n");
    start = get_time();
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, good_increment, &ids[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    double good_time = get_time() - start;
    printf("Time: %.2f ms\n", good_time * 1000);
    
    printf("\nFalse sharing is %.1fx slower!\n", bad_time / good_time);
    
    printf("\n=== WHY THIS HAPPENS ===\n");
    printf("1. Cache coherency: only one core can own a cache line\n");
    printf("2. When one thread writes, others must invalidate\n");
    printf("3. Even though they access DIFFERENT variables!\n");
    printf("4. Solution: pad data to cache line boundaries\n");
    
    printf("\n=== ML IMPLICATIONS ===\n");
    printf("1. Gradient accumulation buffers must be aligned\n");
    printf("2. Per-thread statistics need padding\n");
    printf("3. CUDA shared memory has similar bank conflicts\n");
    
    return 0;
}
