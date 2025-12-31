/*
 * False Sharing Demonstration
 * 
 * Shows how false sharing destroys multi-threaded performance
 * and how to fix it with proper padding/alignment.
 * 
 * Compile: gcc -O3 -pthread 01_false_sharing_demo.c -o false_sharing
 * Run: ./false_sharing
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4
#define ITERATIONS 100000000
#define CACHE_LINE_SIZE 64

// BAD: Counters share cache line
struct bad_counters {
    volatile long counter[NUM_THREADS];  // All in same/adjacent cache lines
};

// GOOD: Each counter on its own cache line
struct good_counters {
    struct {
        volatile long counter;
        char padding[CACHE_LINE_SIZE - sizeof(long)];
    } __attribute__((aligned(CACHE_LINE_SIZE))) data[NUM_THREADS];
};

// C++11 style with alignas (if using C11)
struct alignas_counters {
    _Alignas(CACHE_LINE_SIZE) volatile long counter[NUM_THREADS][CACHE_LINE_SIZE / sizeof(long)];
};

static struct bad_counters bad;
static struct good_counters good;

typedef struct {
    int thread_id;
    int use_padded;
} thread_arg_t;

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void* counter_thread(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    int id = targ->thread_id;
    
    if (targ->use_padded) {
        // GOOD: Each thread writes to its own cache line
        for (long i = 0; i < ITERATIONS; i++) {
            good.data[id].counter++;
        }
    } else {
        // BAD: Threads write to adjacent memory (same cache line)
        for (long i = 0; i < ITERATIONS; i++) {
            bad.counter[id]++;
        }
    }
    
    return NULL;
}

void run_benchmark(int use_padded, const char* name) {
    pthread_t threads[NUM_THREADS];
    thread_arg_t args[NUM_THREADS];
    
    // Reset counters
    for (int i = 0; i < NUM_THREADS; i++) {
        bad.counter[i] = 0;
        good.data[i].counter = 0;
    }
    
    double start = get_time_sec();
    
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].use_padded = use_padded;
        pthread_create(&threads[i], NULL, counter_thread, &args[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double elapsed = get_time_sec() - start;
    double ops_per_sec = (NUM_THREADS * ITERATIONS) / elapsed / 1e6;
    
    printf("%-20s: %.3f sec, %.1f M ops/sec\n", name, elapsed, ops_per_sec);
}

int main() {
    printf("=== False Sharing Demonstration ===\n");
    printf("Threads: %d, Iterations per thread: %d\n\n", NUM_THREADS, ITERATIONS);
    
    printf("Memory layout:\n");
    printf("  bad.counter[0] addr: %p\n", (void*)&bad.counter[0]);
    printf("  bad.counter[1] addr: %p\n", (void*)&bad.counter[1]);
    printf("  Difference: %ld bytes (cache line = %d)\n\n", 
           (long)((char*)&bad.counter[1] - (char*)&bad.counter[0]), CACHE_LINE_SIZE);
    
    printf("  good.data[0].counter addr: %p\n", (void*)&good.data[0].counter);
    printf("  good.data[1].counter addr: %p\n", (void*)&good.data[1].counter);
    printf("  Difference: %ld bytes\n\n",
           (long)((char*)&good.data[1].counter - (char*)&good.data[0].counter));
    
    // Warm up
    run_benchmark(0, "Warmup (bad)");
    run_benchmark(1, "Warmup (good)");
    
    printf("\n=== Benchmark Results ===\n");
    run_benchmark(0, "False Sharing (BAD)");
    run_benchmark(1, "Padded (GOOD)");
    
    printf("\n=== Analysis ===\n");
    printf("False sharing occurs when threads write to different variables\n");
    printf("that share the same cache line. The cache coherency protocol\n");
    printf("(MESI/MOESI) causes the cache line to bounce between cores.\n\n");
    printf("Fix: Ensure frequently-written variables are on separate cache lines.\n");
    printf("Use padding or alignas(%d) for hot variables.\n", CACHE_LINE_SIZE);
    
    return 0;
}
