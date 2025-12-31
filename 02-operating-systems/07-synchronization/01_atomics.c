/**
 * 01_atomics.c - Atomic Operations for Lock-Free Programming
 * 
 * Atomics are the building blocks of:
 * - Lock-free data structures
 * - CUDA atomic operations
 * - Distributed gradient accumulation
 * 
 * Compile: gcc -O2 -pthread -o 01_atomics 01_atomics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4
#define ITERATIONS 1000000

// Non-atomic counter
long regular_counter = 0;

// Atomic counter
atomic_long atomic_counter = 0;

// Mutex-protected counter
long mutex_counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void* increment_regular(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        regular_counter++;  // RACE CONDITION
    }
    return NULL;
}

void* increment_atomic(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    return NULL;
}

void* increment_mutex(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&mutex);
        mutex_counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    printf("=== ATOMIC OPERATIONS ===\n\n");
    
    pthread_t threads[NUM_THREADS];
    double start;
    
    // Regular (broken)
    printf("--- REGULAR INCREMENT (BROKEN) ---\n");
    regular_counter = 0;
    start = get_time();
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_regular, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    double regular_time = get_time() - start;
    printf("Expected: %d, Got: %ld (WRONG!)\n", 
           NUM_THREADS * ITERATIONS, regular_counter);
    printf("Time: %.2f ms\n", regular_time * 1000);
    
    // Atomic
    printf("\n--- ATOMIC INCREMENT ---\n");
    atomic_counter = 0;
    start = get_time();
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_atomic, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    double atomic_time = get_time() - start;
    printf("Expected: %d, Got: %ld (CORRECT)\n",
           NUM_THREADS * ITERATIONS, atomic_load(&atomic_counter));
    printf("Time: %.2f ms\n", atomic_time * 1000);
    
    // Mutex
    printf("\n--- MUTEX INCREMENT ---\n");
    mutex_counter = 0;
    start = get_time();
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_mutex, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    double mutex_time = get_time() - start;
    printf("Expected: %d, Got: %ld (CORRECT)\n",
           NUM_THREADS * ITERATIONS, mutex_counter);
    printf("Time: %.2f ms\n", mutex_time * 1000);
    
    printf("\n=== COMPARISON ===\n");
    printf("Atomic: %.1fx faster than mutex\n", mutex_time / atomic_time);
    
    printf("\n=== CUDA ATOMICS ===\n");
    printf("atomicAdd() - add to global memory\n");
    printf("atomicMax() - find maximum\n");
    printf("atomicCAS() - compare and swap\n");
    printf("Used in: reductions, histograms, attention\n");
    
    return 0;
}
