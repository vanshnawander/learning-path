/**
 * 02_threads_basics.c - POSIX Threads (pthreads)
 * 
 * Threads vs Processes:
 * - Threads share memory (easier communication)
 * - Threads have less overhead
 * - But need synchronization!
 * 
 * Compile: gcc -pthread -o 02_threads_basics 02_threads_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 4

// Shared data
int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Thread function (no mutex)
void* increment_unsafe(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 100000; i++) {
        counter++;  // RACE CONDITION!
    }
    return NULL;
}

// Thread function (with mutex)
void* increment_safe(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&mutex);
        counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    printf("=== THREADS: SHARED MEMORY PARALLELISM ===\n\n");
    
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];
    
    // Unsafe increment (race condition)
    printf("--- UNSAFE INCREMENT (RACE CONDITION) ---\n");
    counter = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, increment_unsafe, &ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Expected: %d\n", NUM_THREADS * 100000);
    printf("Got:      %d (race condition!)\n", counter);
    
    // Safe increment (with mutex)
    printf("\n--- SAFE INCREMENT (MUTEX) ---\n");
    counter = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, increment_safe, &ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Expected: %d\n", NUM_THREADS * 100000);
    printf("Got:      %d (correct!)\n", counter);
    
    printf("\n=== THREAD vs PROCESS FOR ML ===\n");
    printf("Process (multiprocessing):\n");
    printf("  + Avoids Python GIL\n");
    printf("  + Better isolation (one crash doesn't kill all)\n");
    printf("  - Memory not shared (need IPC)\n");
    printf("  - Higher overhead\n");
    printf("\nThreads (threading):\n");
    printf("  + Shared memory\n");
    printf("  + Lower overhead\n");
    printf("  - Python GIL blocks CPU-bound threads\n");
    printf("  - Need careful synchronization\n");
    printf("\nPyTorch DataLoader uses multiprocessing for workers\n");
    printf("But internal C++ uses threads for parallel ops\n");
    
    return 0;
}
