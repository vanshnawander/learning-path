/*
 * POSIX Threads (pthreads) Fundamentals
 * 
 * Comprehensive demonstration of pthread API including:
 * - Thread creation and joining
 * - Mutexes and condition variables
 * - Thread-local storage
 * - Thread attributes
 * 
 * Compile: gcc -pthread 01_pthreads_fundamentals.c -o pthreads_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#define NUM_THREADS 4
#define ITERATIONS 1000000

// ============================================================
// Part 1: Basic Thread Creation
// ============================================================

void* simple_thread(void* arg) {
    int id = *(int*)arg;
    printf("Thread %d: Hello from thread!\n", id);
    return (void*)(long)(id * 10);  // Return value
}

void demo_basic_threads(void) {
    printf("\n=== Basic Thread Creation ===\n");
    
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        int rc = pthread_create(&threads[i], NULL, simple_thread, &thread_ids[i]);
        if (rc) {
            fprintf(stderr, "pthread_create failed: %s\n", strerror(rc));
            exit(1);
        }
    }
    
    // Join threads and get return values
    for (int i = 0; i < NUM_THREADS; i++) {
        void* retval;
        pthread_join(threads[i], &retval);
        printf("Thread %d returned: %ld\n", i, (long)retval);
    }
}

// ============================================================
// Part 2: Mutexes - Protecting Shared Data
// ============================================================

long shared_counter = 0;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;

void* mutex_thread(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&counter_mutex);
        shared_counter++;
        pthread_mutex_unlock(&counter_mutex);
    }
    return NULL;
}

// Without mutex - RACE CONDITION
long unsafe_counter = 0;

void* unsafe_thread(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        unsafe_counter++;  // Data race!
    }
    return NULL;
}

void demo_mutexes(void) {
    printf("\n=== Mutex Protection ===\n");
    
    pthread_t threads[NUM_THREADS];
    
    // Test unsafe counter
    unsafe_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, unsafe_thread, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Unsafe counter: %ld (expected: %d)\n", 
           unsafe_counter, NUM_THREADS * ITERATIONS);
    
    // Test safe counter
    shared_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, mutex_thread, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Safe counter:   %ld (expected: %d)\n", 
           shared_counter, NUM_THREADS * ITERATIONS);
}

// ============================================================
// Part 3: Condition Variables - Producer/Consumer
// ============================================================

#define BUFFER_SIZE 10

typedef struct {
    int buffer[BUFFER_SIZE];
    int count;
    int in;   // Write position
    int out;  // Read position
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} bounded_buffer_t;

bounded_buffer_t bb = {
    .count = 0, .in = 0, .out = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .not_full = PTHREAD_COND_INITIALIZER,
    .not_empty = PTHREAD_COND_INITIALIZER
};

void buffer_put(bounded_buffer_t* b, int item) {
    pthread_mutex_lock(&b->mutex);
    
    // Wait while buffer is full
    while (b->count == BUFFER_SIZE) {
        pthread_cond_wait(&b->not_full, &b->mutex);
    }
    
    b->buffer[b->in] = item;
    b->in = (b->in + 1) % BUFFER_SIZE;
    b->count++;
    
    pthread_cond_signal(&b->not_empty);
    pthread_mutex_unlock(&b->mutex);
}

int buffer_get(bounded_buffer_t* b) {
    pthread_mutex_lock(&b->mutex);
    
    // Wait while buffer is empty
    while (b->count == 0) {
        pthread_cond_wait(&b->not_empty, &b->mutex);
    }
    
    int item = b->buffer[b->out];
    b->out = (b->out + 1) % BUFFER_SIZE;
    b->count--;
    
    pthread_cond_signal(&b->not_full);
    pthread_mutex_unlock(&b->mutex);
    
    return item;
}

void* producer(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 20; i++) {
        int item = id * 100 + i;
        buffer_put(&bb, item);
        printf("Producer %d: put %d\n", id, item);
        usleep(rand() % 10000);
    }
    return NULL;
}

void* consumer(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 20; i++) {
        int item = buffer_get(&bb);
        printf("Consumer %d: got %d\n", id, item);
        usleep(rand() % 10000);
    }
    return NULL;
}

void demo_condition_variables(void) {
    printf("\n=== Condition Variables (Producer/Consumer) ===\n");
    
    pthread_t prod[2], cons[2];
    int ids[] = {0, 1};
    
    pthread_create(&prod[0], NULL, producer, &ids[0]);
    pthread_create(&prod[1], NULL, producer, &ids[1]);
    pthread_create(&cons[0], NULL, consumer, &ids[0]);
    pthread_create(&cons[1], NULL, consumer, &ids[1]);
    
    pthread_join(prod[0], NULL);
    pthread_join(prod[1], NULL);
    pthread_join(cons[0], NULL);
    pthread_join(cons[1], NULL);
}

// ============================================================
// Part 4: Thread-Local Storage
// ============================================================

pthread_key_t tls_key;

void tls_destructor(void* value) {
    printf("TLS destructor called, freeing: %p\n", value);
    free(value);
}

void* tls_thread(void* arg) {
    int id = *(int*)arg;
    
    // Allocate thread-local data
    int* my_data = malloc(sizeof(int));
    *my_data = id * 1000;
    pthread_setspecific(tls_key, my_data);
    
    // Access thread-local data
    int* data = pthread_getspecific(tls_key);
    printf("Thread %d: TLS value = %d\n", id, *data);
    
    return NULL;
}

void demo_thread_local_storage(void) {
    printf("\n=== Thread-Local Storage ===\n");
    
    pthread_key_create(&tls_key, tls_destructor);
    
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, tls_thread, &ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_key_delete(tls_key);
}

// ============================================================
// Part 5: Thread Attributes
// ============================================================

void demo_thread_attributes(void) {
    printf("\n=== Thread Attributes ===\n");
    
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    // Set stack size (minimum 16KB on most systems)
    size_t stack_size = 1024 * 1024;  // 1MB
    pthread_attr_setstacksize(&attr, stack_size);
    
    // Set detach state
    // PTHREAD_CREATE_JOINABLE (default) - must be joined
    // PTHREAD_CREATE_DETACHED - resources freed on exit
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    // Get and print attributes
    pthread_attr_getstacksize(&attr, &stack_size);
    printf("Stack size: %zu bytes\n", stack_size);
    
    int detach_state;
    pthread_attr_getdetachstate(&attr, &detach_state);
    printf("Detach state: %s\n", 
           detach_state == PTHREAD_CREATE_DETACHED ? "detached" : "joinable");
    
    pthread_attr_destroy(&attr);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== POSIX Threads (pthreads) Demonstration ===\n");
    
    demo_basic_threads();
    demo_mutexes();
    demo_thread_attributes();
    demo_thread_local_storage();
    
    // Comment out for shorter output:
    // demo_condition_variables();
    
    printf("\n=== All demos completed ===\n");
    return 0;
}
