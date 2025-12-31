/*
 * Atomic Operations in C11
 * 
 * Demonstrates lock-free programming with C11 atomics.
 * 
 * Compile: gcc -std=c11 -pthread -O2 02_atomic_operations.c -o atomics
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdbool.h>

#define NUM_THREADS 8
#define ITERATIONS 1000000

// ============================================================
// 1. Basic Atomic Counter
// ============================================================

atomic_long atomic_counter = 0;
long regular_counter = 0;

void* increment_atomic(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        atomic_fetch_add(&atomic_counter, 1);
    }
    return NULL;
}

void* increment_regular(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        regular_counter++;  // DATA RACE
    }
    return NULL;
}

void demo_atomic_counter(void) {
    printf("=== Atomic Counter Demo ===\n");
    
    pthread_t threads[NUM_THREADS];
    
    // Test regular counter (race condition)
    regular_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_regular, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    printf("Regular counter: %ld (expected %d, RACE!)\n", 
           regular_counter, NUM_THREADS * ITERATIONS);
    
    // Test atomic counter
    atomic_store(&atomic_counter, 0);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment_atomic, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    printf("Atomic counter:  %ld (expected %d, CORRECT)\n\n", 
           atomic_load(&atomic_counter), NUM_THREADS * ITERATIONS);
}

// ============================================================
// 2. Compare-and-Swap (CAS)
// ============================================================

atomic_int cas_value = 0;

// Atomic maximum using CAS
void atomic_max(atomic_int* target, int value) {
    int current = atomic_load(target);
    while (value > current) {
        if (atomic_compare_exchange_weak(target, &current, value)) {
            break;
        }
        // current is updated with actual value on failure
    }
}

void* cas_max_thread(void* arg) {
    int id = *(int*)arg;
    for (int i = 0; i < 10000; i++) {
        int val = (id * 10000) + i;
        atomic_max(&cas_value, val);
    }
    return NULL;
}

void demo_cas(void) {
    printf("=== Compare-and-Swap Demo ===\n");
    
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];
    
    atomic_store(&cas_value, 0);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, cas_max_thread, &ids[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Max value found: %d (expected %d)\n\n", 
           atomic_load(&cas_value), (NUM_THREADS - 1) * 10000 + 9999);
}

// ============================================================
// 3. Spinlock Implementation
// ============================================================

typedef struct {
    atomic_flag flag;
} spinlock_t;

void spinlock_init(spinlock_t* lock) {
    atomic_flag_clear(&lock->flag);
}

void spinlock_lock(spinlock_t* lock) {
    while (atomic_flag_test_and_set_explicit(&lock->flag, 
                                              memory_order_acquire)) {
        // Spin-wait (optionally add pause instruction)
    }
}

void spinlock_unlock(spinlock_t* lock) {
    atomic_flag_clear_explicit(&lock->flag, memory_order_release);
}

spinlock_t spin;
long protected_counter = 0;

void* spinlock_thread(void* arg) {
    for (int i = 0; i < ITERATIONS / 10; i++) {
        spinlock_lock(&spin);
        protected_counter++;
        spinlock_unlock(&spin);
    }
    return NULL;
}

void demo_spinlock(void) {
    printf("=== Spinlock Demo ===\n");
    
    pthread_t threads[NUM_THREADS];
    
    spinlock_init(&spin);
    protected_counter = 0;
    
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, spinlock_thread, NULL);
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    
    printf("Spinlock-protected counter: %ld (expected %d)\n\n",
           protected_counter, NUM_THREADS * ITERATIONS / 10);
}

// ============================================================
// 4. Lock-Free Stack (Treiber Stack)
// ============================================================

typedef struct node {
    int data;
    struct node* next;
} node_t;

typedef struct {
    _Atomic(node_t*) head;
} lf_stack_t;

void lf_stack_init(lf_stack_t* stack) {
    atomic_store(&stack->head, NULL);
}

void lf_stack_push(lf_stack_t* stack, int value) {
    node_t* new_node = malloc(sizeof(node_t));
    new_node->data = value;
    
    new_node->next = atomic_load_explicit(&stack->head, memory_order_relaxed);
    
    while (!atomic_compare_exchange_weak_explicit(
            &stack->head,
            &new_node->next,
            new_node,
            memory_order_release,
            memory_order_relaxed)) {
        // new_node->next updated to current head on failure
    }
}

bool lf_stack_pop(lf_stack_t* stack, int* value) {
    node_t* old_head = atomic_load_explicit(&stack->head, memory_order_acquire);
    
    while (old_head != NULL) {
        if (atomic_compare_exchange_weak_explicit(
                &stack->head,
                &old_head,
                old_head->next,
                memory_order_acquire,
                memory_order_relaxed)) {
            *value = old_head->data;
            // Note: In production, use hazard pointers or epoch-based reclamation
            free(old_head);
            return true;
        }
        // old_head updated to current head on failure
    }
    return false;
}

lf_stack_t stack;
atomic_int push_count = 0;
atomic_int pop_count = 0;

void* stack_pusher(void* arg) {
    for (int i = 0; i < 10000; i++) {
        lf_stack_push(&stack, i);
        atomic_fetch_add(&push_count, 1);
    }
    return NULL;
}

void* stack_popper(void* arg) {
    int value;
    int local_count = 0;
    for (int i = 0; i < 10000; i++) {
        if (lf_stack_pop(&stack, &value)) {
            local_count++;
        }
    }
    atomic_fetch_add(&pop_count, local_count);
    return NULL;
}

void demo_lock_free_stack(void) {
    printf("=== Lock-Free Stack Demo ===\n");
    
    pthread_t pushers[4], poppers[4];
    
    lf_stack_init(&stack);
    atomic_store(&push_count, 0);
    atomic_store(&pop_count, 0);
    
    for (int i = 0; i < 4; i++) {
        pthread_create(&pushers[i], NULL, stack_pusher, NULL);
        pthread_create(&poppers[i], NULL, stack_popper, NULL);
    }
    
    for (int i = 0; i < 4; i++) {
        pthread_join(pushers[i], NULL);
        pthread_join(poppers[i], NULL);
    }
    
    // Drain remaining
    int value;
    while (lf_stack_pop(&stack, &value)) {
        atomic_fetch_add(&pop_count, 1);
    }
    
    printf("Pushed: %d, Popped: %d\n\n", 
           atomic_load(&push_count), atomic_load(&pop_count));
}

// ============================================================
// 5. Memory Ordering Demo
// ============================================================

atomic_int data = 0;
atomic_bool ready = false;

void* producer(void* arg) {
    data = 42;  // Regular store
    atomic_store_explicit(&ready, true, memory_order_release);
    return NULL;
}

void* consumer(void* arg) {
    while (!atomic_load_explicit(&ready, memory_order_acquire)) {
        // Spin
    }
    int result = data;  // Guaranteed to see 42
    printf("Consumer read data = %d (expected 42)\n", result);
    return NULL;
}

void demo_memory_ordering(void) {
    printf("=== Memory Ordering Demo ===\n");
    
    pthread_t prod, cons;
    
    atomic_store(&data, 0);
    atomic_store(&ready, false);
    
    pthread_create(&cons, NULL, consumer, NULL);
    pthread_create(&prod, NULL, producer, NULL);
    
    pthread_join(prod, NULL);
    pthread_join(cons, NULL);
    printf("\n");
}

int main(void) {
    demo_atomic_counter();
    demo_cas();
    demo_spinlock();
    demo_lock_free_stack();
    demo_memory_ordering();
    
    printf("All demos completed!\n");
    return 0;
}
