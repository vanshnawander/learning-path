/*
 * Thread Pool Implementation
 * 
 * A production-quality thread pool demonstrating:
 * - Work queue with condition variables
 * - Graceful shutdown
 * - Work stealing (optional)
 * 
 * Compile: gcc -pthread -O2 01_thread_pool.c -o thread_pool
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

// Task function type
typedef void (*task_func_t)(void* arg);

// Task structure
typedef struct task {
    task_func_t function;
    void* argument;
    struct task* next;
} task_t;

// Thread pool structure
typedef struct {
    pthread_t* threads;
    int thread_count;
    
    task_t* task_queue_head;
    task_t* task_queue_tail;
    int task_count;
    
    pthread_mutex_t lock;
    pthread_cond_t notify;
    
    bool shutdown;
    bool graceful;
    
    int working_count;
    pthread_cond_t idle;
} thread_pool_t;

// Worker thread function
void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    
    while (true) {
        pthread_mutex_lock(&pool->lock);
        
        // Wait for task or shutdown
        while (pool->task_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->notify, &pool->lock);
        }
        
        // Check for shutdown
        if (pool->shutdown) {
            if (!pool->graceful || pool->task_count == 0) {
                pthread_mutex_unlock(&pool->lock);
                break;
            }
        }
        
        // Get task from queue
        task_t* task = pool->task_queue_head;
        if (task) {
            pool->task_queue_head = task->next;
            if (pool->task_queue_head == NULL) {
                pool->task_queue_tail = NULL;
            }
            pool->task_count--;
            pool->working_count++;
        }
        
        pthread_mutex_unlock(&pool->lock);
        
        // Execute task
        if (task) {
            task->function(task->argument);
            free(task);
            
            pthread_mutex_lock(&pool->lock);
            pool->working_count--;
            if (pool->working_count == 0 && pool->task_count == 0) {
                pthread_cond_signal(&pool->idle);
            }
            pthread_mutex_unlock(&pool->lock);
        }
    }
    
    return NULL;
}

// Create thread pool
thread_pool_t* thread_pool_create(int num_threads) {
    thread_pool_t* pool = calloc(1, sizeof(thread_pool_t));
    if (!pool) return NULL;
    
    pool->thread_count = num_threads;
    pool->threads = calloc(num_threads, sizeof(pthread_t));
    
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->notify, NULL);
    pthread_cond_init(&pool->idle, NULL);
    
    // Start worker threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    
    return pool;
}

// Add task to pool
int thread_pool_submit(thread_pool_t* pool, task_func_t func, void* arg) {
    task_t* task = malloc(sizeof(task_t));
    if (!task) return -1;
    
    task->function = func;
    task->argument = arg;
    task->next = NULL;
    
    pthread_mutex_lock(&pool->lock);
    
    if (pool->shutdown) {
        pthread_mutex_unlock(&pool->lock);
        free(task);
        return -1;
    }
    
    // Add to queue
    if (pool->task_queue_tail) {
        pool->task_queue_tail->next = task;
    } else {
        pool->task_queue_head = task;
    }
    pool->task_queue_tail = task;
    pool->task_count++;
    
    pthread_cond_signal(&pool->notify);
    pthread_mutex_unlock(&pool->lock);
    
    return 0;
}

// Wait for all tasks to complete
void thread_pool_wait(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->lock);
    while (pool->task_count > 0 || pool->working_count > 0) {
        pthread_cond_wait(&pool->idle, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
}

// Destroy thread pool
void thread_pool_destroy(thread_pool_t* pool, bool graceful) {
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = true;
    pool->graceful = graceful;
    pthread_cond_broadcast(&pool->notify);
    pthread_mutex_unlock(&pool->lock);
    
    // Wait for all threads
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    // Free remaining tasks if not graceful
    if (!graceful) {
        task_t* task = pool->task_queue_head;
        while (task) {
            task_t* next = task->next;
            free(task);
            task = next;
        }
    }
    
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->notify);
    pthread_cond_destroy(&pool->idle);
    free(pool->threads);
    free(pool);
}

// ============================================================
// Example Usage
// ============================================================

void example_task(void* arg) {
    int id = *(int*)arg;
    printf("Task %d executing on thread %lu\n", id, pthread_self());
    usleep(100000);  // Simulate work
    free(arg);
}

int main() {
    printf("=== Thread Pool Demo ===\n");
    
    // Create pool with 4 threads
    thread_pool_t* pool = thread_pool_create(4);
    
    // Submit 20 tasks
    for (int i = 0; i < 20; i++) {
        int* arg = malloc(sizeof(int));
        *arg = i;
        thread_pool_submit(pool, example_task, arg);
    }
    
    printf("All tasks submitted, waiting...\n");
    thread_pool_wait(pool);
    
    printf("All tasks completed, destroying pool...\n");
    thread_pool_destroy(pool, true);
    
    printf("Done!\n");
    return 0;
}
