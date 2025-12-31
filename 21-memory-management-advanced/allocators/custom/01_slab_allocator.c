/*
 * Slab Allocator Implementation
 * 
 * A production-quality slab allocator demonstrating:
 * - Fixed-size object allocation
 * - Memory pooling for cache-friendly design
 * - Minimal fragmentation
 * 
 * Compile: gcc -O3 -pthread 01_slab_allocator.c -o slab_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

#define SLAB_PAGE_SIZE 4096
#define CACHE_LINE_SIZE 64
#define SLAB_MIN_OBJECTS 8

// Free object node (stored in free objects themselves)
typedef struct free_object {
    struct free_object* next;
} free_object_t;

// Individual slab
typedef struct slab {
    struct slab* next;
    struct slab* prev;
    uint32_t inuse;
    uint32_t total;
    free_object_t* free_list;
    uint8_t data[] __attribute__((aligned(64)));
} slab_t;

// Slab cache
typedef struct slab_cache {
    char name[32];
    size_t object_size;
    size_t aligned_size;
    uint32_t objects_per_slab;
    size_t slab_size;
    
    slab_t* slabs_partial;
    slab_t* slabs_full;
    slab_t* slabs_empty;
    uint32_t empty_count;
    uint32_t max_empty;
    
    pthread_mutex_t lock;
    uint64_t alloc_count;
    uint64_t free_count;
} slab_cache_t;

static inline size_t align_up(size_t size, size_t align) {
    return (size + align - 1) & ~(align - 1);
}

static slab_t* slab_create(slab_cache_t* cache) {
    slab_t* slab = (slab_t*)aligned_alloc(SLAB_PAGE_SIZE, cache->slab_size);
    if (!slab) return NULL;
    
    slab->next = slab->prev = NULL;
    slab->inuse = 0;
    slab->total = cache->objects_per_slab;
    
    // Build free list
    size_t header = align_up(sizeof(slab_t), CACHE_LINE_SIZE);
    uint8_t* obj_start = (uint8_t*)slab + header;
    
    slab->free_list = NULL;
    for (uint32_t i = 0; i < cache->objects_per_slab; i++) {
        free_object_t* obj = (free_object_t*)(obj_start + i * cache->aligned_size);
        obj->next = slab->free_list;
        slab->free_list = obj;
    }
    return slab;
}

static void slab_list_add(slab_t** list, slab_t* slab) {
    slab->next = *list;
    slab->prev = NULL;
    if (*list) (*list)->prev = slab;
    *list = slab;
}

static void slab_list_remove(slab_t** list, slab_t* slab) {
    if (slab->prev) slab->prev->next = slab->next;
    else *list = slab->next;
    if (slab->next) slab->next->prev = slab->prev;
    slab->next = slab->prev = NULL;
}

slab_cache_t* slab_cache_create(const char* name, size_t obj_size, uint32_t max_empty) {
    slab_cache_t* cache = (slab_cache_t*)calloc(1, sizeof(slab_cache_t));
    if (!cache) return NULL;
    
    strncpy(cache->name, name, sizeof(cache->name) - 1);
    cache->object_size = obj_size;
    cache->aligned_size = align_up(obj_size, sizeof(void*));
    if (cache->aligned_size < sizeof(free_object_t))
        cache->aligned_size = sizeof(free_object_t);
    
    size_t header = align_up(sizeof(slab_t), CACHE_LINE_SIZE);
    cache->slab_size = SLAB_PAGE_SIZE;
    cache->objects_per_slab = (cache->slab_size - header) / cache->aligned_size;
    cache->max_empty = max_empty;
    
    pthread_mutex_init(&cache->lock, NULL);
    return cache;
}

void* slab_alloc(slab_cache_t* cache) {
    pthread_mutex_lock(&cache->lock);
    
    slab_t* slab = cache->slabs_partial;
    if (!slab) {
        if (cache->slabs_empty) {
            slab = cache->slabs_empty;
            slab_list_remove(&cache->slabs_empty, slab);
            cache->empty_count--;
        } else {
            slab = slab_create(cache);
            if (!slab) { pthread_mutex_unlock(&cache->lock); return NULL; }
        }
        slab_list_add(&cache->slabs_partial, slab);
    }
    
    free_object_t* obj = slab->free_list;
    slab->free_list = obj->next;
    slab->inuse++;
    
    if (slab->inuse == slab->total) {
        slab_list_remove(&cache->slabs_partial, slab);
        slab_list_add(&cache->slabs_full, slab);
    }
    
    cache->alloc_count++;
    pthread_mutex_unlock(&cache->lock);
    return (void*)obj;
}

void slab_free_obj(slab_cache_t* cache, void* ptr) {
    if (!ptr) return;
    pthread_mutex_lock(&cache->lock);
    
    // Find slab (simplified - real impl uses page table)
    slab_t* slab = NULL;
    for (slab_t* s = cache->slabs_partial; s; s = s->next) {
        if ((uint8_t*)ptr >= (uint8_t*)s && 
            (uint8_t*)ptr < (uint8_t*)s + cache->slab_size) {
            slab = s; break;
        }
    }
    if (!slab) {
        for (slab_t* s = cache->slabs_full; s; s = s->next) {
            if ((uint8_t*)ptr >= (uint8_t*)s && 
                (uint8_t*)ptr < (uint8_t*)s + cache->slab_size) {
                slab = s; break;
            }
        }
    }
    
    if (slab) {
        bool was_full = (slab->inuse == slab->total);
        free_object_t* obj = (free_object_t*)ptr;
        obj->next = slab->free_list;
        slab->free_list = obj;
        slab->inuse--;
        
        if (was_full) {
            slab_list_remove(&cache->slabs_full, slab);
            slab_list_add(&cache->slabs_partial, slab);
        } else if (slab->inuse == 0) {
            slab_list_remove(&cache->slabs_partial, slab);
            if (cache->empty_count < cache->max_empty) {
                slab_list_add(&cache->slabs_empty, slab);
                cache->empty_count++;
            } else {
                free(slab);
            }
        }
        cache->free_count++;
    }
    pthread_mutex_unlock(&cache->lock);
}

void slab_cache_destroy(slab_cache_t* cache) {
    slab_t *slab, *next;
    for (slab = cache->slabs_partial; slab; slab = next) { next = slab->next; free(slab); }
    for (slab = cache->slabs_full; slab; slab = next) { next = slab->next; free(slab); }
    for (slab = cache->slabs_empty; slab; slab = next) { next = slab->next; free(slab); }
    pthread_mutex_destroy(&cache->lock);
    free(cache);
}

// Test
int main() {
    slab_cache_t* cache = slab_cache_create("test_128", 128, 2);
    
    void* ptrs[1000];
    for (int i = 0; i < 1000; i++) ptrs[i] = slab_alloc(cache);
    printf("Allocated 1000 objects, allocs=%lu\n", cache->alloc_count);
    
    for (int i = 0; i < 500; i++) slab_free_obj(cache, ptrs[i]);
    printf("Freed 500 objects, frees=%lu\n", cache->free_count);
    
    for (int i = 500; i < 1000; i++) slab_free_obj(cache, ptrs[i]);
    printf("Freed remaining, frees=%lu\n", cache->free_count);
    
    slab_cache_destroy(cache);
    return 0;
}
