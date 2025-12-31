/**
 * 02_custom_allocator.c - Building a Simple Pool Allocator
 * 
 * This is the concept behind PyTorch's CUDA caching allocator.
 * Reusing memory is much faster than going to the OS.
 * 
 * Compile: gcc -O2 -o custom_alloc 02_custom_allocator.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ============================================================
// Simple Pool Allocator
// ============================================================

#define POOL_SIZE (64 * 1024 * 1024)  // 64 MB pool
#define BLOCK_SIZE 4096               // 4 KB blocks

typedef struct Block {
    struct Block* next;
} Block;

typedef struct {
    void* pool;           // Raw memory pool
    Block* free_list;     // List of free blocks
    size_t block_size;
    size_t total_blocks;
    size_t used_blocks;
} PoolAllocator;

PoolAllocator* pool_create(size_t pool_size, size_t block_size) {
    PoolAllocator* alloc = malloc(sizeof(PoolAllocator));
    
    // Allocate the pool
    alloc->pool = aligned_alloc(64, pool_size);
    alloc->block_size = block_size;
    alloc->total_blocks = pool_size / block_size;
    alloc->used_blocks = 0;
    
    // Build free list
    alloc->free_list = NULL;
    char* ptr = (char*)alloc->pool;
    for (size_t i = 0; i < alloc->total_blocks; i++) {
        Block* block = (Block*)ptr;
        block->next = alloc->free_list;
        alloc->free_list = block;
        ptr += block_size;
    }
    
    return alloc;
}

void* pool_alloc(PoolAllocator* alloc) {
    if (alloc->free_list == NULL) {
        return NULL;  // Pool exhausted
    }
    
    Block* block = alloc->free_list;
    alloc->free_list = block->next;
    alloc->used_blocks++;
    
    return (void*)block;
}

void pool_free(PoolAllocator* alloc, void* ptr) {
    Block* block = (Block*)ptr;
    block->next = alloc->free_list;
    alloc->free_list = block;
    alloc->used_blocks--;
}

void pool_destroy(PoolAllocator* alloc) {
    free(alloc->pool);
    free(alloc);
}

void pool_stats(PoolAllocator* alloc) {
    printf("Pool: %zu/%zu blocks used (%.1f%%)\n",
           alloc->used_blocks, alloc->total_blocks,
           100.0 * alloc->used_blocks / alloc->total_blocks);
}

// ============================================================
// Benchmark
// ============================================================

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== CUSTOM POOL ALLOCATOR ===\n\n");
    
    printf("This demonstrates why PyTorch uses a caching allocator.\n");
    printf("Reusing memory from a pool is MUCH faster than malloc.\n\n");
    
    // Create pool
    PoolAllocator* pool = pool_create(POOL_SIZE, BLOCK_SIZE);
    printf("Created pool: %d MB, %zu blocks of %d bytes\n\n",
           POOL_SIZE / (1024*1024), pool->total_blocks, BLOCK_SIZE);
    
    int iterations = 100000;
    void* ptrs[1000];
    double start;
    
    // ========================================================
    // Benchmark: malloc/free
    // ========================================================
    printf("--- BENCHMARK: MALLOC/FREE ---\n");
    
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 100; j++) {
            ptrs[j] = malloc(BLOCK_SIZE);
        }
        for (int j = 0; j < 100; j++) {
            free(ptrs[j]);
        }
    }
    double malloc_time = get_time() - start;
    printf("malloc/free: %.2f ms\n", malloc_time * 1000);
    
    // ========================================================
    // Benchmark: Pool allocator
    // ========================================================
    printf("\n--- BENCHMARK: POOL ALLOCATOR ---\n");
    
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 100; j++) {
            ptrs[j] = pool_alloc(pool);
        }
        for (int j = 0; j < 100; j++) {
            pool_free(pool, ptrs[j]);
        }
    }
    double pool_time = get_time() - start;
    printf("pool alloc:  %.2f ms\n", pool_time * 1000);
    
    printf("\nPool allocator is %.1fx faster!\n\n", malloc_time / pool_time);
    
    // ========================================================
    // Demonstrate usage pattern
    // ========================================================
    printf("--- USAGE PATTERN ---\n\n");
    
    // Allocate some blocks
    void* a = pool_alloc(pool);
    void* b = pool_alloc(pool);
    void* c = pool_alloc(pool);
    pool_stats(pool);
    
    // Free and reuse
    pool_free(pool, b);
    pool_stats(pool);
    
    void* d = pool_alloc(pool);  // Reuses b's memory!
    printf("Freed block at %p, new alloc at %p (same!)\n", b, d);
    
    pool_free(pool, a);
    pool_free(pool, c);
    pool_free(pool, d);
    
    pool_destroy(pool);
    
    // ========================================================
    // PyTorch Connection
    // ========================================================
    printf("\n--- PYTORCH CACHING ALLOCATOR ---\n\n");
    
    printf("PyTorch's CUDA allocator does this at scale:\n\n");
    printf("1. SIZE CLASSES:\n");
    printf("   Separate pools for different sizes\n");
    printf("   512B, 1KB, 2KB, 4KB, ... 256MB, large\n\n");
    
    printf("2. BLOCK SPLITTING:\n");
    printf("   Large blocks split to satisfy small requests\n\n");
    
    printf("3. STREAM ORDERING:\n");
    printf("   Tracks which CUDA stream last used a block\n");
    printf("   Only reuses when stream is synchronized\n\n");
    
    printf("4. STATISTICS:\n");
    printf("   torch.cuda.memory_allocated()\n");
    printf("   torch.cuda.memory_reserved()\n");
    printf("   torch.cuda.memory_stats()\n\n");
    
    printf("5. MANUAL CONTROL:\n");
    printf("   torch.cuda.empty_cache() - release to CUDA\n");
    printf("   PYTORCH_CUDA_ALLOC_CONF - tuning options\n");
    
    return 0;
}
