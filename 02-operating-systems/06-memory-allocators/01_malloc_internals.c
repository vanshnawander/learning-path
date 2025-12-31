/**
 * 01_malloc_internals.c - How Memory Allocators Work
 * 
 * Understanding allocators explains:
 * - PyTorch's CUDA caching allocator
 * - Why repeated malloc/free is slow
 * - Memory fragmentation issues
 * 
 * Compile: gcc -O2 -o 01_malloc_internals 01_malloc_internals.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_ALLOCS 100000

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== MEMORY ALLOCATOR INTERNALS ===\n\n");
    
    // Benchmark: many small allocations
    printf("--- MANY SMALL ALLOCATIONS ---\n");
    
    void* ptrs[NUM_ALLOCS];
    double start = get_time();
    
    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptrs[i] = malloc(64);
    }
    double alloc_time = get_time() - start;
    
    start = get_time();
    for (int i = 0; i < NUM_ALLOCS; i++) {
        free(ptrs[i]);
    }
    double free_time = get_time() - start;
    
    printf("%d allocations: %.2f ms (%.0f ns each)\n", 
           NUM_ALLOCS, alloc_time * 1000, (alloc_time / NUM_ALLOCS) * 1e9);
    printf("%d frees:       %.2f ms (%.0f ns each)\n",
           NUM_ALLOCS, free_time * 1000, (free_time / NUM_ALLOCS) * 1e9);
    
    // Reuse pattern (simulates tensor allocation)
    printf("\n--- ALLOCATION REUSE PATTERN ---\n");
    
    void* ptr = NULL;
    start = get_time();
    for (int i = 0; i < NUM_ALLOCS; i++) {
        ptr = malloc(1024);
        free(ptr);
    }
    double reuse_time = get_time() - start;
    
    printf("Alloc+free %d times: %.2f ms\n", NUM_ALLOCS, reuse_time * 1000);
    printf("Modern allocators cache freed blocks!\n");
    
    // Size class demonstration
    printf("\n--- SIZE CLASSES ---\n");
    printf("Allocators round up to size classes:\n");
    
    size_t sizes[] = {1, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 11; i++) {
        void* p = malloc(sizes[i]);
        printf("  Request %4zu bytes → likely allocated ~%zu bytes\n",
               sizes[i], sizes[i] < 16 ? 16 : 
               (sizes[i] <= 32 ? 32 : (sizes[i] + 15) & ~15));
        free(p);
    }
    
    printf("\n=== PYTORCH CUDA ALLOCATOR ===\n");
    printf("PyTorch's caching allocator is similar:\n");
    printf("1. Maintains pools of different size classes\n");
    printf("2. Freed tensors go back to pool, not to CUDA\n");
    printf("3. cudaMalloc/cudaFree are SLOW (~ms)\n");
    printf("4. Pool reuse is fast (~us)\n");
    printf("5. torch.cuda.empty_cache() releases to CUDA\n");
    printf("6. Memory fragmentation can cause OOM\n");
    
    return 0;
}
