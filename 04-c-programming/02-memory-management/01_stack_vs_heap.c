/**
 * 01_stack_vs_heap.c - Understanding Memory Regions
 * 
 * Where your data lives determines performance and lifetime.
 * Critical for ML: tensor allocation, weight storage, activations.
 * 
 * Compile: gcc -o stack_heap 01_stack_vs_heap.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARRAY_SIZE (1024 * 1024)  // 1M elements

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Recursive function to show stack growth
void show_stack_depth(int depth, void* first_frame) {
    int local_var;
    if (depth == 0) {
        printf("Stack grows: %s\n", 
               (void*)&local_var < first_frame ? "downward ↓" : "upward ↑");
        printf("Distance from first frame: %td bytes\n",
               (char*)first_frame - (char*)&local_var);
    }
    if (depth > 0) {
        show_stack_depth(depth - 1, first_frame);
    }
}

int main() {
    printf("=== STACK VS HEAP ===\n\n");
    
    // ========================================================
    // SECTION 1: Stack allocation
    // ========================================================
    printf("--- SECTION 1: STACK ALLOCATION ---\n\n");
    
    int stack_var = 42;
    int stack_array[100];
    
    printf("Stack variable at:     %p\n", (void*)&stack_var);
    printf("Stack array at:        %p\n", (void*)stack_array);
    printf("Stack array size:      %zu bytes\n\n", sizeof(stack_array));
    
    printf("Stack characteristics:\n");
    printf("  + Automatic lifetime (freed when function returns)\n");
    printf("  + Very fast allocation (just move stack pointer)\n");
    printf("  + Cache-friendly (recently used)\n");
    printf("  - Limited size (typically 1-8 MB)\n");
    printf("  - Cannot return pointers to local variables!\n\n");
    
    // Show stack direction
    int first;
    show_stack_depth(10, &first);
    
    // ========================================================
    // SECTION 2: Heap allocation
    // ========================================================
    printf("\n--- SECTION 2: HEAP ALLOCATION ---\n\n");
    
    int* heap_array = malloc(100 * sizeof(int));
    
    printf("Heap array at:         %p\n", (void*)heap_array);
    printf("(Pointer itself on stack: %p)\n\n", (void*)&heap_array);
    
    printf("Heap characteristics:\n");
    printf("  + Unlimited size (limited by RAM)\n");
    printf("  + Controlled lifetime (you decide when to free)\n");
    printf("  + Can be shared between functions\n");
    printf("  - Slower allocation (malloc overhead)\n");
    printf("  - Must manually free (memory leaks!)\n");
    printf("  - Fragmentation possible\n\n");
    
    free(heap_array);
    
    // ========================================================
    // SECTION 3: Allocation speed comparison
    // ========================================================
    printf("--- SECTION 3: ALLOCATION SPEED ---\n\n");
    
    double start;
    int iterations = 100000;
    
    // Stack allocation (in loop - reuses same space)
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        int stack_arr[100];
        stack_arr[0] = i;  // Prevent optimization
        (void)stack_arr[0];
    }
    double stack_time = get_time() - start;
    
    // Heap allocation
    start = get_time();
    for (int i = 0; i < iterations; i++) {
        int* heap_arr = malloc(100 * sizeof(int));
        heap_arr[0] = i;
        free(heap_arr);
    }
    double heap_time = get_time() - start;
    
    printf("Stack allocation: %.2f ms\n", stack_time * 1000);
    printf("Heap allocation:  %.2f ms\n", heap_time * 1000);
    printf("Heap is %.0fx slower!\n\n", heap_time / stack_time);
    
    // ========================================================
    // SECTION 4: Static/Global memory
    // ========================================================
    printf("--- SECTION 4: STATIC/GLOBAL MEMORY ---\n\n");
    
    static int static_var = 100;
    static int static_array[1000];
    
    printf("Static variable at:    %p\n", (void*)&static_var);
    printf("Static array at:       %p\n", (void*)static_array);
    
    printf("\nStatic characteristics:\n");
    printf("  + Lifetime = program lifetime\n");
    printf("  + Initialized to zero by default\n");
    printf("  + No allocation overhead at runtime\n");
    printf("  - Size fixed at compile time\n");
    printf("  - Adds to executable size\n\n");
    
    // ========================================================
    // SECTION 5: ML Implications
    // ========================================================
    printf("--- SECTION 5: ML IMPLICATIONS ---\n\n");
    
    printf("MODEL WEIGHTS:\n");
    printf("  Heap allocated (too large for stack)\n");
    printf("  Or memory-mapped from file\n\n");
    
    printf("ACTIVATIONS:\n");
    printf("  PyTorch uses caching allocator\n");
    printf("  Reuses heap memory to avoid malloc overhead\n\n");
    
    printf("SMALL TEMPORARIES:\n");
    printf("  Stack when possible (faster)\n");
    printf("  Compiler may use registers\n\n");
    
    printf("GPU MEMORY:\n");
    printf("  Separate address space entirely!\n");
    printf("  cudaMalloc is ~1000x slower than malloc\n");
    printf("  PyTorch caching allocator is essential\n");
    
    return 0;
}
