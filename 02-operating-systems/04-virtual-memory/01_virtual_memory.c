/**
 * 01_virtual_memory.c - Virtual Memory and Page Tables
 * 
 * Every ML program uses virtual memory. Understanding it explains:
 * - Why mmap is efficient
 * - What page faults are
 * - How the OS manages memory for large models
 * 
 * Compile: gcc -O2 -o 01_virtual_memory 01_virtual_memory.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#define PAGE_SIZE 4096

int main() {
    printf("=== VIRTUAL MEMORY FUNDAMENTALS ===\n\n");
    
    printf("System page size: %ld bytes\n\n", sysconf(_SC_PAGESIZE));
    
    printf("--- ADDRESS SPACE LAYOUT ---\n");
    
    // Show addresses of different memory regions
    int stack_var = 42;
    static int data_var = 100;
    int* heap_var = malloc(sizeof(int));
    
    printf("Stack variable:  %p\n", (void*)&stack_var);
    printf("Heap variable:   %p\n", (void*)heap_var);
    printf("Data variable:   %p\n", (void*)&data_var);
    printf("Code (main):     %p\n", (void*)main);
    
    printf("\nTypical 64-bit layout (high to low):\n");
    printf("  Kernel space   (0xFFFF...)\n");
    printf("  Stack          (grows down)\n");
    printf("  ...\n");
    printf("  mmap region\n");
    printf("  ...\n");
    printf("  Heap           (grows up)\n");
    printf("  BSS            (uninitialized globals)\n");
    printf("  Data           (initialized globals)\n");
    printf("  Text/Code      (0x0040...)\n");
    
    // Demonstrate page fault behavior
    printf("\n--- PAGE FAULTS ---\n");
    
    size_t size = 100 * 1024 * 1024;  // 100 MB
    
    printf("Allocating %zu MB with mmap...\n", size / (1024*1024));
    
    char* region = mmap(NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (region == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    
    printf("mmap returned: %p\n", region);
    printf("No physical memory allocated yet! (demand paging)\n\n");
    
    printf("Touching first page...\n");
    region[0] = 'A';  // This causes a page fault
    printf("Page fault occurred, one 4KB page now allocated\n\n");
    
    printf("Touching every page (causing %zu page faults)...\n", size / PAGE_SIZE);
    for (size_t i = 0; i < size; i += PAGE_SIZE) {
        region[i] = 'X';
    }
    printf("All %zu pages now backed by physical memory\n", size / PAGE_SIZE);
    
    munmap(region, size);
    
    printf("\n=== ML MEMORY IMPLICATIONS ===\n");
    printf("1. Model loading: weights are demand-paged from disk\n");
    printf("2. Large allocations don't use RAM until touched\n");
    printf("3. OOM can happen later than expected (lazy allocation)\n");
    printf("4. GPU memory is NOT virtual (physical allocation)\n");
    printf("5. Pinned memory bypasses virtual memory for DMA\n");
    
    free(heap_var);
    return 0;
}
