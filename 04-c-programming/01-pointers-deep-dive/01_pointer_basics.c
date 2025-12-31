/**
 * 01_pointer_basics.c - Pointers From First Principles
 * 
 * A pointer is just a variable that holds a memory address.
 * That's it. Everything else follows from this.
 * 
 * Compile: gcc -o pointer_basics 01_pointer_basics.c
 * Run: ./pointer_basics
 */

#include <stdio.h>
#include <stdint.h>

int main() {
    printf("=== POINTERS: THE FOUNDATION ===\n\n");
    
    // ========================================================
    // SECTION 1: What is a pointer?
    // ========================================================
    printf("--- SECTION 1: WHAT IS A POINTER? ---\n\n");
    
    int x = 42;
    int* p = &x;  // p stores the ADDRESS of x
    
    printf("Variable x:\n");
    printf("  Value:   %d\n", x);
    printf("  Address: %p\n", (void*)&x);
    printf("  Size:    %zu bytes\n\n", sizeof(x));
    
    printf("Pointer p:\n");
    printf("  Value (address it holds): %p\n", (void*)p);
    printf("  Its own address:          %p\n", (void*)&p);
    printf("  Size:                     %zu bytes\n", sizeof(p));
    printf("  Dereferenced (*p):        %d\n\n", *p);
    
    // ========================================================
    // SECTION 2: Dereferencing
    // ========================================================
    printf("--- SECTION 2: DEREFERENCING ---\n\n");
    
    printf("Before: x = %d\n", x);
    
    *p = 100;  // Write through pointer
    
    printf("After *p = 100: x = %d\n", x);
    printf("We modified x through the pointer!\n\n");
    
    // ========================================================
    // SECTION 3: Pointer types matter
    // ========================================================
    printf("--- SECTION 3: POINTER TYPES ---\n\n");
    
    printf("Different pointer types:\n");
    printf("  sizeof(char*):   %zu bytes\n", sizeof(char*));
    printf("  sizeof(int*):    %zu bytes\n", sizeof(int*));
    printf("  sizeof(double*): %zu bytes\n", sizeof(double*));
    printf("  sizeof(void*):   %zu bytes\n\n", sizeof(void*));
    
    printf("All pointers are same size (address size).\n");
    printf("But the TYPE tells the compiler:\n");
    printf("  - How many bytes to read/write on dereference\n");
    printf("  - How much to add for pointer arithmetic\n\n");
    
    // Demonstrate
    int arr[4] = {10, 20, 30, 40};
    int* ip = arr;
    char* cp = (char*)arr;
    
    printf("Array: [10, 20, 30, 40]\n");
    printf("int* ip = arr:\n");
    printf("  ip[0] = %d, ip[1] = %d\n", ip[0], ip[1]);
    printf("char* cp = (char*)arr:\n");
    printf("  cp[0] = %d, cp[4] = %d (first byte of each int)\n", 
           (unsigned char)cp[0], (unsigned char)cp[4]);
    
    // ========================================================
    // SECTION 4: NULL pointer
    // ========================================================
    printf("\n--- SECTION 4: NULL POINTER ---\n\n");
    
    int* null_ptr = NULL;
    
    printf("NULL = %p\n", (void*)null_ptr);
    printf("NULL means 'points to nothing'\n");
    printf("Always check before dereferencing!\n\n");
    
    if (null_ptr != NULL) {
        printf("This won't print\n");
    } else {
        printf("null_ptr is NULL - don't dereference!\n");
    }
    
    // ========================================================
    // SECTION 5: Why this matters for ML
    // ========================================================
    printf("\n--- SECTION 5: ML IMPLICATIONS ---\n\n");
    
    printf("1. TENSORS:\n");
    printf("   PyTorch Tensor.data_ptr() returns a raw pointer\n");
    printf("   This is how C extensions access tensor data\n\n");
    
    printf("2. MEMORY MAPPING:\n");
    printf("   mmap() returns a pointer to file contents\n");
    printf("   FFCV accesses data through these pointers\n\n");
    
    printf("3. CUDA:\n");
    printf("   cudaMalloc gives you a DEVICE pointer\n");
    printf("   Cannot dereference on CPU!\n\n");
    
    printf("4. ZERO-COPY:\n");
    printf("   Pass pointers, not data\n");
    printf("   Function modifies original, no copy needed\n");
    
    return 0;
}
