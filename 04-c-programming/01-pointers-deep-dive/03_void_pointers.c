/**
 * 03_void_pointers.c - Generic Programming in C
 * 
 * void* is how C does generic programming.
 * This is used extensively in:
 * - malloc/free (returns/takes void*)
 * - qsort, bsearch (generic algorithms)
 * - PyTorch C API (generic tensor data)
 * 
 * Compile: gcc -o void_ptr 03_void_pointers.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// SECTION 1: void* basics
// ============================================================

void print_bytes(void* ptr, size_t n) {
    unsigned char* bytes = (unsigned char*)ptr;
    printf("Bytes: ");
    for (size_t i = 0; i < n; i++) {
        printf("%02X ", bytes[i]);
    }
    printf("\n");
}

// ============================================================
// SECTION 2: Generic swap function
// ============================================================

void generic_swap(void* a, void* b, size_t size) {
    unsigned char temp[size];  // VLA for temporary storage
    memcpy(temp, a, size);
    memcpy(a, b, size);
    memcpy(b, temp, size);
}

// ============================================================
// SECTION 3: Generic array operations (like PyTorch ops)
// ============================================================

typedef enum { DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE } DType;

void print_array(void* arr, int n, DType dtype) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        switch (dtype) {
            case DTYPE_INT:
                printf("%d", ((int*)arr)[i]);
                break;
            case DTYPE_FLOAT:
                printf("%.2f", ((float*)arr)[i]);
                break;
            case DTYPE_DOUBLE:
                printf("%.2f", ((double*)arr)[i]);
                break;
        }
    }
    printf("]\n");
}

void scale_array(void* arr, int n, double factor, DType dtype) {
    for (int i = 0; i < n; i++) {
        switch (dtype) {
            case DTYPE_INT:
                ((int*)arr)[i] *= (int)factor;
                break;
            case DTYPE_FLOAT:
                ((float*)arr)[i] *= (float)factor;
                break;
            case DTYPE_DOUBLE:
                ((double*)arr)[i] *= factor;
                break;
        }
    }
}

// ============================================================
// SECTION 4: Comparison function for qsort
// ============================================================

int compare_ints(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int compare_floats(const void* a, const void* b) {
    float fa = *(float*)a;
    float fb = *(float*)b;
    return (fa > fb) - (fa < fb);
}

int main() {
    printf("=== VOID POINTERS: GENERIC PROGRAMMING ===\n\n");
    
    // ========================================================
    // Basics
    // ========================================================
    printf("--- VOID* BASICS ---\n\n");
    
    int x = 0x12345678;
    float f = 3.14159f;
    
    void* vp;
    
    vp = &x;
    printf("int x = 0x%X\n", x);
    print_bytes(vp, sizeof(int));
    
    vp = &f;
    printf("float f = %f\n", f);
    print_bytes(vp, sizeof(float));
    
    printf("\nvoid* can point to anything.\n");
    printf("BUT you cannot dereference void* directly!\n");
    printf("Must cast to typed pointer first.\n\n");
    
    // ========================================================
    // Generic swap
    // ========================================================
    printf("--- GENERIC SWAP ---\n\n");
    
    int a = 10, b = 20;
    printf("Before: a=%d, b=%d\n", a, b);
    generic_swap(&a, &b, sizeof(int));
    printf("After:  a=%d, b=%d\n\n", a, b);
    
    double d1 = 1.5, d2 = 2.5;
    printf("Before: d1=%.1f, d2=%.1f\n", d1, d2);
    generic_swap(&d1, &d2, sizeof(double));
    printf("After:  d1=%.1f, d2=%.1f\n\n", d1, d2);
    
    // ========================================================
    // Generic array operations
    // ========================================================
    printf("--- GENERIC ARRAY OPS (LIKE PYTORCH) ---\n\n");
    
    float farr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int iarr[] = {10, 20, 30, 40, 50};
    
    printf("Float array: ");
    print_array(farr, 5, DTYPE_FLOAT);
    
    printf("Int array:   ");
    print_array(iarr, 5, DTYPE_INT);
    
    printf("\nScaling float array by 2.0...\n");
    scale_array(farr, 5, 2.0, DTYPE_FLOAT);
    printf("Result: ");
    print_array(farr, 5, DTYPE_FLOAT);
    
    // ========================================================
    // qsort example
    // ========================================================
    printf("\n--- QSORT WITH VOID* ---\n\n");
    
    int unsorted[] = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    int n = sizeof(unsorted) / sizeof(unsorted[0]);
    
    printf("Before: ");
    print_array(unsorted, n, DTYPE_INT);
    
    qsort(unsorted, n, sizeof(int), compare_ints);
    
    printf("After:  ");
    print_array(unsorted, n, DTYPE_INT);
    
    // ========================================================
    // ML Applications
    // ========================================================
    printf("\n--- ML APPLICATIONS ---\n\n");
    
    printf("1. PYTORCH C API:\n");
    printf("   void* data = tensor.data_ptr();\n");
    printf("   Cast based on tensor.dtype()\n\n");
    
    printf("2. NUMPY C API:\n");
    printf("   void* data = PyArray_DATA(arr);\n");
    printf("   Type from PyArray_TYPE(arr)\n\n");
    
    printf("3. CUDA:\n");
    printf("   void* d_ptr;\n");
    printf("   cudaMalloc(&d_ptr, size);\n\n");
    
    printf("4. MEMORY ALLOCATORS:\n");
    printf("   void* malloc(size_t size);\n");
    printf("   Returns generic pointer, cast to needed type\n");
    
    return 0;
}
