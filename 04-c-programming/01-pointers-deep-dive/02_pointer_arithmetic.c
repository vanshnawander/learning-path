/**
 * 02_pointer_arithmetic.c - Navigating Memory with Pointers
 * 
 * Pointer arithmetic is how you traverse arrays, buffers, tensors.
 * This is fundamental to understanding memory access patterns.
 * 
 * Compile: gcc -o pointer_arith 02_pointer_arithmetic.c
 */

#include <stdio.h>
#include <stdint.h>

int main() {
    printf("=== POINTER ARITHMETIC ===\n\n");
    
    // ========================================================
    // SECTION 1: Basic arithmetic
    // ========================================================
    printf("--- SECTION 1: ADDING TO POINTERS ---\n\n");
    
    int arr[5] = {10, 20, 30, 40, 50};
    int* p = arr;
    
    printf("Array at %p: [10, 20, 30, 40, 50]\n\n", (void*)arr);
    
    printf("Pointer arithmetic (int* adds 4 bytes per increment):\n");
    for (int i = 0; i < 5; i++) {
        printf("  p + %d = %p, *(p + %d) = %d\n", 
               i, (void*)(p + i), i, *(p + i));
    }
    
    printf("\nKEY INSIGHT: p + 1 adds sizeof(*p) bytes, not 1 byte!\n");
    printf("  int* p:    p + 1 adds %zu bytes\n", sizeof(int));
    printf("  double* p: p + 1 adds %zu bytes\n", sizeof(double));
    printf("  char* p:   p + 1 adds %zu byte\n\n", sizeof(char));
    
    // ========================================================
    // SECTION 2: Array indexing IS pointer arithmetic
    // ========================================================
    printf("--- SECTION 2: arr[i] == *(arr + i) ---\n\n");
    
    printf("These are IDENTICAL:\n");
    printf("  arr[2]      = %d\n", arr[2]);
    printf("  *(arr + 2)  = %d\n", *(arr + 2));
    printf("  *(p + 2)    = %d\n", *(p + 2));
    printf("  p[2]        = %d\n", p[2]);
    printf("  2[arr]      = %d (yes, this works!)\n\n", 2[arr]);
    
    printf("arr[i] is just syntactic sugar for *(arr + i)\n\n");
    
    // ========================================================
    // SECTION 3: Stride access (critical for tensors!)
    // ========================================================
    printf("--- SECTION 3: STRIDED ACCESS ---\n\n");
    
    // Simulate a 3x4 row-major matrix
    int matrix[12] = {
        1, 2, 3, 4,     // row 0
        5, 6, 7, 8,     // row 1
        9, 10, 11, 12   // row 2
    };
    
    int rows = 3, cols = 4;
    int* m = matrix;
    
    printf("3x4 matrix (row-major in memory):\n");
    for (int r = 0; r < rows; r++) {
        printf("  ");
        for (int c = 0; c < cols; c++) {
            // m[r * cols + c] == *(m + r * cols + c)
            printf("%2d ", m[r * cols + c]);
        }
        printf("\n");
    }
    
    printf("\nRow stride = %d elements = %zu bytes\n", cols, cols * sizeof(int));
    printf("To access element [r][c]: *(m + r * stride + c)\n\n");
    
    printf("Column-major access (INEFFICIENT for row-major layout):\n");
    printf("  Accessing column 0: ");
    for (int r = 0; r < rows; r++) {
        printf("%d ", m[r * cols + 0]);  // Stride of 4 between accesses
    }
    printf("(stride = %d, cache unfriendly!)\n\n", cols);
    
    // ========================================================
    // SECTION 4: Pointer difference
    // ========================================================
    printf("--- SECTION 4: POINTER SUBTRACTION ---\n\n");
    
    int* start = &arr[0];
    int* end = &arr[4];
    
    ptrdiff_t diff = end - start;
    
    printf("start = %p\n", (void*)start);
    printf("end   = %p\n", (void*)end);
    printf("end - start = %td elements (not bytes!)\n", diff);
    printf("Byte difference = %td\n\n", (char*)end - (char*)start);
    
    // ========================================================
    // SECTION 5: Iterating with pointers
    // ========================================================
    printf("--- SECTION 5: POINTER ITERATION ---\n\n");
    
    printf("Index-based loop:\n  ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    printf("Pointer-based loop:\n  ");
    for (int* ptr = arr; ptr < arr + 5; ptr++) {
        printf("%d ", *ptr);
    }
    printf("\n\n");
    
    printf("Pointer iteration can be faster (no multiply for index)\n");
    printf("Compiler often optimizes index loops to pointer form\n");
    
    // ========================================================
    // SECTION 6: ML Applications
    // ========================================================
    printf("\n--- SECTION 6: ML APPLICATIONS ---\n\n");
    
    printf("1. TENSOR STRIDES:\n");
    printf("   PyTorch tensors store strides for each dimension\n");
    printf("   data[i,j,k] = base + i*stride[0] + j*stride[1] + k*stride[2]\n\n");
    
    printf("2. BATCH PROCESSING:\n");
    printf("   float* batch = data + batch_idx * batch_stride\n\n");
    
    printf("3. CHANNEL ACCESS:\n");
    printf("   NCHW: channel c at (n,c,h,w) = base + n*N + c*C + h*H + w\n");
    printf("   Different layouts have different strides!\n\n");
    
    printf("4. WEIGHT MATRICES:\n");
    printf("   Accessing row r of weight matrix: weights + r * hidden_dim\n");
    
    return 0;
}
