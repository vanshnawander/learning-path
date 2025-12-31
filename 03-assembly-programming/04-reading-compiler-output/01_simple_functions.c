/**
 * 01_simple_functions.c - Understanding Compiler Output
 * 
 * Compile with different options and examine:
 *   gcc -S -O0 01_simple_functions.c -o func_O0.s
 *   gcc -S -O2 01_simple_functions.c -o func_O2.s
 *   gcc -S -O3 -mavx2 01_simple_functions.c -o func_O3.s
 * 
 * Or use Godbolt: https://godbolt.org/
 */

#include <stdint.h>

// ============================================================
// Simple arithmetic - see register usage
// ============================================================

int add(int a, int b) {
    return a + b;
}

/*
Expected O2 output (Intel syntax):
    add:
        lea     eax, [rdi+rsi]    ; eax = rdi + rsi (args in rdi, rsi)
        ret
        
Or AT&T:
        leal    (%rdi,%rsi), %eax
        ret
*/

// ============================================================
// Loop - see loop structure
// ============================================================

int sum_array(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

/*
O0: Naive loop with stack variables
O2: Optimized loop, maybe unrolled
O3 + AVX: Vectorized with vpaddd
*/

// ============================================================
// Conditionals - see branching
// ============================================================

int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

/*
Expected O2 output:
    cmp     edi, esi
    mov     eax, esi
    cmovge  eax, edi      ; Conditional move, no branch!
    ret
*/

// ============================================================
// Memory access - see addressing modes
// ============================================================

void copy_array(float* dst, float* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

/*
O2: Simple load/store loop
O3 + AVX: vmovups for 8 floats at a time
*/

// ============================================================
// Floating point - see SSE/AVX usage
// ============================================================

float dot_product(float* a, float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/*
O3 + AVX:
    vxorps      xmm0, xmm0, xmm0    ; sum = 0
.L5:
    vmovups     ymm1, [rdi+rax]     ; Load 8 floats from a
    vfmadd231ps ymm0, ymm1, [rsi+rax]  ; sum += a * b
    add         rax, 32
    cmp         rax, rcx
    jb          .L5
    ; ... horizontal sum ...
*/

// ============================================================
// Inlining - functions may disappear
// ============================================================

static inline int square(int x) {
    return x * x;
}

int sum_of_squares(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += square(arr[i]);
    }
    return sum;
}

/*
At O2+, square() is inlined - no call instruction
The multiply appears directly in the loop
*/

// ============================================================
// Struct access - see offsets
// ============================================================

typedef struct {
    int x;
    int y;
    int z;
} Point;

int get_z(Point* p) {
    return p->z;
}

/*
    mov     eax, [rdi+8]    ; z is at offset 8 (after x and y)
    ret
*/

// Prevent optimizer from removing unused functions
void* funcs[] = {
    (void*)add, (void*)sum_array, (void*)max,
    (void*)copy_array, (void*)dot_product,
    (void*)sum_of_squares, (void*)get_z
};
