/**
 * functions_main.c - C wrapper for functions demo
 *
 * Compile: gcc -o functions 05_functions.s functions_main.c
 * Run: ./functions
 */

#include <stdio.h>

extern long sysv_calling_convention(long a, long b, long c, long d, long e, long f, long g);
extern long windows_calling_convention(long a, long b, long c, long d);
extern long leaf_function(long* arr, int n);
extern long nested_function(void);
extern long struct_function(void);

int main() {
    printf("=== FUNCTION CALLING CONVENTIONS ===\n\n");

    printf("1. System V AMD64 ABI (Linux/macOS)...\n");
    long result = sysv_calling_convention(1, 2, 3, 4, 5, 6, 7);
    printf("   add7(1,2,3,4,5,6,7) = %ld (expected 28)\n\n", result);

    printf("2. Leaf function (no function calls inside)...\n");
    long arr[] = {10, 20, 30, 40, 50};
    result = leaf_function(arr, 5);
    printf("   sum_array([10,20,30,40,50], 5) = %ld (expected 150)\n\n", result);

    printf("3. Nested function (calls other functions)...\n");
    result = nested_function();
    printf("   nested_function() completed\n\n");

    printf("=== KEY CONCEPTS ===\n");
    printf("- SysV: args in RDI,RSI,RDX,RCX,R8,R9 | Win: RCX,RDX,R8,R9\n");
    printf("- Callee-saved: RBX,RBP,R12-R15 (SysV) | RBX,RBP,RSI,RDI,R12-R15 (Win)\n");
    printf("- Caller-saved: RAX,RCX,RDX,RSI,RDI,R8-R11 (SysV) | RAX,RCX,RDX,R8-R11 (Win)\n");
    printf("- SysV has 128-byte 'red zone' below RSP (leaf functions only)\n");
    printf("- Windows requires 32-byte shadow space\n");
    printf("- Stack must be 16-byte aligned before CALL\n");

    return 0;
}
