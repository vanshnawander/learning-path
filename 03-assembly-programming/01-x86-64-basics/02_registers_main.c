/**
 * 02_registers_main.c - C wrapper for assembly demo
 * 
 * Compile: gcc -o registers 02_registers.s 02_registers_main.c
 * Run: ./registers
 */

#include <stdio.h>

// Declare assembly functions
extern void register_demo(void);
extern long add_numbers(long a, long b, long c, long d, long e, long f);

int main() {
    printf("=== REGISTER DEMONSTRATION ===\n\n");
    
    // Call assembly function
    printf("Calling register_demo()...\n");
    register_demo();
    printf("Done.\n\n");
    
    // Test add_numbers
    printf("Testing add_numbers(1, 2, 3, 4, 5, 6)...\n");
    long result = add_numbers(1, 2, 3, 4, 5, 6);
    printf("Result: %ld (expected 21)\n\n", result);
    
    printf("=== CALLING CONVENTION ===\n");
    printf("Arguments passed in: RDI, RSI, RDX, RCX, R8, R9\n");
    printf("Return value in: RAX\n");
    printf("Callee must preserve: RBX, RBP, R12-R15\n");
    
    return 0;
}
