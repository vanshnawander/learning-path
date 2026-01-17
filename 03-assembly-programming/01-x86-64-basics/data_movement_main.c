/**
 * data_movement_main.c - C wrapper for data movement demo
 *
 * Compile: gcc -o data_movement 03_data_movement.s data_movement_main.c
 * Run: ./data_movement
 */

#include <stdio.h>

extern void data_movement_demo(void);
extern void memory_copy(void);
extern void string_operations(void);
extern void extended_moves(void);

int main() {
    printf("=== DATA MOVEMENT DEMONSTRATION ===\n\n");

    printf("1. Basic MOV instructions...\n");
    data_movement_demo();
    printf("   Done.\n\n");

    printf("2. All addressing modes...\n");
    memory_copy();
    printf("   Done.\n\n");

    printf("3. String operations (MOVS, STOS, LODS, CMPS)...\n");
    string_operations();
    printf("   Done.\n\n");

    printf("4. Conditional moves (CMOV)...\n");
    extended_moves();
    printf("   Done.\n\n");

    printf("=== KEY CONCEPTS ===\n");
    printf("- MOV copies data; LEA calculates addresses (no dereference)\n");
    printf("- 32-bit MOV zero-extends to 64 bits automatically\n");
    printf("- Use MOVS*/STOS*/LODS* for string operations\n");
    printf("- CMOV moves only if condition flags are set\n");
    printf("- Addressing: disp(base, index, scale) = disp + base + index*scale\n");

    return 0;
}
