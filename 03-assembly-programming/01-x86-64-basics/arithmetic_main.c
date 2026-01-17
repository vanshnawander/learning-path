/**
 * arithmetic_main.c - C wrapper for arithmetic demo
 *
 * Compile: gcc -o arithmetic 07_arithmetic.s arithmetic_main.c
 * Run: ./arithmetic
 */

#include <stdio.h>

extern void arithmetic_demo(void);
extern void integer_arithmetic(void);
extern void shifts_and_rotates(void);
extern void bitwise_operations(void);
extern void extended_precision(void);

int main() {
    printf("=== ARITHMETIC OPERATIONS DEMONSTRATION ===\n\n");

    printf("1. Basic arithmetic (ADD, SUB, INC, DEC, NEG)...\n");
    arithmetic_demo();
    printf("   Done.\n\n");

    printf("2. Multiplication and division (IMUL, MUL, IDIV, DIV)...\n");
    integer_arithmetic();
    printf("   Done.\n\n");

    printf("3. Shifts and rotates (SHL, SHR, SAR, ROL, ROR)...\n");
    shifts_and_rotates();
    printf("   Done.\n\n");

    printf("4. Bitwise operations (AND, OR, XOR, NOT, BT*)...\n");
    bitwise_operations();
    printf("   Done.\n\n");

    printf("5. Extended precision (128-bit arithmetic)...\n");
    extended_precision();
    printf("   Done.\n\n");

    printf("=== KEY CONCEPTS ===\n");
    printf("- ADD/SUB set flags: ZF, SF, CF, OF\n");
    printf("- IMUL: RDX:RAX = RAX * src | DIV: RDX:RAX / src\n");
    printf("- SHL/SAL: logical left | SHR: logical right | SAR: arithmetic right\n");
    printf("- XOR reg,reg is faster than MOV $0,reg for zeroing\n");
    printf("- Use ADC/SBB for multi-precision arithmetic\n");
    printf("- Use CMOV for branch-free min/max/clamp\n");

    return 0;
}
