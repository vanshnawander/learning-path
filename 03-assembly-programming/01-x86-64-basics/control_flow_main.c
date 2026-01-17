/**
 * control_flow_main.c - C wrapper for control flow demo
 *
 * Compile: gcc -o control_flow 04_control_flow.s control_flow_main.c
 * Run: ./control_flow
 */

#include <stdio.h>

extern void control_flow_demo(void);
extern void compare_and_jump(void);
extern void loop_patterns(void);
extern void jump_tables(void);

int main() {
    printf("=== CONTROL FLOW DEMONSTRATION ===\n\n");

    printf("1. Flags and comparisons (CMP, TEST)...\n");
    control_flow_demo();
    printf("   Done.\n\n");

    printf("2. Conditional jumps (signed vs unsigned)...\n");
    compare_and_jump();
    printf("   Done.\n\n");

    printf("3. Loop patterns (while, for, do-while)...\n");
    loop_patterns();
    printf("   Done.\n\n");

    printf("4. Jump tables (switch statements)...\n");
    jump_tables();
    printf("   Done.\n\n");

    printf("=== KEY CONCEPTS ===\n");
    printf("- CMP subtracts and sets flags (doesn't store result)\n");
    printf("- TEST does AND and sets flags (doesn't store result)\n");
    printf("- Signed: jl/jle/jg/jge | Unsigned: jb/jbe/ja/jae\n");
    printf("- LOOP decrements RCX and jumps if RCX != 0\n");
    printf("- Jump tables: lea jt(%%rip),%%rax; jmp *(%%rax,%%rdi,8)\n");

    return 0;
}
