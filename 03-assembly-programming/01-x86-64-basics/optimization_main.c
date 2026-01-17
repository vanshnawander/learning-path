/**
 * optimization_main.c - C wrapper for optimization demo
 *
 * Compile: gcc -o optimization 06_optimization.s optimization_main.c
 * Run: ./optimization
 */

#include <stdio.h>

extern void latency_vs_throughput(void);
extern void loop_optimization(void);
extern void instruction_selection(void);

int main() {
    printf("=== ASSEMBLY OPTIMIZATION DEMONSTRATION ===\n\n");

    printf("1. Latency vs Throughput...\n");
    latency_vs_throughput();
    printf("   Done.\n\n");

    printf("2. Loop optimization (unrolling, pipelining)...\n");
    loop_optimization();
    printf("   Done.\n\n");

    printf("3. Instruction selection (LEA, CMOV, SET)...\n");
    instruction_selection();
    printf("   Done.\n\n");

    printf("=== KEY OPTIMIZATION PRINCIPLES ===\n");
    printf("- Dependency chains limit parallelism (break them!)\n");
    printf("- Loop unrolling reduces overhead, increases ILP\n");
    printf("- LEA is faster than multiple ADD for complex math\n");
    printf("- CMOV avoids branch misprediction penalties\n");
    printf("- Sequential memory access = cache hits\n");
    printf("- Prefetch ahead for large working sets\n");
    printf("- Division is slow: use shifts/multiplication\n");

    return 0;
}
