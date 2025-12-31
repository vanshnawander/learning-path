/**
 * 01_syscall_overhead.c - Measuring System Call Overhead
 * 
 * System calls have overhead (user→kernel→user transition).
 * This is why:
 * - mmap beats many read() calls
 * - Batching operations is important
 * - CUDA kernels batch work
 * 
 * Compile: gcc -O2 -o 01_syscall_overhead 01_syscall_overhead.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

#define ITERATIONS 1000000

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== SYSTEM CALL OVERHEAD ===\n\n");
    
    double start, elapsed;
    
    // Measure getpid() - minimal syscall
    printf("--- MINIMAL SYSCALL: getpid() ---\n");
    start = get_time();
    for (int i = 0; i < ITERATIONS; i++) {
        getpid();
    }
    elapsed = get_time() - start;
    
    printf("%d calls in %.2f ms\n", ITERATIONS, elapsed * 1000);
    printf("Per call: %.0f ns\n", (elapsed / ITERATIONS) * 1e9);
    
    // Measure clock_gettime() - fast vsyscall
    printf("\n--- VSYSCALL: clock_gettime() ---\n");
    struct timespec ts;
    start = get_time();
    for (int i = 0; i < ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    elapsed = get_time() - start;
    
    printf("%d calls in %.2f ms\n", ITERATIONS, elapsed * 1000);
    printf("Per call: %.0f ns (faster - uses vDSO)\n", (elapsed / ITERATIONS) * 1e9);
    
    // Measure write to /dev/null
    printf("\n--- WRITE SYSCALL: write() ---\n");
    int fd = open("/dev/null", 1);
    char buf[1] = {'x'};
    
    start = get_time();
    for (int i = 0; i < ITERATIONS; i++) {
        write(fd, buf, 1);
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("%d calls in %.2f ms\n", ITERATIONS, elapsed * 1000);
    printf("Per call: %.0f ns\n", (elapsed / ITERATIONS) * 1e9);
    
    // Compare: function call overhead
    printf("\n--- BASELINE: FUNCTION CALL ---\n");
    volatile int x = 0;
    start = get_time();
    for (int i = 0; i < ITERATIONS; i++) {
        x++;  // Just a memory operation
    }
    elapsed = get_time() - start;
    
    printf("%d operations in %.2f ms\n", ITERATIONS, elapsed * 1000);
    printf("Per operation: %.1f ns\n", (elapsed / ITERATIONS) * 1e9);
    
    printf("\n=== IMPLICATIONS ===\n");
    printf("Syscall overhead: ~100-500 ns\n");
    printf("Function call: ~1 ns\n");
    printf("Ratio: 100-500x!\n\n");
    printf("This is why:\n");
    printf("1. mmap + pointer access beats many read() calls\n");
    printf("2. CUDA batches work into kernel launches\n");
    printf("3. Vectorized operations beat element-wise\n");
    printf("4. DataLoader prefetches to hide syscall latency\n");
    
    return 0;
}
