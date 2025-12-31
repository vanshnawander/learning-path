/**
 * 01_latency_numbers.c - Measure Real System Latencies
 * 
 * These numbers are CRITICAL for understanding ML performance.
 * "Latency numbers every programmer should know" - Jeff Dean
 * 
 * Compile: gcc -O2 -o latency 01_latency_numbers.c -lrt
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define ITERATIONS 10000000
#define CACHE_LINE 64

// High-precision timing
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

double get_cpu_freq_ghz() {
    // Rough estimate - measure 1 second
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    uint64_t tsc_start = rdtsc();
    
    // Busy wait ~100ms
    volatile long sum = 0;
    for (long i = 0; i < 100000000; i++) sum += i;
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t tsc_end = rdtsc();
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
    return (tsc_end - tsc_start) / elapsed / 1e9;
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           LATENCY NUMBERS FOR ML ENGINEERS                     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    double freq = get_cpu_freq_ghz();
    printf("Estimated CPU frequency: %.2f GHz\n", freq);
    printf("1 cycle ≈ %.2f ns\n\n", 1.0 / freq);
    
    // ================================================================
    // 1. Function call overhead
    // ================================================================
    printf("=== FUNCTION CALL OVERHEAD ===\n");
    
    volatile int x = 0;
    uint64_t start = rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        x = x + 1;  // Just increment
    }
    uint64_t end = rdtsc();
    double cycles = (double)(end - start) / ITERATIONS;
    printf("Simple increment: %.1f cycles (%.1f ns)\n", cycles, cycles / freq);
    
    // ================================================================
    // 2. L1 Cache access
    // ================================================================
    printf("\n=== CACHE LATENCIES ===\n");
    
    // Small array that fits in L1
    int* l1_array = aligned_alloc(CACHE_LINE, 32 * 1024);  // 32 KB
    for (int i = 0; i < 32 * 1024 / sizeof(int); i++) {
        l1_array[i] = i + 1;
    }
    
    // Pointer chasing through L1
    volatile int sum = 0;
    start = rdtsc();
    for (int i = 0; i < ITERATIONS; i++) {
        sum += l1_array[i % (32 * 1024 / sizeof(int))];
    }
    end = rdtsc();
    cycles = (double)(end - start) / ITERATIONS;
    printf("L1 cache hit: ~%.1f cycles (~%.1f ns)\n", cycles, cycles / freq);
    
    // ================================================================
    // 3. L3/Main Memory (larger array)
    // ================================================================
    size_t large_size = 256 * 1024 * 1024;  // 256 MB
    int* large_array = aligned_alloc(CACHE_LINE, large_size);
    
    // Initialize with pointer chain (random access pattern)
    size_t n = large_size / sizeof(int);
    for (size_t i = 0; i < n; i++) {
        large_array[i] = (i * 16807) % n;  // Pseudo-random
    }
    
    // Random access (likely cache misses)
    volatile size_t idx = 0;
    start = rdtsc();
    for (int i = 0; i < 100000; i++) {  // Fewer iterations for slow access
        idx = large_array[idx];
    }
    end = rdtsc();
    cycles = (double)(end - start) / 100000;
    printf("Random DRAM access: ~%.0f cycles (~%.0f ns)\n", cycles, cycles / freq);
    
    free(l1_array);
    free(large_array);
    
    // ================================================================
    // 4. System call overhead
    // ================================================================
    printf("\n=== SYSTEM CALL OVERHEAD ===\n");
    
    start = rdtsc();
    for (int i = 0; i < 100000; i++) {
        getpid();  // Simple syscall
    }
    end = rdtsc();
    cycles = (double)(end - start) / 100000;
    printf("getpid() syscall: ~%.0f cycles (~%.0f ns)\n", cycles, cycles / freq);
    
    // ================================================================
    // 5. File I/O (cached)
    // ================================================================
    printf("\n=== FILE I/O ===\n");
    
    int fd = open("/tmp/latency_test", O_RDWR | O_CREAT, 0644);
    char buf[4096] = {0};
    write(fd, buf, 4096);
    fsync(fd);
    
    // Cached read
    lseek(fd, 0, SEEK_SET);
    read(fd, buf, 4096);  // Warm cache
    
    start = rdtsc();
    for (int i = 0; i < 10000; i++) {
        lseek(fd, 0, SEEK_SET);
        read(fd, buf, 4096);
    }
    end = rdtsc();
    cycles = (double)(end - start) / 10000;
    printf("Cached 4KB read: ~%.0f cycles (~%.0f µs)\n", 
           cycles, cycles / freq / 1000);
    
    close(fd);
    unlink("/tmp/latency_test");
    
    // ================================================================
    // Reference numbers
    // ================================================================
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║           REFERENCE LATENCY NUMBERS (2024)                      ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║ L1 cache reference                          ~1 ns              ║\n");
    printf("║ L2 cache reference                          ~4 ns              ║\n");
    printf("║ L3 cache reference                          ~10-20 ns          ║\n");
    printf("║ Main memory (DRAM)                          ~100 ns            ║\n");
    printf("║ NVMe SSD random read                        ~10-100 µs         ║\n");
    printf("║ NVMe SSD sequential read (1MB)              ~100-200 µs        ║\n");
    printf("║ HDD random read                             ~10 ms             ║\n");
    printf("║ PCIe round-trip                             ~1-5 µs            ║\n");
    printf("║ GPU kernel launch                           ~5-10 µs           ║\n");
    printf("║ cudaMemcpy (1 MB)                           ~50-100 µs         ║\n");
    printf("║ Network round-trip (same datacenter)        ~500 µs            ║\n");
    printf("║ Network round-trip (cross-region)           ~50-150 ms         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n=== ML IMPLICATIONS ===\n\n");
    printf("1. GPU kernel launch overhead (~5µs) makes tiny kernels inefficient\n");
    printf("2. PCIe transfer (~1µs + data) means keep data on GPU!\n");
    printf("3. DRAM latency (~100ns) means cache efficiency matters\n");
    printf("4. SSD latency (~10µs) is 100x slower than DRAM\n");
    printf("5. Network latency dominates distributed training\n");
    
    return 0;
}
