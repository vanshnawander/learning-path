/*
 * Memory Bandwidth Measurement Tool
 * 
 * This program measures actual memory bandwidth for different access patterns
 * and compares to theoretical peak bandwidth.
 * 
 * Compile: gcc -O3 -march=native -fopenmp 02_memory_bandwidth_measurement.c -o membw
 * Run: ./membw
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <sys/mman.h>
#include <x86intrin.h>
#endif

#define KB (1024ULL)
#define MB (1024ULL * KB)
#define GB (1024ULL * MB)

#define ARRAY_SIZE (512 * MB)
#define ITERATIONS 10

// Ensure alignment to avoid cache line splits
#define CACHE_LINE_SIZE 64

typedef struct {
    double bandwidth_gbps;
    double latency_ns;
    uint64_t cycles;
} benchmark_result_t;

// High-resolution timer
static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Allocate aligned memory with huge pages if available
void* allocate_aligned(size_t size) {
    void* ptr = NULL;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, CACHE_LINE_SIZE);
#else
    // Try huge pages first (2MB pages)
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    
    if (ptr == MAP_FAILED) {
        // Fall back to regular aligned allocation
        if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) != 0) {
            return NULL;
        }
        printf("Note: Using regular pages (huge pages unavailable)\n");
    } else {
        printf("Note: Using 2MB huge pages\n");
    }
#endif
    
    return ptr;
}

void free_aligned(void* ptr, size_t size) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    munmap(ptr, size);
#endif
}

/*
 * STREAM-like benchmarks
 * These measure sustainable memory bandwidth
 */

// Read bandwidth: a[i] -> sum
benchmark_result_t benchmark_read(double* __restrict a, size_t n) {
    benchmark_result_t result = {0};
    volatile double sum = 0;
    
    // Warm up
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += a[i];
    }
    
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < n; i++) {
            sum += a[i];
        }
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    size_t bytes = n * sizeof(double) * ITERATIONS;
    
    result.bandwidth_gbps = (bytes / elapsed) / GB;
    result.cycles = (end_cycles - start_cycles) / ITERATIONS;
    
    // Prevent optimization
    if (sum == 0.12345) printf("x");
    
    return result;
}

// Write bandwidth: scalar -> a[i]
benchmark_result_t benchmark_write(double* __restrict a, size_t n) {
    benchmark_result_t result = {0};
    double scalar = 3.14159;
    
    // Warm up
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = scalar;
    }
    
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = scalar;
        }
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    size_t bytes = n * sizeof(double) * ITERATIONS;
    
    result.bandwidth_gbps = (bytes / elapsed) / GB;
    result.cycles = (end_cycles - start_cycles) / ITERATIONS;
    
    return result;
}

// Copy bandwidth: a[i] -> b[i]
benchmark_result_t benchmark_copy(double* __restrict a, double* __restrict b, size_t n) {
    benchmark_result_t result = {0};
    
    // Warm up
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        b[i] = a[i];
    }
    
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            b[i] = a[i];
        }
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    // Count both read and write
    size_t bytes = 2 * n * sizeof(double) * ITERATIONS;
    
    result.bandwidth_gbps = (bytes / elapsed) / GB;
    result.cycles = (end_cycles - start_cycles) / ITERATIONS;
    
    return result;
}

// Triad bandwidth: a[i] = b[i] + scalar * c[i]
benchmark_result_t benchmark_triad(double* __restrict a, double* __restrict b, 
                                    double* __restrict c, size_t n) {
    benchmark_result_t result = {0};
    double scalar = 3.0;
    
    // Warm up
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + scalar * c[i];
    }
    
    double start = get_time_sec();
    uint64_t start_cycles = rdtsc();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
    
    uint64_t end_cycles = rdtsc();
    double end = get_time_sec();
    
    double elapsed = end - start;
    // 2 reads + 1 write
    size_t bytes = 3 * n * sizeof(double) * ITERATIONS;
    
    result.bandwidth_gbps = (bytes / elapsed) / GB;
    result.cycles = (end_cycles - start_cycles) / ITERATIONS;
    
    return result;
}

/*
 * Latency measurement using pointer chasing
 * This measures true memory latency without prefetching
 */
benchmark_result_t benchmark_latency(void** chain, size_t chain_length) {
    benchmark_result_t result = {0};
    
    void** p = chain;
    size_t count = chain_length * ITERATIONS;
    
    // Pointer chasing - cannot be prefetched or parallelized
    uint64_t start_cycles = rdtsc();
    
    for (size_t i = 0; i < count; i++) {
        p = (void**)*p;
    }
    
    uint64_t end_cycles = rdtsc();
    
    // Prevent optimization
    if (p == (void**)0x12345) printf("x");
    
    result.cycles = end_cycles - start_cycles;
    // Assuming ~3GHz CPU for ns conversion (adjust for your system)
    result.latency_ns = (double)(end_cycles - start_cycles) / count / 3.0;
    
    return result;
}

// Create a random pointer chain for latency measurement
void** create_pointer_chain(size_t size_bytes) {
    size_t num_pointers = size_bytes / sizeof(void*);
    void** array = (void**)allocate_aligned(size_bytes);
    
    if (!array) return NULL;
    
    // Create random permutation for pointer chasing
    size_t* indices = (size_t*)malloc(num_pointers * sizeof(size_t));
    for (size_t i = 0; i < num_pointers; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (size_t i = num_pointers - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
    
    // Create chain following shuffled indices
    for (size_t i = 0; i < num_pointers - 1; i++) {
        array[indices[i]] = &array[indices[i + 1]];
    }
    array[indices[num_pointers - 1]] = &array[indices[0]]; // Close the loop
    
    free(indices);
    return array;
}

void print_system_info(void) {
    printf("=== System Information ===\n");
    
    #pragma omp parallel
    {
        #pragma omp single
        printf("OpenMP threads: %d\n", omp_get_num_threads());
    }
    
    printf("Array size: %llu MB\n", (unsigned long long)(ARRAY_SIZE / MB));
    printf("Iterations: %d\n", ITERATIONS);
    
#ifndef _WIN32
    // Try to read memory info on Linux
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemTotal:", 9) == 0 ||
                strncmp(line, "HugePages_Total:", 16) == 0) {
                printf("%s", line);
            }
        }
        fclose(meminfo);
    }
#endif
    
    printf("\n");
}

int main(int argc, char** argv) {
    print_system_info();
    
    size_t n = ARRAY_SIZE / sizeof(double);
    
    printf("Allocating arrays...\n");
    double* a = (double*)allocate_aligned(ARRAY_SIZE);
    double* b = (double*)allocate_aligned(ARRAY_SIZE);
    double* c = (double*)allocate_aligned(ARRAY_SIZE);
    
    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize arrays
    printf("Initializing arrays...\n");
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }
    
    printf("\n=== Bandwidth Benchmarks (STREAM-like) ===\n");
    printf("%-15s %15s %15s\n", "Benchmark", "Bandwidth (GB/s)", "Cycles");
    printf("%-15s %15s %15s\n", "---------", "----------------", "------");
    
    benchmark_result_t result;
    
    result = benchmark_read(a, n);
    printf("%-15s %15.2f %15llu\n", "Read", result.bandwidth_gbps, 
           (unsigned long long)result.cycles);
    
    result = benchmark_write(a, n);
    printf("%-15s %15.2f %15llu\n", "Write", result.bandwidth_gbps,
           (unsigned long long)result.cycles);
    
    result = benchmark_copy(a, b, n);
    printf("%-15s %15.2f %15llu\n", "Copy", result.bandwidth_gbps,
           (unsigned long long)result.cycles);
    
    result = benchmark_triad(a, b, c, n);
    printf("%-15s %15.2f %15llu\n", "Triad", result.bandwidth_gbps,
           (unsigned long long)result.cycles);
    
    printf("\n=== Latency Benchmarks (Pointer Chasing) ===\n");
    printf("%-20s %15s\n", "Working Set Size", "Latency (ns)");
    printf("%-20s %15s\n", "----------------", "------------");
    
    // Test different working set sizes to see cache hierarchy
    size_t sizes[] = {
        32 * KB,    // L1 cache
        256 * KB,   // L2 cache
        8 * MB,     // L3 cache
        64 * MB,    // Main memory
        256 * MB    // Main memory (larger)
    };
    
    for (int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        void** chain = create_pointer_chain(sizes[i]);
        if (chain) {
            size_t chain_length = sizes[i] / sizeof(void*);
            result = benchmark_latency(chain, chain_length);
            
            char size_str[32];
            if (sizes[i] >= MB) {
                snprintf(size_str, sizeof(size_str), "%llu MB", 
                         (unsigned long long)(sizes[i] / MB));
            } else {
                snprintf(size_str, sizeof(size_str), "%llu KB",
                         (unsigned long long)(sizes[i] / KB));
            }
            
            printf("%-20s %15.2f\n", size_str, result.latency_ns);
            free_aligned(chain, sizes[i]);
        }
    }
    
    // Cleanup
    free_aligned(a, ARRAY_SIZE);
    free_aligned(b, ARRAY_SIZE);
    free_aligned(c, ARRAY_SIZE);
    
    printf("\n=== Interpretation Guide ===\n");
    printf("DDR4-3200 dual-channel theoretical peak: ~51.2 GB/s\n");
    printf("DDR5-6400 dual-channel theoretical peak: ~102.4 GB/s\n");
    printf("Typical achievable: 70-90%% of peak for streaming\n");
    printf("\nLatency expectations:\n");
    printf("  L1 cache: ~1 ns (4 cycles)\n");
    printf("  L2 cache: ~4 ns (12 cycles)\n");
    printf("  L3 cache: ~12 ns (40 cycles)\n");
    printf("  DRAM:     ~60-100 ns\n");
    
    return 0;
}
