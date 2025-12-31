/**
 * 04_data_movement_costs.c - Understanding Data Movement Overhead
 * 
 * THIS IS THE MOST IMPORTANT FILE FOR ML PERFORMANCE
 * 
 * Data movement (not compute) is usually the bottleneck.
 * Every example includes timing to make this obvious.
 * 
 * Compile: gcc -O3 -o data_movement 04_data_movement_costs.c -lrt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define KB (1024)
#define MB (1024 * KB)
#define GB (1024 * MB)

// High-precision timing
typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

void timer_start(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_stop(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    return (t->end.tv_sec - t->start.tv_sec) + 
           (t->end.tv_nsec - t->start.tv_nsec) * 1e-9;
}

// Prevent compiler from optimizing away
volatile uint64_t sink;

// ============================================================
// TEST 1: Memory Copy Bandwidth
// ============================================================
void test_memory_copy() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 1: MEMORY COPY BANDWIDTH                               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t sizes[] = {64*KB, 256*KB, 1*MB, 16*MB, 64*MB, 256*MB};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("%-12s %-12s %-12s %-15s\n", "Size", "Time (ms)", "BW (GB/s)", "Cache Level");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t size = sizes[i];
        char* src = aligned_alloc(64, size);
        char* dst = aligned_alloc(64, size);
        memset(src, 'A', size);
        
        // Warm up
        memcpy(dst, src, size);
        
        Timer t;
        int iterations = (256*MB) / size;  // Scale iterations
        if (iterations < 10) iterations = 10;
        
        timer_start(&t);
        for (int j = 0; j < iterations; j++) {
            memcpy(dst, src, size);
        }
        double elapsed = timer_stop(&t);
        
        double total_bytes = (double)size * iterations * 2;  // Read + Write
        double bandwidth = total_bytes / elapsed / 1e9;
        double time_ms = elapsed / iterations * 1000;
        
        const char* cache_level;
        if (size <= 32*KB) cache_level = "L1 Cache";
        else if (size <= 256*KB) cache_level = "L2 Cache";
        else if (size <= 8*MB) cache_level = "L3 Cache";
        else cache_level = "DRAM";
        
        printf("%-12zu %-12.3f %-12.2f %s\n", 
               size/KB, time_ms, bandwidth, cache_level);
        
        free(src);
        free(dst);
    }
    
    printf("\n");
    printf("📊 KEY INSIGHT: Bandwidth drops 3-10x when data exceeds cache!\n");
    printf("   L1→L2: ~2x slower | L3→DRAM: ~3-5x slower\n");
    printf("   This is why batch size and tiling matter in ML!\n");
}

// ============================================================
// TEST 2: Sequential vs Random Access
// ============================================================
void test_access_patterns() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 2: SEQUENTIAL vs RANDOM ACCESS                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t size = 64 * MB;
    size_t num_elements = size / sizeof(int);
    int* data = aligned_alloc(64, size);
    
    // Initialize
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = i;
    }
    
    Timer t;
    size_t accesses = 10000000;
    
    // Sequential access
    printf("Sequential access (stride 1):\n");
    timer_start(&t);
    uint64_t sum = 0;
    for (size_t i = 0; i < accesses; i++) {
        sum += data[i % num_elements];
    }
    sink = sum;
    double seq_time = timer_stop(&t);
    printf("  Time: %.3f ms for %zu accesses\n", seq_time * 1000, accesses);
    printf("  Rate: %.2f million accesses/sec\n", accesses / seq_time / 1e6);
    
    // Strided access (stride = cache line)
    printf("\nStrided access (stride 16 = 64 bytes = cache line):\n");
    timer_start(&t);
    sum = 0;
    for (size_t i = 0; i < accesses; i++) {
        sum += data[(i * 16) % num_elements];
    }
    sink = sum;
    double stride_time = timer_stop(&t);
    printf("  Time: %.3f ms for %zu accesses\n", stride_time * 1000, accesses);
    printf("  Rate: %.2f million accesses/sec\n", accesses / stride_time / 1e6);
    
    // Random access
    printf("\nRandom access:\n");
    size_t* indices = malloc(accesses * sizeof(size_t));
    for (size_t i = 0; i < accesses; i++) {
        indices[i] = rand() % num_elements;
    }
    
    timer_start(&t);
    sum = 0;
    for (size_t i = 0; i < accesses; i++) {
        sum += data[indices[i]];
    }
    sink = sum;
    double random_time = timer_stop(&t);
    printf("  Time: %.3f ms for %zu accesses\n", random_time * 1000, accesses);
    printf("  Rate: %.2f million accesses/sec\n", accesses / random_time / 1e6);
    
    printf("\n📊 KEY INSIGHT:\n");
    printf("  Sequential vs Random: %.1fx faster\n", random_time / seq_time);
    printf("  This is why contiguous tensors matter in PyTorch!\n");
    printf("  This is why FFCV uses quasi-random sampling!\n");
    
    free(data);
    free(indices);
}

// ============================================================
// TEST 3: Cache Line Effects (False Sharing Preview)
// ============================================================
void test_cache_line_effects() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 3: CACHE LINE EFFECTS                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t size = 1 * MB;
    char* data = aligned_alloc(64, size);
    memset(data, 0, size);
    
    Timer t;
    size_t iterations = 10000000;
    
    // Access same cache line repeatedly
    printf("Accessing SAME cache line repeatedly:\n");
    timer_start(&t);
    for (size_t i = 0; i < iterations; i++) {
        data[i % 64] = i;  // All within 64-byte cache line
    }
    double same_line = timer_stop(&t);
    printf("  Time: %.3f ms\n", same_line * 1000);
    
    // Access different cache lines
    printf("\nAccessing DIFFERENT cache lines (stride 64):\n");
    timer_start(&t);
    for (size_t i = 0; i < iterations; i++) {
        data[(i * 64) % size] = i;  // Different cache line each time
    }
    double diff_line = timer_stop(&t);
    printf("  Time: %.3f ms\n", diff_line * 1000);
    
    printf("\n📊 KEY INSIGHT:\n");
    printf("  Same cache line: %.1fx faster\n", diff_line / same_line);
    printf("  CPU loads 64 bytes at a time (cache line)\n");
    printf("  Group related data together!\n");
    
    free(data);
}

// ============================================================
// TEST 4: Compute vs Memory Time
// ============================================================
void test_compute_vs_memory() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 4: COMPUTE vs MEMORY TIME (Memory Bound Reality)       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t size = 64 * MB;
    size_t num_floats = size / sizeof(float);
    float* data = aligned_alloc(64, size);
    
    // Initialize
    for (size_t i = 0; i < num_floats; i++) {
        data[i] = (float)i;
    }
    
    Timer t;
    
    // Pure memory (just read)
    printf("Pure memory read (load all elements):\n");
    timer_start(&t);
    float sum = 0;
    for (size_t i = 0; i < num_floats; i++) {
        sum += data[i];
    }
    sink = (uint64_t)sum;
    double read_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", read_time * 1000);
    printf("  Bandwidth: %.2f GB/s\n", size / read_time / 1e9);
    
    // Memory + light compute (1 multiply)
    printf("\nMemory + 1 multiply per element:\n");
    timer_start(&t);
    sum = 0;
    for (size_t i = 0; i < num_floats; i++) {
        sum += data[i] * 1.5f;
    }
    sink = (uint64_t)sum;
    double mul_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", mul_time * 1000);
    printf("  Overhead: %.1f%%\n", (mul_time - read_time) / read_time * 100);
    
    // Memory + heavy compute (10 multiplies)
    printf("\nMemory + 10 operations per element:\n");
    timer_start(&t);
    sum = 0;
    for (size_t i = 0; i < num_floats; i++) {
        float x = data[i];
        x = x * 1.1f; x = x * 1.1f; x = x * 1.1f; x = x * 1.1f; x = x * 1.1f;
        x = x * 1.1f; x = x * 1.1f; x = x * 1.1f; x = x * 1.1f; x = x * 1.1f;
        sum += x;
    }
    sink = (uint64_t)sum;
    double heavy_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", heavy_time * 1000);
    printf("  Overhead: %.1f%%\n", (heavy_time - read_time) / read_time * 100);
    
    printf("\n📊 KEY INSIGHT:\n");
    printf("  Adding compute barely increases time!\n");
    printf("  This means: MEMORY is the bottleneck, not compute.\n");
    printf("  This is why Flash Attention recomputes instead of reloading!\n");
    
    free(data);
}

// ============================================================
// TEST 5: File I/O Costs
// ============================================================
void test_file_io() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TEST 5: FILE I/O COSTS                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t size = 10 * MB;
    char* data = malloc(size);
    memset(data, 'X', size);
    
    Timer t;
    
    // Write test
    printf("Writing %zu MB to disk:\n", size / MB);
    timer_start(&t);
    FILE* f = fopen("/tmp/test_io.bin", "wb");
    fwrite(data, 1, size, f);
    fflush(f);
    fclose(f);
    double write_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", write_time * 1000);
    printf("  Throughput: %.2f GB/s\n", size / write_time / 1e9);
    
    // Read test
    printf("\nReading %zu MB from disk:\n", size / MB);
    timer_start(&t);
    f = fopen("/tmp/test_io.bin", "rb");
    size_t read = fread(data, 1, size, f);
    fclose(f);
    double read_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", read_time * 1000);
    printf("  Throughput: %.2f GB/s\n", size / read_time / 1e9);
    
    // Small reads (worst case for data loading)
    printf("\nReading 1000 × 10KB chunks (simulates small images):\n");
    timer_start(&t);
    f = fopen("/tmp/test_io.bin", "rb");
    for (int i = 0; i < 1000; i++) {
        fseek(f, (rand() % (size / 10240)) * 10240, SEEK_SET);
        fread(data, 1, 10240, f);
    }
    fclose(f);
    double small_time = timer_stop(&t);
    printf("  Time: %.3f ms\n", small_time * 1000);
    printf("  Throughput: %.2f GB/s (with seek overhead)\n", 
           (1000 * 10240) / small_time / 1e9);
    
    printf("\n📊 KEY INSIGHT:\n");
    printf("  Small random reads are MUCH slower than sequential!\n");
    printf("  This is why image datasets should be packed (FFCV, WebDataset)!\n");
    printf("  This is why prefetching and caching matter!\n");
    
    remove("/tmp/test_io.bin");
    free(data);
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  DATA MOVEMENT COSTS - THE FOUNDATION OF ML PERFORMANCE      █\n");
    printf("█                                                              █\n");
    printf("█  Understanding these numbers will change how you code.       █\n");
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    test_memory_copy();
    test_access_patterns();
    test_cache_line_effects();
    test_compute_vs_memory();
    test_file_io();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    SUMMARY FOR ML                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. Memory bandwidth, not compute, is usually the limit      ║\n");
    printf("║                                                              ║\n");
    printf("║  2. Sequential access is 10-100x faster than random          ║\n");
    printf("║                                                              ║\n");
    printf("║  3. Keep data in cache (small working set, tiling)           ║\n");
    printf("║                                                              ║\n");
    printf("║  4. Pack small files into large sequential reads             ║\n");
    printf("║                                                              ║\n");
    printf("║  5. Recompute is often faster than reload (Flash Attention)  ║\n");
    printf("║                                                              ║\n");
    printf("║  PROFILE YOUR CODE TO VERIFY THESE NUMBERS ON YOUR HARDWARE  ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
