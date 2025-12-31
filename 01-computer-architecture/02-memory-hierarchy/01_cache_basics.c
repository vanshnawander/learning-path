/**
 * 01_cache_basics.c - Understanding CPU Cache Behavior
 * 
 * The memory hierarchy is the #1 factor in performance.
 * Understanding cache behavior is essential for:
 * - FFCV's data layout decisions
 * - PyTorch tensor memory layout (contiguous vs strided)
 * - GPU memory coalescing
 * - Data loading pipeline optimization
 * 
 * Compile: gcc -O2 -o 01_cache_basics 01_cache_basics.c
 * Run: ./01_cache_basics
 * 
 * Try with perf: perf stat -e cache-misses,cache-references ./01_cache_basics
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define KB (1024)
#define MB (1024 * KB)

// High-resolution timer
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Prevent compiler from optimizing away the result
void escape(void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

/**
 * EXPERIMENT 1: Sequential vs Random Access
 * 
 * Sequential access exploits:
 * 1. Spatial locality - cache lines are 64 bytes
 * 2. Hardware prefetching - CPU predicts next access
 * 
 * Random access defeats both!
 */
void experiment_sequential_vs_random(size_t size) {
    printf("\n=== EXPERIMENT 1: SEQUENTIAL VS RANDOM ACCESS ===\n");
    printf("Array size: %zu MB\n\n", size / MB);
    
    int* array = malloc(size);
    size_t n = size / sizeof(int);
    
    // Initialize
    for (size_t i = 0; i < n; i++) {
        array[i] = i;
    }
    
    // Create random permutation for random access
    size_t* random_indices = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) {
        random_indices[i] = i;
    }
    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = random_indices[i];
        random_indices[i] = random_indices[j];
        random_indices[j] = tmp;
    }
    
    // Sequential access
    volatile long sum = 0;
    double start = get_time();
    for (size_t i = 0; i < n; i++) {
        sum += array[i];
    }
    double sequential_time = get_time() - start;
    escape((void*)&sum);
    
    // Random access
    sum = 0;
    start = get_time();
    for (size_t i = 0; i < n; i++) {
        sum += array[random_indices[i]];
    }
    double random_time = get_time() - start;
    escape((void*)&sum);
    
    printf("Sequential access: %.3f ms (%.2f GB/s)\n", 
           sequential_time * 1000,
           (size / (1e9)) / sequential_time);
    printf("Random access:     %.3f ms (%.2f GB/s)\n", 
           random_time * 1000,
           (size / (1e9)) / random_time);
    printf("Random is %.1fx slower!\n", random_time / sequential_time);
    
    printf("\nWHY: Sequential access allows prefetching and uses full cache lines.\n");
    printf("     Random access causes cache misses on nearly every access.\n");
    printf("\nML IMPLICATION: This is why FFCV uses quasi-random sampling!\n");
    printf("     True random access to files is extremely slow.\n");
    
    free(array);
    free(random_indices);
}

/**
 * EXPERIMENT 2: Cache Line Effects
 * 
 * CPU caches work in units of cache lines (typically 64 bytes).
 * Accessing one byte loads the entire line.
 */
void experiment_cache_line() {
    printf("\n=== EXPERIMENT 2: CACHE LINE EFFECTS ===\n\n");
    
    size_t size = 64 * MB;
    char* array = malloc(size);
    memset(array, 0, size);
    
    printf("Stride  Time(ms)  Bandwidth   Cache Efficiency\n");
    printf("------  --------  ----------  ----------------\n");
    
    // Access with different strides
    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    double baseline = 0;
    
    for (int s = 0; s < 11; s++) {
        int stride = strides[s];
        size_t accesses = size / stride;
        
        volatile long sum = 0;
        double start = get_time();
        
        for (size_t i = 0; i < accesses; i++) {
            sum += array[i * stride];
        }
        
        double elapsed = get_time() - start;
        escape((void*)&sum);
        
        double bandwidth = (accesses * sizeof(char)) / (elapsed * 1e9);
        
        if (s == 0) baseline = elapsed;
        double efficiency = baseline / elapsed;
        
        printf("%5d   %8.2f  %6.2f GB/s  ", stride, elapsed * 1000, bandwidth);
        
        // Visual indicator
        if (stride <= 64) {
            printf("█████████ Full line used\n");
        } else {
            int bars = (int)(10 * 64.0 / stride);
            for (int i = 0; i < bars; i++) printf("█");
            for (int i = bars; i < 10; i++) printf("░");
            printf(" Only %d/64 bytes used\n", 64 / (stride > 64 ? stride/64 : 1));
        }
    }
    
    printf("\nWHY: Cache line is 64 bytes. Stride > 64 wastes loaded data.\n");
    printf("\nML IMPLICATION:\n");
    printf("  - Tensor layouts matter! NCHW vs NHWC affects cache efficiency.\n");
    printf("  - Batch dimension should be contiguous for efficient loading.\n");
    printf("  - This is why we 'contiguous()' tensors before operations.\n");
    
    free(array);
}

/**
 * EXPERIMENT 3: Cache Size Detection
 * 
 * By varying working set size, we can see L1/L2/L3 boundaries.
 */
void experiment_cache_sizes() {
    printf("\n=== EXPERIMENT 3: DETECTING CACHE SIZES ===\n\n");
    
    printf("Size     Time(ms)  Bandwidth    Level\n");
    printf("-------  --------  ----------   -----\n");
    
    size_t sizes[] = {
        4*KB, 8*KB, 16*KB, 32*KB,      // L1
        64*KB, 128*KB, 256*KB, 512*KB, // L2
        1*MB, 2*MB, 4*MB, 8*MB,        // L3
        16*MB, 32*MB, 64*MB            // RAM
    };
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        int* array = malloc(size);
        size_t n = size / sizeof(int);
        
        // Initialize
        for (size_t i = 0; i < n; i++) {
            array[i] = i + 1;
        }
        
        // Warm up cache
        volatile long sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += array[i];
        }
        
        // Measure multiple passes
        int passes = (size < MB) ? 1000 : 100;
        double start = get_time();
        
        for (int p = 0; p < passes; p++) {
            for (size_t i = 0; i < n; i++) {
                sum += array[i];
            }
        }
        
        double elapsed = get_time() - start;
        escape((void*)&sum);
        
        double bandwidth = (size * passes) / (elapsed * 1e9);
        
        // Determine cache level
        const char* level;
        if (size <= 32*KB) level = "L1 (~4 cycles)";
        else if (size <= 256*KB) level = "L2 (~12 cycles)";
        else if (size <= 8*MB) level = "L3 (~40 cycles)";
        else level = "RAM (~100+ cycles)";
        
        if (size >= MB) {
            printf("%4zu MB  %8.2f  %6.2f GB/s   %s\n", 
                   size/MB, elapsed * 1000, bandwidth, level);
        } else {
            printf("%4zu KB  %8.2f  %6.2f GB/s   %s\n", 
                   size/KB, elapsed * 1000, bandwidth, level);
        }
        
        free(array);
    }
    
    printf("\nWHY: Each cache level has different size and latency.\n");
    printf("     When working set exceeds cache size, performance drops.\n");
    printf("\nML IMPLICATION:\n");
    printf("  - Batch size affects whether activations fit in cache\n");
    printf("  - Tiling/blocking algorithms keep working set in cache\n");
    printf("  - This is exactly what Flash Attention does!\n");
}

/**
 * EXPERIMENT 4: Row-Major vs Column-Major Access
 * 
 * This demonstrates why memory layout matters for matrices.
 */
void experiment_matrix_access() {
    printf("\n=== EXPERIMENT 4: ROW-MAJOR VS COLUMN-MAJOR ===\n\n");
    
    int N = 4096;  // 4K x 4K matrix = 64 MB
    float* matrix = malloc(N * N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N * N; i++) {
        matrix[i] = 1.0f;
    }
    
    volatile float sum;
    double start, elapsed;
    
    // Row-major access (how C stores 2D arrays)
    sum = 0;
    start = get_time();
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col];  // Sequential in memory!
        }
    }
    elapsed = get_time() - start;
    escape((void*)&sum);
    double row_major_time = elapsed;
    
    // Column-major access (strided access)
    sum = 0;
    start = get_time();
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < N; row++) {
            sum += matrix[row * N + col];  // Stride of N elements!
        }
    }
    elapsed = get_time() - start;
    escape((void*)&sum);
    double col_major_time = elapsed;
    
    printf("Matrix size: %d x %d (%.0f MB)\n\n", N, N, 
           N * N * sizeof(float) / (float)MB);
    printf("Row-major access:    %.2f ms\n", row_major_time * 1000);
    printf("Column-major access: %.2f ms\n", col_major_time * 1000);
    printf("Column-major is %.1fx slower!\n", col_major_time / row_major_time);
    
    printf("\nMEMORY LAYOUT:\n");
    printf("Row-major (C/PyTorch): [row0_col0, row0_col1, row0_col2, ...]\n");
    printf("Col-major (Fortran):   [row0_col0, row1_col0, row2_col0, ...]\n");
    
    printf("\nML IMPLICATION:\n");
    printf("  - PyTorch tensors are row-major (last dim contiguous)\n");
    printf("  - Matrix multiply: (M,K) @ (K,N) - inner dims must align\n");
    printf("  - transpose().contiguous() copies data for correct layout\n");
    printf("  - cuBLAS expects column-major, PyTorch handles conversion\n");
    
    free(matrix);
}

int main() {
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║     CACHE BEHAVIOR: THE FOUNDATION OF PERFORMANCE      ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    printf("\nModern CPUs have a memory hierarchy:\n");
    printf("  Registers: ~1 cycle,   ~KB\n");
    printf("  L1 Cache:  ~4 cycles,  32-64 KB\n");
    printf("  L2 Cache:  ~12 cycles, 256 KB - 1 MB\n");
    printf("  L3 Cache:  ~40 cycles, 8-64 MB\n");
    printf("  RAM:       ~100 cycles, GBs\n");
    printf("  SSD:       ~10000 cycles, TBs\n");
    printf("\nThe goal: Keep frequently accessed data in faster memory!\n");
    
    srand(42);
    
    experiment_sequential_vs_random(64 * MB);
    experiment_cache_line();
    experiment_cache_sizes();
    experiment_matrix_access();
    
    printf("\n=== KEY TAKEAWAYS FOR ML ===\n\n");
    printf("1. Sequential access >> Random access (10-100x faster)\n");
    printf("2. Use full cache lines - stride of 1 is optimal\n");
    printf("3. Keep working set in cache - tile your computations\n");
    printf("4. Memory layout matters - row-major vs column-major\n");
    printf("5. This applies to GPU too! (coalescing = GPU cache lines)\n");
    
    return 0;
}
