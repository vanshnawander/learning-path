/*
 * NUMA Performance Benchmark
 * 
 * Measures local vs remote memory access latency and bandwidth
 * Demonstrates the NUMA effect on multi-socket systems
 * 
 * Compile: gcc -O3 -fopenmp -lnuma 02_numa_benchmark.c -o numa_bench
 * Run: numactl --hardware && ./numa_bench
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <omp.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#define KB (1024ULL)
#define MB (1024ULL * KB)
#define GB (1024ULL * MB)

#define TEST_SIZE (256 * MB)
#define ITERATIONS 5
#define LATENCY_CHAIN_SIZE (64 * MB)

typedef struct {
    double bandwidth_gbps;
    double latency_ns;
    int source_node;
    int memory_node;
} numa_result_t;

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

#ifdef __linux__
// Allocate memory on a specific NUMA node
void* numa_alloc_on_node(size_t size, int node) {
    void* ptr = numa_alloc_onnode(size, node);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %zu bytes on node %d\n", size, node);
        return NULL;
    }
    
    // Touch all pages to ensure allocation
    memset(ptr, 0, size);
    
    return ptr;
}

// Bind current thread to a specific NUMA node
void bind_to_node(int node) {
    struct bitmask* mask = numa_allocate_cpumask();
    numa_node_to_cpus(node, mask);
    
    // Find first CPU in this node
    for (int cpu = 0; cpu < numa_num_configured_cpus(); cpu++) {
        if (numa_bitmask_isbitset(mask, cpu)) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
            break;
        }
    }
    
    numa_free_cpumask(mask);
}

// Create pointer chain for latency measurement
void** create_chain_on_node(size_t size, int node) {
    size_t num_pointers = size / sizeof(void*);
    void** array = (void**)numa_alloc_on_node(size, node);
    
    if (!array) return NULL;
    
    // Create random permutation
    size_t* indices = (size_t*)malloc(num_pointers * sizeof(size_t));
    for (size_t i = 0; i < num_pointers; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    srand(42);  // Fixed seed for reproducibility
    for (size_t i = num_pointers - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
    
    // Create chain
    for (size_t i = 0; i < num_pointers - 1; i++) {
        array[indices[i]] = &array[indices[i + 1]];
    }
    array[indices[num_pointers - 1]] = &array[indices[0]];
    
    free(indices);
    return array;
}

// Measure memory bandwidth from CPU on source_node to memory on mem_node
numa_result_t measure_bandwidth(int source_node, int mem_node, size_t size) {
    numa_result_t result = {0};
    result.source_node = source_node;
    result.memory_node = mem_node;
    
    // Bind to source node
    bind_to_node(source_node);
    
    // Allocate on memory node
    double* array = (double*)numa_alloc_on_node(size, mem_node);
    if (!array) {
        result.bandwidth_gbps = -1;
        return result;
    }
    
    size_t n = size / sizeof(double);
    
    // Initialize
    for (size_t i = 0; i < n; i++) {
        array[i] = 1.0;
    }
    
    // Warm up
    volatile double sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += array[i];
    }
    
    // Benchmark read bandwidth
    double start = get_time_sec();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += array[i];
        }
    }
    
    double elapsed = get_time_sec() - start;
    
    // Prevent optimization
    if (sum == 0.12345) printf("x");
    
    size_t bytes = n * sizeof(double) * ITERATIONS;
    result.bandwidth_gbps = (bytes / elapsed) / GB;
    
    numa_free(array, size);
    return result;
}

// Measure memory latency from CPU on source_node to memory on mem_node
numa_result_t measure_latency(int source_node, int mem_node) {
    numa_result_t result = {0};
    result.source_node = source_node;
    result.memory_node = mem_node;
    
    // Bind to source node
    bind_to_node(source_node);
    
    // Create chain on memory node
    void** chain = create_chain_on_node(LATENCY_CHAIN_SIZE, mem_node);
    if (!chain) {
        result.latency_ns = -1;
        return result;
    }
    
    size_t chain_length = LATENCY_CHAIN_SIZE / sizeof(void*);
    void** p = chain;
    
    // Warm up
    for (size_t i = 0; i < chain_length; i++) {
        p = (void**)*p;
    }
    
    // Benchmark
    p = chain;
    uint64_t start = rdtsc();
    
    for (size_t i = 0; i < chain_length; i++) {
        p = (void**)*p;
    }
    
    uint64_t cycles = rdtsc() - start;
    
    // Prevent optimization
    if (p == (void**)0x12345) printf("x");
    
    // Assuming ~2.5 GHz (adjust for your CPU)
    result.latency_ns = (double)cycles / chain_length / 2.5;
    
    numa_free(chain, LATENCY_CHAIN_SIZE);
    return result;
}

void print_numa_topology(void) {
    printf("=== NUMA Topology ===\n");
    
    int num_nodes = numa_max_node() + 1;
    printf("Number of NUMA nodes: %d\n", num_nodes);
    
    for (int i = 0; i < num_nodes; i++) {
        long free_mem;
        long total_mem = numa_node_size(i, &free_mem);
        printf("Node %d: %ld MB total, %ld MB free\n", 
               i, total_mem / (1024*1024), free_mem / (1024*1024));
        
        printf("  CPUs: ");
        struct bitmask* cpumask = numa_allocate_cpumask();
        numa_node_to_cpus(i, cpumask);
        for (int cpu = 0; cpu < numa_num_configured_cpus(); cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                printf("%d ", cpu);
            }
        }
        printf("\n");
        numa_free_cpumask(cpumask);
    }
    
    printf("\nNUMA distances:\n");
    printf("      ");
    for (int j = 0; j < num_nodes; j++) {
        printf("Node%-2d ", j);
    }
    printf("\n");
    
    for (int i = 0; i < num_nodes; i++) {
        printf("Node%d ", i);
        for (int j = 0; j < num_nodes; j++) {
            printf("%-6d ", numa_distance(i, j));
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available on this system\n");
        fprintf(stderr, "This benchmark requires a multi-socket NUMA system\n");
        return 1;
    }
    
    print_numa_topology();
    
    int num_nodes = numa_max_node() + 1;
    
    if (num_nodes < 2) {
        printf("Warning: Only 1 NUMA node detected.\n");
        printf("NUMA effects won't be visible on single-socket systems.\n");
        printf("Running anyway to demonstrate the measurement technique.\n\n");
    }
    
    printf("=== Bandwidth Benchmark ===\n");
    printf("Test size: %llu MB\n", (unsigned long long)(TEST_SIZE / MB));
    printf("Iterations: %d\n\n", ITERATIONS);
    
    printf("%-12s %-12s %15s\n", "CPU Node", "Mem Node", "Bandwidth (GB/s)");
    printf("%-12s %-12s %15s\n", "--------", "--------", "----------------");
    
    for (int src = 0; src < num_nodes; src++) {
        for (int mem = 0; mem < num_nodes; mem++) {
            numa_result_t result = measure_bandwidth(src, mem, TEST_SIZE);
            
            const char* locality = (src == mem) ? "(local)" : "(remote)";
            printf("%-12d %-12d %15.2f %s\n", 
                   src, mem, result.bandwidth_gbps, locality);
        }
    }
    
    printf("\n=== Latency Benchmark ===\n");
    printf("Chain size: %llu MB\n\n", (unsigned long long)(LATENCY_CHAIN_SIZE / MB));
    
    printf("%-12s %-12s %15s\n", "CPU Node", "Mem Node", "Latency (ns)");
    printf("%-12s %-12s %15s\n", "--------", "--------", "------------");
    
    for (int src = 0; src < num_nodes; src++) {
        for (int mem = 0; mem < num_nodes; mem++) {
            numa_result_t result = measure_latency(src, mem);
            
            const char* locality = (src == mem) ? "(local)" : "(remote)";
            printf("%-12d %-12d %15.2f %s\n", 
                   src, mem, result.latency_ns, locality);
        }
    }
    
    printf("\n=== Interpretation ===\n");
    printf("Local access should be faster than remote access.\n");
    printf("Typical NUMA ratio (remote/local):\n");
    printf("  Latency: 1.3x - 2.0x\n");
    printf("  Bandwidth: 0.5x - 0.8x\n");
    printf("\nIf you don't see this difference, possible reasons:\n");
    printf("  1. Single-socket system (no NUMA)\n");
    printf("  2. Memory interleaving enabled in BIOS\n");
    printf("  3. SNC/NPS settings grouping memory differently\n");
    
    return 0;
}

#else // Non-Linux systems

int main(int argc, char** argv) {
    printf("This NUMA benchmark requires Linux with libnuma.\n");
    printf("On Windows, use Windows Performance Toolkit for NUMA analysis.\n");
    return 1;
}

#endif
