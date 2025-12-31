/*
 * Advanced OpenMP Features
 * 
 * Covers: Tasks, SIMD, target offload, and advanced scheduling
 * 
 * Compile: gcc -fopenmp -O3 -march=native 01_openmp_advanced.c -o omp_advanced
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 10000000

// ============================================================
// 1. OpenMP Tasks - Dynamic Parallelism
// ============================================================

long fibonacci_task(int n) {
    if (n < 2) return n;
    
    long x, y;
    
    if (n < 20) {  // Cutoff for sequential
        return fibonacci_task(n-1) + fibonacci_task(n-2);
    }
    
    #pragma omp task shared(x)
    x = fibonacci_task(n - 1);
    
    #pragma omp task shared(y)
    y = fibonacci_task(n - 2);
    
    #pragma omp taskwait
    return x + y;
}

void demo_tasks(void) {
    printf("=== OpenMP Tasks (Fibonacci) ===\n");
    
    long result;
    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        result = fibonacci_task(40);
    }
    
    double elapsed = omp_get_wtime() - start;
    printf("fib(40) = %ld, time = %.3f sec\n\n", result, elapsed);
}

// ============================================================
// 2. Task Dependencies
// ============================================================

void demo_task_dependencies(void) {
    printf("=== Task Dependencies ===\n");
    
    int a = 0, b = 0, c = 0;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task depend(out: a)
            {
                a = 1;
                printf("Task A: a = %d\n", a);
            }
            
            #pragma omp task depend(out: b)
            {
                b = 2;
                printf("Task B: b = %d\n", b);
            }
            
            #pragma omp task depend(in: a, b) depend(out: c)
            {
                c = a + b;
                printf("Task C: c = a + b = %d\n", c);
            }
        }
    }
    printf("\n");
}

// ============================================================
// 3. SIMD Vectorization
// ============================================================

void demo_simd(void) {
    printf("=== OpenMP SIMD ===\n");
    
    float *a = aligned_alloc(64, N * sizeof(float));
    float *b = aligned_alloc(64, N * sizeof(float));
    float *c = aligned_alloc(64, N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(N - i);
    }
    
    double start = omp_get_wtime();
    
    // SIMD vectorization
    #pragma omp simd aligned(a, b, c: 64)
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * b[i] + sqrtf(a[i]);
    }
    
    double simd_time = omp_get_wtime() - start;
    
    // Parallel + SIMD
    start = omp_get_wtime();
    
    #pragma omp parallel for simd aligned(a, b, c: 64)
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * b[i] + sqrtf(a[i]);
    }
    
    double parallel_simd_time = omp_get_wtime() - start;
    
    printf("SIMD only:     %.3f ms\n", simd_time * 1000);
    printf("Parallel+SIMD: %.3f ms\n\n", parallel_simd_time * 1000);
    
    free(a); free(b); free(c);
}

// ============================================================
// 4. Reduction with Custom Operators
// ============================================================

typedef struct {
    double sum;
    double min;
    double max;
} stats_t;

void init_stats(stats_t* s) {
    s->sum = 0.0;
    s->min = 1e30;
    s->max = -1e30;
}

void combine_stats(stats_t* out, stats_t* in) {
    out->sum += in->sum;
    if (in->min < out->min) out->min = in->min;
    if (in->max > out->max) out->max = in->max;
}

#pragma omp declare reduction(stats_reduce: stats_t: \
    combine_stats(&omp_out, &omp_in)) \
    initializer(init_stats(&omp_priv))

void demo_custom_reduction(void) {
    printf("=== Custom Reduction ===\n");
    
    double* data = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        data[i] = sin((double)i / 1000.0) * 100.0;
    }
    
    stats_t stats;
    init_stats(&stats);
    
    #pragma omp parallel for reduction(stats_reduce: stats)
    for (int i = 0; i < N; i++) {
        stats.sum += data[i];
        if (data[i] < stats.min) stats.min = data[i];
        if (data[i] > stats.max) stats.max = data[i];
    }
    
    printf("Sum: %.2f, Min: %.2f, Max: %.2f\n\n", 
           stats.sum, stats.min, stats.max);
    
    free(data);
}

// ============================================================
// 5. Schedule Comparison
// ============================================================

void demo_scheduling(void) {
    printf("=== Schedule Comparison ===\n");
    
    int* work = malloc(1000 * sizeof(int));
    // Variable work per iteration
    for (int i = 0; i < 1000; i++) {
        work[i] = (i % 100) * 1000;  // 0 to 99000 iterations
    }
    
    volatile long dummy = 0;
    double start;
    
    // Static scheduling
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:dummy)
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < work[i]; j++) dummy++;
    }
    printf("Static:      %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    // Dynamic scheduling
    dummy = 0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 10) reduction(+:dummy)
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < work[i]; j++) dummy++;
    }
    printf("Dynamic(10): %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    // Guided scheduling
    dummy = 0;
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided) reduction(+:dummy)
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < work[i]; j++) dummy++;
    }
    printf("Guided:      %.3f ms\n\n", (omp_get_wtime() - start) * 1000);
    
    free(work);
}

// ============================================================
// 6. Nested Parallelism
// ============================================================

void demo_nested_parallelism(void) {
    printf("=== Nested Parallelism ===\n");
    
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    
    #pragma omp parallel num_threads(2)
    {
        int outer_id = omp_get_thread_num();
        
        #pragma omp parallel num_threads(2)
        {
            int inner_id = omp_get_thread_num();
            printf("Outer %d, Inner %d\n", outer_id, inner_id);
        }
    }
    printf("\n");
}

int main(void) {
    printf("OpenMP version: %d\n", _OPENMP);
    printf("Max threads: %d\n\n", omp_get_max_threads());
    
    demo_tasks();
    demo_task_dependencies();
    demo_simd();
    demo_custom_reduction();
    demo_scheduling();
    demo_nested_parallelism();
    
    return 0;
}
