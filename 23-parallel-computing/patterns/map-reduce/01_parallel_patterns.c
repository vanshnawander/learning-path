/*
 * Parallel Programming Patterns
 * 
 * Demonstrates core parallel patterns: Map, Reduce, Scan, Stencil
 * 
 * Compile: gcc -fopenmp -O3 -march=native 01_parallel_patterns.c -o patterns -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define N 10000000

// ============================================================
// 1. MAP Pattern - Apply function to each element
// ============================================================

void map_sequential(float* out, float* in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf(in[i]) * cosf(in[i]);
    }
}

void map_parallel(float* out, float* in, int n) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        out[i] = sinf(in[i]) * cosf(in[i]);
    }
}

void demo_map(void) {
    printf("=== MAP Pattern ===\n");
    
    float* in = aligned_alloc(64, N * sizeof(float));
    float* out = aligned_alloc(64, N * sizeof(float));
    
    for (int i = 0; i < N; i++) in[i] = (float)i / N;
    
    double start = omp_get_wtime();
    map_sequential(out, in, N);
    printf("Sequential: %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    start = omp_get_wtime();
    map_parallel(out, in, N);
    printf("Parallel:   %.3f ms\n\n", (omp_get_wtime() - start) * 1000);
    
    free(in); free(out);
}

// ============================================================
// 2. REDUCE Pattern - Combine all elements to single value
// ============================================================

double reduce_sequential(double* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

double reduce_parallel(double* data, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Tree reduction (manual implementation)
double reduce_tree(double* data, int n) {
    double* temp = malloc(n * sizeof(double));
    memcpy(temp, data, n * sizeof(double));
    
    int size = n;
    while (size > 1) {
        int half = (size + 1) / 2;
        #pragma omp parallel for
        for (int i = 0; i < size / 2; i++) {
            temp[i] = temp[2*i] + temp[2*i + 1];
        }
        if (size % 2) temp[half - 1] = temp[size - 1];
        size = half;
    }
    
    double result = temp[0];
    free(temp);
    return result;
}

void demo_reduce(void) {
    printf("=== REDUCE Pattern ===\n");
    
    double* data = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) data[i] = 1.0;
    
    double start = omp_get_wtime();
    double sum1 = reduce_sequential(data, N);
    printf("Sequential: %.3f ms (sum = %.0f)\n", 
           (omp_get_wtime() - start) * 1000, sum1);
    
    start = omp_get_wtime();
    double sum2 = reduce_parallel(data, N);
    printf("Parallel:   %.3f ms (sum = %.0f)\n", 
           (omp_get_wtime() - start) * 1000, sum2);
    
    start = omp_get_wtime();
    double sum3 = reduce_tree(data, N);
    printf("Tree:       %.3f ms (sum = %.0f)\n\n", 
           (omp_get_wtime() - start) * 1000, sum3);
    
    free(data);
}

// ============================================================
// 3. SCAN (Prefix Sum) Pattern
// ============================================================

// Exclusive prefix sum: out[i] = sum(in[0..i-1])
void scan_sequential(int* out, int* in, int n) {
    out[0] = 0;
    for (int i = 1; i < n; i++) {
        out[i] = out[i-1] + in[i-1];
    }
}

// Blelloch parallel scan (work-efficient)
void scan_parallel(int* out, int* in, int n) {
    memcpy(out, in, n * sizeof(int));
    
    // Up-sweep (reduce)
    for (int d = 0; d < (int)log2(n); d++) {
        int stride = 1 << (d + 1);
        #pragma omp parallel for
        for (int i = stride - 1; i < n; i += stride) {
            out[i] += out[i - (1 << d)];
        }
    }
    
    out[n-1] = 0;  // Clear last element
    
    // Down-sweep
    for (int d = (int)log2(n) - 1; d >= 0; d--) {
        int stride = 1 << (d + 1);
        #pragma omp parallel for
        for (int i = stride - 1; i < n; i += stride) {
            int temp = out[i - (1 << d)];
            out[i - (1 << d)] = out[i];
            out[i] += temp;
        }
    }
}

void demo_scan(void) {
    printf("=== SCAN (Prefix Sum) Pattern ===\n");
    
    int size = 1 << 20;  // Power of 2 for simplicity
    int* in = malloc(size * sizeof(int));
    int* out1 = malloc(size * sizeof(int));
    int* out2 = malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) in[i] = 1;
    
    double start = omp_get_wtime();
    scan_sequential(out1, in, size);
    printf("Sequential: %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    start = omp_get_wtime();
    scan_parallel(out2, in, size);
    printf("Parallel:   %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < size && errors < 5; i++) {
        if (out1[i] != out2[i]) {
            printf("Mismatch at %d: %d vs %d\n", i, out1[i], out2[i]);
            errors++;
        }
    }
    printf("Verification: %s\n\n", errors ? "FAILED" : "PASSED");
    
    free(in); free(out1); free(out2);
}

// ============================================================
// 4. STENCIL Pattern - Neighborhood operations
// ============================================================

void stencil_1d_sequential(float* out, float* in, int n) {
    for (int i = 1; i < n - 1; i++) {
        out[i] = 0.25f * in[i-1] + 0.5f * in[i] + 0.25f * in[i+1];
    }
    out[0] = in[0];
    out[n-1] = in[n-1];
}

void stencil_1d_parallel(float* out, float* in, int n) {
    #pragma omp parallel for simd
    for (int i = 1; i < n - 1; i++) {
        out[i] = 0.25f * in[i-1] + 0.5f * in[i] + 0.25f * in[i+1];
    }
    out[0] = in[0];
    out[n-1] = in[n-1];
}

// 2D 5-point stencil (Jacobi iteration)
void stencil_2d(float* out, float* in, int nx, int ny) {
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            int idx = j * nx + i;
            out[idx] = 0.25f * (in[idx-1] + in[idx+1] + 
                                in[idx-nx] + in[idx+nx]);
        }
    }
}

void demo_stencil(void) {
    printf("=== STENCIL Pattern ===\n");
    
    // 1D stencil
    float* in1d = malloc(N * sizeof(float));
    float* out1d = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) in1d[i] = (float)i;
    
    double start = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++) {
        stencil_1d_sequential(out1d, in1d, N);
        float* tmp = in1d; in1d = out1d; out1d = tmp;
    }
    printf("1D Sequential (10 iter): %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    for (int i = 0; i < N; i++) in1d[i] = (float)i;
    start = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++) {
        stencil_1d_parallel(out1d, in1d, N);
        float* tmp = in1d; in1d = out1d; out1d = tmp;
    }
    printf("1D Parallel (10 iter):   %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    // 2D stencil
    int nx = 4096, ny = 4096;
    float* in2d = calloc(nx * ny, sizeof(float));
    float* out2d = calloc(nx * ny, sizeof(float));
    
    start = omp_get_wtime();
    for (int iter = 0; iter < 10; iter++) {
        stencil_2d(out2d, in2d, nx, ny);
        float* tmp = in2d; in2d = out2d; out2d = tmp;
    }
    printf("2D Parallel %dx%d (10 iter): %.3f ms\n\n", 
           nx, ny, (omp_get_wtime() - start) * 1000);
    
    free(in1d); free(out1d);
    free(in2d); free(out2d);
}

// ============================================================
// 5. PIPELINE Pattern
// ============================================================

#define STAGES 4
#define ITEMS 100

typedef struct {
    int data;
    int stage;
} work_item_t;

work_item_t items[ITEMS];

void process_stage(work_item_t* item, int stage) {
    // Simulate work
    volatile int sum = 0;
    for (int i = 0; i < 100000; i++) sum += i;
    item->data += stage;
    item->stage = stage;
}

void demo_pipeline(void) {
    printf("=== PIPELINE Pattern ===\n");
    
    for (int i = 0; i < ITEMS; i++) {
        items[i].data = i;
        items[i].stage = -1;
    }
    
    double start = omp_get_wtime();
    
    // Sequential pipeline
    for (int i = 0; i < ITEMS; i++) {
        for (int s = 0; s < STAGES; s++) {
            process_stage(&items[i], s);
        }
    }
    printf("Sequential: %.3f ms\n", (omp_get_wtime() - start) * 1000);
    
    // Parallel pipeline using OpenMP tasks
    for (int i = 0; i < ITEMS; i++) {
        items[i].data = i;
        items[i].stage = -1;
    }
    
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < ITEMS; i++) {
                #pragma omp task depend(out: items[i])
                process_stage(&items[i], 0);
                
                #pragma omp task depend(in: items[i]) depend(out: items[i])
                process_stage(&items[i], 1);
                
                #pragma omp task depend(in: items[i]) depend(out: items[i])
                process_stage(&items[i], 2);
                
                #pragma omp task depend(in: items[i])
                process_stage(&items[i], 3);
            }
        }
    }
    printf("Parallel:   %.3f ms\n\n", (omp_get_wtime() - start) * 1000);
}

int main(void) {
    printf("Parallel Patterns Demo\n");
    printf("Threads: %d\n\n", omp_get_max_threads());
    
    demo_map();
    demo_reduce();
    demo_scan();
    demo_stencil();
    demo_pipeline();
    
    return 0;
}
