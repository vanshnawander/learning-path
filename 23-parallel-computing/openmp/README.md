# OpenMP

Shared-memory parallel programming for CPUs.

## Basic Usage

```c
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Hello from thread %d\n", tid);
    }
    return 0;
}
```

Compile: `gcc -fopenmp program.c`

## Parallel For

```c
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
}
```

## Reduction

```c
int sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += a[i];
}
```

## Scheduling

```c
// Static: divide equally
#pragma omp parallel for schedule(static)

// Dynamic: on-demand distribution
#pragma omp parallel for schedule(dynamic, chunk_size)

// Guided: decreasing chunk sizes
#pragma omp parallel for schedule(guided)
```

## Critical Sections

```c
#pragma omp parallel
{
    #pragma omp critical
    {
        // Only one thread at a time
        shared_counter++;
    }
}
```

## Atomic Operations

```c
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    #pragma omp atomic
    sum += a[i];
}
```

## SIMD

```c
#pragma omp simd
for (int i = 0; i < n; i++) {
    a[i] = b[i] * c[i];
}
```

## Thread Control

```c
omp_set_num_threads(8);  // Set thread count
int num = omp_get_num_threads();  // Get thread count
int max = omp_get_max_threads();  // Get max threads
```

## Environment Variables
```bash
export OMP_NUM_THREADS=8
export OMP_SCHEDULE="dynamic,100"
```
