/*
 * Structure of Arrays (SoA) vs Array of Structures (AoS)
 * 
 * Demonstrates cache efficiency differences between data layouts.
 * SoA is typically better for SIMD and when accessing few fields.
 * AoS is better when accessing all fields of an entity together.
 * 
 * Compile: gcc -O3 -march=native -fopenmp 01_soa_vs_aos.c -o soa_aos -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 10000000
#define ITERATIONS 10

// AoS: Array of Structures
typedef struct {
    float x, y, z;       // Position
    float vx, vy, vz;    // Velocity
    float mass;
    int id;
    char padding[24];    // Pad to 64 bytes (cache line)
} Particle_AoS;

// SoA: Structure of Arrays
typedef struct {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* id;
} Particles_SoA;

Particle_AoS* aos;
Particles_SoA soa;

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_aos(void) {
    aos = (Particle_AoS*)aligned_alloc(64, N * sizeof(Particle_AoS));
    for (int i = 0; i < N; i++) {
        aos[i].x = (float)i * 0.1f;
        aos[i].y = (float)i * 0.2f;
        aos[i].z = (float)i * 0.3f;
        aos[i].vx = 1.0f;
        aos[i].vy = 2.0f;
        aos[i].vz = 3.0f;
        aos[i].mass = 1.0f;
        aos[i].id = i;
    }
}

void init_soa(void) {
    soa.x = (float*)aligned_alloc(64, N * sizeof(float));
    soa.y = (float*)aligned_alloc(64, N * sizeof(float));
    soa.z = (float*)aligned_alloc(64, N * sizeof(float));
    soa.vx = (float*)aligned_alloc(64, N * sizeof(float));
    soa.vy = (float*)aligned_alloc(64, N * sizeof(float));
    soa.vz = (float*)aligned_alloc(64, N * sizeof(float));
    soa.mass = (float*)aligned_alloc(64, N * sizeof(float));
    soa.id = (int*)aligned_alloc(64, N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        soa.x[i] = (float)i * 0.1f;
        soa.y[i] = (float)i * 0.2f;
        soa.z[i] = (float)i * 0.3f;
        soa.vx[i] = 1.0f;
        soa.vy[i] = 2.0f;
        soa.vz[i] = 3.0f;
        soa.mass[i] = 1.0f;
        soa.id[i] = i;
    }
}

// Test 1: Update positions (access 6 fields per particle)
void update_positions_aos(float dt) {
    for (int i = 0; i < N; i++) {
        aos[i].x += aos[i].vx * dt;
        aos[i].y += aos[i].vy * dt;
        aos[i].z += aos[i].vz * dt;
    }
}

void update_positions_soa(float dt) {
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        soa.x[i] += soa.vx[i] * dt;
    }
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        soa.y[i] += soa.vy[i] * dt;
    }
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        soa.z[i] += soa.vz[i] * dt;
    }
}

// Test 2: Compute distances (access only x,y,z)
float sum_distances_aos(void) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += sqrtf(aos[i].x*aos[i].x + aos[i].y*aos[i].y + aos[i].z*aos[i].z);
    }
    return sum;
}

float sum_distances_soa(void) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += sqrtf(soa.x[i]*soa.x[i] + soa.y[i]*soa.y[i] + soa.z[i]*soa.z[i]);
    }
    return sum;
}

// Test 3: Sum all masses (access only 1 field)
float sum_mass_aos(void) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += aos[i].mass;
    }
    return sum;
}

float sum_mass_soa(void) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += soa.mass[i];
    }
    return sum;
}

int main() {
    printf("=== SoA vs AoS Data Layout Comparison ===\n");
    printf("Particles: %d\n", N);
    printf("AoS struct size: %zu bytes\n", sizeof(Particle_AoS));
    printf("SoA total size: %zu bytes\n\n", 8 * N * sizeof(float));
    
    init_aos();
    init_soa();
    
    double start, elapsed;
    float result;
    
    printf("%-25s %12s %12s %10s\n", "Operation", "AoS (ms)", "SoA (ms)", "Speedup");
    printf("%-25s %12s %12s %10s\n", "---------", "--------", "--------", "-------");
    
    // Test 1: Update positions
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) update_positions_aos(0.01f);
    double aos_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) update_positions_soa(0.01f);
    double soa_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    printf("%-25s %12.2f %12.2f %10.2fx\n", "Update positions", aos_time, soa_time, aos_time/soa_time);
    
    // Test 2: Compute distances
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) result = sum_distances_aos();
    aos_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) result = sum_distances_soa();
    soa_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    printf("%-25s %12.2f %12.2f %10.2fx\n", "Sum distances (x,y,z)", aos_time, soa_time, aos_time/soa_time);
    
    // Test 3: Sum masses (1 field only)
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) result = sum_mass_aos();
    aos_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    start = get_time_sec();
    for (int i = 0; i < ITERATIONS; i++) result = sum_mass_soa();
    soa_time = (get_time_sec() - start) * 1000 / ITERATIONS;
    
    printf("%-25s %12.2f %12.2f %10.2fx\n", "Sum masses (1 field)", aos_time, soa_time, aos_time/soa_time);
    
    printf("\n=== Analysis ===\n");
    printf("SoA advantages:\n");
    printf("  - Better cache utilization when accessing few fields\n");
    printf("  - Enables SIMD vectorization (contiguous data)\n");
    printf("  - Less memory bandwidth wasted on unused fields\n");
    printf("\nAoS advantages:\n");
    printf("  - Better when accessing all fields of one entity\n");
    printf("  - Simpler code, natural object-oriented style\n");
    printf("  - Better locality for entity-centric operations\n");
    printf("\nAoSoA (Array of Structures of Arrays):\n");
    printf("  - Hybrid: group of N entities in SoA layout\n");
    printf("  - Best of both: SIMD-friendly + entity locality\n");
    
    // Cleanup
    free(aos);
    free(soa.x); free(soa.y); free(soa.z);
    free(soa.vx); free(soa.vy); free(soa.vz);
    free(soa.mass); free(soa.id);
    
    return 0;
}
