/**
 * 01_data_oriented.c - Data-Oriented Design for ML
 * 
 * Structure your data for how it's accessed, not how you think about it.
 * This is critical for cache efficiency and SIMD.
 * 
 * Compile: gcc -O3 -mavx2 -o data_oriented 01_data_oriented.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define N 1000000

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
// Object-Oriented Style: Array of Structures (AoS)
// ============================================================

typedef struct {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;
    int active;
} Particle_AoS;

void update_aos(Particle_AoS* particles, int n, float dt) {
    for (int i = 0; i < n; i++) {
        if (particles[i].active) {
            particles[i].x += particles[i].vx * dt;
            particles[i].y += particles[i].vy * dt;
            particles[i].z += particles[i].vz * dt;
        }
    }
}

// ============================================================
// Data-Oriented Style: Structure of Arrays (SoA)
// ============================================================

typedef struct {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* active;
    int count;
} Particles_SoA;

Particles_SoA* create_soa(int n) {
    Particles_SoA* p = malloc(sizeof(Particles_SoA));
    p->x = aligned_alloc(32, n * sizeof(float));
    p->y = aligned_alloc(32, n * sizeof(float));
    p->z = aligned_alloc(32, n * sizeof(float));
    p->vx = aligned_alloc(32, n * sizeof(float));
    p->vy = aligned_alloc(32, n * sizeof(float));
    p->vz = aligned_alloc(32, n * sizeof(float));
    p->mass = aligned_alloc(32, n * sizeof(float));
    p->active = aligned_alloc(32, n * sizeof(int));
    p->count = n;
    return p;
}

void update_soa(Particles_SoA* p, float dt) {
    for (int i = 0; i < p->count; i++) {
        p->x[i] += p->vx[i] * dt;
        p->y[i] += p->vy[i] * dt;
        p->z[i] += p->vz[i] * dt;
    }
}

void update_soa_simd(Particles_SoA* p, float dt) {
    __m256 vdt = _mm256_set1_ps(dt);
    
    for (int i = 0; i < p->count; i += 8) {
        // X update
        __m256 x = _mm256_load_ps(p->x + i);
        __m256 vx = _mm256_load_ps(p->vx + i);
        x = _mm256_fmadd_ps(vx, vdt, x);
        _mm256_store_ps(p->x + i, x);
        
        // Y update
        __m256 y = _mm256_load_ps(p->y + i);
        __m256 vy = _mm256_load_ps(p->vy + i);
        y = _mm256_fmadd_ps(vy, vdt, y);
        _mm256_store_ps(p->y + i, y);
        
        // Z update
        __m256 z = _mm256_load_ps(p->z + i);
        __m256 vz = _mm256_load_ps(p->vz + i);
        z = _mm256_fmadd_ps(vz, vdt, z);
        _mm256_store_ps(p->z + i, z);
    }
}

int main() {
    printf("=== DATA-ORIENTED DESIGN ===\n\n");
    
    printf("Particles: %d\n\n", N);
    
    // Initialize AoS
    Particle_AoS* aos = malloc(N * sizeof(Particle_AoS));
    for (int i = 0; i < N; i++) {
        aos[i].x = aos[i].y = aos[i].z = 0;
        aos[i].vx = aos[i].vy = aos[i].vz = 1.0f;
        aos[i].mass = 1.0f;
        aos[i].active = 1;
    }
    
    // Initialize SoA
    Particles_SoA* soa = create_soa(N);
    for (int i = 0; i < N; i++) {
        soa->x[i] = soa->y[i] = soa->z[i] = 0;
        soa->vx[i] = soa->vy[i] = soa->vz[i] = 1.0f;
        soa->mass[i] = 1.0f;
        soa->active[i] = 1;
    }
    
    double start;
    int iterations = 100;
    
    // Benchmark AoS
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        update_aos(aos, N, 0.016f);
    }
    double aos_time = (get_time() - start) / iterations;
    
    // Benchmark SoA scalar
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        update_soa(soa, 0.016f);
    }
    double soa_time = (get_time() - start) / iterations;
    
    // Benchmark SoA SIMD
    start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        update_soa_simd(soa, 0.016f);
    }
    double soa_simd_time = (get_time() - start) / iterations;
    
    printf("Update time per iteration:\n");
    printf("  AoS:        %.2f ms\n", aos_time * 1000);
    printf("  SoA:        %.2f ms (%.1fx vs AoS)\n", 
           soa_time * 1000, aos_time / soa_time);
    printf("  SoA + SIMD: %.2f ms (%.1fx vs AoS)\n",
           soa_simd_time * 1000, aos_time / soa_simd_time);
    
    printf("\n=== WHY SoA IS FASTER ===\n\n");
    printf("AoS Memory layout:\n");
    printf("  [x0,y0,z0,vx0,vy0,vz0,m0,a0][x1,y1,z1,...]...\n");
    printf("  Stride between x values = 32 bytes (whole struct)\n");
    printf("  Cache line loads unused data (mass, active)\n\n");
    
    printf("SoA Memory layout:\n");
    printf("  x:  [x0,x1,x2,x3,x4,x5,x6,x7,...]\n");
    printf("  vx: [vx0,vx1,vx2,vx3,vx4,vx5,vx6,vx7,...]\n");
    printf("  Contiguous! Every byte loaded is used\n");
    printf("  Perfect for SIMD (8 floats = 32 bytes)\n");
    
    printf("\n=== ML APPLICATION ===\n\n");
    printf("PyTorch tensors ARE SoA:\n");
    printf("  weights: [w0,w1,w2,...] contiguous floats\n");
    printf("  NOT: [w0,grad0,mom0][w1,grad1,mom1]...\n\n");
    printf("This is why tensor operations are fast!\n");
    
    free(aos);
    free(soa->x); free(soa->y); free(soa->z);
    free(soa->vx); free(soa->vy); free(soa->vz);
    free(soa->mass); free(soa->active);
    free(soa);
    
    return 0;
}
