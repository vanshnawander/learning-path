/**
 * 01_soa_vs_aos.c - Array of Structures vs Structure of Arrays
 * 
 * Data layout determines cache efficiency.
 * This is why NCHW vs NHWC matters in PyTorch!
 * 
 * Compile: gcc -O2 -o 01_soa_vs_aos 01_soa_vs_aos.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (10 * 1000 * 1000)

// Array of Structures (AoS)
typedef struct {
    float x, y, z, w;
} PointAoS;

// Structure of Arrays (SoA)
typedef struct {
    float* x;
    float* y;
    float* z;
    float* w;
} PointsSoA;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== DATA LAYOUT: AoS vs SoA ===\n\n");
    
    // Allocate AoS
    PointAoS* aos = malloc(N * sizeof(PointAoS));
    for (int i = 0; i < N; i++) {
        aos[i].x = i; aos[i].y = i; aos[i].z = i; aos[i].w = i;
    }
    
    // Allocate SoA
    PointsSoA soa;
    soa.x = malloc(N * sizeof(float));
    soa.y = malloc(N * sizeof(float));
    soa.z = malloc(N * sizeof(float));
    soa.w = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        soa.x[i] = i; soa.y[i] = i; soa.z[i] = i; soa.w[i] = i;
    }
    
    double start;
    volatile float sum;
    
    // Access only X component (AoS)
    printf("--- ACCESS ONLY X COMPONENT ---\n");
    sum = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        sum += aos[i].x;  // Loads 16 bytes, uses 4
    }
    double aos_time = get_time() - start;
    
    // Access only X component (SoA)
    sum = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        sum += soa.x[i];  // Uses all loaded bytes
    }
    double soa_time = get_time() - start;
    
    printf("AoS (single field): %.2f ms\n", aos_time * 1000);
    printf("SoA (single field): %.2f ms\n", soa_time * 1000);
    printf("SoA is %.1fx faster (better cache use)\n\n", aos_time / soa_time);
    
    // Access ALL components
    printf("--- ACCESS ALL COMPONENTS ---\n");
    sum = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        sum += aos[i].x + aos[i].y + aos[i].z + aos[i].w;
    }
    aos_time = get_time() - start;
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < N; i++) {
        sum += soa.x[i] + soa.y[i] + soa.z[i] + soa.w[i];
    }
    soa_time = get_time() - start;
    
    printf("AoS (all fields): %.2f ms\n", aos_time * 1000);
    printf("SoA (all fields): %.2f ms\n", soa_time * 1000);
    printf("Similar when accessing all fields\n");
    
    printf("\n=== TENSOR LAYOUT IN PYTORCH ===\n");
    printf("NCHW (batch, channel, height, width):\n");
    printf("  - Channels are contiguous\n");
    printf("  - Good for per-channel operations\n");
    printf("  - Default in PyTorch\n\n");
    printf("NHWC (batch, height, width, channel):\n");
    printf("  - Spatial positions are contiguous\n");
    printf("  - Better for convolutions (Tensor Cores)\n");
    printf("  - channels_last memory format\n\n");
    printf("torch.contiguous() rearranges to match stride order\n");
    
    free(aos);
    free(soa.x); free(soa.y); free(soa.z); free(soa.w);
    
    return 0;
}
