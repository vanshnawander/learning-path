/**
 * 02_mmap_dataloader.c - Building a Simple mmap-based DataLoader
 * 
 * This shows the core concept behind FFCV's fast data loading.
 * We build a minimal dataset format and loader from scratch.
 * 
 * Compile: gcc -O2 -o 02_mmap_dataloader 02_mmap_dataloader.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <time.h>

// Simple dataset format (like a minimal .beton)
typedef struct {
    uint32_t magic;           // Magic number for validation
    uint32_t num_samples;     // Number of samples
    uint32_t sample_size;     // Size of each sample in bytes
    uint32_t header_size;     // Size of this header
} DatasetHeader;

typedef struct {
    int fd;
    size_t file_size;
    void* mapped_data;
    DatasetHeader* header;
    void* samples;  // Pointer to start of sample data
} MappedDataset;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Create a dataset file
void create_dataset(const char* path, int num_samples, int sample_size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    
    DatasetHeader header = {
        .magic = 0xBET0CAFE,
        .num_samples = num_samples,
        .sample_size = sample_size,
        .header_size = sizeof(DatasetHeader)
    };
    
    write(fd, &header, sizeof(header));
    
    // Write sample data
    float* sample = malloc(sample_size);
    for (int i = 0; i < num_samples; i++) {
        // Fill with recognizable pattern
        for (int j = 0; j < sample_size / sizeof(float); j++) {
            sample[j] = i * 1000 + j;
        }
        write(fd, sample, sample_size);
    }
    
    free(sample);
    close(fd);
}

// Open and mmap a dataset
MappedDataset* open_dataset(const char* path) {
    MappedDataset* ds = malloc(sizeof(MappedDataset));
    
    ds->fd = open(path, O_RDONLY);
    if (ds->fd < 0) { perror("open"); return NULL; }
    
    // Get file size
    ds->file_size = lseek(ds->fd, 0, SEEK_END);
    
    // Memory map the entire file
    ds->mapped_data = mmap(NULL, ds->file_size, PROT_READ, MAP_PRIVATE, ds->fd, 0);
    if (ds->mapped_data == MAP_FAILED) { perror("mmap"); return NULL; }
    
    // Parse header
    ds->header = (DatasetHeader*)ds->mapped_data;
    
    // Validate magic
    if (ds->header->magic != 0xBET0CAFE) {
        fprintf(stderr, "Invalid dataset file!\n");
        return NULL;
    }
    
    // Point to samples
    ds->samples = (char*)ds->mapped_data + ds->header->header_size;
    
    printf("Opened dataset: %u samples, %u bytes each\n",
           ds->header->num_samples, ds->header->sample_size);
    
    return ds;
}

// Get sample by index - O(1) access!
void* get_sample(MappedDataset* ds, uint32_t index) {
    if (index >= ds->header->num_samples) return NULL;
    return (char*)ds->samples + (size_t)index * ds->header->sample_size;
}

void close_dataset(MappedDataset* ds) {
    munmap(ds->mapped_data, ds->file_size);
    close(ds->fd);
    free(ds);
}

int main() {
    printf("=== MMAP-BASED DATALOADER FROM SCRATCH ===\n\n");
    
    const char* path = "/tmp/test_dataset.bin";
    int num_samples = 10000;
    int sample_size = 3 * 224 * 224 * sizeof(float);  // ImageNet-like
    
    // Create dataset
    printf("Creating dataset with %d samples of %d bytes each...\n",
           num_samples, sample_size);
    double start = get_time();
    create_dataset(path, num_samples, sample_size);
    printf("Created in %.2f ms\n\n", (get_time() - start) * 1000);
    
    // Open with mmap
    MappedDataset* ds = open_dataset(path);
    if (!ds) return 1;
    
    // Sequential access benchmark
    printf("\n--- SEQUENTIAL ACCESS ---\n");
    volatile float sum = 0;
    start = get_time();
    for (int i = 0; i < num_samples; i++) {
        float* sample = (float*)get_sample(ds, i);
        sum += sample[0];  // Just touch first element
    }
    double seq_time = get_time() - start;
    printf("Sequential: %.2f ms (%.0f samples/sec)\n", 
           seq_time * 1000, num_samples / seq_time);
    
    // Random access benchmark
    printf("\n--- RANDOM ACCESS ---\n");
    int* indices = malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; i++) indices[i] = i;
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < num_samples; i++) {
        float* sample = (float*)get_sample(ds, indices[i]);
        sum += sample[0];
    }
    double rand_time = get_time() - start;
    printf("Random: %.2f ms (%.0f samples/sec)\n",
           rand_time * 1000, num_samples / rand_time);
    
    printf("\nRandom is %.1fx slower than sequential\n", rand_time / seq_time);
    
    printf("\n=== KEY INSIGHTS ===\n");
    printf("1. mmap gives O(1) access to any sample\n");
    printf("2. But random access causes page faults\n");
    printf("3. FFCV's 'quasi-random' balances randomness and locality\n");
    printf("4. OS page cache makes repeated access fast\n");
    
    close_dataset(ds);
    free(indices);
    unlink(path);
    
    return 0;
}
