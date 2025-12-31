/**
 * 06_c_mmap_loader.c - Pure C Memory-Mapped Data Loader
 * 
 * The foundation of all fast data loaders. This is what FFCV does.
 * Every operation is timed.
 * 
 * Compile: gcc -O3 -o mmap_loader 06_c_mmap_loader.c -lpthread -lrt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ============================================================
// TIMING
// ============================================================

typedef struct {
    struct timespec start;
} Timer;

void timer_start(Timer* t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_stop_ms(Timer* t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - t->start.tv_sec) * 1000.0 +
           (end.tv_nsec - t->start.tv_nsec) / 1e6;
}

// ============================================================
// DATASET FILE FORMAT (like .beton)
// ============================================================

// Header structure at start of file
typedef struct {
    uint32_t magic;          // File identifier
    uint32_t version;        // Format version
    uint64_t num_samples;    // Total samples
    uint64_t sample_size;    // Bytes per sample
    uint64_t data_offset;    // Offset to data start
    uint64_t index_offset;   // Offset to sample index
} DatasetHeader;

#define MAGIC 0x4D4C4454  // "MLDT"

// ============================================================
// MEMORY-MAPPED DATASET
// ============================================================

typedef struct {
    void* mapped_data;       // mmap'd file
    size_t file_size;        // Total file size
    DatasetHeader header;    // Parsed header
    uint64_t* index;         // Sample offsets
    int fd;                  // File descriptor
} MmapDataset;

// Create dataset file (preparation step)
int create_dataset_file(const char* path, size_t num_samples, size_t sample_size) {
    printf("Creating dataset file: %s\n", path);
    printf("  Samples: %zu\n", num_samples);
    printf("  Sample size: %zu bytes\n", sample_size);
    
    Timer t;
    timer_start(&t);
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("fopen");
        return -1;
    }
    
    // Write header
    DatasetHeader header = {
        .magic = MAGIC,
        .version = 1,
        .num_samples = num_samples,
        .sample_size = sample_size,
        .data_offset = sizeof(DatasetHeader) + num_samples * sizeof(uint64_t),
        .index_offset = sizeof(DatasetHeader)
    };
    fwrite(&header, sizeof(header), 1, f);
    
    // Write index (offsets to each sample)
    for (size_t i = 0; i < num_samples; i++) {
        uint64_t offset = header.data_offset + i * sample_size;
        fwrite(&offset, sizeof(offset), 1, f);
    }
    
    // Write sample data (random for this example)
    uint8_t* sample = aligned_alloc(64, sample_size);
    for (size_t i = 0; i < num_samples; i++) {
        // Generate random sample data
        for (size_t j = 0; j < sample_size; j++) {
            sample[j] = rand() % 256;
        }
        fwrite(sample, 1, sample_size, f);
        
        if (i % 10000 == 0 && i > 0) {
            printf("  Written %zu samples...\n", i);
        }
    }
    free(sample);
    
    fclose(f);
    
    double elapsed = timer_stop_ms(&t);
    size_t total_size = header.data_offset + num_samples * sample_size;
    printf("  Total size: %.2f MB\n", total_size / (1024.0 * 1024.0));
    printf("  Write time: %.2f ms\n", elapsed);
    printf("  Write speed: %.2f GB/s\n", total_size / (elapsed / 1000.0) / 1e9);
    
    return 0;
}

// Open dataset with mmap
MmapDataset* open_dataset(const char* path) {
    printf("\nOpening dataset: %s\n", path);
    
    Timer t;
    timer_start(&t);
    
    MmapDataset* ds = calloc(1, sizeof(MmapDataset));
    
#ifdef __linux__
    // Open file
    ds->fd = open(path, O_RDONLY);
    if (ds->fd < 0) {
        perror("open");
        free(ds);
        return NULL;
    }
    
    // Get file size
    struct stat st;
    fstat(ds->fd, &st);
    ds->file_size = st.st_size;
    
    // Memory map entire file
    ds->mapped_data = mmap(NULL, ds->file_size, PROT_READ, 
                           MAP_SHARED | MAP_POPULATE, ds->fd, 0);
    if (ds->mapped_data == MAP_FAILED) {
        perror("mmap");
        close(ds->fd);
        free(ds);
        return NULL;
    }
    
    // Advise kernel about random access pattern
    madvise(ds->mapped_data, ds->file_size, MADV_RANDOM);
    
    // Parse header
    memcpy(&ds->header, ds->mapped_data, sizeof(DatasetHeader));
    
    // Check magic
    if (ds->header.magic != MAGIC) {
        printf("Invalid file format!\n");
        munmap(ds->mapped_data, ds->file_size);
        close(ds->fd);
        free(ds);
        return NULL;
    }
    
    // Get index pointer
    ds->index = (uint64_t*)((uint8_t*)ds->mapped_data + ds->header.index_offset);
    
#else
    printf("mmap not available, using fallback\n");
    // Fallback: read entire file into memory
    FILE* f = fopen(path, "rb");
    if (!f) {
        free(ds);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    ds->file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    ds->mapped_data = aligned_alloc(64, ds->file_size);
    fread(ds->mapped_data, 1, ds->file_size, f);
    fclose(f);
    
    memcpy(&ds->header, ds->mapped_data, sizeof(DatasetHeader));
    ds->index = (uint64_t*)((uint8_t*)ds->mapped_data + ds->header.index_offset);
#endif
    
    double elapsed = timer_stop_ms(&t);
    printf("  Samples: %lu\n", ds->header.num_samples);
    printf("  Sample size: %lu bytes\n", ds->header.sample_size);
    printf("  File size: %.2f MB\n", ds->file_size / (1024.0 * 1024.0));
    printf("  Open time: %.2f ms\n", elapsed);
    
    return ds;
}

// Get sample (zero-copy with mmap!)
const void* get_sample(MmapDataset* ds, size_t idx) {
    if (idx >= ds->header.num_samples) return NULL;
    return (uint8_t*)ds->mapped_data + ds->index[idx];
}

// Close dataset
void close_dataset(MmapDataset* ds) {
    if (!ds) return;
    
#ifdef __linux__
    if (ds->mapped_data && ds->mapped_data != MAP_FAILED) {
        munmap(ds->mapped_data, ds->file_size);
    }
    if (ds->fd >= 0) {
        close(ds->fd);
    }
#else
    free(ds->mapped_data);
#endif
    
    free(ds);
}

// ============================================================
// BATCH LOADING
// ============================================================

typedef struct {
    float* data;
    int* labels;
    size_t batch_size;
    size_t sample_size;
} Batch;

Batch* create_batch(size_t batch_size, size_t sample_size) {
    Batch* b = malloc(sizeof(Batch));
    b->batch_size = batch_size;
    b->sample_size = sample_size;
    b->data = aligned_alloc(64, batch_size * sample_size);
    b->labels = malloc(batch_size * sizeof(int));
    return b;
}

void free_batch(Batch* b) {
    free(b->data);
    free(b->labels);
    free(b);
}

// Load batch with timing
double load_batch(MmapDataset* ds, size_t* indices, size_t batch_size, Batch* batch) {
    Timer t;
    timer_start(&t);
    
    for (size_t i = 0; i < batch_size; i++) {
        const void* sample = get_sample(ds, indices[i]);
        // Copy to batch (simulate decode/transform)
        memcpy((uint8_t*)batch->data + i * ds->header.sample_size,
               sample, ds->header.sample_size);
        batch->labels[i] = indices[i] % 1000;
    }
    
    return timer_stop_ms(&t);
}

// ============================================================
// BENCHMARKS
// ============================================================

void benchmark_random_access(MmapDataset* ds) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RANDOM ACCESS BENCHMARK                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t num_accesses = 10000;
    size_t* indices = malloc(num_accesses * sizeof(size_t));
    
    // Generate random indices
    for (size_t i = 0; i < num_accesses; i++) {
        indices[i] = rand() % ds->header.num_samples;
    }
    
    Timer t;
    
    // Cold access (no page cache)
    printf("Cold random access (%zu samples):\n", num_accesses);
    timer_start(&t);
    volatile uint8_t sum = 0;
    for (size_t i = 0; i < num_accesses; i++) {
        const uint8_t* sample = get_sample(ds, indices[i]);
        sum += sample[0];  // Touch first byte
    }
    double cold_time = timer_stop_ms(&t);
    printf("  Time: %.2f ms\n", cold_time);
    printf("  Throughput: %.0f samples/sec\n", num_accesses / (cold_time / 1000.0));
    
    // Hot access (in page cache)
    printf("\nHot random access (same indices):\n");
    timer_start(&t);
    for (size_t i = 0; i < num_accesses; i++) {
        const uint8_t* sample = get_sample(ds, indices[i]);
        sum += sample[0];
    }
    double hot_time = timer_stop_ms(&t);
    printf("  Time: %.2f ms\n", hot_time);
    printf("  Throughput: %.0f samples/sec\n", num_accesses / (hot_time / 1000.0));
    printf("  Speedup: %.1fx (page cache effect)\n", cold_time / hot_time);
    
    free(indices);
}

void benchmark_batch_loading(MmapDataset* ds) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  BATCH LOADING BENCHMARK                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    size_t batch_sizes[] = {16, 32, 64, 128, 256};
    int num_sizes = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    printf("%-10s %-15s %-15s %-15s\n", "Batch", "Time (ms)", "Samples/sec", "Bandwidth");
    printf("─────────────────────────────────────────────────────────────\n");
    
    for (int b = 0; b < num_sizes; b++) {
        size_t batch_size = batch_sizes[b];
        Batch* batch = create_batch(batch_size, ds->header.sample_size);
        
        // Generate indices
        size_t* indices = malloc(batch_size * sizeof(size_t));
        for (size_t i = 0; i < batch_size; i++) {
            indices[i] = rand() % ds->header.num_samples;
        }
        
        // Warmup
        load_batch(ds, indices, batch_size, batch);
        
        // Benchmark
        int iterations = 100;
        Timer t;
        timer_start(&t);
        for (int i = 0; i < iterations; i++) {
            load_batch(ds, indices, batch_size, batch);
        }
        double elapsed = timer_stop_ms(&t);
        
        double avg_time = elapsed / iterations;
        double samples_per_sec = batch_size * iterations / (elapsed / 1000.0);
        double bandwidth = samples_per_sec * ds->header.sample_size / 1e9;
        
        printf("%-10zu %-15.3f %-15.0f %.2f GB/s\n",
               batch_size, avg_time, samples_per_sec, bandwidth);
        
        free(indices);
        free_batch(batch);
    }
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█  C MEMORY-MAPPED DATA LOADER                                 █\n");
    printf("█  The foundation of fast ML data loading                      █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    const char* dataset_path = "/tmp/test_dataset.bin";
    size_t num_samples = 50000;
    size_t sample_size = 224 * 224 * 3;  // RGB image size
    
    // Create test dataset
    printf("\n─── Creating Test Dataset ───\n");
    if (create_dataset_file(dataset_path, num_samples, sample_size) < 0) {
        return 1;
    }
    
    // Open with mmap
    printf("\n─── Opening Dataset ───\n");
    MmapDataset* ds = open_dataset(dataset_path);
    if (!ds) {
        return 1;
    }
    
    // Run benchmarks
    benchmark_random_access(ds);
    benchmark_batch_loading(ds);
    
    // Cleanup
    close_dataset(ds);
    remove(dataset_path);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAYS                                               ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. MMAP IS ZERO-COPY                                        ║\n");
    printf("║     • No explicit read() calls                               ║\n");
    printf("║     • Data accessed via pointer arithmetic                   ║\n");
    printf("║     • OS handles page faults and caching                     ║\n");
    printf("║                                                              ║\n");
    printf("║  2. SINGLE FILE >> MANY FILES                                ║\n");
    printf("║     • One mmap setup vs thousands of opens                   ║\n");
    printf("║     • Better sequential I/O when possible                    ║\n");
    printf("║                                                              ║\n");
    printf("║  3. PAGE CACHE HELPS REPEATED ACCESS                         ║\n");
    printf("║     • Hot data is in memory                                  ║\n");
    printf("║     • But first access is disk-speed                         ║\n");
    printf("║                                                              ║\n");
    printf("║  4. USE MADVISE FOR HINTS                                    ║\n");
    printf("║     • MADV_RANDOM for shuffled access                        ║\n");
    printf("║     • MADV_SEQUENTIAL for streaming                          ║\n");
    printf("║     • MADV_WILLNEED for prefetch                             ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
