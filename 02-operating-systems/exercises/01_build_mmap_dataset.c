/**
 * EXERCISE: Build Your Own Memory-Mapped Dataset Format
 * 
 * Create a simple dataset format similar to FFCV's .beton:
 * 1. Header with metadata
 * 2. Index for O(1) sample lookup
 * 3. Contiguous sample data
 * 
 * Your task: Complete the TODOs
 * 
 * Compile: gcc -O2 -o mmap_dataset 01_build_mmap_dataset.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

#define MAGIC 0xDATA5E7F

// Dataset header
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_samples;
    uint32_t sample_size;
    uint64_t data_offset;  // Where sample data starts
} Header;

// TODO 1: Implement this function
// Create a dataset file with num_samples, each of sample_size bytes
void create_dataset(const char* path, int num_samples, int sample_size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); exit(1); }
    
    // Write header
    Header header = {
        .magic = MAGIC,
        .version = 1,
        .num_samples = num_samples,
        .sample_size = sample_size,
        .data_offset = sizeof(Header)
    };
    write(fd, &header, sizeof(header));
    
    // TODO: Write sample data
    // Each sample should contain recognizable data
    // e.g., sample i should have data that identifies it as sample i
    
    // YOUR CODE HERE
    
    close(fd);
    printf("Created dataset: %s\n", path);
}

// TODO 2: Implement this function
// Open dataset and return mmap'd pointer
void* open_dataset(const char* path, Header** header_out, size_t* size_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return NULL; }
    
    // Get file size
    *size_out = lseek(fd, 0, SEEK_END);
    
    // TODO: mmap the file
    // YOUR CODE HERE
    void* mapped = NULL;  // Replace with mmap call
    
    // Validate header
    *header_out = (Header*)mapped;
    if ((*header_out)->magic != MAGIC) {
        fprintf(stderr, "Invalid dataset!\n");
        return NULL;
    }
    
    return mapped;
}

// TODO 3: Implement this function
// Get pointer to sample i (O(1) access)
void* get_sample(void* mapped, Header* header, int index) {
    if (index >= header->num_samples) return NULL;
    
    // TODO: Calculate offset and return pointer
    // YOUR CODE HERE
    
    return NULL;  // Replace with correct pointer
}

int main() {
    printf("=== BUILD YOUR OWN DATASET FORMAT ===\n\n");
    
    const char* path = "/tmp/my_dataset.bin";
    int num_samples = 1000;
    int sample_size = 1024;  // 1KB per sample
    
    // Create dataset
    create_dataset(path, num_samples, sample_size);
    
    // Open and verify
    Header* header;
    size_t size;
    void* mapped = open_dataset(path, &header, &size);
    
    if (mapped) {
        printf("Opened: %d samples, %d bytes each\n", 
               header->num_samples, header->sample_size);
        
        // Test random access
        for (int i = 0; i < 5; i++) {
            int idx = rand() % num_samples;
            void* sample = get_sample(mapped, header, idx);
            if (sample) {
                printf("Sample %d: first byte = %d\n", idx, *(char*)sample);
            }
        }
        
        munmap(mapped, size);
    }
    
    unlink(path);
    
    printf("\n=== NEXT STEPS ===\n");
    printf("1. Implement the TODOs\n");
    printf("2. Add variable-length samples (requires index array)\n");
    printf("3. Add compression support\n");
    printf("4. Benchmark vs reading individual files\n");
    
    return 0;
}
