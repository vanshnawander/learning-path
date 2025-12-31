/**
 * 02_shared_tensor.c - Sharing Tensors Between Processes
 * 
 * This is how PyTorch DataLoader workers share data with main process.
 * Zero-copy tensor sharing using shared memory.
 * 
 * Compile: gcc -O2 -o shared_tensor 02_shared_tensor.c -lrt -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>

#define SHM_NAME "/tensor_shm"

// Simple tensor structure
typedef struct {
    int ndim;
    int shape[4];
    int strides[4];
    size_t data_offset;  // Offset to actual data
    size_t nbytes;
} TensorHeader;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== SHARED MEMORY TENSOR ===\n\n");
    
    // Tensor: 32 x 3 x 224 x 224 (batch of images)
    int batch = 32, channels = 3, height = 224, width = 224;
    size_t tensor_size = batch * channels * height * width * sizeof(float);
    size_t total_size = sizeof(TensorHeader) + tensor_size;
    
    printf("Creating shared tensor: [%d, %d, %d, %d]\n", 
           batch, channels, height, width);
    printf("Size: %.2f MB\n\n", tensor_size / (1024.0 * 1024.0));
    
    // Create shared memory
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { perror("shm_open"); return 1; }
    
    ftruncate(shm_fd, total_size);
    
    // Map shared memory
    void* shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) { perror("mmap"); return 1; }
    
    // Setup tensor header
    TensorHeader* header = (TensorHeader*)shm_ptr;
    header->ndim = 4;
    header->shape[0] = batch;
    header->shape[1] = channels;
    header->shape[2] = height;
    header->shape[3] = width;
    
    // Row-major strides
    header->strides[3] = 1;
    header->strides[2] = width;
    header->strides[1] = height * width;
    header->strides[0] = channels * height * width;
    
    header->data_offset = sizeof(TensorHeader);
    header->nbytes = tensor_size;
    
    // Pointer to tensor data
    float* tensor_data = (float*)((char*)shm_ptr + header->data_offset);
    
    // Fork worker process
    pid_t pid = fork();
    
    if (pid == 0) {
        // WORKER PROCESS (like DataLoader worker)
        printf("[Worker] Loading data into shared tensor...\n");
        
        double start = get_time();
        
        // Simulate loading images
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * header->strides[0] +
                                  c * header->strides[1] +
                                  h * header->strides[2] +
                                  w * header->strides[3];
                        tensor_data[idx] = (float)(b * 1000 + c * 100 + h + w) / 10000.0f;
                    }
                }
            }
        }
        
        double elapsed = get_time() - start;
        printf("[Worker] Loaded in %.2f ms\n", elapsed * 1000);
        
        exit(0);
        
    } else {
        // MAIN PROCESS (training loop)
        wait(NULL);  // Wait for worker
        
        printf("[Main] Reading shared tensor...\n");
        
        double start = get_time();
        
        // Verify data (no copy needed!)
        volatile float sum = 0;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * header->strides[0] +
                                  c * header->strides[1] +
                                  h * header->strides[2] +
                                  w * header->strides[3];
                        sum += tensor_data[idx];
                    }
                }
            }
        }
        
        double elapsed = get_time() - start;
        printf("[Main] Verified in %.2f ms, sum = %.2f\n", 
               elapsed * 1000, sum);
        printf("[Main] NO COPY - accessed worker's data directly!\n");
    }
    
    // Cleanup
    munmap(shm_ptr, total_size);
    shm_unlink(SHM_NAME);
    
    printf("\n=== PYTORCH DATALOADER CONNECTION ===\n\n");
    
    printf("PyTorch does exactly this:\n\n");
    printf("1. multiprocessing.Queue with shared memory\n");
    printf("2. Workers load tensors into shared memory\n");
    printf("3. Main process gets pointer, no copy\n");
    printf("4. torch.multiprocessing handles tensor serialization\n\n");
    
    printf("Key PyTorch APIs:\n");
    printf("  tensor.share_memory_() - move to shared memory\n");
    printf("  tensor.is_shared() - check if shared\n");
    printf("  torch.multiprocessing.Queue - shared memory queue\n");
    
    return 0;
}
