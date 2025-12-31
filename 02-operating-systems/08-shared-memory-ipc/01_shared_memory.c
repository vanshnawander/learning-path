/**
 * 01_shared_memory.c - Shared Memory for Inter-Process Communication
 * 
 * Shared memory is how PyTorch DataLoader workers communicate!
 * Zero-copy data sharing between processes.
 * 
 * Compile: gcc -O2 -o 01_shared_memory 01_shared_memory.c -lrt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define SHM_NAME "/pytorch_style_shm"
#define SHM_SIZE (1024 * 1024)  // 1 MB

int main() {
    printf("=== SHARED MEMORY FOR IPC ===\n\n");
    
    // Create shared memory object
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { perror("shm_open"); return 1; }
    
    // Set size
    ftruncate(shm_fd, SHM_SIZE);
    
    // Map into address space
    float* shared_data = mmap(NULL, SHM_SIZE, 
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED, shm_fd, 0);
    if (shared_data == MAP_FAILED) { perror("mmap"); return 1; }
    
    printf("Shared memory created: %s (%d bytes)\n", SHM_NAME, SHM_SIZE);
    printf("Address: %p\n\n", shared_data);
    
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process (like a DataLoader worker)
        printf("Worker: Loading data into shared memory...\n");
        
        // Simulate loading a batch
        int batch_size = SHM_SIZE / sizeof(float);
        for (int i = 0; i < batch_size; i++) {
            shared_data[i] = (float)i / batch_size;
        }
        
        printf("Worker: Wrote %d floats\n", batch_size);
        exit(0);
        
    } else {
        // Parent process (main training process)
        wait(NULL);  // Wait for worker
        
        printf("Main: Reading from shared memory...\n");
        
        // Verify data
        float sum = 0;
        int batch_size = SHM_SIZE / sizeof(float);
        for (int i = 0; i < batch_size; i++) {
            sum += shared_data[i];
        }
        
        printf("Main: Sum of %d floats = %.2f\n", batch_size, sum);
        printf("Main: NO COPY HAPPENED - zero-copy!\n");
    }
    
    // Cleanup
    munmap(shared_data, SHM_SIZE);
    shm_unlink(SHM_NAME);
    
    printf("\n=== PYTORCH DATALOADER ===\n");
    printf("1. Workers load data into shared memory\n");
    printf("2. Main process reads directly - no copy\n");
    printf("3. multiprocessing.Queue uses shared memory\n");
    printf("4. torch.multiprocessing adds tensor support\n");
    printf("5. pin_memory=True for faster GPU transfer\n");
    
    return 0;
}
