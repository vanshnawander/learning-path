/**
 * 01_fork_basics.c - Process Creation with fork()
 * 
 * Understanding fork() is essential for:
 * - PyTorch DataLoader multiprocessing
 * - Why tensor sharing between workers is tricky
 * - Copy-on-write semantics
 * 
 * Compile: gcc -o 01_fork_basics 01_fork_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <string.h>

int main() {
    printf("=== FORK: PROCESS CREATION ===\n\n");
    
    printf("Parent PID: %d\n\n", getpid());
    
    // Simple fork
    printf("--- BASIC FORK ---\n");
    
    int x = 100;
    
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork failed");
        return 1;
    } else if (pid == 0) {
        // Child process
        printf("Child (PID %d): x = %d\n", getpid(), x);
        x = 200;
        printf("Child modified x to %d\n", x);
        exit(0);
    } else {
        // Parent process
        wait(NULL);  // Wait for child
        printf("Parent: x = %d (unchanged!)\n", x);
    }
    
    printf("\n--- COPY-ON-WRITE ---\n");
    printf("fork() doesn't copy memory immediately!\n");
    printf("Pages are marked read-only and shared.\n");
    printf("Only when written, a copy is made (COW).\n");
    printf("This is why fork() is fast even with large memory.\n");
    
    // Shared memory between processes
    printf("\n--- SHARED MEMORY ---\n");
    
    // Allocate shared memory
    int* shared = mmap(NULL, sizeof(int), 
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *shared = 0;
    
    pid = fork();
    
    if (pid == 0) {
        // Child increments shared counter
        for (int i = 0; i < 1000; i++) {
            (*shared)++;
        }
        printf("Child added 1000\n");
        exit(0);
    } else {
        // Parent also increments
        for (int i = 0; i < 1000; i++) {
            (*shared)++;
        }
        wait(NULL);
        printf("Shared counter: %d (may be < 2000 due to races!)\n", *shared);
    }
    
    munmap(shared, sizeof(int));
    
    printf("\n=== PYTORCH DATALOADER CONNECTION ===\n");
    printf("1. DataLoader uses fork() to create worker processes\n");
    printf("2. Each worker loads different samples\n");
    printf("3. Data is sent back via multiprocessing.Queue\n");
    printf("4. num_workers=0 means no fork, runs in main process\n");
    printf("5. fork() after CUDA init can cause issues!\n");
    printf("   → Use 'spawn' or 'forkserver' start methods\n");
    
    return 0;
}
