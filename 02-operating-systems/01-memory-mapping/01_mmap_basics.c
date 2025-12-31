/**
 * 01_mmap_basics.c - Memory Mapping: The Foundation of FFCV
 * 
 * mmap() is THE key system call for efficient data loading:
 * - FFCV .beton files are memory-mapped
 * - Avoids read() system call overhead
 * - OS handles caching automatically
 * - Enables zero-copy data access
 * 
 * Compile: gcc -O2 -o 01_mmap_basics 01_mmap_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define FILE_SIZE (100 * 1024 * 1024)  // 100 MB

int main() {
    printf("=== MEMORY MAPPING: HOW FFCV LOADS DATA ===\n\n");
    
    // Create a test file
    const char* filename = "/tmp/mmap_test.bin";
    
    printf("Creating %d MB test file...\n", FILE_SIZE / (1024*1024));
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }
    
    // Allocate and write data
    char* write_buf = malloc(FILE_SIZE);
    for (int i = 0; i < FILE_SIZE; i++) {
        write_buf[i] = (char)(i % 256);
    }
    write(fd, write_buf, FILE_SIZE);
    free(write_buf);
    
    // METHOD 1: Traditional read()
    printf("\n--- METHOD 1: TRADITIONAL read() ---\n");
    
    lseek(fd, 0, SEEK_SET);
    char* read_buf = malloc(FILE_SIZE);
    
    double start = get_time();
    ssize_t bytes = read(fd, read_buf, FILE_SIZE);
    double read_time = get_time() - start;
    
    printf("read() %zd bytes in %.2f ms\n", bytes, read_time * 1000);
    printf("Bandwidth: %.2f GB/s\n", (FILE_SIZE / 1e9) / read_time);
    
    // Access data
    volatile long sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE; i += 64) {
        sum += read_buf[i];
    }
    double access_time = get_time() - start;
    printf("Access time: %.2f ms\n", access_time * 1000);
    
    free(read_buf);
    
    // METHOD 2: Memory mapping
    printf("\n--- METHOD 2: MEMORY MAPPING (mmap) ---\n");
    
    start = get_time();
    char* mapped = mmap(NULL, FILE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    double mmap_time = get_time() - start;
    
    if (mapped == MAP_FAILED) { perror("mmap"); return 1; }
    
    printf("mmap() call: %.4f ms (just sets up mapping!)\n", mmap_time * 1000);
    
    // Access data (this is when pages are actually loaded)
    sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE; i += 64) {
        sum += mapped[i];  // Page fault loads data on demand
    }
    access_time = get_time() - start;
    printf("First access (page faults): %.2f ms\n", access_time * 1000);
    
    // Second access (data is now in page cache)
    sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE; i += 64) {
        sum += mapped[i];
    }
    access_time = get_time() - start;
    printf("Second access (cached): %.2f ms\n", access_time * 1000);
    
    // Advise the kernel about access pattern
    printf("\n--- MADVISE: HINTS TO THE KERNEL ---\n");
    
    madvise(mapped, FILE_SIZE, MADV_SEQUENTIAL);
    printf("MADV_SEQUENTIAL: Tell kernel we'll read sequentially\n");
    printf("  → Kernel will prefetch ahead, drop behind\n");
    
    madvise(mapped, FILE_SIZE, MADV_WILLNEED);
    printf("MADV_WILLNEED: Pre-fault all pages now\n");
    
    munmap(mapped, FILE_SIZE);
    close(fd);
    unlink(filename);
    
    printf("\n=== WHY MMAP FOR ML DATA LOADING ===\n");
    printf("1. Zero-copy: Data goes directly to user space\n");
    printf("2. Lazy loading: Only load pages you actually access\n");
    printf("3. OS caching: Automatic, intelligent caching\n");
    printf("4. Shared memory: Multiple processes share same pages\n");
    printf("5. Random access: O(1) access to any offset\n");
    
    printf("\n=== FFCV .beton FORMAT ===\n");
    printf("- Header at known offset (mmap + seek)\n");
    printf("- Sample index for O(1) lookup\n");
    printf("- Data blocks memory-mapped\n");
    printf("- Quasi-random access within blocks\n");
    
    return 0;
}
