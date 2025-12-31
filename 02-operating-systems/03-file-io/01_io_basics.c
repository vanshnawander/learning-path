/**
 * 01_io_basics.c - File I/O: Buffered vs Direct vs Async
 * 
 * Understanding I/O is critical for data loading performance.
 * Different I/O methods have vastly different characteristics.
 * 
 * Compile: gcc -O2 -o 01_io_basics 01_io_basics.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define FILE_SIZE (100 * 1024 * 1024)  // 100 MB
#define BLOCK_SIZE (4 * 1024)          // 4 KB blocks

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== FILE I/O METHODS ===\n\n");
    
    const char* path = "/tmp/io_test.bin";
    char* buffer = aligned_alloc(4096, BLOCK_SIZE);
    memset(buffer, 'A', BLOCK_SIZE);
    
    // Create test file
    printf("Creating %d MB test file...\n", FILE_SIZE / (1024*1024));
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    for (int i = 0; i < FILE_SIZE / BLOCK_SIZE; i++) {
        write(fd, buffer, BLOCK_SIZE);
    }
    fsync(fd);
    close(fd);
    
    double start, elapsed;
    
    // Method 1: Standard read() with small buffer
    printf("\n--- METHOD 1: SMALL BUFFER READS ---\n");
    fd = open(path, O_RDONLY);
    
    start = get_time();
    size_t total = 0;
    while (read(fd, buffer, BLOCK_SIZE) > 0) {
        total += BLOCK_SIZE;
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("Read %zu bytes in %.2f ms\n", total, elapsed * 1000);
    printf("Bandwidth: %.2f MB/s\n", (total / 1e6) / elapsed);
    printf("System calls: %d\n", FILE_SIZE / BLOCK_SIZE);
    
    // Method 2: Large buffer reads
    printf("\n--- METHOD 2: LARGE BUFFER READS ---\n");
    size_t large_size = 1024 * 1024;  // 1 MB
    char* large_buffer = malloc(large_size);
    
    fd = open(path, O_RDONLY);
    start = get_time();
    total = 0;
    ssize_t n;
    while ((n = read(fd, large_buffer, large_size)) > 0) {
        total += n;
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("Read %zu bytes in %.2f ms\n", total, elapsed * 1000);
    printf("Bandwidth: %.2f MB/s\n", (total / 1e6) / elapsed);
    printf("System calls: %d\n", FILE_SIZE / (int)large_size);
    
    // Method 3: stdio buffered I/O
    printf("\n--- METHOD 3: STDIO BUFFERED (fread) ---\n");
    FILE* fp = fopen(path, "rb");
    
    start = get_time();
    total = 0;
    while (fread(buffer, 1, BLOCK_SIZE, fp) > 0) {
        total += BLOCK_SIZE;
    }
    elapsed = get_time() - start;
    fclose(fp);
    
    printf("Read %zu bytes in %.2f ms\n", total, elapsed * 1000);
    printf("Bandwidth: %.2f MB/s\n", (total / 1e6) / elapsed);
    printf("stdio handles buffering internally\n");
    
    printf("\n=== I/O FOR ML DATA LOADING ===\n");
    printf("1. Small reads = many syscalls = slow\n");
    printf("2. Large sequential reads = good\n");
    printf("3. mmap avoids read() syscall entirely\n");
    printf("4. Random small reads = worst case\n");
    printf("5. FFCV: mmap + sequential-ish access\n");
    
    free(buffer);
    free(large_buffer);
    unlink(path);
    
    return 0;
}
