/**
 * 01_buffered_io.c - I/O Buffering Strategies
 * 
 * How you read/write data dramatically affects performance.
 * Critical for data loading pipelines.
 * 
 * Compile: gcc -O2 -o buffered_io 01_buffered_io.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define FILE_SIZE (50 * 1024 * 1024)  // 50 MB

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== I/O BUFFERING STRATEGIES ===\n\n");
    
    const char* filename = "/tmp/io_test.bin";
    
    // Create test file
    printf("Creating %d MB test file...\n\n", FILE_SIZE / (1024*1024));
    
    char* buffer = aligned_alloc(4096, FILE_SIZE);
    memset(buffer, 'X', FILE_SIZE);
    
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    write(fd, buffer, FILE_SIZE);
    fsync(fd);
    close(fd);
    
    double start, elapsed;
    
    // ========================================================
    // Test 1: Tiny reads (worst case)
    // ========================================================
    printf("--- TEST 1: TINY READS (1 byte) ---\n");
    
    fd = open(filename, O_RDONLY);
    char byte;
    int count = 0;
    int sample_size = 100000;  // Only read 100K bytes (too slow otherwise)
    
    start = get_time();
    for (int i = 0; i < sample_size; i++) {
        read(fd, &byte, 1);
        count++;
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("Read %d bytes one at a time\n", count);
    printf("Time: %.2f ms\n", elapsed * 1000);
    printf("Throughput: %.2f KB/s (TERRIBLE!)\n", count / elapsed / 1024);
    printf("Each read() is a syscall!\n\n");
    
    // ========================================================
    // Test 2: Small buffer
    // ========================================================
    printf("--- TEST 2: SMALL BUFFER (4KB) ---\n");
    
    fd = open(filename, O_RDONLY);
    size_t buf_size = 4096;
    char* small_buf = malloc(buf_size);
    size_t total = 0;
    
    start = get_time();
    ssize_t n;
    while ((n = read(fd, small_buf, buf_size)) > 0) {
        total += n;
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("Read %zu bytes with 4KB buffer\n", total);
    printf("Time: %.2f ms\n", elapsed * 1000);
    printf("Throughput: %.2f MB/s\n", total / elapsed / (1024*1024));
    printf("Syscalls: %zu\n\n", total / buf_size);
    
    free(small_buf);
    
    // ========================================================
    // Test 3: Large buffer
    // ========================================================
    printf("--- TEST 3: LARGE BUFFER (1MB) ---\n");
    
    fd = open(filename, O_RDONLY);
    buf_size = 1024 * 1024;
    char* large_buf = malloc(buf_size);
    total = 0;
    
    start = get_time();
    while ((n = read(fd, large_buf, buf_size)) > 0) {
        total += n;
    }
    elapsed = get_time() - start;
    close(fd);
    
    printf("Read %zu bytes with 1MB buffer\n", total);
    printf("Time: %.2f ms\n", elapsed * 1000);
    printf("Throughput: %.2f MB/s\n", total / elapsed / (1024*1024));
    printf("Syscalls: %zu\n\n", total / buf_size);
    
    free(large_buf);
    
    // ========================================================
    // Test 4: stdio buffering (fread)
    // ========================================================
    printf("--- TEST 4: STDIO BUFFERED (fread) ---\n");
    
    FILE* fp = fopen(filename, "rb");
    buf_size = 4096;
    small_buf = malloc(buf_size);
    total = 0;
    
    start = get_time();
    while ((n = fread(small_buf, 1, buf_size, fp)) > 0) {
        total += n;
    }
    elapsed = get_time() - start;
    fclose(fp);
    
    printf("Read %zu bytes with fread (4KB requests)\n", total);
    printf("Time: %.2f ms\n", elapsed * 1000);
    printf("Throughput: %.2f MB/s\n", total / elapsed / (1024*1024));
    printf("stdio buffers internally (default 8KB)\n\n");
    
    free(small_buf);
    
    // ========================================================
    // Summary
    // ========================================================
    printf("=== SUMMARY FOR DATA LOADING ===\n\n");
    
    printf("RECOMMENDATIONS:\n");
    printf("1. Never read 1 byte at a time!\n");
    printf("2. Buffer size 64KB-1MB is usually optimal\n");
    printf("3. Use mmap for random access\n");
    printf("4. Use large sequential reads for streaming\n");
    printf("5. Consider async I/O (io_uring) for overlap\n\n");
    
    printf("PYTORCH DATALOADER:\n");
    printf("- Workers read data in parallel\n");
    printf("- prefetch_factor controls lookahead\n");
    printf("- num_workers should match I/O parallelism\n\n");
    
    printf("FFCV:\n");
    printf("- mmap for .beton files\n");
    printf("- No read() syscalls during training\n");
    printf("- OS handles buffering automatically\n");
    
    free(buffer);
    unlink(filename);
    
    return 0;
}
