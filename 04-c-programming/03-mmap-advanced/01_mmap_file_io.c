/**
 * 01_mmap_file_io.c - Advanced Memory Mapping for Data Loading
 * 
 * mmap is THE technique for fast data loading:
 * - FFCV uses it for .beton files
 * - Database engines use it
 * - Shared memory between processes
 * 
 * Compile: gcc -O2 -o mmap_file 01_mmap_file_io.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#define FILE_SIZE (100 * 1024 * 1024)  // 100 MB

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== ADVANCED MEMORY MAPPING ===\n\n");
    
    const char* filename = "/tmp/mmap_test.bin";
    
    // ========================================================
    // Create test file
    // ========================================================
    printf("Creating %d MB test file...\n", FILE_SIZE / (1024*1024));
    
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }
    
    // Extend file to desired size
    if (ftruncate(fd, FILE_SIZE) < 0) {
        perror("ftruncate");
        return 1;
    }
    
    // Write some data
    float* write_buf = malloc(FILE_SIZE);
    for (int i = 0; i < FILE_SIZE / sizeof(float); i++) {
        write_buf[i] = (float)i;
    }
    pwrite(fd, write_buf, FILE_SIZE, 0);
    free(write_buf);
    
    printf("Done.\n\n");
    
    // ========================================================
    // Method 1: Traditional read()
    // ========================================================
    printf("--- METHOD 1: TRADITIONAL read() ---\n");
    
    float* read_buf = malloc(FILE_SIZE);
    
    double start = get_time();
    lseek(fd, 0, SEEK_SET);
    read(fd, read_buf, FILE_SIZE);
    double read_time = get_time() - start;
    
    // Access data
    volatile float sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE / sizeof(float); i += 16) {
        sum += read_buf[i];
    }
    double access_time = get_time() - start;
    
    printf("read() time:   %.2f ms\n", read_time * 1000);
    printf("Access time:   %.2f ms\n", access_time * 1000);
    printf("Total:         %.2f ms\n\n", (read_time + access_time) * 1000);
    
    free(read_buf);
    
    // ========================================================
    // Method 2: mmap with different flags
    // ========================================================
    printf("--- METHOD 2: MMAP VARIATIONS ---\n\n");
    
    // 2a: MAP_PRIVATE (copy-on-write)
    start = get_time();
    float* priv = mmap(NULL, FILE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    double mmap_time = get_time() - start;
    
    if (priv == MAP_FAILED) { perror("mmap"); return 1; }
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE / sizeof(float); i += 16) {
        sum += priv[i];  // First access triggers page faults
    }
    access_time = get_time() - start;
    
    printf("MAP_PRIVATE:\n");
    printf("  mmap() call: %.4f ms (just sets up mapping)\n", mmap_time * 1000);
    printf("  First access: %.2f ms (page faults load data)\n", access_time * 1000);
    
    // Second access (data now in page cache)
    sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE / sizeof(float); i += 16) {
        sum += priv[i];
    }
    double cached_time = get_time() - start;
    printf("  Cached access: %.2f ms\n\n", cached_time * 1000);
    
    munmap(priv, FILE_SIZE);
    
    // 2b: MAP_POPULATE (prefault all pages)
    start = get_time();
    float* pop = mmap(NULL, FILE_SIZE, PROT_READ, 
                      MAP_PRIVATE | MAP_POPULATE, fd, 0);
    mmap_time = get_time() - start;
    
    if (pop == MAP_FAILED) { perror("mmap populate"); return 1; }
    
    printf("MAP_POPULATE:\n");
    printf("  mmap() call: %.2f ms (loads ALL pages)\n", mmap_time * 1000);
    
    sum = 0;
    start = get_time();
    for (int i = 0; i < FILE_SIZE / sizeof(float); i += 16) {
        sum += pop[i];
    }
    access_time = get_time() - start;
    printf("  Access time: %.2f ms (already loaded)\n\n", access_time * 1000);
    
    munmap(pop, FILE_SIZE);
    
    // ========================================================
    // Method 3: madvise hints
    // ========================================================
    printf("--- METHOD 3: MADVISE HINTS ---\n\n");
    
    float* adv = mmap(NULL, FILE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    
    printf("MADV_SEQUENTIAL: Tell kernel we read sequentially\n");
    madvise(adv, FILE_SIZE, MADV_SEQUENTIAL);
    printf("  Kernel will prefetch ahead, drop behind\n\n");
    
    printf("MADV_RANDOM: Tell kernel we access randomly\n");
    printf("  Disables prefetching\n\n");
    
    printf("MADV_WILLNEED: Prefetch these pages now\n");
    madvise(adv, FILE_SIZE, MADV_WILLNEED);
    printf("  Triggers async read of pages\n\n");
    
    printf("MADV_DONTNEED: We're done with these pages\n");
    printf("  Allows kernel to reclaim memory\n\n");
    
    munmap(adv, FILE_SIZE);
    
    // ========================================================
    // Cleanup
    // ========================================================
    close(fd);
    unlink(filename);
    
    // ========================================================
    // Summary
    // ========================================================
    printf("=== MMAP FOR ML DATA LOADING ===\n\n");
    
    printf("FFCV .beton Strategy:\n");
    printf("1. mmap entire file (fast, just sets up mapping)\n");
    printf("2. Read index to get sample offsets\n");
    printf("3. Access samples via pointer arithmetic\n");
    printf("4. OS handles caching automatically\n");
    printf("5. Quasi-random access for locality\n\n");
    
    printf("Why mmap beats read():\n");
    printf("- No user/kernel buffer copy\n");
    printf("- No syscall per access\n");
    printf("- Automatic caching by OS\n");
    printf("- Shared between processes\n");
    printf("- Random access is O(1)\n");
    
    return 0;
}
