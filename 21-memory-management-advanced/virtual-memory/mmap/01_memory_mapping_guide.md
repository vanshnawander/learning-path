# Memory Mapping (mmap) Deep Dive

## What is mmap?

`mmap` creates a mapping between a file (or anonymous memory) and the process's virtual address space. The kernel handles paging data in/out transparently.

```
Process Virtual Address Space
┌────────────────────────────────────────┐
│            Stack                        │
├────────────────────────────────────────┤
│              ↓                          │
│                                         │
│              ↑                          │
├────────────────────────────────────────┤
│   Memory-mapped region (mmap)          │ ◄── File contents
├────────────────────────────────────────┤
│            Heap                         │
├────────────────────────────────────────┤
│           .data/.bss                    │
├────────────────────────────────────────┤
│            .text                        │
└────────────────────────────────────────┘
```

## Types of Mappings

### File-Backed vs Anonymous

```c
// File-backed mapping
int fd = open("data.bin", O_RDWR);
void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                 MAP_SHARED, fd, 0);

// Anonymous mapping (no file)
void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
```

### Shared vs Private

```c
// MAP_SHARED: Changes visible to other processes, written to file
void* shared = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);

// MAP_PRIVATE: Copy-on-write, changes private to process
void* private = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE, fd, 0);
```

## mmap System Call

```c
#include <sys/mman.h>

void *mmap(void *addr, size_t length, int prot, int flags,
           int fd, off_t offset);

// Parameters:
// addr   - Suggested address (usually NULL for kernel to choose)
// length - Size of mapping
// prot   - Protection: PROT_READ, PROT_WRITE, PROT_EXEC, PROT_NONE
// flags  - MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS, MAP_FIXED, etc.
// fd     - File descriptor (-1 for anonymous)
// offset - Offset in file (must be page-aligned)

int munmap(void *addr, size_t length);  // Unmap region
```

## Common Use Cases

### 1. Large File Processing

```c
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// Process large file without loading entirely into memory
void process_large_file(const char* filename) {
    int fd = open(filename, O_RDONLY);
    
    struct stat sb;
    fstat(fd, &sb);
    
    // Map entire file
    char* data = mmap(NULL, sb.st_size, PROT_READ, 
                      MAP_PRIVATE, fd, 0);
    
    // Advise kernel about access pattern
    madvise(data, sb.st_size, MADV_SEQUENTIAL);
    
    // Process data (pages loaded on demand)
    size_t count = 0;
    for (size_t i = 0; i < sb.st_size; i++) {
        if (data[i] == '\n') count++;
    }
    
    munmap(data, sb.st_size);
    close(fd);
}
```

### 2. Inter-Process Communication (IPC)

```c
// Process A: Create shared memory
int fd = shm_open("/my_shared", O_CREAT | O_RDWR, 0666);
ftruncate(fd, 4096);
void* shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);

// Write data
strcpy(shared, "Hello from Process A");

// Process B: Open shared memory
int fd = shm_open("/my_shared", O_RDWR, 0666);
void* shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);

// Read data
printf("Received: %s\n", (char*)shared);

// Cleanup
shm_unlink("/my_shared");
```

### 3. Custom Memory Allocator

```c
// Large allocation using mmap (like malloc for big blocks)
void* my_alloc(size_t size) {
    if (size >= 128 * 1024) {  // Use mmap for large allocations
        void* ptr = mmap(NULL, size + sizeof(size_t),
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) return NULL;
        
        // Store size for munmap
        *(size_t*)ptr = size + sizeof(size_t);
        return (char*)ptr + sizeof(size_t);
    }
    return malloc(size);  // Regular malloc for small allocations
}

void my_free(void* ptr) {
    // Check if mmap'd (implementation detail)
    size_t* size_ptr = (size_t*)((char*)ptr - sizeof(size_t));
    munmap(size_ptr, *size_ptr);
}
```

## madvise: Advising the Kernel

```c
#include <sys/mman.h>

int madvise(void *addr, size_t length, int advice);

// Advice values:
// MADV_NORMAL     - Default behavior
// MADV_SEQUENTIAL - Sequential access (aggressive readahead)
// MADV_RANDOM     - Random access (disable readahead)
// MADV_WILLNEED   - Will need soon (prefetch)
// MADV_DONTNEED   - Won't need soon (can drop pages)
// MADV_HUGEPAGE   - Use huge pages if possible (THP)
// MADV_NOHUGEPAGE - Don't use huge pages
```

### Practical madvise Usage

```c
// Sequential file scan
madvise(data, size, MADV_SEQUENTIAL);

// Random access (e.g., hash table)
madvise(data, size, MADV_RANDOM);

// Prefetch data we'll need
madvise(next_chunk, chunk_size, MADV_WILLNEED);

// Release memory back to OS
madvise(unused_region, region_size, MADV_DONTNEED);
```

## Performance Considerations

### mmap vs read()

| Factor | mmap | read() |
|--------|------|--------|
| Copies | 0 (direct mapping) | 1 (kernel→user) |
| Small random reads | Slower (page faults) | Faster |
| Large sequential | Faster | Similar |
| Memory pressure | Pages can be evicted | Must fit in memory |
| Modification | Direct | read-modify-write |

### When to Use mmap

**Good for:**
- Large files > 100MB
- Random access patterns
- Shared memory IPC
- Memory-mapped databases
- Read-only data

**Avoid for:**
- Small files
- Files on network filesystems (NFS)
- When you need precise I/O error handling
- High-frequency small writes

## Common Pitfalls

### 1. Page Fault Overhead

```c
// BAD: Many page faults on first access
char* data = mmap(...);
for (int i = 0; i < size; i += 4096) {
    sum += data[i];  // Page fault every 4KB!
}

// BETTER: Prefetch or use madvise
madvise(data, size, MADV_WILLNEED);  // Prefetch all pages
```

### 2. Memory Pressure

```c
// mmap'd regions can be evicted under memory pressure
// For critical data, use mlock() to prevent eviction
mlock(critical_data, size);  // Pin in RAM

// Remember to unlock
munlock(critical_data, size);
```

### 3. Synchronization

```c
// Changes may not be immediately written to file
// Use msync() to force writes
msync(data, size, MS_SYNC);   // Synchronous flush
msync(data, size, MS_ASYNC);  // Asynchronous flush
```

## References

- `man mmap`, `man madvise`, `man msync`
- "The Linux Programming Interface" - Kerrisk
- "Understanding the Linux Kernel" - Bovet & Cesati
