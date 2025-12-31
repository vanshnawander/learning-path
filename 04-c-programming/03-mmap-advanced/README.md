# Advanced Memory Mapping

Deep dive into mmap - the foundation of FFCV and efficient data loading.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_mmap_file_io.c` | mmap vs read() comparison |
| `02_shared_tensor.c` | Sharing tensors between processes |

## Why mmap is Essential

### Traditional I/O
```
File → Kernel Buffer → User Buffer → Process
       [copy 1]        [copy 2]
```

### Memory-Mapped I/O
```
File → Page Cache ← Process (direct pointer access)
       [shared]
```

## Key APIs

### mmap()
```c
void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);

// Example
void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
float value = ((float*)data)[1000];  // Direct access!
```

### madvise()
```c
madvise(ptr, len, MADV_SEQUENTIAL);  // Reading sequentially
madvise(ptr, len, MADV_RANDOM);      // Random access
madvise(ptr, len, MADV_WILLNEED);    // Prefetch now
madvise(ptr, len, MADV_DONTNEED);    // Done with this
```

### Shared Memory
```c
int fd = shm_open("/name", O_CREAT | O_RDWR, 0666);
ftruncate(fd, size);
void* ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// Now 'ptr' is shared between processes!
```

## Connection to ML Systems

| System | mmap Use |
|--------|----------|
| FFCV | .beton file access |
| PyTorch DataLoader | Worker shared memory |
| numpy.memmap | Large array access |
| HuggingFace | Model weight loading |

## Exercises

1. Compare read() vs mmap() for random access
2. Build a shared memory queue
3. Implement quasi-random access pattern
