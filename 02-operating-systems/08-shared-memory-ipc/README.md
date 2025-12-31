# Shared Memory and IPC

How processes communicate - the foundation of DataLoader workers.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_shared_memory.c` | POSIX shared memory example |

## IPC Methods

| Method | Speed | Complexity | Use Case |
|--------|-------|------------|----------|
| Shared Memory | Fastest | Medium | Large data |
| Pipes | Medium | Low | Streaming |
| Sockets | Slowest | High | Network |
| Files | Slow | Low | Persistence |

## Shared Memory

```c
// Create
int fd = shm_open("/name", O_CREAT | O_RDWR, 0666);
ftruncate(fd, size);

// Map
void* ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, 
                 MAP_SHARED, fd, 0);

// Use
ptr[0] = data;  // Visible to other processes!

// Cleanup
munmap(ptr, size);
shm_unlink("/name");
```

## PyTorch DataLoader

```python
DataLoader(dataset, num_workers=4, ...)
```

### How It Works
1. Main process forks workers
2. Workers load data into shared memory
3. Main process reads directly (zero-copy!)
4. `torch.multiprocessing` handles tensor sharing

### Key Classes
- `multiprocessing.Queue` - message passing
- `multiprocessing.shared_memory` - raw shared memory
- `torch.multiprocessing` - tensor-aware sharing

## pin_memory=True

```python
DataLoader(..., pin_memory=True)
```

- Allocates page-locked (pinned) memory
- Enables DMA transfer to GPU
- Faster CPU→GPU copy
- Uses more RAM (can't be swapped)
