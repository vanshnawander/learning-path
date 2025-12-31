# Memory Mapping (mmap)

The foundation of efficient data loading - how FFCV achieves its speed.

## Files in This Directory

| File | Description | Key Concept |
|------|-------------|-------------|
| `01_mmap_basics.c` | mmap fundamentals | Mapping files to memory |
| `02_mmap_dataloader.c` | Build a mini DataLoader | FFCV-style access |

## What is mmap?

`mmap()` maps a file directly into your process's address space.
Instead of `read()` copying data, you access file contents via pointers.

```c
void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
// Now data[i] reads directly from file!
```

## Why mmap for ML?

### Traditional read()
```
File → Kernel Buffer → User Buffer → Process
       (copy)          (copy)
```

### With mmap
```
File → Page Cache ← Process (direct access)
       (shared)
```

## Key System Calls

| Call | Purpose |
|------|---------|
| `mmap()` | Create mapping |
| `munmap()` | Remove mapping |
| `madvise()` | Hint access pattern |
| `msync()` | Flush changes to disk |

## FFCV Connection

1. `.beton` file is memory-mapped
2. Index gives O(1) sample offsets
3. Quasi-random access within pages
4. OS handles caching automatically

## Exercises

1. Compare read() vs mmap() for large files
2. Experiment with MADV_SEQUENTIAL vs MADV_RANDOM
3. Build a dataset format with header + data
