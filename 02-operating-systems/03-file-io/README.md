# File I/O Deep Dive

Understanding I/O is critical for data loading performance.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_io_basics.c` | Buffered vs direct I/O comparison |

## I/O Methods

### 1. Buffered I/O (stdio)
```c
FILE* f = fopen("data.bin", "rb");
fread(buffer, size, 1, f);
```
- Library handles buffering
- Good for small reads

### 2. System Calls (read/write)
```c
int fd = open("data.bin", O_RDONLY);
read(fd, buffer, size);
```
- Direct kernel calls
- More control

### 3. Memory Mapping (mmap)
```c
void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
```
- Zero-copy access
- Best for random access

### 4. Direct I/O (O_DIRECT)
```c
int fd = open("data.bin", O_RDONLY | O_DIRECT);
```
- Bypasses page cache
- For when you manage caching

### 5. Async I/O (io_uring, libaio)
- Non-blocking I/O
- Overlap I/O with compute

## Performance Hierarchy

```
mmap (cached)     > 10 GB/s
Large read()      ~5 GB/s
Small read() loop ~100 MB/s (syscall overhead)
```

## ML Data Loading

| Method | Use Case |
|--------|----------|
| mmap | FFCV .beton files |
| Buffered | Small config files |
| Async I/O | Streaming datasets |
