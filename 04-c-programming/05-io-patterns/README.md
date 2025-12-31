# I/O Patterns for Data Loading

Choosing the right I/O strategy is critical for ML data pipelines.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_buffered_io.c` | Comparing I/O strategies |

## I/O Methods Ranked

| Method | Speed | Use Case |
|--------|-------|----------|
| mmap (cached) | ★★★★★ | Random access, FFCV |
| Large read() | ★★★★☆ | Sequential streaming |
| fread() | ★★★☆☆ | General purpose |
| Small read() | ★☆☆☆☆ | Never do this! |

## Buffer Size Matters

```
1 byte reads:   ~100 KB/s  (syscall per byte!)
4 KB reads:     ~500 MB/s
1 MB reads:     ~2 GB/s
mmap:           ~5+ GB/s
```

## Key Insight: Syscall Overhead

Each `read()` call:
1. Switch to kernel mode (~100+ cycles)
2. Copy to kernel buffer
3. Copy to user buffer
4. Switch back to user mode

**Solution**: Minimize syscalls!
- Large buffers
- mmap (no read syscalls)
- Async I/O (overlap)

## PyTorch DataLoader Strategy

```python
DataLoader(
    dataset,
    num_workers=4,      # Parallel I/O
    prefetch_factor=2,  # Lookahead
    pin_memory=True,    # Fast GPU transfer
)
```

Workers read in parallel, main process trains.

## FFCV Strategy

```python
# mmap entire file once
Loader(beton_file, ...)

# During training:
# - No read() syscalls
# - OS manages caching
# - Quasi-random for locality
```
