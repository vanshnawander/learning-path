# Processes and Threads

Understanding concurrency - the foundation of parallel data loading.

## Files in This Directory

| File | Description | Key Concept |
|------|-------------|-------------|
| `01_fork_basics.c` | Process creation | Copy-on-write, isolation |
| `02_threads_basics.c` | POSIX threads | Shared memory, races |

## Process vs Thread

| Aspect | Process | Thread |
|--------|---------|--------|
| Memory | Separate | Shared |
| Creation cost | High | Low |
| Communication | IPC needed | Direct |
| Crash isolation | Yes | No |

## PyTorch DataLoader

```python
DataLoader(dataset, num_workers=4)
```

- `num_workers=0`: Single process, no fork
- `num_workers>0`: Fork worker processes
- Workers load data in parallel
- Data sent via multiprocessing.Queue

## Why Processes over Threads in Python?

**Python GIL (Global Interpreter Lock)**:
- Only one thread can execute Python at a time
- Threads don't parallelize CPU-bound work
- Processes bypass GIL completely

## Start Methods

```python
mp.set_start_method('fork')    # Fast, shares memory (Linux default)
mp.set_start_method('spawn')   # Safe with CUDA (Windows default)
mp.set_start_method('forkserver')  # Compromise
```

**Warning**: `fork` after CUDA initialization can cause issues!

## Exercises

1. Measure fork() overhead for different memory sizes
2. Demonstrate race condition with threads
3. Compare process vs thread for CPU-bound work
