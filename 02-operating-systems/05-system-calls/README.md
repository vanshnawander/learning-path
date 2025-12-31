# System Calls

The interface between user programs and the kernel.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_syscall_overhead.c` | Measuring syscall cost |

## What is a System Call?

Programs can't directly access hardware.
System calls ask the kernel to do privileged operations.

```
User Space        Kernel Space
    |                  |
    |--- syscall() --->|
    |                  | (privileged operation)
    |<-- return -------|
    |                  |
```

## Common System Calls

| Category | Examples |
|----------|----------|
| Files | open, read, write, close |
| Memory | mmap, brk, mprotect |
| Process | fork, exec, wait, exit |
| Network | socket, bind, connect |

## Syscall Overhead

```
Syscall cost: ~100-500 ns
Function call: ~1 ns
Ratio: 100-500x slower!
```

This is why:
- mmap beats many read() calls
- Batching is important
- Buffered I/O exists

## Reducing Syscall Overhead

1. **Batch operations**: One big read vs many small
2. **mmap**: Access file without read syscalls
3. **vDSO**: Some calls (gettimeofday) avoid kernel
4. **io_uring**: Batch async I/O operations

## ML Data Loading

```python
# Bad: many small reads
for i in range(1000):
    data = file.read(100)

# Good: one large read
data = file.read(100000)

# Best: memory map
data = mmap.mmap(fd, size)
```
