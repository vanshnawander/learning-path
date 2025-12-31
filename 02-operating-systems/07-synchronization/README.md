# Synchronization Primitives

Coordinating access to shared resources.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_atomics.c` | Atomic operations for lock-free code |

## Why Synchronization?

```c
// Thread 1          // Thread 2
counter++;           counter++;
// Both read 0, increment to 1, write 1
// Result: 1 instead of 2!
```

## Synchronization Methods

### 1. Mutex (Mutual Exclusion)
```c
pthread_mutex_lock(&mutex);
counter++;  // Only one thread at a time
pthread_mutex_unlock(&mutex);
```
- Simple, safe
- Can be slow (kernel involvement)

### 2. Spinlock
```c
while (atomic_flag_test_and_set(&lock)) { }
counter++;
atomic_flag_clear(&lock);
```
- Busy-waits (wastes CPU)
- Fast for short critical sections

### 3. Atomic Operations
```c
atomic_fetch_add(&counter, 1);
```
- Lock-free
- Hardware supported
- Best for simple operations

### 4. Read-Write Lock
```c
pthread_rwlock_rdlock(&rwlock);  // Multiple readers OK
pthread_rwlock_wrlock(&rwlock);  // Exclusive writer
```

## CUDA Atomics

```cuda
atomicAdd(&global_var, value);
atomicMax(&global_var, value);
atomicCAS(&global_var, compare, value);
```

Used in:
- Reduction operations
- Histogram computation
- Attention score accumulation

## Performance Comparison

```
Atomic add:  ~10-50 ns
Mutex:       ~100-500 ns
Spinlock:    ~10-100 ns (wastes CPU)
```
