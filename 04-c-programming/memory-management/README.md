# C Memory Management

Understanding heap, stack, and manual memory control.

## Memory Layout

```
High Address
┌─────────────────┐
│      Stack      │ ← Local variables, function calls
│        ↓        │   (grows downward)
├─────────────────┤
│                 │
│     (free)      │
│                 │
├─────────────────┤
│        ↑        │
│      Heap       │ ← Dynamic allocation
│                 │   (grows upward)
├─────────────────┤
│      BSS        │ ← Uninitialized globals
├─────────────────┤
│      Data       │ ← Initialized globals
├─────────────────┤
│      Text       │ ← Program code
└─────────────────┘
Low Address
```

## Stack vs Heap

| Aspect | Stack | Heap |
|--------|-------|------|
| Speed | Fast | Slower |
| Size | Limited (~8MB) | Large |
| Lifetime | Automatic | Manual |
| Access | LIFO | Random |

## Allocation Functions

```c
#include <stdlib.h>

// Allocate uninitialized memory
void *ptr = malloc(size);

// Allocate and zero-initialize
void *ptr = calloc(count, size);

// Resize allocation
void *new_ptr = realloc(ptr, new_size);

// Free memory
free(ptr);
```

## Common Bugs

### 1. Memory Leak
```c
void leak() {
    int *p = malloc(sizeof(int));
    // Forgot to free!
}
```

### 2. Use After Free
```c
int *p = malloc(sizeof(int));
free(p);
*p = 42;  // BUG!
```

### 3. Double Free
```c
int *p = malloc(sizeof(int));
free(p);
free(p);  // BUG!
```

### 4. Buffer Overflow
```c
int arr[10];
arr[10] = 42;  // BUG: out of bounds
```

## Valgrind

Detect memory errors:
```bash
valgrind --leak-check=full ./program
```

## AddressSanitizer

Compile-time instrumentation:
```bash
gcc -fsanitize=address program.c
./a.out
```

## Best Practices
1. Always pair malloc with free
2. Set pointers to NULL after freeing
3. Use Valgrind/ASan in development
4. Consider RAII patterns (C++ wrapper)
