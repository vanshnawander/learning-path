# 04 - C Programming Deep Dive

Mastering C is essential for understanding ML systems at the lowest level.

## 📁 Directory Structure

```
04-c-programming/
├── 01-pointers-deep-dive/
│   ├── README.md
│   ├── 01_pointer_basics.c
│   ├── 02_pointer_arithmetic.c
│   └── 03_void_pointers.c
├── 02-memory-management/
│   ├── 01_stack_vs_heap.c
│   └── 02_custom_allocator.c
├── 03-mmap-advanced/
│   ├── 01_mmap_file_io.c
│   └── 02_shared_tensor.c
├── 04-struct-patterns/
│   └── 01_data_oriented.c
├── 05-io-patterns/
│   └── 01_buffered_io.c
└── memory-management/
    └── README.md
```

## 🎯 Learning Objectives

After completing this module, you will:

- [ ] Master pointer arithmetic and memory navigation
- [ ] Understand stack vs heap allocation
- [ ] Build custom memory allocators (like PyTorch's)
- [ ] Use mmap for zero-copy data access
- [ ] Share memory between processes (DataLoader pattern)
- [ ] Apply data-oriented design for cache efficiency
- [ ] Choose optimal I/O strategies

## 🔗 Connection to ML Systems

| C Concept | ML Application |
|-----------|----------------|
| Pointer arithmetic | Tensor stride access |
| void* | Generic tensor data (dtype) |
| Custom allocator | PyTorch CUDA caching allocator |
| mmap | FFCV .beton file access |
| Shared memory | DataLoader worker communication |
| Data-oriented design | Tensor memory layout |
| Buffered I/O | Efficient data loading |

## 📖 Recommended Order

### Week 1: Pointers
1. `01-pointers-deep-dive/01_pointer_basics.c`
2. `01-pointers-deep-dive/02_pointer_arithmetic.c`
3. `01-pointers-deep-dive/03_void_pointers.c`

### Week 2: Memory
1. `02-memory-management/01_stack_vs_heap.c`
2. `02-memory-management/02_custom_allocator.c`

### Week 3: Advanced I/O
1. `03-mmap-advanced/01_mmap_file_io.c`
2. `03-mmap-advanced/02_shared_tensor.c`
3. `05-io-patterns/01_buffered_io.c`

### Week 4: Optimization
1. `04-struct-patterns/01_data_oriented.c`

## 🛠️ Compilation

```bash
# Basic compilation
gcc -o program program.c

# With optimization
gcc -O2 -o program program.c

# With SIMD
gcc -O3 -mavx2 -o program program.c

# With debugging
gcc -g -o program program.c

# With shared memory library
gcc -o program program.c -lrt -lpthread
```

## 📚 Key Resources

- "The C Programming Language" - K&R
- "Expert C Programming" - Peter van der Linden
- "Computer Systems: A Programmer's Perspective" - Bryant & O'Hallaron
