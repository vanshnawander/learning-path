# 04 - C Programming

The language of systems programming - essential for understanding low-level code.

## 📚 Topics Covered

### Core C Language
- **Types**: Primitives, structs, unions, enums
- **Pointers**: Arithmetic, function pointers, void pointers
- **Memory**: Stack vs heap, lifetime
- **Preprocessor**: Macros, conditional compilation
- **Bit Manipulation**: Bitfields, masks, shifts

### Memory Management
- **malloc/free**: Heap allocation
- **Memory Layout**: Text, data, BSS, heap, stack
- **Memory Debugging**: Valgrind, AddressSanitizer
- **Custom Allocators**: Pool allocators, arena allocators

### Advanced C
- **Undefined Behavior**: Common pitfalls
- **Volatile & Restrict**: Compiler hints
- **Inline Assembly**: asm blocks
- **ABI Compatibility**: Struct packing, alignment

### Build Systems
- **Compilation Process**: Preprocessing, compilation, linking
- **Makefiles**: Dependencies, patterns
- **CMake**: Modern C build system
- **Static vs Dynamic Linking**: Libraries

### C Standard Library
- **stdio.h**: File I/O
- **stdlib.h**: Memory, utilities
- **string.h**: String manipulation
- **pthread.h**: POSIX threads

## 🎯 Learning Objectives

- [ ] Write memory-safe C code
- [ ] Debug memory issues with Valgrind
- [ ] Understand linking and loading
- [ ] Use CMake for building projects
- [ ] Read C code in major projects

## 💻 Practical Exercises

1. Implement a dynamic array
2. Write a hash table
3. Build a simple memory allocator
4. Create a thread-safe queue

## 📖 Resources

### Books
- "The C Programming Language" (K&R)
- "Expert C Programming: Deep C Secrets"
- "21st Century C" - Ben Klemens

### Online
- Beej's Guide to C Programming
- Modern C (free online book)

## 📁 Structure

```
04-c-programming/
├── fundamentals/
│   ├── types/
│   ├── pointers/
│   └── memory/
├── memory-management/
│   ├── allocators/
│   ├── debugging/
│   └── valgrind/
├── advanced/
│   ├── undefined-behavior/
│   ├── inline-assembly/
│   └── optimization/
└── build-systems/
    ├── makefiles/
    └── cmake/
```

## ⏱️ Estimated Time: 4-5 weeks
