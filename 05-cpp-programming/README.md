# 05 - C++ Programming

Modern C++ for high-performance systems and ML frameworks.

## 📚 Topics Covered

### Modern C++ (C++17/20/23)
- **Auto & Type Inference**: decltype, auto
- **Move Semantics**: rvalue references, std::move
- **Smart Pointers**: unique_ptr, shared_ptr, weak_ptr
- **Lambdas**: Captures, generic lambdas
- **Structured Bindings**: Tuple unpacking

### Templates & Metaprogramming
- **Function Templates**: Type deduction
- **Class Templates**: Generic containers
- **Variadic Templates**: Parameter packs
- **SFINAE**: Enable_if, concepts (C++20)
- **Compile-Time Computation**: constexpr, consteval

### Memory & Performance
- **RAII**: Resource management
- **Custom Allocators**: std::allocator interface
- **Memory Pools**: Arena allocators
- **Cache-Friendly Code**: Data-oriented design
- **Move vs Copy**: Performance implications

### Standard Library
- **Containers**: vector, unordered_map, array
- **Algorithms**: STL algorithms, execution policies
- **Ranges (C++20)**: Lazy evaluation, views
- **Concurrency**: std::thread, std::async, atomics

### C++ in ML Frameworks
- **PyTorch C++ (libtorch)**: ATen, c10
- **pybind11**: Python bindings
- **Eigen**: Linear algebra library

## 🎯 Learning Objectives

- [ ] Write modern C++ with move semantics
- [ ] Use templates for generic programming
- [ ] Implement RAII patterns
- [ ] Understand PyTorch's C++ codebase
- [ ] Create Python bindings with pybind11

## 💻 Practical Exercises

1. Implement a smart pointer from scratch
2. Write a compile-time matrix library
3. Create Python bindings for C++ code
4. Build a thread-safe data structure

## 📖 Resources

### Books
- "Effective Modern C++" - Scott Meyers
- "C++ Concurrency in Action" - Anthony Williams
- "A Tour of C++" - Bjarne Stroustrup

### Online
- CppCon talks on YouTube
- cppreference.com

## 📁 Structure

```
05-cpp-programming/
├── modern-cpp/
│   ├── move-semantics/
│   ├── smart-pointers/
│   └── lambdas/
├── templates/
│   ├── basics/
│   ├── metaprogramming/
│   └── concepts/
├── memory-performance/
│   ├── allocators/
│   ├── data-oriented-design/
│   └── optimization/
├── concurrency/
│   ├── threads/
│   ├── atomics/
│   └── async/
└── ml-frameworks/
    ├── pybind11/
    └── libtorch/
```

## ⏱️ Estimated Time: 5-6 weeks
