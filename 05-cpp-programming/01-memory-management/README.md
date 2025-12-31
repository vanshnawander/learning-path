# C++ Memory Management

Modern C++ provides powerful memory management that's essential for high-performance ML systems.

## Why C++ for ML Systems?

| Library | Language | Why |
|---------|----------|-----|
| PyTorch | C++ core | Performance-critical ops |
| TensorFlow | C++ | Execution engine |
| ONNX Runtime | C++ | Inference optimization |
| llama.cpp | C++ | Efficient LLM inference |

## Files in This Directory

| File | Description |
|------|-------------|
| `01_raii.cpp` | Resource Acquisition Is Initialization |
| `02_smart_pointers.cpp` | unique_ptr, shared_ptr, weak_ptr |
| `03_move_semantics.cpp` | Move vs Copy for zero-cost transfers |
| `04_custom_allocator.cpp` | Arena allocators for ML |
| `05_memory_pools.cpp` | PyTorch-style caching allocator |

## Key Concepts

### RAII (Resource Acquisition Is Initialization)
```cpp
class Tensor {
    float* data;
public:
    Tensor(size_t n) : data(new float[n]) {}
    ~Tensor() { delete[] data; }  // Automatic cleanup!
};
```

### Smart Pointers
```cpp
auto weights = std::make_unique<float[]>(1000000);
// Automatically freed when out of scope - no leaks!
```

### Move Semantics
```cpp
Tensor a = load_weights();  // Expensive
Tensor b = std::move(a);    // FREE! Just pointer swap
```

## Connection to ML

| C++ Feature | ML Use Case |
|-------------|-------------|
| RAII | CUDA memory management |
| unique_ptr | Exclusive tensor ownership |
| shared_ptr | Shared model weights |
| Move semantics | Zero-copy tensor passing |
| Custom allocator | PyTorch caching allocator |
