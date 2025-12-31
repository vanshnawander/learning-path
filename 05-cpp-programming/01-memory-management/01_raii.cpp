/**
 * 01_raii.cpp - Resource Acquisition Is Initialization
 * 
 * RAII is THE fundamental C++ pattern for resource management.
 * Used in PyTorch, CUDA, and every serious C++ codebase.
 * 
 * Compile: g++ -std=c++17 -O2 -o raii 01_raii.cpp
 */

#include <iostream>
#include <cstdlib>
#include <stdexcept>

// ============================================================
// SECTION 1: The Problem - Manual Memory Management
// ============================================================

void bad_example() {
    float* data = new float[1000];
    
    // What if this throws?
    // process(data);  
    
    // What if we return early?
    // if (condition) return;  // LEAK!
    
    delete[] data;  // Easy to forget!
}

// ============================================================
// SECTION 2: RAII Solution - Automatic Cleanup
// ============================================================

class Buffer {
private:
    float* data_;
    size_t size_;
    
public:
    // Constructor acquires resource
    explicit Buffer(size_t size) : size_(size) {
        data_ = new float[size];
        std::cout << "Buffer: Allocated " << size << " floats\n";
    }
    
    // Destructor releases resource - ALWAYS called!
    ~Buffer() {
        delete[] data_;
        std::cout << "Buffer: Freed " << size_ << " floats\n";
    }
    
    // Delete copy (we'll add move later)
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Accessors
    float* data() { return data_; }
    size_t size() const { return size_; }
    float& operator[](size_t i) { return data_[i]; }
};

void demonstrate_raii() {
    std::cout << "=== RAII DEMONSTRATION ===\n\n";
    
    {
        Buffer buf(1000);
        buf[0] = 3.14f;
        std::cout << "Using buffer: buf[0] = " << buf[0] << "\n";
        // Destructor called automatically at end of scope!
    }
    
    std::cout << "After scope - buffer already freed!\n\n";
}

// ============================================================
// SECTION 3: RAII with Exceptions
// ============================================================

void may_throw() {
    throw std::runtime_error("Something went wrong!");
}

void demonstrate_exception_safety() {
    std::cout << "=== EXCEPTION SAFETY ===\n\n";
    
    try {
        Buffer buf(500);
        std::cout << "Buffer created\n";
        
        may_throw();  // This throws!
        
        std::cout << "This won't print\n";
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
        // Buffer destructor was STILL called!
    }
    
    std::cout << "No memory leak despite exception!\n\n";
}

// ============================================================
// SECTION 4: CUDA-style RAII wrapper
// ============================================================

// Simulated CUDA functions
void* cuda_malloc(size_t size) {
    std::cout << "  cudaMalloc(" << size << ")\n";
    return std::malloc(size);
}

void cuda_free(void* ptr) {
    std::cout << "  cudaFree()\n";
    std::free(ptr);
}

class CudaBuffer {
private:
    void* ptr_;
    size_t size_;
    
public:
    explicit CudaBuffer(size_t size) : size_(size) {
        ptr_ = cuda_malloc(size);
        if (!ptr_) throw std::bad_alloc();
    }
    
    ~CudaBuffer() {
        if (ptr_) cuda_free(ptr_);
    }
    
    // Move constructor
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cuda_free(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // No copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    void* get() { return ptr_; }
    size_t size() const { return size_; }
};

void demonstrate_cuda_raii() {
    std::cout << "=== CUDA-STYLE RAII ===\n\n";
    
    {
        CudaBuffer gpu_mem(1024 * 1024);  // 1MB
        std::cout << "GPU memory allocated\n";
        
        // Use gpu_mem...
        
    }  // Automatically freed!
    
    std::cout << "GPU memory freed automatically\n\n";
}

// ============================================================
// SECTION 5: RAII for File Handles
// ============================================================

#include <fstream>

void demonstrate_file_raii() {
    std::cout << "=== FILE HANDLE RAII ===\n\n";
    
    // std::fstream uses RAII
    {
        std::ofstream file("/tmp/test.txt");
        file << "Hello, RAII!\n";
        // File automatically closed at end of scope
    }
    
    std::cout << "File automatically closed\n";
    std::cout << "This is how PyTorch checkpoint saving works!\n\n";
}

int main() {
    demonstrate_raii();
    demonstrate_exception_safety();
    demonstrate_cuda_raii();
    demonstrate_file_raii();
    
    std::cout << "=== KEY TAKEAWAYS ===\n\n";
    std::cout << "1. Constructor acquires, destructor releases\n";
    std::cout << "2. Cleanup happens even on exceptions\n";
    std::cout << "3. No manual delete/free needed\n";
    std::cout << "4. Used everywhere: CUDA, files, locks, etc.\n";
    std::cout << "5. PyTorch tensors use RAII for memory\n";
    
    return 0;
}
