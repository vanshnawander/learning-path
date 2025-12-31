/**
 * 03_move_semantics.cpp - Zero-Cost Ownership Transfer
 * 
 * Move semantics eliminate unnecessary copies.
 * Critical for performance in ML frameworks.
 * 
 * Compile: g++ -std=c++17 -O2 -o move 03_move_semantics.cpp
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <utility>

// ============================================================
// SECTION 1: The Problem - Expensive Copies
// ============================================================

class HeavyTensor {
    float* data_;
    size_t size_;
    static int copy_count_;
    static int move_count_;
    
public:
    explicit HeavyTensor(size_t size) : size_(size) {
        data_ = new float[size];
        std::memset(data_, 0, size * sizeof(float));
    }
    
    // Copy constructor - EXPENSIVE!
    HeavyTensor(const HeavyTensor& other) : size_(other.size_) {
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
        copy_count_++;
        std::cout << "  COPY: Allocated and copied " << size_ << " floats\n";
    }
    
    // Move constructor - CHEAP!
    HeavyTensor(HeavyTensor&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
        move_count_++;
        std::cout << "  MOVE: Just swapped pointers (zero cost)\n";
    }
    
    // Copy assignment
    HeavyTensor& operator=(const HeavyTensor& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new float[size_];
            std::memcpy(data_, other.data_, size_ * sizeof(float));
            copy_count_++;
        }
        return *this;
    }
    
    // Move assignment
    HeavyTensor& operator=(HeavyTensor&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            move_count_++;
        }
        return *this;
    }
    
    ~HeavyTensor() {
        delete[] data_;
    }
    
    size_t size() const { return size_; }
    
    static void reset_counts() { copy_count_ = move_count_ = 0; }
    static int copies() { return copy_count_; }
    static int moves() { return move_count_; }
};

int HeavyTensor::copy_count_ = 0;
int HeavyTensor::move_count_ = 0;

// ============================================================
// SECTION 2: Demonstrating Copy vs Move
// ============================================================

HeavyTensor create_tensor(size_t size) {
    HeavyTensor t(size);
    return t;  // Move (or NRVO)
}

void demonstrate_copy_vs_move() {
    std::cout << "=== COPY VS MOVE ===\n\n";
    
    HeavyTensor::reset_counts();
    
    std::cout << "Creating tensor:\n";
    HeavyTensor a(1000000);
    
    std::cout << "\nCopying tensor (expensive):\n";
    HeavyTensor b = a;  // Copy
    
    std::cout << "\nMoving tensor (free):\n";
    HeavyTensor c = std::move(a);  // Move
    
    std::cout << "\nReturning from function:\n";
    HeavyTensor d = create_tensor(1000000);  // Move or NRVO
    
    std::cout << "\nTotal copies: " << HeavyTensor::copies() << "\n";
    std::cout << "Total moves: " << HeavyTensor::moves() << "\n\n";
}

// ============================================================
// SECTION 3: std::move is just a cast
// ============================================================

void demonstrate_move_cast() {
    std::cout << "=== std::move IS JUST A CAST ===\n\n";
    
    int x = 42;
    
    // std::move doesn't move anything!
    // It just casts to rvalue reference
    int&& rref = std::move(x);
    
    std::cout << "x = " << x << " (still valid!)\n";
    std::cout << "rref = " << rref << "\n";
    
    // The actual move happens when you construct/assign
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::cout << "v1.size() before move: " << v1.size() << "\n";
    
    std::vector<int> v2 = std::move(v1);  // HERE the move happens
    std::cout << "v1.size() after move: " << v1.size() << " (moved-from)\n";
    std::cout << "v2.size(): " << v2.size() << "\n\n";
}

// ============================================================
// SECTION 4: Perfect Forwarding
// ============================================================

template<typename T>
class Container {
    T value_;
    
public:
    // Perfect forwarding constructor
    template<typename U>
    Container(U&& val) : value_(std::forward<U>(val)) {
        std::cout << "Container constructed\n";
    }
};

void demonstrate_perfect_forwarding() {
    std::cout << "=== PERFECT FORWARDING ===\n\n";
    
    HeavyTensor::reset_counts();
    
    HeavyTensor t(100);
    
    std::cout << "Passing lvalue (will copy):\n";
    Container<HeavyTensor> c1(t);
    
    std::cout << "\nPassing rvalue (will move):\n";
    Container<HeavyTensor> c2(HeavyTensor(100));
    
    std::cout << "\nCopies: " << HeavyTensor::copies() << "\n";
    std::cout << "Moves: " << HeavyTensor::moves() << "\n\n";
}

// ============================================================
// SECTION 5: Performance Benchmark
// ============================================================

void benchmark() {
    std::cout << "=== PERFORMANCE BENCHMARK ===\n\n";
    
    const size_t SIZE = 10000000;  // 10M elements
    const int ITERATIONS = 100;
    
    // Benchmark copy
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        std::vector<float> v1(SIZE, 1.0f);
        std::vector<float> v2 = v1;  // Copy
        (void)v2[0];  // Prevent optimization
    }
    auto copy_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark move
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        std::vector<float> v1(SIZE, 1.0f);
        std::vector<float> v2 = std::move(v1);  // Move
        (void)v2[0];  // Prevent optimization
    }
    auto move_time = std::chrono::high_resolution_clock::now() - start;
    
    auto copy_ms = std::chrono::duration<double, std::milli>(copy_time).count();
    auto move_ms = std::chrono::duration<double, std::milli>(move_time).count();
    
    std::cout << "Vector size: " << SIZE << " floats (" 
              << SIZE * sizeof(float) / (1024*1024) << " MB)\n";
    std::cout << "Copy time: " << copy_ms << " ms\n";
    std::cout << "Move time: " << move_ms << " ms\n";
    std::cout << "Speedup: " << copy_ms / move_ms << "x\n\n";
}

int main() {
    demonstrate_copy_vs_move();
    demonstrate_move_cast();
    demonstrate_perfect_forwarding();
    benchmark();
    
    std::cout << "=== ML IMPLICATIONS ===\n\n";
    std::cout << "1. Return tensors by value - compiler uses move/NRVO\n";
    std::cout << "2. Pass large objects as const& or &&\n";
    std::cout << "3. Use std::move for explicit ownership transfer\n";
    std::cout << "4. PyTorch tensors use move for zero-copy\n";
    std::cout << "5. Model weights should be moved, not copied\n";
    
    return 0;
}
