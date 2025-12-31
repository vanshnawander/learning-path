/**
 * 02_smart_pointers.cpp - Modern C++ Memory Management
 * 
 * Smart pointers eliminate memory leaks and make ownership explicit.
 * Essential for understanding PyTorch C++ internals.
 * 
 * Compile: g++ -std=c++17 -O2 -o smart_ptr 02_smart_pointers.cpp
 */

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// ============================================================
// SECTION 1: unique_ptr - Exclusive Ownership
// ============================================================

class Tensor {
    std::string name_;
    size_t size_;
    
public:
    Tensor(const std::string& name, size_t size) 
        : name_(name), size_(size) {
        std::cout << "Tensor '" << name_ << "' created (" << size_ << " elements)\n";
    }
    
    ~Tensor() {
        std::cout << "Tensor '" << name_ << "' destroyed\n";
    }
    
    void process() {
        std::cout << "Processing tensor '" << name_ << "'\n";
    }
    
    const std::string& name() const { return name_; }
};

void demonstrate_unique_ptr() {
    std::cout << "=== unique_ptr: EXCLUSIVE OWNERSHIP ===\n\n";
    
    // Create with make_unique (preferred)
    auto weights = std::make_unique<Tensor>("weights", 1000000);
    weights->process();
    
    // Cannot copy unique_ptr!
    // auto copy = weights;  // COMPILE ERROR!
    
    // Can move (transfer ownership)
    auto transferred = std::move(weights);
    std::cout << "Ownership transferred\n";
    
    // weights is now nullptr
    if (!weights) {
        std::cout << "Original pointer is now null\n";
    }
    
    transferred->process();
    
    // For arrays
    auto buffer = std::make_unique<float[]>(1000);
    buffer[0] = 3.14f;
    
    std::cout << "\n";
}

// ============================================================
// SECTION 2: shared_ptr - Shared Ownership
// ============================================================

void demonstrate_shared_ptr() {
    std::cout << "=== shared_ptr: SHARED OWNERSHIP ===\n\n";
    
    // Multiple owners allowed
    std::shared_ptr<Tensor> model;
    
    {
        auto layer1 = std::make_shared<Tensor>("shared_weights", 5000);
        std::cout << "Reference count: " << layer1.use_count() << "\n";
        
        model = layer1;  // Now two owners
        std::cout << "Reference count: " << layer1.use_count() << "\n";
        
        {
            auto layer2 = layer1;  // Third owner
            std::cout << "Reference count: " << layer1.use_count() << "\n";
        }
        
        std::cout << "After inner scope: " << layer1.use_count() << "\n";
    }
    
    std::cout << "After outer scope: " << model.use_count() << "\n";
    model->process();  // Still valid!
    
    std::cout << "\n";
}

// ============================================================
// SECTION 3: weak_ptr - Non-owning Reference
// ============================================================

void demonstrate_weak_ptr() {
    std::cout << "=== weak_ptr: NON-OWNING REFERENCE ===\n\n";
    
    std::weak_ptr<Tensor> cache;
    
    {
        auto tensor = std::make_shared<Tensor>("cached_data", 100);
        cache = tensor;  // weak_ptr doesn't increase ref count
        
        std::cout << "Inside scope - ref count: " << tensor.use_count() << "\n";
        
        // Convert weak_ptr to shared_ptr to use
        if (auto locked = cache.lock()) {
            locked->process();
        }
    }
    
    // Tensor destroyed, weak_ptr knows it's invalid
    if (cache.expired()) {
        std::cout << "Cache entry expired (object destroyed)\n";
    }
    
    // Safe check before use
    if (auto locked = cache.lock()) {
        locked->process();  // Won't execute
    } else {
        std::cout << "Cannot lock - object no longer exists\n";
    }
    
    std::cout << "\n";
}

// ============================================================
// SECTION 4: PyTorch-style Intrusive Pointers
// ============================================================

// PyTorch uses intrusive_ptr for tensors (ref count in object)
class IntrusiveTensor {
    mutable std::atomic<int> refcount_{0};
    std::string name_;
    
public:
    IntrusiveTensor(const std::string& name) : name_(name) {
        std::cout << "IntrusiveTensor '" << name_ << "' created\n";
    }
    
    ~IntrusiveTensor() {
        std::cout << "IntrusiveTensor '" << name_ << "' destroyed\n";
    }
    
    void add_ref() const { refcount_++; }
    void release() const {
        if (--refcount_ == 0) {
            delete this;
        }
    }
    int refcount() const { return refcount_.load(); }
};

// Simple intrusive pointer
template<typename T>
class intrusive_ptr {
    T* ptr_;
    
public:
    intrusive_ptr(T* p = nullptr) : ptr_(p) {
        if (ptr_) ptr_->add_ref();
    }
    
    ~intrusive_ptr() {
        if (ptr_) ptr_->release();
    }
    
    intrusive_ptr(const intrusive_ptr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->add_ref();
    }
    
    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
};

void demonstrate_intrusive_ptr() {
    std::cout << "=== INTRUSIVE POINTERS (PyTorch style) ===\n\n";
    
    {
        intrusive_ptr<IntrusiveTensor> t1(new IntrusiveTensor("tensor"));
        std::cout << "Refcount: " << t1->refcount() << "\n";
        
        {
            intrusive_ptr<IntrusiveTensor> t2 = t1;
            std::cout << "Refcount: " << t1->refcount() << "\n";
        }
        
        std::cout << "Refcount: " << t1->refcount() << "\n";
    }
    
    std::cout << "\n";
}

// ============================================================
// SECTION 5: Custom Deleters
// ============================================================

void demonstrate_custom_deleter() {
    std::cout << "=== CUSTOM DELETERS ===\n\n";
    
    // Custom deleter for CUDA memory (simulated)
    auto cuda_deleter = [](float* p) {
        std::cout << "Custom deleter: cudaFree() called\n";
        delete[] p;
    };
    
    {
        std::unique_ptr<float[], decltype(cuda_deleter)> 
            gpu_data(new float[1000], cuda_deleter);
        
        gpu_data[0] = 3.14f;
        std::cout << "GPU data[0] = " << gpu_data[0] << "\n";
    }
    
    std::cout << "\n";
}

int main() {
    demonstrate_unique_ptr();
    demonstrate_shared_ptr();
    demonstrate_weak_ptr();
    demonstrate_intrusive_ptr();
    demonstrate_custom_deleter();
    
    std::cout << "=== SUMMARY ===\n\n";
    std::cout << "unique_ptr: Single owner, zero overhead\n";
    std::cout << "  Use for: Local tensors, exclusive resources\n\n";
    std::cout << "shared_ptr: Multiple owners, reference counted\n";
    std::cout << "  Use for: Shared model weights, caches\n\n";
    std::cout << "weak_ptr: Non-owning, can check validity\n";
    std::cout << "  Use for: Caches, breaking cycles\n\n";
    std::cout << "intrusive_ptr: PyTorch's choice for Tensor\n";
    std::cout << "  Use for: High-performance ref counting\n";
    
    return 0;
}
