/**
 * 05_cpp_dataloader.cpp - High-Performance C++ Data Loader
 * 
 * Shows how to build optimized data loading from scratch in C++.
 * This is what FFCV, DALI, and other fast loaders do internally.
 * 
 * Compile: g++ -std=c++17 -O3 -pthread -o cpp_loader 05_cpp_dataloader.cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <cstring>
#include <memory>

// For mmap
#ifdef __linux__
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ============================================================
// TIMING UTILITIES
// ============================================================

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// ============================================================
// THREAD-SAFE QUEUE (for prefetching)
// ============================================================

template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t max_size) : max_size_(max_size), done_(false) {}
    
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || done_; });
        if (done_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }
    
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    void set_done() {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t max_size_;
    std::atomic<bool> done_;
};

// ============================================================
// SAMPLE AND BATCH STRUCTURES
// ============================================================

struct Sample {
    std::vector<float> data;
    int label;
    size_t index;
};

struct Batch {
    std::vector<float> data;  // Flattened batch
    std::vector<int> labels;
    size_t batch_size;
    size_t sample_size;
};

// ============================================================
// MEMORY-MAPPED DATA SOURCE
// ============================================================

class MmapDataSource {
public:
    MmapDataSource(const std::string& data_path, size_t num_samples, size_t sample_size)
        : num_samples_(num_samples), sample_size_(sample_size), data_(nullptr) {
        
#ifdef __linux__
        // Open file
        fd_ = open(data_path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "Failed to open file: " << data_path << std::endl;
            return;
        }
        
        // Memory map
        size_t total_size = num_samples * sample_size * sizeof(float);
        data_ = static_cast<float*>(mmap(nullptr, total_size, PROT_READ, 
                                          MAP_SHARED, fd_, 0));
        if (data_ == MAP_FAILED) {
            std::cerr << "mmap failed" << std::endl;
            data_ = nullptr;
            return;
        }
        
        // Advise kernel about access pattern
        madvise(data_, total_size, MADV_RANDOM);  // Random access
        
        std::cout << "Memory-mapped " << total_size / (1024*1024) << " MB" << std::endl;
#else
        std::cout << "mmap not available on this platform, using simulation" << std::endl;
        // Simulate with allocated memory
        simulated_data_.resize(num_samples * sample_size);
        for (size_t i = 0; i < simulated_data_.size(); i++) {
            simulated_data_[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        data_ = simulated_data_.data();
#endif
    }
    
    ~MmapDataSource() {
#ifdef __linux__
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, num_samples_ * sample_size_ * sizeof(float));
        }
        if (fd_ >= 0) {
            close(fd_);
        }
#endif
    }
    
    const float* get_sample(size_t index) const {
        if (!data_ || index >= num_samples_) return nullptr;
        return data_ + index * sample_size_;
    }
    
    size_t num_samples() const { return num_samples_; }
    size_t sample_size() const { return sample_size_; }

private:
    size_t num_samples_;
    size_t sample_size_;
    float* data_;
    int fd_ = -1;
    std::vector<float> simulated_data_;  // Fallback for non-Linux
};

// ============================================================
// DATA LOADER WITH PREFETCHING
// ============================================================

class CppDataLoader {
public:
    CppDataLoader(size_t num_samples, size_t sample_size, size_t batch_size,
                  int num_workers, int prefetch_factor)
        : num_samples_(num_samples)
        , sample_size_(sample_size)
        , batch_size_(batch_size)
        , num_workers_(num_workers)
        , prefetch_queue_(prefetch_factor)
        , current_epoch_(0)
        , samples_loaded_(0)
        , running_(false) {
        
        // Generate indices
        indices_.resize(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            indices_[i] = i;
        }
        
        // Simulate data (in real use, this would be mmap'd)
        data_.resize(num_samples * sample_size);
        labels_.resize(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            labels_[i] = i % 1000;  // 1000 classes
            for (size_t j = 0; j < sample_size; j++) {
                data_[i * sample_size + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
    
    void start() {
        running_ = true;
        shuffle();
        
        // Start worker threads
        for (int i = 0; i < num_workers_; i++) {
            workers_.emplace_back(&CppDataLoader::worker_fn, this, i);
        }
    }
    
    void stop() {
        running_ = false;
        prefetch_queue_.set_done();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    bool get_batch(Batch& batch) {
        return prefetch_queue_.pop(batch);
    }
    
    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
        samples_loaded_ = 0;
    }
    
    size_t num_batches() const {
        return num_samples_ / batch_size_;
    }

private:
    void worker_fn(int worker_id) {
        while (running_) {
            // Get batch indices
            size_t start_idx = samples_loaded_.fetch_add(batch_size_);
            if (start_idx >= num_samples_) {
                break;  // Epoch done
            }
            
            // Create batch
            Batch batch;
            batch.batch_size = std::min(batch_size_, num_samples_ - start_idx);
            batch.sample_size = sample_size_;
            batch.data.resize(batch.batch_size * sample_size_);
            batch.labels.resize(batch.batch_size);
            
            // Load samples
            for (size_t i = 0; i < batch.batch_size; i++) {
                size_t sample_idx = indices_[start_idx + i];
                
                // Copy data (simulates decode/transform)
                std::memcpy(batch.data.data() + i * sample_size_,
                           data_.data() + sample_idx * sample_size_,
                           sample_size_ * sizeof(float));
                
                batch.labels[i] = labels_[sample_idx];
            }
            
            // Push to prefetch queue
            prefetch_queue_.push(std::move(batch));
        }
    }

    size_t num_samples_;
    size_t sample_size_;
    size_t batch_size_;
    int num_workers_;
    
    std::vector<size_t> indices_;
    std::vector<float> data_;
    std::vector<int> labels_;
    
    ThreadSafeQueue<Batch> prefetch_queue_;
    std::vector<std::thread> workers_;
    
    int current_epoch_;
    std::atomic<size_t> samples_loaded_;
    std::atomic<bool> running_;
};

// ============================================================
// BENCHMARKS
// ============================================================

void benchmark_dataloader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  C++ DATALOADER BENCHMARK                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    size_t num_samples = 50000;
    size_t sample_size = 224 * 224 * 3;  // RGB image
    size_t batch_size = 64;
    
    std::cout << "Configuration:\n";
    std::cout << "  Samples: " << num_samples << "\n";
    std::cout << "  Sample size: " << sample_size << " floats ("
              << sample_size * sizeof(float) / 1024 << " KB)\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Total data: " << num_samples * sample_size * sizeof(float) / (1024*1024) 
              << " MB\n\n";
    
    // Test different worker counts
    std::vector<int> worker_counts = {1, 2, 4, 8};
    
    std::cout << "Workers  Batches/sec  Samples/sec  Time/batch(ms)\n";
    std::cout << "─────────────────────────────────────────────────\n";
    
    for (int num_workers : worker_counts) {
        CppDataLoader loader(num_samples, sample_size, batch_size, num_workers, 4);
        
        // Warmup
        loader.start();
        Batch batch;
        for (int i = 0; i < 10; i++) {
            loader.get_batch(batch);
        }
        loader.stop();
        
        // Benchmark
        loader.start();
        
        Timer timer;
        timer.start();
        
        int num_batches = 100;
        for (int i = 0; i < num_batches; i++) {
            if (!loader.get_batch(batch)) break;
        }
        
        double elapsed = timer.stop_ms();
        loader.stop();
        
        double batches_per_sec = num_batches / (elapsed / 1000.0);
        double samples_per_sec = batches_per_sec * batch_size;
        double ms_per_batch = elapsed / num_batches;
        
        std::cout << num_workers << "        " 
                  << static_cast<int>(batches_per_sec) << "          "
                  << static_cast<int>(samples_per_sec) << "        "
                  << ms_per_batch << "\n";
    }
}

void benchmark_memcpy() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  MEMORY COPY BENCHMARK (Critical for Data Loading)           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::vector<size_t> sizes = {64*1024, 256*1024, 1024*1024, 16*1024*1024};
    
    std::cout << "Size        Bandwidth (GB/s)\n";
    std::cout << "────────────────────────────\n";
    
    for (size_t size : sizes) {
        std::vector<char> src(size, 'A');
        std::vector<char> dst(size);
        
        // Warmup
        std::memcpy(dst.data(), src.data(), size);
        
        Timer timer;
        int iterations = 100;
        
        timer.start();
        for (int i = 0; i < iterations; i++) {
            std::memcpy(dst.data(), src.data(), size);
        }
        double elapsed = timer.stop_ms();
        
        double bandwidth = (size * iterations) / (elapsed / 1000.0) / 1e9;
        
        std::cout << size / 1024 << " KB      " << bandwidth << "\n";
    }
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "\n";
    std::cout << "████████████████████████████████████████████████████████████████\n";
    std::cout << "█  HIGH-PERFORMANCE C++ DATA LOADER                            █\n";
    std::cout << "████████████████████████████████████████████████████████████████\n";
    
    benchmark_memcpy();
    benchmark_dataloader();
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  KEY C++ DATA LOADING TECHNIQUES                             ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  1. MEMORY MAPPING (mmap)                                    ║\n";
    std::cout << "║     • Zero-copy access to file data                          ║\n";
    std::cout << "║     • OS handles page caching                                ║\n";
    std::cout << "║     • Use madvise() for access pattern hints                 ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  2. PREFETCHING WITH THREAD POOL                             ║\n";
    std::cout << "║     • Multiple workers load in parallel                      ║\n";
    std::cout << "║     • Queue holds ready batches                              ║\n";
    std::cout << "║     • Training never waits for I/O                           ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  3. PINNED MEMORY (for GPU transfer)                         ║\n";
    std::cout << "║     • cudaHostAlloc for faster DMA                           ║\n";
    std::cout << "║     • Enables non-blocking transfers                         ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  4. SEQUENTIAL I/O                                           ║\n";
    std::cout << "║     • Pack data into single file                             ║\n";
    std::cout << "║     • Shuffle indices, not files                             ║\n";
    std::cout << "║     • Maximize disk throughput                               ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}
