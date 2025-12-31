/*
 * Modern C++ Threading (C++11/14/17/20)
 * 
 * Demonstrates std::thread, std::async, std::mutex, and synchronization
 * 
 * Compile: g++ -std=c++17 -pthread -O2 01_cpp_threading.cpp -o cpp_threading
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <vector>
#include <queue>
#include <chrono>
#include <functional>

using namespace std::chrono_literals;

// ============================================================
// 1. Basic std::thread Usage
// ============================================================

void basic_thread_demo() {
    std::cout << "\n=== Basic std::thread ===\n";
    
    // Lambda thread
    std::thread t1([]() {
        std::cout << "Thread 1: Hello from lambda!\n";
    });
    
    // Function thread
    auto func = [](int id, const std::string& msg) {
        std::cout << "Thread " << id << ": " << msg << "\n";
    };
    std::thread t2(func, 2, "Hello with arguments");
    
    // Get hardware concurrency
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    
    t1.join();
    t2.join();
}

// ============================================================
// 2. std::mutex and RAII Locks
// ============================================================

class Counter {
    int value = 0;
    mutable std::mutex mtx;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);  // RAII lock
        ++value;
    }
    
    int get() const {
        std::lock_guard<std::mutex> lock(mtx);
        return value;
    }
};

void mutex_demo() {
    std::cout << "\n=== Mutex Demo ===\n";
    
    Counter counter;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 10000; j++) {
                counter.increment();
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    std::cout << "Final count: " << counter.get() 
              << " (expected: 100000)\n";
}

// ============================================================
// 3. std::shared_mutex (Reader-Writer Lock)
// ============================================================

class ThreadSafeMap {
    std::map<int, std::string> data;
    mutable std::shared_mutex mtx;
    
public:
    void write(int key, const std::string& value) {
        std::unique_lock<std::shared_mutex> lock(mtx);  // Exclusive
        data[key] = value;
    }
    
    std::string read(int key) const {
        std::shared_lock<std::shared_mutex> lock(mtx);  // Shared
        auto it = data.find(key);
        return (it != data.end()) ? it->second : "";
    }
};

void shared_mutex_demo() {
    std::cout << "\n=== Shared Mutex (RW Lock) ===\n";
    
    ThreadSafeMap map;
    map.write(1, "one");
    map.write(2, "two");
    
    // Multiple readers can run concurrently
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; i++) {
        readers.emplace_back([&map, i]() {
            for (int j = 0; j < 1000; j++) {
                map.read(1);
                map.read(2);
            }
            std::cout << "Reader " << i << " done\n";
        });
    }
    
    for (auto& t : readers) t.join();
}

// ============================================================
// 4. std::condition_variable
// ============================================================

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(std::move(value));
        }
        cv.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !queue.empty(); });
        T value = std::move(queue.front());
        queue.pop();
        return value;
    }
    
    bool try_pop(T& value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        if (!cv.wait_for(lock, timeout, [this]() { return !queue.empty(); })) {
            return false;
        }
        value = std::move(queue.front());
        queue.pop();
        return true;
    }
};

void condition_variable_demo() {
    std::cout << "\n=== Condition Variable ===\n";
    
    ThreadSafeQueue<int> queue;
    
    // Producer
    std::thread producer([&queue]() {
        for (int i = 0; i < 10; i++) {
            queue.push(i);
            std::cout << "Produced: " << i << "\n";
            std::this_thread::sleep_for(50ms);
        }
    });
    
    // Consumer
    std::thread consumer([&queue]() {
        for (int i = 0; i < 10; i++) {
            int value = queue.pop();
            std::cout << "Consumed: " << value << "\n";
        }
    });
    
    producer.join();
    consumer.join();
}

// ============================================================
// 5. std::async and std::future
// ============================================================

int compute_expensive(int n) {
    std::this_thread::sleep_for(100ms);  // Simulate work
    return n * n;
}

void async_demo() {
    std::cout << "\n=== std::async Demo ===\n";
    
    // Launch async tasks
    auto future1 = std::async(std::launch::async, compute_expensive, 5);
    auto future2 = std::async(std::launch::async, compute_expensive, 10);
    auto future3 = std::async(std::launch::deferred, compute_expensive, 15);
    
    std::cout << "Tasks launched...\n";
    
    // Get results (blocks until ready)
    std::cout << "Result 1: " << future1.get() << "\n";
    std::cout << "Result 2: " << future2.get() << "\n";
    std::cout << "Result 3 (deferred): " << future3.get() << "\n";  // Executes here
}

// ============================================================
// 6. std::atomic
// ============================================================

void atomic_demo() {
    std::cout << "\n=== std::atomic Demo ===\n";
    
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter]() {
            for (int j = 0; j < 10000; j++) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    std::cout << "Atomic counter: " << counter.load() 
              << " (expected: 100000)\n";
    
    // Compare-and-swap
    int expected = 100000;
    bool success = counter.compare_exchange_strong(expected, 0);
    std::cout << "CAS success: " << std::boolalpha << success << "\n";
}

// ============================================================
// 7. std::jthread (C++20) - Cooperative Cancellation
// ============================================================

#if __cplusplus >= 202002L
void jthread_demo() {
    std::cout << "\n=== std::jthread (C++20) ===\n";
    
    std::jthread worker([](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            std::cout << "Working...\n";
            std::this_thread::sleep_for(100ms);
        }
        std::cout << "Stop requested, cleaning up...\n";
    });
    
    std::this_thread::sleep_for(350ms);
    // worker.request_stop() called automatically on destruction
    // or explicitly: worker.request_stop();
}
#endif

// ============================================================
// 8. Parallel Algorithms (C++17)
// ============================================================

#include <algorithm>
#include <execution>
#include <numeric>

void parallel_algorithms_demo() {
    std::cout << "\n=== Parallel Algorithms (C++17) ===\n";
    
    std::vector<int> data(10000000);
    std::iota(data.begin(), data.end(), 0);
    
    // Sequential
    auto start = std::chrono::high_resolution_clock::now();
    auto sum_seq = std::reduce(std::execution::seq, 
                               data.begin(), data.end(), 0LL);
    auto seq_time = std::chrono::high_resolution_clock::now() - start;
    
    // Parallel
    start = std::chrono::high_resolution_clock::now();
    auto sum_par = std::reduce(std::execution::par, 
                               data.begin(), data.end(), 0LL);
    auto par_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Sequential: " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(seq_time).count() << " ms\n";
    std::cout << "Parallel:   " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(par_time).count() << " ms\n";
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Modern C++ Threading Demo ===\n";
    
    basic_thread_demo();
    mutex_demo();
    shared_mutex_demo();
    condition_variable_demo();
    async_demo();
    atomic_demo();
    
#if __cplusplus >= 202002L
    jthread_demo();
#endif
    
    parallel_algorithms_demo();
    
    std::cout << "\nAll demos completed!\n";
    return 0;
}
