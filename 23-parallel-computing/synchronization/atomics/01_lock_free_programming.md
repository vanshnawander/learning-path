# Lock-Free Programming

## What is Lock-Free?

**Lock-free**: At least one thread makes progress in finite steps, regardless of other threads' state.

**Wait-free**: Every thread makes progress in finite steps (stronger guarantee).

**Obstruction-free**: A thread makes progress if run in isolation.

```
Progress Guarantees (strongest to weakest):
Wait-free > Lock-free > Obstruction-free > Blocking (mutex)
```

## Why Lock-Free?

| Problem with Locks | Lock-Free Solution |
|-------------------|-------------------|
| Priority inversion | No thread can block another |
| Deadlock | No locks to deadlock |
| Convoying | No waiting in queue |
| Signal safety | Safe in signal handlers |

## Atomic Operations

### Hardware Support

Modern CPUs provide atomic instructions:

| Operation | x86 | ARM | Description |
|-----------|-----|-----|-------------|
| Load | MOV | LDR | Atomic read |
| Store | MOV | STR | Atomic write |
| Exchange | XCHG | SWP | Atomic swap |
| CAS | CMPXCHG | LDXR/STXR | Compare-and-swap |
| Fetch-Add | LOCK XADD | LDADD | Atomic increment |

### C11 Atomics

```c
#include <stdatomic.h>

// Atomic types
atomic_int counter = 0;
_Atomic(void*) ptr = NULL;

// Basic operations
atomic_store(&counter, 10);
int val = atomic_load(&counter);

// Read-modify-write
atomic_fetch_add(&counter, 1);  // Returns old value
atomic_fetch_sub(&counter, 1);
atomic_fetch_or(&counter, 0xFF);
atomic_fetch_and(&counter, 0x0F);

// Compare-and-swap (CAS)
int expected = 5;
bool success = atomic_compare_exchange_strong(&counter, &expected, 10);
// If counter == expected: set counter = 10, return true
// Else: set expected = counter, return false
```

### C++ Atomics

```cpp
#include <atomic>

std::atomic<int> counter{0};
std::atomic<Node*> head{nullptr};

// Operations
counter.store(10);
int val = counter.load();
int old = counter.fetch_add(1);

// CAS
int expected = 5;
bool success = counter.compare_exchange_strong(expected, 10);

// Weak CAS (may spuriously fail - use in loops)
while (!counter.compare_exchange_weak(expected, expected + 1)) {
    // expected updated with current value
}
```

## Memory Ordering

### The Problem

```cpp
// Thread 1          // Thread 2
x = 1;               while (!ready);
ready = true;        assert(x == 1);  // Can fail!
```

Without memory ordering, compiler/CPU may reorder operations.

### Memory Order Options

```cpp
// From weakest to strongest:

memory_order_relaxed  // No ordering, just atomicity
memory_order_consume  // Data-dependent reads ordered (rarely used)
memory_order_acquire  // Reads after this see writes before release
memory_order_release  // Writes before this visible after acquire
memory_order_acq_rel  // Both acquire and release
memory_order_seq_cst  // Total ordering (default, safest, slowest)
```

### Common Patterns

```cpp
// Pattern 1: Release-Acquire (message passing)
std::atomic<bool> ready{false};
int data;

// Producer
data = 42;                                    // Non-atomic
ready.store(true, std::memory_order_release); // Release

// Consumer  
while (!ready.load(std::memory_order_acquire)); // Acquire
assert(data == 42);  // Guaranteed!

// Pattern 2: Relaxed counters (no ordering needed)
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);

// Pattern 3: Sequential consistency (simple, safe)
std::atomic<int> x{0}, y{0};
// Thread 1: x.store(1);
// Thread 2: y.store(1);
// Thread 3: if (x.load() == 1 && y.load() == 0) ...
// Thread 4: if (y.load() == 1 && x.load() == 0) ...
// With seq_cst: threads 3 and 4 cannot both succeed
```

## Lock-Free Data Structures

### Lock-Free Stack (Treiber Stack)

```cpp
template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T val) : data(val), next(nullptr) {}
    };
    
    std::atomic<Node*> head{nullptr};
    
public:
    void push(T val) {
        Node* new_node = new Node(val);
        new_node->next = head.load(std::memory_order_relaxed);
        
        // CAS loop
        while (!head.compare_exchange_weak(
                   new_node->next, new_node,
                   std::memory_order_release,
                   std::memory_order_relaxed));
    }
    
    bool pop(T& result) {
        Node* old_head = head.load(std::memory_order_acquire);
        
        while (old_head) {
            if (head.compare_exchange_weak(
                    old_head, old_head->next,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                result = old_head->data;
                // Note: Memory reclamation is complex!
                // delete old_head; // DANGER: ABA problem
                return true;
            }
        }
        return false;
    }
};
```

### The ABA Problem

```
Thread 1:              Thread 2:
1. Read head = A       
2. Get A->next = B     
                       3. Pop A
                       4. Pop B  
                       5. Push A (reuse)
3. CAS(head, A, B)     
   Succeeds! But B is gone!
```

**Solutions**:
- Hazard pointers
- Epoch-based reclamation
- Tagged pointers (ABA counter)

### Lock-Free Queue (Michael-Scott)

```cpp
template<typename T>
class LockFreeQueue {
    struct Node {
        T data;
        std::atomic<Node*> next{nullptr};
    };
    
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node();
        head.store(dummy);
        tail.store(dummy);
    }
    
    void enqueue(T val) {
        Node* new_node = new Node();
        new_node->data = val;
        
        while (true) {
            Node* last = tail.load(std::memory_order_acquire);
            Node* next = last->next.load(std::memory_order_acquire);
            
            if (last == tail.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    if (last->next.compare_exchange_weak(
                            next, new_node,
                            std::memory_order_release)) {
                        tail.compare_exchange_strong(
                            last, new_node,
                            std::memory_order_release);
                        return;
                    }
                } else {
                    // Help advance tail
                    tail.compare_exchange_weak(
                        last, next,
                        std::memory_order_release);
                }
            }
        }
    }
    
    bool dequeue(T& result) {
        while (true) {
            Node* first = head.load(std::memory_order_acquire);
            Node* last = tail.load(std::memory_order_acquire);
            Node* next = first->next.load(std::memory_order_acquire);
            
            if (first == head.load(std::memory_order_acquire)) {
                if (first == last) {
                    if (next == nullptr) return false;
                    tail.compare_exchange_weak(last, next);
                } else {
                    result = next->data;
                    if (head.compare_exchange_weak(
                            first, next,
                            std::memory_order_release)) {
                        // Reclaim first
                        return true;
                    }
                }
            }
        }
    }
};
```

## Industry Best Practices

### When to Use Lock-Free

**Good fit**:
- Very high contention
- Real-time requirements
- Signal handlers
- Simple data structures (counters, stacks)

**Avoid when**:
- Complex invariants
- Need blocking semantics
- Debugging/maintenance concerns
- Performance isn't critical

### Guidelines

1. **Start with locks** - Optimize only if needed
2. **Use standard library** - `std::atomic`, `tbb::concurrent_*`
3. **Test thoroughly** - Use ThreadSanitizer, stress tests
4. **Measure** - Lock-free isn't always faster
5. **Document** - Memory orderings are subtle

## References

- "C++ Concurrency in Action" - Anthony Williams
- "The Art of Multiprocessor Programming" - Herlihy & Shavit
- "Is Parallel Programming Hard?" - Paul McKenney
