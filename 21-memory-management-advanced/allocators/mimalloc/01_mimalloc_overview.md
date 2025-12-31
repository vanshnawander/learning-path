# mimalloc: Microsoft's High-Performance Allocator

## Overview

mimalloc is a compact, high-performance allocator from Microsoft Research. Key characteristics:

- **~10K lines of code** - Simple, auditable
- **Excellent performance** - Often beats jemalloc/tcmalloc
- **Low fragmentation** - Free list sharding
- **Secure mode** - Guard pages, randomization
- **Portable** - Windows, Linux, macOS, WASM

## Core Design: Free List Sharding

The key innovation is **multi-level free list sharding**:

```
Traditional allocator:
┌─────────────────────────────────────────────┐
│ One big free list per size class            │
│ [obj]→[obj]→[obj]→[obj]→[obj]→[obj]→...   │
│                                             │
│ Problem: Lock contention, cache misses      │
└─────────────────────────────────────────────┘

mimalloc:
┌─────────────────────────────────────────────┐
│ Free list sharded per "mimalloc page"       │
│                                             │
│ Page 0: [obj]→[obj]→[obj]                  │
│ Page 1: [obj]→[obj]                        │
│ Page 2: [obj]→[obj]→[obj]→[obj]            │
│                                             │
│ Benefit: Better locality, less contention   │
└─────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Heap                                 │
│  (One heap per thread by default)                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │                    Pages                            │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐              │    │
│  │  │ Page 0  │ │ Page 1  │ │ Page 2  │  ...         │    │
│  │  │ 64KB    │ │ 64KB    │ │ 64KB    │              │    │
│  │  │         │ │         │ │         │              │    │
│  │  │ size=32 │ │ size=64 │ │ size=32 │              │    │
│  │  └─────────┘ └─────────┘ └─────────┘              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Segments (Coarse allocation)          │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │  Segment: 4MB of pages                    │     │    │
│  │  │  [Page][Page][Page][Page]...             │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Page Structure

Each mimalloc page (default 64KB):
- Contains blocks of **one size class**
- Has its own free list
- Thread-local free list + delayed free list

```c
// Simplified page structure
struct mi_page_s {
    uint8_t segment_idx;
    uint8_t segment_in_use:1;
    uint8_t is_reset:1;
    uint8_t is_committed:1;
    
    uint16_t capacity;      // Total blocks
    uint16_t reserved;      // Reserved blocks
    
    mi_block_t* free;       // Thread-local free list
    mi_block_t* local_free; // Freed by this thread
    _Atomic(mi_block_t*) xthread_free; // Freed by other threads
    
    uint32_t used;          // Blocks in use
    uint32_t block_size;    // Size of each block
    
    mi_heap_t* heap;        // Owning heap
    struct mi_page_s* next; // Next page in queue
    struct mi_page_s* prev;
};
```

### Dual Free Lists

Key to mimalloc's concurrency:

```
┌─────────────────────────────────────────────────────────────┐
│                    Page Free Lists                           │
│                                                              │
│  local_free: [obj]→[obj]→[obj]                              │
│      │       (Freed by owning thread, no sync needed)       │
│      │                                                       │
│  xthread_free: [obj]→[obj]                                  │
│      │         (Freed by other threads, uses CAS)           │
│      ↓                                                       │
│  On allocation:                                              │
│    1. Try local_free first                                  │
│    2. If empty, atomically swap xthread_free to local_free  │
│    3. Allocate from local_free                              │
└─────────────────────────────────────────────────────────────┘
```

**Why this works:**
- Owning thread: Lock-free allocation/free
- Other threads: Single CAS to push to xthread_free
- No complex synchronization needed

## Size Classes

```
Bin   Size    Objects/64KB
0     8       8192
1     16      4096
2     24      2730
3     32      2048
4     40      1638
5     48      1365
...
~70   1MB     N/A (large allocation)
```

Size classes follow pattern:
- 8-byte aligned up to 128 bytes
- Then ~12.5% spacing
- Good balance of fragmentation vs. memory overhead

## Usage

### Drop-in Replacement

```bash
# Linux (LD_PRELOAD)
LD_PRELOAD=/usr/lib/libmimalloc.so ./myprogram

# Windows (via DLL injection or link)
# Link with mimalloc-override.dll

# CMake
find_package(mimalloc REQUIRED)
target_link_libraries(myapp mimalloc)
```

### Explicit API

```c
#include <mimalloc.h>

void* p = mi_malloc(100);
void* q = mi_zalloc(100);     // Zero-initialized
void* r = mi_realloc(p, 200);
mi_free(r);

// Aligned allocation
void* a = mi_malloc_aligned(1024, 64);  // 64-byte aligned

// Heap-specific allocation
mi_heap_t* heap = mi_heap_new();
void* h = mi_heap_malloc(heap, 100);
mi_heap_destroy(heap);  // Frees all allocations in heap
```

### Configuration

```bash
# Environment variables
MIMALLOC_SHOW_STATS=1    # Print stats on exit
MIMALLOC_VERBOSE=1       # Verbose output
MIMALLOC_LARGE_OS_PAGES=1 # Use huge pages
MIMALLOC_ARENA_EAGER_COMMIT=1 # Eager commit
MIMALLOC_PAGE_RESET=1    # Reset pages to reduce RSS
```

## Secure Mode

```c
// Compile with MI_SECURE=4 for full security
// Or at runtime:
mi_option_set(mi_option_secure, 4);

// Security features:
// - Guard pages around allocations
// - Randomized page addresses
// - Encoded free lists (prevents use-after-free exploits)
// - Double-free detection
// - Heap overflow detection
```

## Performance Characteristics

| Metric | mimalloc | jemalloc | tcmalloc | glibc |
|--------|----------|----------|----------|-------|
| Single-thread alloc | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Multi-thread alloc | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Fragmentation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Memory overhead | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Code size | 10KB | 100KB+ | 100KB+ | N/A |

## When to Use mimalloc

**Good fit:**
- General-purpose replacement for malloc
- Multi-threaded applications
- Memory-intensive workloads
- Need low fragmentation
- Want security features

**Consider alternatives:**
- jemalloc: Better profiling/introspection
- tcmalloc: Extremely latency-sensitive (slightly faster p99)
- Custom allocator: Very specific allocation patterns

## References

- GitHub: https://github.com/microsoft/mimalloc
- Paper: "mimalloc: Free List Sharding in Action" (MSR)
- Technical Report: https://www.microsoft.com/en-us/research/publication/mimalloc-free-list-sharding-in-action/
