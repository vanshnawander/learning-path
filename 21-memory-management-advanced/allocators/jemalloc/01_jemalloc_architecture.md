# jemalloc Architecture

## Overview

jemalloc is a general-purpose memory allocator originally developed for FreeBSD and later adopted by Facebook, Firefox, Redis, and many other high-performance systems. It emphasizes:

- **Scalability**: Minimizes lock contention via thread caches and multiple arenas
- **Fragmentation avoidance**: Size classes and slab allocation
- **Introspection**: Rich statistics and profiling

## Hierarchical Structure

```
┌─────────────────────────────────────────────────────────────┐
│                        jemalloc                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Thread Cache                        │   │
│  │  Per-thread, lock-free allocation for small objects   │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                     │   │
│  │  │tc[0]│ │tc[1]│ │tc[2]│ │ ... │  (size classes)     │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                      Arenas                           │   │
│  │  Multiple arenas to reduce contention                 │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │Arena 0 │ │Arena 1 │ │Arena 2 │ │Arena N │        │   │
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘        │   │
│  │      │          │          │          │              │   │
│  └──────┼──────────┼──────────┼──────────┼──────────────┘   │
│         │          │          │          │                   │
│         ↓          ↓          ↓          ↓                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Extents/Slabs                       │   │
│  │  Contiguous memory regions from OS                    │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │ Page runs: Small objects packed in slabs    │     │   │
│  │  │ Large allocations: Dedicated page runs      │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Size Classes

jemalloc uses carefully chosen size classes to minimize internal fragmentation:

```
Small Size Classes (< 14KB default):
┌─────────────────────────────────────────────────────────────┐
│ Tiny: 8, 16 bytes (quantum = 8 or 16)                       │
│ Quantum-spaced: 32, 48, 64, 80, 96, 112, 128               │
│ Sub-page: 160, 192, 224, 256, 320, 384, 448, 512, ...      │
│ (spacing increases: 16 → 32 → 64 → 128 → ...)              │
└─────────────────────────────────────────────────────────────┘

Large Size Classes (>= 14KB, < chunk size):
┌─────────────────────────────────────────────────────────────┐
│ Page-multiple sizes: 16KB, 20KB, 24KB, 28KB, 32KB, ...     │
│ Each size class is multiple of page size                    │
└─────────────────────────────────────────────────────────────┘

Huge Allocations (>= chunk size, default 2MB):
┌─────────────────────────────────────────────────────────────┐
│ Allocated as one or more chunks directly                    │
│ Managed separately from arena slabs                         │
└─────────────────────────────────────────────────────────────┘
```

### Size Class Design Philosophy

```
Target: ~25% maximum internal fragmentation
Strategy: lg(size) determines spacing

For size s:
  - Find smallest size class >= s
  - Internal fragmentation = (class_size - s) / class_size

Example:
  Request 100 bytes → class 112 → 10.7% waste
  Request 1000 bytes → class 1024 → 2.4% waste
  Request 5000 bytes → class 5120 → 2.4% waste
```

## Thread Cache (tcache)

```c
// Per-thread structure
struct tcache_s {
    // Ticker for periodic maintenance
    ticker_t gc_ticker;
    
    // Bins for each small size class
    tcache_bin_t bins_small[NBINS];
    
    // Optional bins for large classes
    tcache_bin_t bins_large[...];
};

struct tcache_bin_s {
    // Stack of cached objects
    void **avail;       // Stack top
    uint32_t ncached;   // Current count
    uint32_t low_water; // Min since last GC
    uint32_t lg_fill_div; // Fill count = max >> lg_fill_div
};
```

### tcache Operation

```
Allocation (small):
1. Check tcache bin for size class
   └─> If available: pop from stack, return (lock-free!)
2. If tcache empty:
   └─> Refill from arena bin (batch fill)
   └─> Return one, cache rest

Free (small):
1. If tcache bin not full:
   └─> Push to stack (lock-free!)
2. If tcache full:
   └─> Flush half to arena bin
   └─> Push freed object
```

### tcache Garbage Collection

```
Periodic GC (every N allocations):
1. For each bin:
   └─> Reduce cached count toward low_water
   └─> Return excess to arena
2. Reset low_water marks

Goal: Reclaim unused cached objects
      Adapt to changing allocation patterns
```

## Arenas

```c
struct arena_s {
    // Arena index
    unsigned ind;
    
    // Statistics
    arena_stats_t stats;
    
    // Bins for small allocations
    bin_t bins[NBINS];
    
    // Extents: available page runs
    extent_heap_t extents_dirty;   // Recently freed
    extent_heap_t extents_muzzy;   // Purged but not released
    extent_heap_t extents_retained; // Clean, reusable
    
    // Large allocation tracking
    extent_list_t large;
    
    // Decay timers for purging
    arena_decay_t decay_dirty;
    arena_decay_t decay_muzzy;
};
```

### Arena Assignment

```
Thread → Arena mapping:
1. Round-robin assignment to balance load
2. Number of arenas = f(CPU cores):
   - Default: 4 * cores (Linux)
   - Configurable via opt.narenas

Thread affinity:
- Thread sticks to assigned arena
- Reduces cache thrashing
- Can be overridden per-allocation
```

## Slabs (Page Runs)

```
Small allocation slabs:
┌─────────────────────────────────────────────────────────────┐
│                    Page Run (e.g., 4 pages)                  │
├─────────────────────────────────────────────────────────────┤
│ ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐          │
│ │obj0││obj1││obj2││obj3││obj4││obj5││obj6││obj7│ ...       │
│ └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘          │
├─────────────────────────────────────────────────────────────┤
│ Metadata: bitmap tracking free/allocated objects            │
│ All objects in slab are same size class                     │
└─────────────────────────────────────────────────────────────┘
```

### Slab Bins

```c
struct bin_s {
    // Lock for this bin
    malloc_mutex_t lock;
    
    // Current slab for allocations
    slab_t *slabcur;
    
    // Heap of non-full slabs (sorted by address)
    slab_heap_t slabs_nonfull;
    
    // Stats
    bin_stats_t stats;
};
```

### Allocation from Slab

```
bin_malloc(bin, size_class):
1. Lock bin
2. Try slabcur (current slab)
   └─> If has free object: allocate, return
3. If slabcur full or NULL:
   └─> Get slab from slabs_nonfull heap
   └─> If heap empty: allocate new slab
4. Allocate from slab's free bitmap
5. Unlock bin
6. Return object
```

## Extent Management

```
Extent: Contiguous range of pages

┌─────────────────────────────────────────────────────────────┐
│                      Extent Lifecycle                        │
│                                                              │
│   [OS Memory]                                                │
│       │                                                      │
│       ↓ (mmap/VirtualAlloc)                                  │
│   [Retained] ←───────────────────────────┐                   │
│       │                                   │                  │
│       ↓ (commit)                         │                   │
│   [Dirty] ←──────────────────────────────┤                   │
│       │                                   │ (free)           │
│       ↓ (purge: madvise DONTNEED)       │                   │
│   [Muzzy] ─────────────────────────→     │                   │
│       │                                   │                  │
│       ↓ (decommit/unmap)                 │                   │
│   [Released to OS]                        │                  │
└─────────────────────────────────────────────────────────────┘
```

### Extent Decay

```
Background decay:
- Dirty → Muzzy: After dirty_decay_ms (default: 10 seconds)
- Muzzy → Released: After muzzy_decay_ms (default: 10 seconds)

Benefit: Gradual memory release
         Quick reuse of recently freed memory
         Smooth RSS behavior

Configuration:
  MALLOC_CONF="dirty_decay_ms:5000,muzzy_decay_ms:5000"
```

## Configuration Options

```bash
# Via environment variable
export MALLOC_CONF="opt1:val1,opt2:val2"

# Key options:
narenas:<n>          # Number of arenas
dirty_decay_ms:<ms>  # Dirty page decay time
muzzy_decay_ms:<ms>  # Muzzy page decay time
background_thread:true  # Enable background threads
tcache:false         # Disable thread cache
prof:true            # Enable heap profiling
prof_leak:true       # Report leaks on exit
stats_print:true     # Print stats on exit
```

## Profiling and Statistics

```c
// Enable at compile time and runtime
// MALLOC_CONF="prof:true"

// Dump heap profile
mallctl("prof.dump", NULL, NULL, NULL, 0);

// Get statistics
size_t allocated, active, mapped;
size_t len = sizeof(size_t);
mallctl("stats.allocated", &allocated, &len, NULL, 0);
mallctl("stats.active", &active, &len, NULL, 0);
mallctl("stats.mapped", &mapped, &len, NULL, 0);

// Print detailed stats
malloc_stats_print(NULL, NULL, NULL);
```

### Example Statistics Output

```
___ Begin jemalloc statistics ___
Version: "5.3.0"
...
Arenas: 8
Quantum: 16
Page size: 4096
...
bins:           size ind    allocated      nmalloc     nrequests
                  8    0        48000        12000         24000
                 16    1       128000        16000         32000
                 32    2       256000        16000         28000
...
large:          size ind    allocated      nmalloc     nrequests
               16384   0       163840           10            15
               20480   1        81920            4             6
...
```

## jemalloc vs glibc malloc

| Feature | jemalloc | glibc malloc |
|---------|----------|--------------|
| Thread cache | Full-featured | tcache (2.26+) |
| Arenas | 4*cores default | 8*cores default |
| Size classes | ~200 classes | ~130 classes |
| Fragmentation | Lower | Higher |
| Statistics | Comprehensive | Basic |
| Profiling | Built-in | mtrace |
| Memory return | Decay-based | Threshold |
| Huge pages | Transparent | Manual |

## When to Use jemalloc

**Good fit**:
- Long-running servers
- Multi-threaded workloads
- Memory-intensive applications
- Need profiling/debugging

**Consider alternatives**:
- Embedded systems (mimalloc smaller)
- Extreme latency requirements (tcmalloc)
- Simple applications (glibc sufficient)

## References

- jemalloc GitHub: https://github.com/jemalloc/jemalloc
- "A Scalable Concurrent malloc" - Jason Evans
- jemalloc documentation and wiki
