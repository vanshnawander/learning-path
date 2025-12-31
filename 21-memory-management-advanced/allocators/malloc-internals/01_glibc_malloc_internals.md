# glibc malloc Internals

## Overview

glibc malloc (ptmalloc2) is the default memory allocator on Linux systems. Understanding its internals is crucial for performance optimization and debugging memory issues.

## Core Concepts

### Arena

An arena is a memory region managed by malloc. Each arena has its own:
- Free lists (bins)
- Memory regions (heaps)
- Synchronization (mutex)

```
┌─────────────────────────────────────────────────────┐
│                     Process                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Arena 0 │ │ Arena 1 │ │ Arena 2 │ │ Arena N │   │
│  │ (main)  │ │(thread) │ │(thread) │ │(thread) │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │         │
│       v           v           v           v         │
│  ┌─────────────────────────────────────────────┐   │
│  │           Per-Arena Free Lists (Bins)        │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐           │   │
│  │  │Fastbins│ │SmallBin│ │LargeBin│ ...       │   │
│  │  └────────┘ └────────┘ └────────┘           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Multi-arena design**:
- Main arena: Uses `brk()` for heap growth
- Thread arenas: Uses `mmap()` for heap allocation
- Number of arenas: `8 * num_cores` (64-bit) or `2 * num_cores` (32-bit)
- Threads attach to arenas to reduce contention

### Chunk Structure

Every allocation is wrapped in a chunk with metadata:

```
Allocated Chunk:
┌─────────────────────────────────────────┐
│ prev_size (if previous chunk is free)   │  8 bytes (64-bit)
├─────────────────────────────────────────┤
│ size | flags (A|M|P)                    │  8 bytes
├─────────────────────────────────────────┤
│                                         │
│           User Data                     │
│                                         │
├─────────────────────────────────────────┤
│ (next chunk's prev_size uses this space)│
└─────────────────────────────────────────┘

Free Chunk:
┌─────────────────────────────────────────┐
│ prev_size                               │
├─────────────────────────────────────────┤
│ size | flags                            │
├─────────────────────────────────────────┤
│ fd (forward pointer)                    │
├─────────────────────────────────────────┤
│ bk (backward pointer)                   │
├─────────────────────────────────────────┤
│ fd_nextsize (large bins only)           │
├─────────────────────────────────────────┤
│ bk_nextsize (large bins only)           │
├─────────────────────────────────────────┤
│         (unused space)                  │
└─────────────────────────────────────────┘

Flags (lowest 3 bits of size):
  P (PREV_INUSE): Previous chunk is allocated
  M (IS_MMAPPED): Chunk was allocated via mmap
  A (NON_MAIN_ARENA): Chunk belongs to non-main arena
```

### Minimum Chunk Size

```c
// 64-bit system
#define MIN_CHUNK_SIZE  32  // 16 bytes header + 16 bytes min data
#define MALLOC_ALIGNMENT 16

// Minimum allocation: 32 bytes (including overhead)
// User gets: 24 bytes usable (next chunk borrows 8 bytes)
```

## Bin Organization

### Fastbins (LIFO, No Coalescing)

```
Size Range: 32, 48, 64, 80, 96, 112, 128 bytes (7 bins, 64-bit)

fastbins[0] ──> chunk_32 ──> chunk_32 ──> chunk_32 ──> NULL
fastbins[1] ──> chunk_48 ──> chunk_48 ──> NULL
fastbins[2] ──> chunk_64 ──> NULL
...

Characteristics:
- Single-linked list (only fd pointer)
- No coalescing with neighbors
- Very fast allocation/deallocation
- Can cause fragmentation
```

**Why LIFO?** Recently freed chunks are likely still in cache.

### Small Bins (FIFO, Exact Size)

```
Size Range: 32 to 1008 bytes (62 bins)
Each bin: One exact size (multiples of 16 bytes)

smallbins[0] = 32 bytes
smallbins[1] = 48 bytes
smallbins[2] = 64 bytes
...
smallbins[61] = 1008 bytes

Structure: Circular doubly-linked list
┌──────────────────────────────────────────────────┐
│  bin_head <──> chunk1 <──> chunk2 <──> bin_head  │
└──────────────────────────────────────────────────┘

Allocate from: front (oldest)
Free to: back (newest)
```

### Large Bins (Sorted by Size)

```
Size Range: >= 1024 bytes (63 bins)
Bins cover size ranges that increase logarithmically:

bins[64-95]:   32 bins,  64 byte range each (1024-3072)
bins[96-111]:  16 bins, 512 byte range each
bins[112-119]:  8 bins, 4096 byte range each
bins[120-123]:  4 bins, 32768 byte range each
bins[124-125]:  2 bins, 262144 byte range each
bins[126]:      1 bin,  remaining sizes

Within each bin: sorted by size (descending)
Uses fd_nextsize/bk_nextsize for skip list of unique sizes
```

### Unsorted Bin

```
bin[1] = Unsorted bin (single bin)

Purpose: Temporary holding area
- Recently freed chunks go here first
- Chunks consolidated during malloc scan
- Best-fit search checks unsorted bin

Flow:
1. free() puts chunk in unsorted bin
2. Next malloc() scans unsorted bin
3. Chunks either:
   - Satisfy request (return to user)
   - Move to appropriate small/large bin
```

### Tcache (Thread-Local Cache) - glibc 2.26+

```
Per-thread cache:
┌────────────────────────────────────────┐
│ Thread-Local Storage                   │
│  ┌──────────────────────────────────┐  │
│  │ tcache_perthread_struct          │  │
│  │  entries[64]: 64 size classes    │  │
│  │  counts[64]: chunks per bin      │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

Size range: 32 to 1040 bytes (64 bins)
Max chunks per bin: 7 (default)

Benefits:
- No locking (thread-local)
- Very fast path for common sizes
- Checked before fastbins
```

## Allocation Algorithm

```
malloc(size) flow:

1. Check tcache
   └─> If match found, return (no lock)

2. If size <= 64 bytes (fastbin range)
   └─> Check fastbins
       └─> If found, return

3. If size <= 1008 bytes (small bin range)
   └─> Check exact-size small bin
       └─> If found, return

4. Consolidate fastbins into unsorted bin

5. Scan unsorted bin
   └─> For each chunk:
       - If exact match, return
       - If last remainder and large enough, split and return
       - Otherwise, sort into small/large bins

6. If size > 1008 bytes
   └─> Search large bins (best fit)
       └─> If found, possibly split and return

7. Check small bins again (may have been populated)

8. Binmap-guided search of larger bins
   └─> Find smallest bin with chunks
       └─> Split chunk if needed

9. Use top chunk (wilderness)
   └─> Split from top of heap
   └─> If top too small, extend heap (brk/mmap)

10. Last resort: mmap() for large requests
```

## Free Algorithm

```
free(ptr) flow:

1. If chunk fits in tcache
   └─> Add to tcache (no lock)
   └─> Return

2. If chunk is mmap'd
   └─> munmap() and return

3. If chunk fits in fastbin
   └─> Add to fastbin head
   └─> Return

4. Coalesce with neighbors:
   └─> Check PREV_INUSE flag
       └─> If previous is free, merge backward
   └─> Check next chunk's PREV_INUSE
       └─> If next is free (not top), merge forward

5. If next chunk is top chunk
   └─> Merge with top

6. Otherwise
   └─> Add to unsorted bin
```

## Key Tuning Parameters

```c
// Tunable via mallopt() or environment variables

M_MXFAST (default: 64*sizeof(size_t) = 128 bytes on 64-bit)
  Maximum fastbin size

M_TRIM_THRESHOLD (default: 128KB)
  Amount of free memory at top before returning to OS

M_TOP_PAD (default: 0)
  Extra padding when extending heap

M_MMAP_THRESHOLD (default: 128KB)
  Size above which malloc uses mmap instead of brk

M_MMAP_MAX (default: 65536)
  Maximum number of mmap regions

M_ARENA_MAX (default: 8 * cores)
  Maximum number of arenas

M_ARENA_TEST (default: 8 on 64-bit)
  Number of arenas to try before creating new one
```

### Environment Variables

```bash
# Disable tcache (debugging)
export GLIBC_TUNABLES=glibc.malloc.tcache_count=0

# Limit arenas
export MALLOC_ARENA_MAX=4

# Debug output
export MALLOC_CHECK_=3

# Use mmap for all allocations
export MALLOC_MMAP_THRESHOLD_=0
```

## Memory Layout Example

```
Process Virtual Address Space:
┌─────────────────────────────────────┐ High addresses
│            Stack                     │ (grows down)
│               ↓                      │
├─────────────────────────────────────┤
│         (unmapped)                   │
├─────────────────────────────────────┤
│                                      │
│    Memory-mapped regions             │
│    (mmap'd allocations, libs)        │
│               ↑                      │
├─────────────────────────────────────┤
│         (unmapped)                   │
├─────────────────────────────────────┤
│               ↑                      │
│            Heap                      │ (grows up)
│    [main arena heap - brk]           │
├─────────────────────────────────────┤ brk / program break
│            BSS                       │
├─────────────────────────────────────┤
│            Data                      │
├─────────────────────────────────────┤
│            Text                      │
└─────────────────────────────────────┘ Low addresses
```

## Debugging malloc

```c
// Enable internal checks
#include <malloc.h>

// Heap consistency check
malloc_check(0);  // or MALLOC_CHECK_=1 env var

// Print arena statistics
malloc_stats();

// Detailed heap info
malloc_info(0, stdout);

// Using mtrace for leak detection
#include <mcheck.h>
mtrace();  // Enable tracing
// ... allocations ...
muntrace();  // Disable and write log
// Run: mtrace ./program mtrace.log
```

## Performance Characteristics

| Operation | Best Case | Worst Case | Notes |
|-----------|-----------|------------|-------|
| malloc (tcache hit) | O(1), no lock | - | Most common |
| malloc (fastbin) | O(1) | O(1) | Lock per arena |
| malloc (small bin) | O(1) | O(1) | FIFO |
| malloc (large bin) | O(log n) | O(n) | Sorted search |
| malloc (mmap) | O(1) | - | System call overhead |
| free (tcache) | O(1), no lock | - | Most common |
| free (fastbin) | O(1) | O(1) | No coalescing |
| free (coalesce) | O(1) | O(1) | Check neighbors |

## Common Issues

1. **Fragmentation**: Fastbins don't coalesce
   - Fix: `malloc_trim(0)` or tune M_MXFAST

2. **Arena contention**: Many threads, few arenas
   - Fix: Increase M_ARENA_MAX or use tcmalloc/jemalloc

3. **Memory not returned to OS**: Top chunk not trimmed
   - Fix: `malloc_trim(0)` or lower M_TRIM_THRESHOLD

4. **mmap overhead**: Many large allocations
   - Fix: Adjust M_MMAP_THRESHOLD

## References

- glibc malloc source: `malloc/malloc.c`
- "Understanding glibc malloc" - sploitfun
- "Malloc Internals" - glibc wiki
