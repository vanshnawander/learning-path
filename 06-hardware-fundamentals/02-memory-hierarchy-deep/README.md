# Memory Hierarchy Deep Dive

Understanding the memory hierarchy is THE most important concept for performance.

## The Memory Wall Problem

```
                    Performance Growth (per year)
CPU Speed:          ~50% (historically)
Memory Bandwidth:   ~10%
Memory Latency:     ~7%

Result: CPUs wait for memory most of the time!
```

## Complete Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐  ← Registers (< 1 ns, ~KB)                       │
│  │    CPU    │                                                   │
│  │ Registers │  Fastest, smallest                                │
│  └─────┬─────┘                                                   │
│        │                                                         │
│  ┌─────▼─────┐  ← L1 Cache (~1 ns, 32-64 KB per core)           │
│  │  L1 I/D   │                                                   │
│  │  Cache    │  Split: Instructions / Data                       │
│  └─────┬─────┘                                                   │
│        │                                                         │
│  ┌─────▼─────┐  ← L2 Cache (~4 ns, 256KB-1MB per core)          │
│  │    L2     │                                                   │
│  │   Cache   │  Unified, per-core                                │
│  └─────┬─────┘                                                   │
│        │                                                         │
│  ┌─────▼─────┐  ← L3 Cache (~10-20 ns, 8-64 MB shared)          │
│  │    L3     │                                                   │
│  │   Cache   │  Shared across all cores (LLC)                    │
│  └─────┬─────┘                                                   │
│        │                                                         │
│  ┌─────▼─────┐  ← Main Memory (~100 ns, 32-512 GB)              │
│  │   DRAM    │                                                   │
│  │   DDR5    │  High capacity, moderate speed                    │
│  └─────┬─────┘                                                   │
│        │                                                         │
│  ┌─────▼─────┐  ← Storage (~10 µs SSD, ~10 ms HDD)              │
│  │   SSD/    │                                                   │
│  │   HDD     │  Persistent, huge capacity                        │
│  └───────────┘                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Cache Line: The Unit of Transfer

```
Cache Line = 64 bytes (typically)

When you access ONE byte:
┌────────────────────────────────────────────────────────────┐
│  The ENTIRE 64-byte line is loaded from memory!            │
│                                                             │
│  Memory: [████████████████████████████████████████████████] │
│          └──────────── 64 bytes ──────────────┘             │
│                        ↓                                    │
│  Cache:  [████████████████████████████████████████████████] │
│                                                             │
│  You access: [█]                                            │
│  But you get: [████████████████████████████████████████████] │
└────────────────────────────────────────────────────────────┘

IMPLICATION: Sequential access is essentially FREE after first access!
```

## Files in This Directory

| File | Description |
|------|-------------|
| `01_cache_line_effects.c` | Measure cache line impact |
| `02_cache_associativity.c` | N-way set associative caches |
| `03_tlb_misses.c` | Translation Lookaside Buffer |
| `04_numa_effects.c` | Non-Uniform Memory Access |
| `05_memory_bandwidth.c` | Measure real bandwidth |
| `06_prefetching_deep.c` | Hardware vs software prefetch |

## Cache Organization

### Direct Mapped
```
Address → Exactly one cache location
Simple but conflicts are common
```

### N-Way Set Associative (Modern CPUs)
```
Address → One of N locations in a "set"
Typical: 8-way or 16-way associative

Set index = (Address / Cache_Line_Size) % Num_Sets

Example: 8-way, 32KB L1D, 64B lines
  Num_Sets = 32KB / (8 * 64B) = 64 sets
```

## DRAM: How It Really Works

```
DRAM Chip Organization:
┌─────────────────────────────────────────┐
│            Bank 0        Bank 1         │
│         ┌─────────┐   ┌─────────┐       │
│         │ Row 0   │   │ Row 0   │       │
│         │ Row 1   │   │ Row 1   │       │
│         │ ...     │   │ ...     │       │
│         │ Row N   │   │ Row N   │       │
│         └────┬────┘   └────┬────┘       │
│              │             │            │
│         Row Buffer    Row Buffer        │
│         (open row)    (open row)        │
│              │             │            │
│              └──────┬──────┘            │
│                     │                   │
│               Data Bus (64-bit)         │
└─────────────────────────────────────────┘

Access patterns:
- Same row (row buffer hit):    ~10 ns
- Different row (row buffer miss): ~50 ns
- Different bank (bank conflict): ~100 ns
```

## Bandwidth vs Latency

```
Bandwidth = How much data per second (GB/s)
Latency = How long until first byte (ns)

Analogy:
  Bandwidth = Width of the highway
  Latency = Distance to destination

You can have:
  - High bandwidth, high latency (HDD bulk read)
  - Low bandwidth, low latency (L1 cache)
  - High bandwidth, low latency (GPU HBM - expensive!)
```

## ML Implications

### 1. Tensor Layout Matters
```python
# NCHW: Channel-first (PyTorch default)
# Memory: [B0C0H0W0, B0C0H0W1, ..., B0C0H1W0, ...]
#         Spatial locality in width dimension

# NHWC: Channel-last (TensorFlow, better for some ops)  
# Memory: [B0H0W0C0, B0H0W0C1, ..., B0H0W1C0, ...]
#         Channel values contiguous
```

### 2. Batch Size Selection
```
Small batch: More cache hits, less parallelism
Large batch: Better GPU utilization, more memory
Optimal: Fits in L3 cache during compute
```

### 3. Tiling/Blocking
```
Instead of:
  for i in range(N):
      for j in range(M):
          C[i,j] = A[i,:] @ B[:,j]

Do:
  for ii in range(0, N, TILE):
      for jj in range(0, M, TILE):
          # Process TILE x TILE block
          # Reuse data in cache!
```
