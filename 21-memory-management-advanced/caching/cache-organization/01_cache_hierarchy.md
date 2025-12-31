# Cache Hierarchy and Organization

## Modern Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU Core 0                              │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │  L1 I-Cache │  │  L1 D-Cache │  32-64 KB, 4-cycle latency   │
│  │   32 KB     │  │   32 KB     │                              │
│  └──────┬──────┘  └──────┬──────┘                              │
│         └────────┬───────┘                                      │
│                  ▼                                              │
│         ┌───────────────┐                                       │
│         │   L2 Cache    │  256-512 KB, 12-cycle latency        │
│         │    256 KB     │  (private per core)                  │
│         └───────┬───────┘                                       │
└─────────────────┼───────────────────────────────────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────────┐
│                 ▼                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    L3 Cache (LLC)                       │    │
│  │                      30-50 MB                           │    │
│  │                   (shared by all cores)                 │    │
│  │                   30-50 cycle latency                   │    │
│  └────────────────────────────┬───────────────────────────┘    │
│                               │                                  │
│  ┌────────────────────────────┴───────────────────────────┐    │
│  │                    Memory Controller                    │    │
│  └────────────────────────────┬───────────────────────────┘    │
└───────────────────────────────┼─────────────────────────────────┘
                                ▼
                    ┌───────────────────┐
                    │    Main Memory    │  100-300 cycle latency
                    │    (DRAM)         │
                    └───────────────────┘
```

## Cache Line Structure

```
Cache Line (64 bytes on x86):
┌────────────────────────────────────────────────────────────────┐
│ Tag │ Index │ Offset │ Data (64 bytes)                         │
└────────────────────────────────────────────────────────────────┘

Address breakdown (48-bit address, 32KB 8-way L1):
┌─────────────────────┬──────────┬────────┐
│       Tag           │  Index   │ Offset │
│     (36 bits)       │ (6 bits) │(6 bits)│
└─────────────────────┴──────────┴────────┘
         ↓                 ↓         ↓
   Identifies line    Set number  Byte within
                      (64 sets)   cache line
```

## Set-Associative Cache

```
8-way set-associative cache (32 KB):
- 64 sets
- 8 ways per set
- 64 bytes per line
- Total: 64 × 8 × 64 = 32768 bytes

Set 0:  [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]
Set 1:  [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]
Set 2:  [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]
...
Set 63: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]

Address → Index → Search all ways in set → Hit/Miss
```

## Cache Lookup Process

```c
// Pseudocode for cache lookup
uint64_t address = 0x7FFE1234AB80;

// Extract fields
uint64_t offset = address & 0x3F;           // Bottom 6 bits
uint64_t index = (address >> 6) & 0x3F;     // Next 6 bits (for 64 sets)
uint64_t tag = address >> 12;               // Remaining bits

// Search set
for (int way = 0; way < 8; way++) {
    if (cache[index][way].valid && cache[index][way].tag == tag) {
        // HIT: Return data
        return cache[index][way].data[offset];
    }
}
// MISS: Fetch from lower level
```

## Replacement Policies

### LRU (Least Recently Used)
```
Access order: A, B, C, D, E, F, G, H, I (8-way cache)

After H: [A][B][C][D][E][F][G][H]  (MRU→LRU order: H,G,F,E,D,C,B,A)
After I: [I][B][C][D][E][F][G][H]  (Evict A, insert I)
After A: [I][A][C][D][E][F][G][H]  (Evict B, insert A)

True LRU requires log2(N!) bits per set - expensive!
```

### Pseudo-LRU (Tree-based)
```
Binary tree for 8 ways:
          [0]
         /   \
       [1]   [2]
       / \   / \
     [3][4][5][6]
     /\ /\ /\ /\
    0 1 2 3 4 5 6 7  ← Ways

Bits point to "less recently used" subtree
Only 7 bits needed vs 15 for true LRU
```

### RRIP (Re-Reference Interval Prediction)
```
Each line has 2-3 bit RRIP value:
- 0: Near-immediate re-reference expected
- 3: Distant re-reference expected

On miss: Scan for RRIP=3, evict. Else increment all and retry.
On hit: Set RRIP=0

Better than LRU for scan-resistant behavior.
```

## Write Policies

### Write-Back vs Write-Through

```
Write-Back (most common):
┌──────┐     ┌──────┐     ┌──────┐
│ CPU  │────►│Cache │     │Memory│
│write │     │dirty │     │stale │
└──────┘     └──────┘     └──────┘
                │
                ▼ (on eviction)
             ┌──────┐
             │Memory│
             │update│
             └──────┘

Write-Through:
┌──────┐     ┌──────┐     ┌──────┐
│ CPU  │────►│Cache │────►│Memory│
│write │     │clean │     │update│
└──────┘     └──────┘     └──────┘
```

### Write Allocate vs No-Write-Allocate

```
Write-Allocate (on write miss):
1. Fetch line from memory
2. Update line in cache
3. Mark dirty

No-Write-Allocate (on write miss):
1. Write directly to memory
2. Don't bring into cache
```

## Cache Coherence (Multi-core)

### MESI Protocol

```
States:
- M (Modified): Dirty, exclusive
- E (Exclusive): Clean, exclusive  
- S (Shared): Clean, shared
- I (Invalid): Not present

State transitions:
┌─────────────────────────────────────────────────────────────┐
│                          MESI                                │
│                                                              │
│         ┌────────────────────────────────────┐              │
│         │              Modified              │              │
│         │     (dirty, only this cache)      │              │
│         └───────────┬───────────────────────┘              │
│                     │ Other core read                       │
│                     ▼                                        │
│         ┌────────────────────────────────────┐              │
│         │              Shared                │              │
│ Write   │     (clean, multiple caches)      │◄─┐           │
│ hit     └───────────┬───────────────────────┘  │ Read      │
│    │                │ Write (invalidate others)│ miss      │
│    │                ▼                          │            │
│    │    ┌────────────────────────────────────┐│            │
│    └───►│             Exclusive              ││            │
│         │     (clean, only this cache)      │─┘            │
│         └───────────┬───────────────────────┘              │
│                     │ Eviction                              │
│                     ▼                                        │
│         ┌────────────────────────────────────┐              │
│         │              Invalid               │              │
│         │          (not present)             │              │
│         └────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Querying Cache Info (Linux)

```bash
# Cache sizes and organization
getconf -a | grep CACHE

# Detailed cache info
lscpu | grep -i cache

# Per-core cache topology
cat /sys/devices/system/cpu/cpu0/cache/index*/size
cat /sys/devices/system/cpu/cpu0/cache/index*/type
cat /sys/devices/system/cpu/cpu0/cache/index*/ways_of_associativity

# Example output:
# index0: 32K L1 data, 8-way
# index1: 32K L1 instruction, 8-way
# index2: 256K L2, 4-way
# index3: 30720K L3, 20-way
```

## Performance Implications

| Access Type | Latency | Bandwidth |
|-------------|---------|-----------|
| L1 hit | 4 cycles | ~1 TB/s |
| L2 hit | 12 cycles | ~500 GB/s |
| L3 hit | 40 cycles | ~200 GB/s |
| DRAM | 200 cycles | ~50 GB/s |

**Working set sizing:**
- < L1: Ideal performance
- < L2: Good performance
- < L3: Acceptable
- > L3: Memory-bound
