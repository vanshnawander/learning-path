# Hardware and Software Prefetching

## Why Prefetching Matters

Memory latency is 100-300 cycles. Prefetching hides this by fetching data before it's needed.

```
Without prefetching:
CPU: [compute]----[STALL waiting for memory]----[compute]----[STALL]...

With prefetching:
CPU: [compute]----[compute]----[compute]----[compute]...
          ↑            ↑            ↑
       prefetch     prefetch     prefetch
       issued       issued       issued
```

## Hardware Prefetching

Modern CPUs have automatic prefetchers:

### Types of Hardware Prefetchers

| Prefetcher | Pattern | Description |
|------------|---------|-------------|
| **Stride** | A, A+S, A+2S | Detects constant stride access |
| **Stream** | A, A+1, A+2 | Sequential access (L1/L2) |
| **Adjacent Line** | Line N → N+1 | Fetches neighboring cache lines |
| **DCU IP** | Per-instruction | Tracks patterns per load instruction |

### Intel Prefetcher Controls

```bash
# Check prefetcher status (requires MSR access)
rdmsr 0x1A4

# Bit meanings:
# Bit 0: L2 hardware prefetcher disable
# Bit 1: L2 adjacent cache line prefetcher disable
# Bit 2: DCU hardware prefetcher disable
# Bit 3: DCU IP prefetcher disable

# Disable all prefetchers (for benchmarking)
wrmsr -a 0x1A4 0xF

# Enable all prefetchers
wrmsr -a 0x1A4 0x0
```

## Software Prefetching

### x86 Prefetch Instructions

```c
#include <xmmintrin.h>  // SSE intrinsics

// Prefetch to L1 cache (temporal - will be reused)
_mm_prefetch((char*)ptr, _MM_HINT_T0);

// Prefetch to L2 cache
_mm_prefetch((char*)ptr, _MM_HINT_T1);

// Prefetch to L3 cache
_mm_prefetch((char*)ptr, _MM_HINT_T2);

// Non-temporal prefetch (streaming, won't pollute cache)
_mm_prefetch((char*)ptr, _MM_HINT_NTA);

// GCC built-in (portable)
__builtin_prefetch(ptr, 0, 3);  // read, high locality
__builtin_prefetch(ptr, 1, 0);  // write, no locality
```

### Prefetch Distance Calculation

```
Optimal prefetch distance = Memory latency × Computation rate

Example:
- Memory latency: 200 cycles
- Loop iteration: 50 cycles
- Distance = 200 / 50 = 4 iterations ahead

for (int i = 0; i < N; i++) {
    __builtin_prefetch(&data[i + 4], 0, 3);  // 4 iterations ahead
    process(data[i]);
}
```

## Practical Examples

### Array Processing with Prefetch

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000
#define PREFETCH_DISTANCE 16

void process_no_prefetch(double* data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i] * data[i];
    }
}

void process_with_prefetch(double* data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        __builtin_prefetch(&data[i + PREFETCH_DISTANCE], 0, 3);
        sum += data[i] * data[i];
    }
}

// Benchmark shows 10-30% improvement for memory-bound code
```

### Linked List Traversal

```c
struct Node {
    struct Node* next;
    int data;
    char padding[56];  // Ensure different cache lines
};

// Without prefetch: each node access waits for memory
void traverse_no_prefetch(struct Node* head) {
    while (head) {
        process(head->data);
        head = head->next;
    }
}

// With prefetch: overlap memory access
void traverse_with_prefetch(struct Node* head) {
    struct Node* lookahead = head;
    
    // Build up prefetch distance
    for (int i = 0; i < 4 && lookahead; i++) {
        lookahead = lookahead->next;
    }
    
    while (head) {
        if (lookahead) {
            __builtin_prefetch(lookahead, 0, 3);
            lookahead = lookahead->next;
        }
        process(head->data);
        head = head->next;
    }
}
```

### Matrix Operations

```c
// Prefetch next row while processing current row
void matrix_row_sum(double* matrix, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Prefetch next row
        if (i + 1 < rows) {
            for (int j = 0; j < cols; j += 8) {  // Cache line = 64 bytes = 8 doubles
                __builtin_prefetch(&matrix[(i+1)*cols + j], 0, 3);
            }
        }
        
        // Process current row
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i*cols + j];
        }
        result[i] = sum;
    }
}
```

## When Prefetching Helps/Hurts

### Helps When:
- Memory-bound workloads
- Predictable access patterns
- Large working sets
- Pointer chasing (linked structures)

### Hurts When:
- Cache already has data (wasted bandwidth)
- Unpredictable patterns (wrong prefetches)
- Compute-bound code (no memory stalls)
- Small working sets (fits in cache)

## Measuring Prefetch Effectiveness

```bash
# Count prefetch instructions with perf
perf stat -e L1-dcache-prefetches,L1-dcache-prefetch-misses ./program

# Check cache miss rates before/after
perf stat -e cache-misses,cache-references ./program_no_prefetch
perf stat -e cache-misses,cache-references ./program_with_prefetch
```

## Best Practices

1. **Profile first** - Only prefetch if memory-bound
2. **Tune distance** - Too early wastes cache, too late doesn't help
3. **Don't over-prefetch** - Bandwidth is limited
4. **Consider hardware prefetcher** - Often sufficient for sequential access
5. **Test on target hardware** - Effectiveness varies by CPU

## References

- Intel Optimization Manual
- "What Every Programmer Should Know About Memory" - Drepper
- Agner Fog's optimization guides
