# NUMA: Non-Uniform Memory Access

## What is NUMA?

NUMA (Non-Uniform Memory Access) is a memory architecture where memory access time depends on the memory location relative to the processor. In a NUMA system, each processor (or group of processors) has "local" memory that it can access faster than "remote" memory attached to other processors.

```
Traditional SMP (Symmetric Multi-Processing):
┌─────────────────────────────────────────┐
│          Shared Memory Bus              │
└─────┬─────┬─────┬─────┬─────┬─────┬────┘
      │     │     │     │     │     │
    ┌─┴─┐ ┌─┴─┐ ┌─┴─┐ ┌─┴─┐ ┌───────┴───────┐
    │CPU│ │CPU│ │CPU│ │CPU│ │    Memory     │
    └───┘ └───┘ └───┘ └───┘ └───────────────┘
    
    Problem: Memory bus becomes bottleneck

NUMA Architecture:
┌──────────────────┐         ┌──────────────────┐
│     Node 0       │         │     Node 1       │
│  ┌───┐  ┌───┐   │         │   ┌───┐  ┌───┐  │
│  │CPU│  │CPU│   │◄───────►│   │CPU│  │CPU│  │
│  └───┘  └───┘   │ Inter-  │   └───┘  └───┘  │
│       │         │ connect │         │       │
│  ┌────┴────┐    │ (QPI/   │    ┌────┴────┐  │
│  │ Memory  │    │  UPI/   │    │ Memory  │  │
│  │ (Local) │    │ Infini) │    │ (Local) │  │
│  └─────────┘    │         │    └─────────┘  │
└──────────────────┘         └──────────────────┘
```

## NUMA Latency and Bandwidth

### Typical Latency Numbers (Modern Intel/AMD)

| Access Type | Latency (approx) | Relative |
|-------------|------------------|----------|
| Local DRAM | 80-100 ns | 1.0x |
| Remote DRAM (1 hop) | 130-150 ns | 1.5-1.7x |
| Remote DRAM (2 hops) | 180-220 ns | 2.0-2.5x |

### NUMA Distance Matrix

The Linux kernel exposes NUMA distances via sysfs:

```bash
$ cat /sys/devices/system/node/node*/distance
# Example output for 2-node system:
# node0: 10 21
# node1: 21 10

# Example for 4-node system:
# node0: 10 21 21 21
# node1: 21 10 21 21
# node2: 21 21 10 21
# node3: 21 21 21 10
```

The numbers represent relative distances (10 = local, higher = further).

## AMD EPYC: CCX/CCD/NPS Complexity

AMD EPYC processors have additional NUMA-like behavior within a socket:

```
EPYC Socket (Zen 3/4):
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │  CCD 0  │  │  CCD 1  │  │  CCD 2  │  │  CCD 3  ││
│  │┌───┬───┐│  │┌───┬───┐│  │┌───┬───┐│  │┌───┬───┐││
│  ││CCX│CCX││  ││CCX│CCX││  ││CCX│CCX││  ││CCX│CCX│││
│  │└───┴───┘│  │└───┴───┘│  │└───┴───┘│  │└───┴───┘││
│  │   L3    │  │   L3    │  │   L3    │  │   L3    ││
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘│
│       │            │            │            │     │
│       └────────────┼────────────┼────────────┘     │
│                    │ Infinity   │                  │
│              ┌─────┴─────┬──────┴─────┐           │
│              │ Memory    │  Memory    │           │
│              │ Channel   │  Channel   │           │
│              └───────────┴────────────┘           │
└─────────────────────────────────────────────────────┘

NPS (NUMA Per Socket) Settings:
- NPS1: Entire socket = 1 NUMA node
- NPS2: Socket split into 2 NUMA nodes  
- NPS4: Socket split into 4 NUMA nodes
```

**NPS Impact**:
- **NPS1**: Simplest programming, all memory appears local, but higher average latency
- **NPS4**: Best for NUMA-aware code, lowest local latency, but requires careful placement

## Intel Xeon: Sub-NUMA Clustering (SNC)

Intel offers Sub-NUMA Clustering (SNC) which splits the LLC into regions:

```
Xeon with SNC2 enabled:
┌─────────────────────────────────────────┐
│ NUMA Node 0          │ NUMA Node 1      │
│ (Half of LLC)        │ (Half of LLC)    │
│ Memory Controllers   │ Memory Ctrls    │
│ 0, 1                 │ 2, 3            │
└─────────────────────────────────────────┘
```

**When to use SNC**:
- Enable for latency-sensitive, NUMA-aware workloads
- Disable for non-NUMA-aware or memory-bandwidth-bound workloads

## Detecting NUMA Topology

### Linux Commands

```bash
# Full NUMA topology
numactl --hardware

# Example output:
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23
# node 0 size: 64301 MB
# node 0 free: 62156 MB
# node 1 cpus: 8 9 10 11 12 13 14 15 24 25 26 27 28 29 30 31
# node 1 size: 64507 MB
# node 1 free: 63822 MB
# node distances:
# node   0   1
#   0:  10  21
#   1:  21  10

# CPU to NUMA node mapping
lscpu | grep NUMA

# Detailed topology
lstopo-no-graphics
```

### Programmatic Detection (Linux)

```c
#include <numa.h>
#include <stdio.h>

int main() {
    if (numa_available() < 0) {
        printf("NUMA not available\n");
        return 1;
    }
    
    int num_nodes = numa_max_node() + 1;
    printf("NUMA nodes: %d\n", num_nodes);
    
    // Get memory per node
    for (int i = 0; i < num_nodes; i++) {
        long free_mem;
        long total_mem = numa_node_size(i, &free_mem);
        printf("Node %d: %ld MB total, %ld MB free\n", 
               i, total_mem / (1024*1024), free_mem / (1024*1024));
    }
    
    // Get CPUs per node
    for (int i = 0; i < num_nodes; i++) {
        struct bitmask* cpumask = numa_allocate_cpumask();
        numa_node_to_cpus(i, cpumask);
        printf("Node %d CPUs: ", i);
        for (int cpu = 0; cpu < numa_num_configured_cpus(); cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                printf("%d ", cpu);
            }
        }
        printf("\n");
        numa_free_cpumask(cpumask);
    }
    
    return 0;
}
```

## NUMA Memory Policies

### Linux Memory Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `default` | Allocate on local node | General purpose |
| `bind` | Only allocate from specific nodes | Dedicated memory pools |
| `interleave` | Round-robin across nodes | Bandwidth-intensive, shared |
| `preferred` | Prefer specific node, fallback to others | Soft affinity |

### Setting Policies

```bash
# Run with memory interleaved across all nodes
numactl --interleave=all ./myprogram

# Run on specific node (CPU + memory)
numactl --cpunodebind=0 --membind=0 ./myprogram

# Run on node 0 CPUs with memory preferred on node 0
numactl --cpunodebind=0 --preferred=0 ./myprogram
```

### Programmatic Control

```c
#include <numa.h>
#include <numaif.h>

// Allocate on specific node
void* ptr = numa_alloc_onnode(size, node_id);

// Allocate with interleaving
void* ptr = numa_alloc_interleaved(size);

// Set memory policy for future allocations
numa_set_interleave_mask(numa_all_nodes_ptr);

// Move existing pages to a different node
numa_move_pages(pid, count, pages, nodes, status, MPOL_MF_MOVE);

// Get the node where a page resides
int node = numa_node_of_cpu(sched_getcpu());
```

## NUMA Optimization Strategies

### 1. First-Touch Policy (Default)

Memory is allocated on the node where it's first accessed:

```c
// Thread 0 initializes, so memory goes to node 0
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    array[i] = 0;  // First touch determines placement
}

// Better: Parallel first-touch for distributed data
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    // Each thread touches its portion
    // Memory ends up on each thread's local node
    array[i] = 0;
}
```

### 2. Memory Interleaving

For bandwidth-intensive shared data:

```c
// Interleave across all NUMA nodes
struct bitmask* all_nodes = numa_allocate_nodemask();
numa_bitmask_setall(all_nodes);
numa_set_interleave_mask(all_nodes);

void* shared_data = malloc(large_size);

numa_set_interleave_mask(numa_no_nodes_ptr);  // Reset
```

### 3. Explicit Placement

For performance-critical allocations:

```c
// Allocate on the node where this thread runs
int node = numa_node_of_cpu(sched_getcpu());
void* local_data = numa_alloc_onnode(size, node);

// Or use specific node
void* node0_data = numa_alloc_onnode(size, 0);
```

### 4. Thread-Data Affinity

Ensure threads run near their data:

```c
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int node = tid / threads_per_node;
    
    // Bind this thread to the node
    numa_run_on_node(node);
    
    // Work on data that's on this node
    process_local_partition(data + tid * partition_size);
}
```

## Monitoring NUMA Performance

```bash
# Watch NUMA statistics
numastat -c
numastat -p <pid>

# Example output:
#                   Node 0     Node 1
# numa_hit         12345678   11234567
# numa_miss             234        345
# numa_foreign          345        234
# local_node       12345444   11234222
# other_node            234        345

# Using perf
perf stat -e 'node-loads,node-load-misses,node-stores,node-store-misses' ./myprogram
```

## Anti-Patterns to Avoid

1. **Serial initialization, parallel computation**
   - All memory ends up on one node
   - Fix: Parallel first-touch

2. **Ignoring NUMA with `malloc()`**
   - Default policy may not be optimal
   - Fix: Use `numa_alloc_*` or set policy

3. **Thread migration**
   - Threads moving to different nodes lose locality
   - Fix: Pin threads with `pthread_setaffinity_np()` or `numactl`

4. **Memory pooling across nodes**
   - Pool returns memory from wrong node
   - Fix: Per-node memory pools

## References

- "What Every Programmer Should Know About Memory" - Ulrich Drepper
- NUMA API (libnuma) documentation
- Intel and AMD processor documentation
