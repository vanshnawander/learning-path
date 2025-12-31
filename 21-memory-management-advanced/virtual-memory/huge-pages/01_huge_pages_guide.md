# Huge Pages: Deep Dive

## Why Huge Pages Matter

Standard 4KB pages create overhead:
- **TLB pressure**: TLB can only cache ~1500 entries
- **Page table size**: 4-level page walks for each miss
- **4KB × 1500 = 6MB** addressable with full TLB hit

With 2MB huge pages:
- **2MB × 1500 = 3TB** addressable with full TLB hit
- Fewer page table entries
- Fewer TLB misses

## Page Sizes by Architecture

| Architecture | Standard | Large | Huge/Giant |
|--------------|----------|-------|------------|
| x86-64 | 4KB | 2MB | 1GB |
| ARM64 | 4KB | 2MB | 1GB |
| POWER | 4KB | 64KB | 16MB |

## Linux Huge Page Types

### 1. HugeTLB (Traditional Huge Pages)

Pre-allocated, reserved pool of huge pages.

```bash
# Check current huge page configuration
cat /proc/meminfo | grep Huge
# HugePages_Total:     128
# HugePages_Free:      128
# HugePages_Rsvd:        0
# Hugepagesize:       2048 kB

# Reserve huge pages at boot (grub)
# Add to kernel command line:
hugepages=128 hugepagesz=2M

# Or at runtime:
echo 128 > /proc/sys/vm/nr_hugepages

# For 1GB pages (must be at boot):
# hugepagesz=1G hugepages=4
```

**Using HugeTLB in Code:**

```c
#include <sys/mman.h>

// Method 1: mmap with MAP_HUGETLB
void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

// Method 2: Mount hugetlbfs and use files
// mount -t hugetlbfs none /mnt/huge
int fd = open("/mnt/huge/myfile", O_CREAT | O_RDWR, 0600);
ftruncate(fd, size);
void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```

### 2. Transparent Huge Pages (THP)

Kernel automatically promotes/demotes pages.

```bash
# Check THP status
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# Options:
# always - THP for all processes
# madvise - THP only when requested via madvise()
# never - Disable THP

# Set THP mode
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
```

**Using THP in Code:**

```c
#include <sys/mman.h>

void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

// Request THP for this region
madvise(ptr, size, MADV_HUGEPAGE);

// Or prevent THP
madvise(ptr, size, MADV_NOHUGEPAGE);
```

## THP Pitfalls

### 1. Allocation Stalls

THP can cause latency spikes:
```bash
# Check compaction activity
cat /proc/vmstat | grep compact
# compact_stall 12345  <- THP waiting for compaction
```

**Fix**: Use `madvise` mode instead of `always`.

### 2. Memory Bloat

THP can waste memory for sparse access:
```
Request: 100KB
Without THP: 25 × 4KB pages = 100KB
With THP: 1 × 2MB page = 2MB (1.9MB wasted!)
```

**Fix**: Disable THP for sparse allocations.

### 3. khugepaged CPU Usage

Background daemon consumes CPU collapsing pages.

```bash
# Check khugepaged activity
cat /sys/kernel/mm/transparent_hugepage/khugepaged/pages_collapsed

# Tune scan interval (milliseconds)
echo 10000 > /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs
```

## Industry Best Practices

### Database Workloads (Redis, PostgreSQL)

```bash
# Disable THP for databases (causes latency variance)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Use explicit HugeTLB for buffer pools
# PostgreSQL: huge_pages = on in postgresql.conf
```

### HPC / Scientific Computing

```bash
# Use explicit 1GB pages for large arrays
# Reserve at boot: hugepagesz=1G hugepages=16

# In code:
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
void* ptr = mmap(NULL, 16ULL*1024*1024*1024, 
                 PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                 -1, 0);
```

### Machine Learning

```bash
# THP works well for large tensor allocations
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# PyTorch/JAX allocators can benefit
# But monitor for fragmentation issues
```

## Measuring THP Effectiveness

```bash
# TLB misses with perf
perf stat -e dTLB-load-misses,dTLB-store-misses ./myprogram

# Page fault statistics
perf stat -e page-faults,major-faults ./myprogram

# THP statistics
cat /proc/vmstat | grep thp
# thp_fault_alloc 12345
# thp_fault_fallback 100  <- Failed to allocate THP
# thp_collapse_alloc 5000
```

## NUMA + Huge Pages

```bash
# Allocate huge pages on specific NUMA node
echo 64 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 64 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages

# In code with libnuma
#include <numa.h>
void* ptr = numa_alloc_onnode(size, node);
madvise(ptr, size, MADV_HUGEPAGE);
```

## Summary: When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Databases | HugeTLB or THP=never |
| HPC large arrays | 1GB HugeTLB |
| ML training | THP=madvise |
| General server | THP=madvise |
| Latency-critical | HugeTLB (predictable) |
| Embedded/low-mem | THP=never |
