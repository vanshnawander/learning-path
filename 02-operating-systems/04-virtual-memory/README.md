# Virtual Memory

Every process has its own address space - how the OS manages memory.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_virtual_memory.c` | Page faults, demand paging |

## Key Concepts

### Virtual vs Physical Memory
- **Virtual**: What your program sees (0x0000... to 0xFFFF...)
- **Physical**: Actual RAM
- **Page table**: Maps virtual → physical

### Pages
- Memory divided into pages (typically 4KB)
- Each page can be:
  - In RAM (valid)
  - On disk (swapped)
  - Not allocated (fault)

### Demand Paging
Memory isn't allocated until accessed:
```c
char* p = malloc(1GB);  // Just reserves address space
p[0] = 'x';             // NOW physical memory allocated
```

## Page Faults

When accessing unmapped page:
1. CPU traps to kernel
2. Kernel allocates physical page
3. Updates page table
4. Resumes program

**Cost**: ~1000+ cycles (vs ~100 for RAM access)

## Huge Pages

Regular pages: 4KB
Huge pages: 2MB or 1GB

Benefits:
- Fewer page table entries
- Fewer TLB misses
- Better for large allocations (ML models!)

```bash
# Enable huge pages on Linux
echo 1024 > /proc/sys/vm/nr_hugepages
```

## ML Implications

1. **Model loading**: Weights paged in on first access
2. **GPU memory**: NOT virtual! Physical allocation.
3. **Pinned memory**: Locked in RAM, no swapping
4. **OOM**: Can happen later than expected (lazy alloc)
