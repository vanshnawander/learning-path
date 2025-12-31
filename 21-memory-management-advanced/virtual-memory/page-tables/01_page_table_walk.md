# x86-64 Page Table Walk

## Virtual Address Translation

x86-64 uses 4-level page tables to translate 48-bit virtual addresses to physical addresses.

### Virtual Address Layout (4KB pages)

```
63         48 47       39 38       30 29       21 20       12 11        0
┌─────────────┬──────────┬──────────┬──────────┬──────────┬─────────────┐
│   Sign Ext  │   PML4   │   PDPT   │    PD    │    PT    │   Offset    │
│   (16 bits) │  (9 bits)│  (9 bits)│  (9 bits)│  (9 bits)│  (12 bits)  │
└─────────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
                  ↓           ↓          ↓          ↓           ↓
              512 entries 512 entries 512 entries 512 entries  4096 bytes
```

### Page Table Walk Process

```
CR3 Register (Page Map Level 4 Base Address)
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Step 1: PML4 (Page Map Level 4)                                    │
│   Physical address = CR3 + (PML4_index * 8)                        │
│   Read PML4 entry → Contains PDPT physical base address            │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Step 2: PDPT (Page Directory Pointer Table)                        │
│   Physical address = PDPT_base + (PDPT_index * 8)                  │
│   Read PDPT entry → Contains PD physical base address              │
│   OR: 1GB huge page (if PS bit set)                                │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Step 3: PD (Page Directory)                                        │
│   Physical address = PD_base + (PD_index * 8)                      │
│   Read PD entry → Contains PT physical base address                │
│   OR: 2MB huge page (if PS bit set)                                │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Step 4: PT (Page Table)                                            │
│   Physical address = PT_base + (PT_index * 8)                      │
│   Read PT entry → Contains 4KB page physical base address          │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Final: Physical Address                                            │
│   Physical address = Page_base + Offset                            │
└────────────────────────────────────────────────────────────────────┘
```

### Page Table Entry Format

```
63  62       52 51                     12 11  9 8 7 6 5 4 3 2 1 0
┌───┬──────────┬────────────────────────┬─────┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│NX │ Available│  Physical Page Number  │ AVL │G│S│D│A│C│T│U│W│P│
└───┴──────────┴────────────────────────┴─────┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

P  (0): Present - page is in memory
W  (1): Writable - page is writable
U  (2): User - accessible from user mode
WT (3): Write-through caching
CD (4): Cache disable
A  (5): Accessed - set by CPU on access
D  (6): Dirty - set by CPU on write (PT entry only)
PS (7): Page size - 2MB/1GB page (PD/PDPT entry)
G  (8): Global - don't flush from TLB on CR3 write
NX (63): No execute - page cannot contain code
```

### TLB (Translation Lookaside Buffer)

```
Virtual Address ──► TLB Lookup ──► Hit? ──► Physical Address
                        │
                        │ Miss
                        ▼
                   Page Table Walk (expensive!)
                        │
                        ▼
                   Update TLB
                        │
                        ▼
                   Physical Address

TLB Hit: ~1 cycle
TLB Miss + Page Walk: ~100-1000 cycles (depending on cache state)
```

### TLB Organization (Typical Modern CPU)

| Level | Entries | Page Sizes | Associativity |
|-------|---------|------------|---------------|
| L1 DTLB | 64 | 4KB | 4-way |
| L1 DTLB | 32 | 2MB/1GB | 4-way |
| L1 ITLB | 128 | 4KB | 8-way |
| L2 TLB | 1536 | 4KB/2MB | 12-way |

### Page Walk Cache (PWC)

Modern CPUs cache intermediate page table entries:

```
PML4 Cache: Caches PML4 entries
PDPT Cache: Caches PDPT entries  
PD Cache:   Caches PD entries

Benefit: 4-level walk becomes 1-level if upper levels cached
```

### Huge Pages and TLB Coverage

```
4KB pages:
  TLB entries: 1536
  Coverage: 1536 × 4KB = 6MB

2MB pages:
  TLB entries: 1536 (shared L2)
  Coverage: 1536 × 2MB = 3GB

1GB pages:
  TLB entries: 4 (typical)
  Coverage: 4 × 1GB = 4GB
```

### Calculating Page Table Memory Overhead

```
For 48-bit virtual address space (256 TB):
  PML4: 1 page = 4KB
  PDPT: 512 pages = 2MB (if all present)
  PD:   512 × 512 = 262K pages = 1GB
  PT:   512^3 = 134M pages = 512GB

But sparse allocation means most aren't needed!

Typical process with 100MB code + 1GB heap + 8MB stack:
  Actual page table size: ~4-8 MB
```

### Linux Page Table APIs

```c
// Walk page tables (kernel code)
pgd_t *pgd = pgd_offset(mm, address);
if (pgd_none(*pgd)) return NULL;

p4d_t *p4d = p4d_offset(pgd, address);
if (p4d_none(*p4d)) return NULL;

pud_t *pud = pud_offset(p4d, address);
if (pud_none(*pud)) return NULL;
if (pud_large(*pud)) return pud;  // 1GB huge page

pmd_t *pmd = pmd_offset(pud, address);
if (pmd_none(*pmd)) return NULL;
if (pmd_large(*pmd)) return pmd;  // 2MB huge page

pte_t *pte = pte_offset(pmd, address);
return pte;
```

### Performance Implications

1. **TLB misses are expensive**
   - Each miss requires memory accesses
   - Use huge pages for large working sets

2. **Page table memory**
   - Sparse allocations are efficient
   - Dense allocations might warrant huge pages

3. **PCID (Process Context ID)**
   - Avoids TLB flush on context switch
   - Linux enables by default on modern CPUs

4. **5-Level Paging**
   - 57-bit virtual addresses (128 PB)
   - Additional level adds latency
   - Only enabled when needed

## References

- Intel SDM Volume 3, Chapter 4: Paging
- AMD64 Architecture Manual, Chapter 5
- "Understanding the Linux Virtual Memory Manager" - Gorman
