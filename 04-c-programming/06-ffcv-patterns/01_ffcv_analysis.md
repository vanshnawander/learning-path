# FFCV Code Analysis: Optimization Patterns

This document analyzes the FFCV source code to understand its optimization techniques.

## Key Files Studied

- `ffcv/memory_allocator.py` - Page-based allocation
- `ffcv/reader.py` - Memory-mapped file reading
- `ffcv/writer.py` - Efficient file writing

## Pattern 1: Page-Based Memory Allocation

From `memory_allocator.py`:

```python
class MemoryAllocator():
    def __init__(self, fname, offset_start, page_size):
        self.page_size = page_size
        self.page_data = np.zeros(self.page_size, '<u1')
        
    def malloc(self, size):
        if size > self.space_left_in_page:
            self.flush_page()
            # Book the next available page
            with self.next_page_allocated.get_lock():
                self.my_page = self.next_page_allocated.value
                self.next_page_allocated.value = self.my_page + 1
```

**Key Insights:**
1. **Page-aligned writes** - Data written in fixed-size pages
2. **Lock-free page allocation** - Atomic counter for page IDs
3. **Spin-lock for ordering** - Ensures sequential file writes
4. **Zero-copy buffers** - Returns numpy view into page buffer

## Pattern 2: Header-Based File Format

From `reader.py`:

```python
def read_header(self):
    header = np.fromfile(self._fname, dtype=HeaderType, count=1)[0]
    self.num_samples = header['num_samples']
    self.page_size = header['page_size']
    
def read_allocation_table(self):
    offset = self.header['alloc_table_ptr']
    alloc_table = np.fromfile(self._fname, dtype=ALLOC_TABLE_TYPE, offset=offset)
```

**Key Insights:**
1. **Fixed header at offset 0** - Quick metadata access
2. **Allocation table** - O(1) sample lookup by index
3. **numpy fromfile** - Efficient binary parsing
4. **Immutable after read** - `setflags(write=False)`

## Pattern 3: Memory-Mapped Access

The loader uses `np.memmap` under the hood:

```python
# Conceptually:
data = np.memmap(fname, dtype='uint8', mode='r')
sample = data[offset:offset+size]  # Zero-copy slice!
```

**Benefits:**
- No `read()` syscalls during training
- OS handles caching automatically
- Multiple workers share same pages

## Pattern 4: Quasi-Random Sampling

FFCV doesn't use truly random access:

```python
# Bad: True random (cache-hostile)
indices = np.random.permutation(n)

# Good: Quasi-random (cache-friendly)
# Group samples by page, shuffle within groups
```

**Why:** Samples on same page load together.

## C Implementation Ideas

### 1. Page-Aligned Writer

```c
typedef struct {
    int fd;
    size_t page_size;
    uint8_t* page_buffer;
    size_t page_offset;
    size_t current_page;
} PageWriter;

void* page_malloc(PageWriter* w, size_t size) {
    if (size > w->page_size - w->page_offset) {
        flush_page(w);
        w->current_page++;
        w->page_offset = 0;
    }
    void* ptr = w->page_buffer + w->page_offset;
    w->page_offset += size;
    return ptr;
}
```

### 2. Memory-Mapped Reader

```c
typedef struct {
    void* data;
    size_t size;
    uint64_t* alloc_table;
    int num_samples;
} MappedDataset;

void* get_sample(MappedDataset* ds, int idx) {
    uint64_t offset = ds->alloc_table[idx];
    return (char*)ds->data + offset;
}
```

## Performance Implications

| Technique | Speedup | Why |
|-----------|---------|-----|
| Page alignment | 2-3x | Matches OS page size |
| Memory mapping | 5-10x | No syscall overhead |
| Quasi-random | 2-5x | Cache locality |
| Zero-copy | 10x+ | No data movement |

## Exercises

1. Implement the C writer above
2. Add variable-length sample support
3. Benchmark vs naive file reading
4. Add multi-threaded writing
