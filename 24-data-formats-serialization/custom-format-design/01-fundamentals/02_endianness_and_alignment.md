# Endianness and Memory Alignment

## Why This Matters for Data Formats

When you write binary data, you're writing bytes. But how are multi-byte values stored? And where should they be placed in memory? These questions are critical for:

- **Cross-platform compatibility** (ARM vs x86)
- **Performance** (aligned access is faster)
- **Correctness** (misaligned access can crash)

## Endianness: Byte Order

### The Problem

The number `0x12345678` (4 bytes) can be stored two ways:

```
Memory Address:    0x00  0x01  0x02  0x03
                   ────  ────  ────  ────
Big Endian:        0x12  0x34  0x56  0x78   (Most significant first)
Little Endian:     0x78  0x56  0x34  0x12   (Least significant first)
```

### Platform Differences

| Architecture | Endianness |
|--------------|------------|
| x86, x86-64 | Little Endian |
| ARM (most modes) | Little Endian |
| ARM (legacy) | Big Endian |
| PowerPC | Big Endian |
| SPARC | Big Endian |
| Network protocols | Big Endian ("network order") |

### FFCV's Solution: Fix Endianness

From `ffcv/types.py`:
```python
# Note: Using '<' prefix forces little-endian
# This makes files portable between architectures

HeaderType = np.dtype([
    ('version', '<u2'),      # '<' = little-endian, 'u2' = unsigned 2-byte
    ('num_fields', '<u2'),
    ('page_size', '<u4'),    # '<u4' = little-endian unsigned 4-byte
    ('num_samples', '<u8'),
    ('alloc_table_ptr', '<u8')
], align=True)
```

### NumPy Endianness Notation

| Prefix | Meaning |
|--------|---------|
| `<` | Little-endian |
| `>` | Big-endian |
| `=` | Native (platform-dependent, avoid!) |
| `\|` | Not applicable (single byte) |

```python
import numpy as np
import sys

# Check system endianness
print(sys.byteorder)  # 'little' on x86

# Creating little-endian data (portable)
data = np.array([0x12345678], dtype='<u4')
print(data.tobytes())  # b'\x78\x56\x34\x12'

# Creating big-endian data
data = np.array([0x12345678], dtype='>u4')
print(data.tobytes())  # b'\x12\x34\x56\x78'

# Converting between endianness
data = np.array([0x12345678], dtype='<u4')
swapped = data.byteswap().newbyteorder()
```

## Memory Alignment

### The Problem

CPUs access memory in chunks (typically 4 or 8 bytes). When data isn't aligned to these boundaries, bad things happen:

```
Memory:    0x00  0x01  0x02  0x03  0x04  0x05  0x06  0x07
           ────────────────────  ────────────────────────
                Word 0                  Word 1

Aligned 4-byte value at 0x00: ✓ One memory access
Unaligned 4-byte value at 0x01: ✗ Two memory accesses (or crash!)
```

### Alignment Rules

| Type | Size | Natural Alignment |
|------|------|-------------------|
| uint8 | 1 byte | 1 byte |
| uint16 | 2 bytes | 2 bytes |
| uint32 | 4 bytes | 4 bytes |
| uint64 | 8 bytes | 8 bytes |
| float32 | 4 bytes | 4 bytes |
| float64 | 8 bytes | 8 bytes |
| pointer | 8 bytes (64-bit) | 8 bytes |

### Struct Padding

C structs add padding to maintain alignment:

```c
// Without optimization
struct BadLayout {
    uint8_t  a;    // 1 byte  @ offset 0
    // 7 bytes padding (to align b to 8)
    uint64_t b;    // 8 bytes @ offset 8
    uint8_t  c;    // 1 byte  @ offset 16
    // 7 bytes padding (total size must be multiple of largest alignment)
};
// Total: 24 bytes for 10 bytes of data!

// Better layout
struct GoodLayout {
    uint64_t b;    // 8 bytes @ offset 0
    uint8_t  a;    // 1 byte  @ offset 8
    uint8_t  c;    // 1 byte  @ offset 9
    // 6 bytes padding
};
// Total: 16 bytes for 10 bytes of data
```

### NumPy Struct Alignment

```python
import numpy as np

# Without alignment (packed)
packed_dtype = np.dtype([
    ('a', 'u1'),
    ('b', '<u8'),
    ('c', 'u1'),
], align=False)
print(f"Packed size: {packed_dtype.itemsize}")  # 10 bytes

# With alignment (padded)
aligned_dtype = np.dtype([
    ('a', 'u1'),
    ('b', '<u8'),
    ('c', 'u1'),
], align=True)
print(f"Aligned size: {aligned_dtype.itemsize}")  # 24 bytes

# Check field offsets
for name in aligned_dtype.names:
    print(f"{name}: offset {aligned_dtype.fields[name][1]}")
```

### FFCV Uses Aligned Types

```python
# From ffcv/types.py - note align=True
HeaderType = np.dtype([
    ('version', '<u2'),
    ('num_fields', '<u2'),
    ('page_size', '<u4'),
    ('num_samples', '<u8'),
    ('alloc_table_ptr', '<u8')
], align=True)
```

## Page Alignment

Beyond individual value alignment, FFCV aligns data to **page boundaries**.

### Why Pages Matter

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORY PAGE ALIGNMENT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Standard page size: 4 KB (4096 bytes)                              │
│  Huge page size: 2 MB (2,097,152 bytes)                             │
│                                                                      │
│  Benefits of page alignment:                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. mmap() works at page granularity                         │    │
│  │ 2. OS page cache operates on pages                          │    │
│  │ 3. DMA transfers prefer page-aligned buffers                │    │
│  │ 4. Huge pages reduce TLB misses                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  FFCV page size: 8 MB (2^23) = 4 × huge page                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### FFCV's Page Alignment

From `ffcv/writer.py`:
```python
MIN_PAGE_SIZE = 1 << 21  # 2 MB (huge page size)
MAX_PAGE_SIZE = 1 << 32  # 4 GB

# Default: 4 × MIN_PAGE_SIZE = 8 MB
page_size = 4 * MIN_PAGE_SIZE
```

### Alignment Utility Function

```python
def align_to_page(ptr: int, page_size: int) -> int:
    """Round up to next page boundary."""
    if ptr % page_size == 0:
        return ptr
    return ptr + page_size - (ptr % page_size)

# Example
offset = 1000
page_size = 4096
aligned = align_to_page(offset, page_size)
print(f"{offset} -> {aligned}")  # 1000 -> 4096
```

## Practical Example: Designing Aligned Structures

```python
import numpy as np

def design_header():
    """Design an optimally aligned header."""
    
    # Group fields by size for minimal padding
    # Largest first, then smaller ones
    header = np.dtype([
        # 8-byte fields first
        ('num_samples', '<u8'),
        ('data_offset', '<u8'),
        ('index_offset', '<u8'),
        
        # 4-byte fields
        ('magic', '<u4'),
        ('page_size', '<u4'),
        ('flags', '<u4'),
        
        # 2-byte fields
        ('version', '<u2'),
        ('num_fields', '<u2'),
        
        # 1-byte fields last
        ('compression', 'u1'),
        ('reserved', 'u1', 3),  # Pad to 4-byte boundary
    ], align=True)
    
    print(f"Header size: {header.itemsize} bytes")
    
    # Show layout
    for name in header.names:
        offset = header.fields[name][1]
        size = header.fields[name][0].itemsize
        print(f"  {name:20s}: offset={offset:3d}, size={size}")
    
    return header

# Verify alignment
header = design_header()
```

## Common Pitfalls

### 1. Platform-Dependent Code
```python
# BAD: Uses native byte order
data = np.array([1, 2, 3], dtype='u4')  # Platform-dependent!

# GOOD: Explicit byte order
data = np.array([1, 2, 3], dtype='<u4')  # Always little-endian
```

### 2. Packed Structs for Performance
```python
# BAD: Packed struct (unaligned access)
dtype = np.dtype([('a', 'u1'), ('b', '<u8')], align=False)

# GOOD: Aligned struct
dtype = np.dtype([('a', 'u1'), ('b', '<u8')], align=True)
```

### 3. Forgetting Page Alignment for mmap
```python
# BAD: Data starts at arbitrary offset
data_start = header_size + metadata_size

# GOOD: Data aligned to page boundary
data_start = align_to_page(header_size + metadata_size, page_size)
```

## Exercises

1. Calculate the size of this struct with and without alignment:
   ```python
   np.dtype([
       ('flags', 'u1'),
       ('count', '<u4'),
       ('value', '<f8'),
       ('tag', 'u1'),
   ])
   ```

2. Reorder the fields above for minimal padding.

3. Write a function that verifies all fields in a dtype are naturally aligned.
