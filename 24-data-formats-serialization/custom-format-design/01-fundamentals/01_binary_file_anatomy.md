# Binary File Anatomy: Understanding File Format Structure from Scratch

## What is a Binary File?

A binary file is a sequence of bytes (8-bit values, 0-255). Unlike text files which store human-readable characters, binary files store data in formats optimized for machine processing. Every file on your computer, from images to executables to databases, is ultimately bytes. The question is: **how are those bytes organized?**

### Text vs Binary: A Physical Example

```
TEXT FILE ("hello"):
Bytes: 0x68 0x65 0x6C 0x6C 0x6F
       'h'  'e'  'l'  'l'  'o'
Each byte represents one character from the ASCII table.

BINARY FILE (integer 123456789):
Bytes: 0x15 0xCD 0x5B 0x07
This is the number 123456789 stored as a 32-bit little-endian integer.
A text file would need 9 bytes ("123456789"), binary needs only 4.
```

Binary is more compact. It's also more complex because the meaning of a byte depends on its position and the format's rules.

## Byte Order (Endianness)

Before we can store a multi-byte number, we must decide which end comes first.

### Little-Endian (x86, ARM, most GPUs)
The **least significant byte** is stored at the lowest address.
```
Number: 0x12345678
Memory address: 0  1  2  3
Bytes:          78 56 34 12  (Least significant first)
```

### Big-Endian (Network protocols, Java)
The **most significant byte** is stored at the lowest address.
```
Number: 0x12345678
Memory address: 0  1  2  3
Bytes:          12 34 56 78  (Most significant first)
```

**Why it matters for ML datasets**: If you create a format on an Intel machine (little-endian) and read it on an older PowerPC (big-endian), the numbers will be garbage. The standard practice is to pick one (usually little-endian, as NumPy's `'<'` prefix indicates) and document it in the header.

## NumPy Dtypes: The Foundation

NumPy's `dtype` (data type) is the primary way we define the structure of binary data in Python.

### Scalar Dtypes

| dtype | C equivalent | Size | Description |
|-------|--------------|------|-------------|
| `'u1'` or `np.uint8` | `unsigned char` | 1 byte | Unsigned integer 0-255 |
| `'i1'` or `np.int8` | `signed char` | 1 byte | Signed integer -128 to 127 |
| `'<u2'` or `np.uint16` | `unsigned short` | 2 bytes | Little-endian unsigned 0-65535 |
| `'>u2'` | `unsigned short` | 2 bytes | Big-endian unsigned 0-65535 |
| `'<i4'` or `np.int32` | `int` | 4 bytes | Little-endian signed |
| `'<u8'` or `np.uint64` | `unsigned long long` | 8 bytes | Little-endian unsigned |
| `'<f4'` or `np.float32` | `float` | 4 bytes | Single-precision float |
| `'<f8'` or `np.float64` | `double` | 8 bytes | Double-precision float |
| `'Sn'` or `np.string_` | `char[n]` | n bytes | Fixed-length byte string |

### Structured Dtypes (Compound Data)

A structured dtype is like a C `struct`. It defines a group of named fields with specific types.

```python
import numpy as np

# Define a structured dtype
PersonType = np.dtype([
    ('id', '<u4'),       # 4-byte unsigned int (offset 0)
    ('age', '<u1'),      # 1-byte unsigned int (offset 4)
    # Note: Padding may be inserted here for alignment!
    ('height', '<f4'),   # 4-byte float (offset 8 after padding)
    ('name', 'S32'),     # 32-byte string (offset 12)
], align=True)  # `align=True` adds padding for faster CPU access

print(f"PersonType size: {PersonType.itemsize} bytes")
# This might be 44 bytes due to alignment padding, not 4+1+4+32=41

# Create an array
people = np.zeros(2, dtype=PersonType)
people[0] = (1, 25, 1.75, b'Alice')
people[1] = (2, 30, 1.80, b'Bob')

# Write to file
people.tofile('people.bin')

# Read back
loaded = np.fromfile('people.bin', dtype=PersonType)
print(loaded[0]['name'])  # b'Alice'
```

### Why Alignment Matters

Modern CPUs read memory most efficiently when data is aligned to its natural boundary:
- 2-byte values should start at even addresses.
- 4-byte values should start at addresses divisible by 4.
- 8-byte values should start at addresses divisible by 8.

If you ignore alignment:
1. **Slow reads**: CPU may need two memory fetches instead of one.
2. **Bus errors**: Some architectures (older ARM) will crash on unaligned access.
3. **Broken `mmap`**: Memory mapping requires proper alignment for zero-copy views.

The `align=True` parameter in `np.dtype()` automatically inserts padding bytes to satisfy these constraints.

## The Universal Binary File Structure

Every well-designed binary format follows a common conceptual pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BINARY FILE LAYOUT                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 1. MAGIC BYTES (4-8 bytes)                                             ││
│  │    Purpose: Identify the file format without relying on the extension. ││
│  │    Example: b'\x89PNG\r\n\x1a\n' for PNG files.                        ││
│  │    Why 0x89? It's > 127, so text editors won't try to display it.     ││
│  │    Why \r\n? Catches DOS/Unix line ending corruption during transfer. ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 2. HEADER (fixed size, typically 24-256 bytes)                         ││
│  │    Purpose: Store all pointers and counts needed to parse the file.    ││
│  │    Contents:                                                            ││
│  │    - Format version (for backward/forward compatibility)               ││
│  │    - Number of samples/records                                          ││
│  │    - Number of fields/columns                                           ││
│  │    - Pointers (byte offsets) to other sections                         ││
│  │    - Feature flags (compression, encryption, endianness)               ││
│  │    MUST BE FIXED SIZE: So you can read it without knowing the contents.││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 3. SCHEMA / FIELD DESCRIPTORS (variable size)                          ││
│  │    Purpose: Define the structure of each field/column.                  ││
│  │    Contents per field:                                                  ││
│  │    - Name (e.g., "image", "label")                                     ││
│  │    - Type ID (e.g., 10 = RGB Image, 20 = Audio Waveform)              ││
│  │    - Type-specific arguments (e.g., JPEG quality, sample rate)        ││
│  │    - Metadata dtype description                                         ││
│  │    The header tells you how many field descriptors to read.            ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 4. METADATA TABLE (fixed size per sample)                              ││
│  │    Purpose: Enable O(1) lookup of any sample's information.            ││
│  │    Layout: A contiguous array of (num_samples, row_size) where row_size││
│  │            is the sum of all fields' metadata sizes.                    ││
│  │    Contents per sample, per field:                                      ││
│  │    - For fixed data (e.g., labels): Store the value directly.          ││
│  │    - For variable data (e.g., images): Store a pointer and size.       ││
│  │    Access pattern: metadata[sample_id] gives you all info for a sample.││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 5. DATA REGION (variable size, page-aligned)                           ││
│  │    Purpose: Store the actual bulk data (images, audio, text).          ││
│  │    Organization: Typically divided into large "pages" (e.g., 2MB) for: ││
│  │    - Parallel writing without file locks                               ││
│  │    - Alignment with OS page cache (4KB or 2MB huge pages)             ││
│  │    - Sequential I/O performance                                         ││
│  │    Pointers in the Metadata Table point into this region.              ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ 6. ALLOCATION TABLE (fixed size per allocation)                        ││
│  │    Purpose: Map data pointers to their sizes (for the reader).         ││
│  │    Layout: An array of (pointer, size) pairs, sorted by pointer.       ││
│  │    Why needed: When reading variable-length data, you must know where  ││
│  │                it ends. The metadata stores the pointer; this table    ││
│  │                stores the size. Separation allows binary search.       ││
│  │    Written last, after the data region's size is known.                ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Order?

### Writing Order
1.  Write **placeholder header** (we don't know all counts/pointers yet).
2.  Write **field descriptors** (we know the schema).
3.  Reserve space for **metadata table** (we know its size: `num_samples * row_size`).
4.  Write **data region** (fill in metadata pointers as we go).
5.  Write **allocation table** (now we know all sizes).
6.  **Seek back** to the start and rewrite the final header and metadata.

### Reading Order
1.  Read **magic bytes**: Validate format.
2.  Read **header**: Get version, counts, all pointers.
3.  Seek to `field_desc_ptr`, read **field descriptors**: Know how to parse.
4.  Seek to `metadata_ptr`, read **metadata table**: Ready for random access.
5.  Seek to `alloc_table_ptr`, read **allocation table**: Know data sizes.
6.  Now any sample can be accessed in O(1) by consulting the metadata table.

## FFCV's `.beton` File Layout

Let's examine FFCV's actual implementation from `ffcv/types.py`:

```python
import numpy as np

# Header: 24 bytes at offset 0 (FFCV has no separate magic bytes)
HeaderType = np.dtype([
    ('version', '<u2'),        # Format version number (2 bytes)
    ('num_fields', '<u2'),     # How many data fields (image, label, etc.)
    ('page_size', '<u4'),      # Memory page size for alignment
    ('num_samples', '<u8'),    # Total count of samples
    ('alloc_table_ptr', '<u8') # File offset where the allocation table begins
], align=True)

# Field Descriptor: Describes how to interpret one field
FieldDescType = np.dtype([
    ('type_id', '<u1'),          # Identifier for the field type (1 byte)
    ('name', ('<u1', 16)),       # Field name, fixed 16-byte string
    ('arguments', ('<u1', 1024)) # Type-specific parameters (e.g., JPEG quality)
], align=True)

# Allocation Table Entry: Maps pointer to size for variable-length data
ALLOC_TABLE_TYPE = np.dtype([
    ('sample_id', '<u8'),  # Which sample this allocation belongs to
    ('ptr', '<u8'),        # Byte offset into the data region
    ('size', '<u8'),       # Size of this data chunk in bytes
])
```

### File Layout Calculation Example

```python
def calculate_layout(num_samples, num_fields, metadata_row_size, page_size=2*1024*1024):
    """
    Calculate exact byte offsets for each section of the file.
    """
    
    # Section 1: Header (starts at byte 0)
    header_start = 0
    header_size = HeaderType.itemsize  # 24 bytes for FFCV's header
    
    # Section 2: Field Descriptors (immediately after header)
    field_desc_start = header_size
    field_desc_total_size = num_fields * FieldDescType.itemsize  # 1041 bytes each
    
    # Section 3: Metadata Table (after field descriptors)
    metadata_start = field_desc_start + field_desc_total_size
    metadata_total_size = num_samples * metadata_row_size
    
    # Section 4: Data Region (align to page boundary for mmap efficiency)
    data_start_unaligned = metadata_start + metadata_total_size
    # Round up to next page boundary
    data_start = (data_start_unaligned + page_size - 1) // page_size * page_size
    # Data size is determined during writing
    
    # Section 5: Allocation Table (at the very end, after all data)
    # Location stored in header['alloc_table_ptr']
    
    print(f"Header:           0 - {header_size}")
    print(f"Field Descriptors: {field_desc_start} - {field_desc_start + field_desc_total_size}")
    print(f"Metadata Table:   {metadata_start} - {data_start_unaligned}")
    print(f"Data Region:      {data_start} - (variable)")
    
    return {
        'header': (header_start, header_size),
        'fields': (field_desc_start, field_desc_total_size),
        'metadata': (metadata_start, metadata_total_size),
        'data': data_start,
    }

# Example: 1 million samples, 2 fields (image + label)
# Assume metadata_row_size = 24 bytes (8-byte pointer, 4-byte size, etc. × 2 fields)
layout = calculate_layout(
    num_samples=1_000_000, 
    num_fields=2, 
    metadata_row_size=24
)
```

## Key Design Decisions and Trade-offs

### Fixed-Size vs Variable-Size Regions

| Region | Size | Why |
|--------|------|-----|
| Header | **Fixed** | You must be able to read it *before* knowing anything else. |
| Field Descriptors | Fixed per field | The number of types is bounded; each type has fixed-size arguments. |
| Metadata Table | Fixed per sample | **Enables O(1) random access**. `metadata[i]` is always at a known offset. |
| Data Region | **Variable** | Actual content (images, audio) varies in size. |
| Allocation Table | Fixed per entry | Simple (ptr, size) pairs for efficient lookup. |

### What Goes in Metadata vs Data?

**In Metadata (fast, O(1) access):**
- Scalar labels (class ID, score).
- Pointers to variable-length data.
- Sizes of variable-length data.
- Small fixed-size values (e.g., bounding box as 4 floats).
- Encoding flags (e.g., "this image is JPEG compressed").

**In Data Region (variable, needs pointer):**
- Images (resolution varies).
- Audio waveforms (duration varies).
- Text (token count varies).
- Large embeddings.

### The Metadata/Pointer Split

A common confusion: why not store the data size *with* the metadata pointer?

FFCV splits them:
- `metadata['data_ptr']` is in the per-sample metadata.
- `sizes` are in a separate allocation table, sorted for binary search.

**Reason**: It allows a generic, highly-optimized `read(ptr, mem_state)` function in Numba. The reader doesn't need to know *which* field it's reading; it just looks up the size by pointer.

## Reading and Writing: The Full Flow

### Writing a Sample (Simplified)

```python
def write_sample(file, sample_data, metadata_table, sample_idx, alloc_table):
    for field_name, field in fields.items():
        value = sample_data[field_name]
        
        # Encode the value (e.g., JPEG-compress an image)
        encoded_bytes = field.encode(value)
        
        # Get current file position (this is the "pointer")
        data_ptr = file.tell()
        
        # Write the encoded data to the data region
        file.write(encoded_bytes)
        
        # Store the pointer in the metadata table
        metadata_table[sample_idx][field_name]['data_ptr'] = data_ptr
        metadata_table[sample_idx][field_name]['data_size'] = len(encoded_bytes)
        
        # Record in the allocation table
        alloc_table.add(data_ptr, len(encoded_bytes))
```

### Reading a Sample (Simplified)

```python
def read_sample(mmap_view, metadata_table, sample_idx, alloc_table_sorted):
    sample = {}
    for field_name in fields:
        # Get pointer and size from metadata
        ptr = metadata_table[sample_idx][field_name]['data_ptr']
        
        # Look up size from allocation table (binary search)
        idx = np.searchsorted(alloc_table_sorted['ptr'], ptr)
        size = alloc_table_sorted['size'][idx]
        
        # Get a VIEW into the mmap (zero-copy!)
        raw_bytes = mmap_view[ptr : ptr + size]
        
        # Decode (e.g., JPEG-decompress)
        sample[field_name] = fields[field_name].decode(raw_bytes)
    
    return sample
```

## Exercises

1.  **Calculate Offsets**: For a dataset with 10,000,000 samples and 3 fields where each field's metadata is 16 bytes, calculate the exact byte offset of the data region assuming a 2MB page size and a 64-byte header.

2.  **Design a Header**: Create a NumPy dtype for a header that includes: magic bytes (8), major/minor version (2 each), flags (4), sample count (8), field count (2), page size (4), and 5 region pointers (8 each). Ensure it's a power-of-2 size with appropriate padding.

3.  **Endianness Bug Hunt**: Write a script that creates a file with `'>u4'` (big-endian) integers, then reads it back with `'<u4'` (little-endian). Observe the garbage values and understand why.

4.  **Alignment Test**: Create a struct dtype *without* `align=True` and measure the read time of 1 million entries vs. with `align=True`. (Use `timeit`.)
