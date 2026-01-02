# NumPy Structured Arrays for Format Design

## Why NumPy for Binary Formats?

NumPy's dtype system is perfect for defining binary formats because:
1. **Memory layout control** - exact byte positions
2. **Zero-copy file I/O** - `np.fromfile()` and `np.memmap()`
3. **Structured access** - named fields like a database
4. **Cross-platform** - handles endianness

## Basic Structured Arrays

```python
import numpy as np

# Define a record type
person_dtype = np.dtype([
    ('name', 'U20'),      # Unicode string, 20 chars
    ('age', '<i4'),       # Little-endian 32-bit int
    ('salary', '<f8'),    # Little-endian 64-bit float
])

# Create records
people = np.array([
    ('Alice', 30, 75000.0),
    ('Bob', 25, 65000.0),
], dtype=person_dtype)

# Access by field name
print(people['name'])    # ['Alice' 'Bob']
print(people['age'])     # [30 25]

# Access individual record
print(people[0])         # ('Alice', 30, 75000.)
print(people[0]['age'])  # 30
```

## Type Codes Reference

### Numeric Types
| Code | Type | Size | Range |
|------|------|------|-------|
| `'<u1'` | uint8 | 1 byte | 0 to 255 |
| `'<u2'` | uint16 | 2 bytes | 0 to 65,535 |
| `'<u4'` | uint32 | 4 bytes | 0 to 4.3 billion |
| `'<u8'` | uint64 | 8 bytes | 0 to 18 quintillion |
| `'<i1'` | int8 | 1 byte | -128 to 127 |
| `'<i2'` | int16 | 2 bytes | -32,768 to 32,767 |
| `'<i4'` | int32 | 4 bytes | ±2.1 billion |
| `'<i8'` | int64 | 8 bytes | ±9 quintillion |
| `'<f4'` | float32 | 4 bytes | ~7 decimal digits |
| `'<f8'` | float64 | 8 bytes | ~15 decimal digits |

### Fixed-Size Arrays
```python
# Fixed-size array within struct
dtype = np.dtype([
    ('id', '<u4'),
    ('vector', '<f4', (3,)),      # 3-element float32 array
    ('matrix', '<f8', (4, 4)),    # 4x4 float64 matrix
])

data = np.zeros(1, dtype=dtype)
data[0]['vector'] = [1.0, 2.0, 3.0]
data[0]['matrix'] = np.eye(4)
```

## FFCV's Type Definitions

### Header Type
```python
# From ffcv/types.py
HeaderType = np.dtype([
    ('version', '<u2'),          # Format version (2 bytes)
    ('num_fields', '<u2'),       # Number of data fields (2 bytes)
    ('page_size', '<u4'),        # Page size for alignment (4 bytes)
    ('num_samples', '<u8'),      # Total number of samples (8 bytes)
    ('alloc_table_ptr', '<u8')   # Pointer to allocation table (8 bytes)
], align=True)

print(f"Header size: {HeaderType.itemsize} bytes")  # 24 bytes
```

### Field Descriptor Type
```python
# Each field is described by this structure
FieldDescType = np.dtype([
    ('type_id', '<u1'),              # Field type (1 byte)
    ('name', ('<u1', 16)),           # Field name as bytes (16 bytes)
    ('arguments', ('<u1', 1024)),    # Field-specific arguments (1024 bytes)
], align=True)

# Type IDs map to handlers
TYPE_ID_HANDLER = {
    0: 'FloatField',
    1: 'IntField',
    2: 'RGBImageField',
    3: 'BytesField',
    4: 'NDArrayField',
    5: 'JSONField',
    6: 'TorchTensorField',
    255: None,  # Custom field
}
```

### Allocation Table Type
```python
ALLOC_TABLE_TYPE = np.dtype([
    ('sample_id', '<u8'),   # Which sample this allocation is for
    ('ptr', '<u8'),         # File offset where data lives
    ('size', '<u8'),        # Size of the data in bytes
])
```

## Building Composite Metadata Types

FFCV builds per-sample metadata dynamically:

```python
def get_metadata_type(handlers):
    """Build composite dtype from field handlers."""
    # Each field contributes its metadata type
    field_dtypes = []
    for handler in handlers:
        field_dtypes.append(('', handler.metadata_type))
    
    return np.dtype(field_dtypes, align=True)

# Example: Image + Label dataset
class IntField:
    @property
    def metadata_type(self):
        return np.dtype('<i8')

class RGBImageField:
    @property
    def metadata_type(self):
        return np.dtype([
            ('mode', '<u1'),      # 0=jpg, 1=raw
            ('width', '<u2'),     # Image width
            ('height', '<u2'),    # Image height
            ('data_ptr', '<u8'),  # Pointer to image data
        ])

# Combined metadata per sample
metadata_type = np.dtype([
    ('f0', RGBImageField().metadata_type),  # Image metadata
    ('f1', IntField().metadata_type),       # Label
], align=True)
```

## File I/O with Structured Arrays

### Writing
```python
import numpy as np

# Define header
HeaderType = np.dtype([
    ('magic', '<u4'),
    ('version', '<u2'),
    ('num_samples', '<u8'),
], align=True)

# Create and write
header = np.zeros(1, dtype=HeaderType)
header['magic'] = 0x42455445  # "BETE" in ASCII
header['version'] = 1
header['num_samples'] = 1000

with open('data.bin', 'wb') as f:
    f.write(header.tobytes())
```

### Reading
```python
# Read back
header = np.fromfile('data.bin', dtype=HeaderType, count=1)
print(f"Magic: 0x{header['magic'][0]:08X}")
print(f"Version: {header['version'][0]}")
print(f"Samples: {header['num_samples'][0]}")
```

### Memory Mapping
```python
# Memory map for zero-copy access
mmap_data = np.memmap('data.bin', dtype='u1', mode='r')

# Interpret specific region as structured type
header = mmap_data[:HeaderType.itemsize].view(HeaderType)
```

### Reading at Offset
```python
# Read metadata table at specific offset
metadata_offset = 100
num_samples = 1000

metadata = np.fromfile(
    'data.bin',
    dtype=metadata_type,
    count=num_samples,
    offset=metadata_offset
)
```

## Advanced: Nested Structures

```python
# Vector type
Vector3D = np.dtype([
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
])

# Bounding box using nested types
BoundingBox = np.dtype([
    ('min', Vector3D),
    ('max', Vector3D),
])

# Sample with bounding box
SampleType = np.dtype([
    ('id', '<u4'),
    ('bbox', BoundingBox),
    ('label', '<u2'),
])

# Usage
sample = np.zeros(1, dtype=SampleType)
sample['bbox']['min']['x'] = 0.0
sample['bbox']['max']['x'] = 100.0
```

## Serializing Field Arguments

FFCV fields store their configuration in a fixed-size byte array:

```python
import json

ARG_TYPE = np.dtype([('', '<u1', 1024)])  # 1024 bytes for arguments

class NDArrayField:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
    
    def to_binary(self):
        """Serialize field config to bytes."""
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        
        # Serialize shape
        shape_bytes = np.array(self.shape, dtype='<u8').tobytes()
        
        # Serialize dtype as JSON
        dtype_json = json.dumps(self.dtype.str).encode('ascii')
        
        # Pack into result
        header_size = 8 * len(self.shape) + 8  # shape + dtype_length
        result[0][:len(shape_bytes)] = np.frombuffer(shape_bytes, dtype='u1')
        result[0][len(shape_bytes):len(shape_bytes)+8] = len(dtype_json)
        result[0][len(shape_bytes)+8:len(shape_bytes)+8+len(dtype_json)] = \
            np.frombuffer(dtype_json, dtype='u1')
        
        return result
    
    @staticmethod
    def from_binary(binary):
        """Deserialize field config from bytes."""
        # Reverse the process
        shape = []
        # ... parsing logic
        pass
```

## Exercises

1. Create a structured dtype for an audio sample with:
   - Sample rate (uint32)
   - Number of channels (uint8)
   - Duration in ms (uint32)
   - Data pointer (uint64)
   - Encoding type (uint8: 0=raw, 1=mp3, 2=flac)

2. Write functions to serialize and deserialize a Python dict to a fixed-size byte array (like FFCV's field arguments).

3. Create a memory-mapped reader that can access sample N's metadata in O(1) time.

```python
# Starter code
def create_audio_metadata_dtype():
    """Define audio sample metadata."""
    return np.dtype([
        # Your code here
    ], align=True)

def mmap_sample_metadata(filename, sample_idx, metadata_dtype, header_size):
    """Get metadata for sample N using mmap."""
    # Your code here
    pass
```
