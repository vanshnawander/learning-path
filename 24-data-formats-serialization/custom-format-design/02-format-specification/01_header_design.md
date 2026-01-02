# Header and Format Specification Design: A Deep Dive

## What is a Header?

A file header is a fixed block of bytes at the beginning of a file that acts as a **table of contents**. Without a header, a reader would have to parse the entire file sequentially, guessing its structure. The header answers basic questions immediately:
- Is this file the format I expect?
- What version of the format is it?
- How many records are there?
- Where do I find different sections of the file?

## Design Principle 1: Magic Bytes

### What They Are
Magic bytes (or a "file signature") are the first few bytes of a file that uniquely identify the file type. They are checked *before* reading any other data.

### Why They Exist
1.  **No reliance on file extension**: The file `data.txt` could secretly be an executable. Magic bytes reveal the truth.
2.  **Early failure**: If the magic doesn't match, stop immediately. Don't waste time parsing garbage.
3.  **Corruption detection**: If the magic is wrong, the file is likely corrupted or truncated.

### Designing Good Magic Bytes

The PNG format uses a brilliantly designed 8-byte magic:
```
0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
 |    |P   |N   |G  |\r  |\n  |^Z  |\n
```

*   `0x89`: A byte with the high bit set (> 127). If a program tries to interpret this as ASCII text, it will fail or display garbage. This signals "not a text file."
*   `PNG`: Human-readable identifier if you open in a hex editor.
*   `0x0D 0x0A` (`\r\n`): The DOS line ending. If a file transfer program incorrectly converts line endings, this sequence will be corrupted, and the magic check will fail.
*   `0x1A` (`^Z`): The DOS "end-of-file" marker. If you `type` the file in CMD, it stops here, preventing terminal garbage.
*   `0x0A` (`\n`): The Unix line ending. Detects Unix-to-DOS conversion errors.

**For our ML format, we can use a similar pattern:**
```python
MAGIC_BYTES = b'\x89MLDF\r\n\x1a\n'  # "ML Data Format" (8 bytes)
```

## Design Principle 2: Versioning

Formats evolve. You might add new field types, change metadata layouts, or add compression. Without versioning, old readers will choke on new files (and vice-versa).

### Semantic Versioning for File Formats

| Version Part | Meaning in Code | Meaning in Files |
|--------------|-----------------|------------------|
| **Major** | Breaking API changes | File structure is incompatible. Old reader MUST reject. |
| **Minor** | Backward-compatible additions | New features added. Old reader SHOULD still work (ignores new parts). |
| **Patch** | Bug fixes | No file format change. Only writer/reader code changed. |

### Version Checking Logic

```python
def can_read_file(file_major, file_minor, reader_major, reader_minor):
    """
    Determine if this reader can parse the file.
    """
    if file_major > reader_major:
        # File is from a newer major version. We cannot understand its structure.
        return False, "File is from a newer major version. Upgrade your reader."
    
    if file_major < reader_major:
        # File is from an older major version. We *might* support reading old formats
        # if we have legacy code, but typically this is also rejected.
        return False, "File is from an older major version. Use an older reader."
    
    # Same major version. Minor version differences are okay.
    if file_minor > reader_minor:
        # File has features we don't know about, but we can ignore them.
        print(f"Warning: File is version {file_major}.{file_minor}, reader supports up to {reader_major}.{reader_minor}. Some features may be ignored.")
    
    return True, "OK"
```

### Where to Store Version

In the **header**, as two 16-bit unsigned integers:
```python
HeaderType = np.dtype([
    ('version_major', '<u2'),  # 2 bytes (0-65535 major versions possible)
    ('version_minor', '<u2'),  # 2 bytes
    # ...
], align=True)
```

## Design Principle 3: Self-Description

A file should contain everything needed to parse it. This includes:
1.  Counts: How many samples? How many fields?
2.  Schema: What are the fields and their types?
3.  Offsets: Where do different sections start?

**Anti-pattern**: Requiring a separate "schema file" or external metadata to parse the data file. If they get separated, the data is useless.

## Design Principle 4: Fixed-Size Header

The header size **must be known before reading the file**. Otherwise, you have a chicken-and-egg problem: "I need to read the header to find out how big the header is."

### Common Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed Absolute Size** | Header is always exactly N bytes. | Simplest. | Limits future expansion. |
| **Size-Prefixed** | First 4 bytes = header size. | Flexible. | Slightly more complex reading logic. |
| **Hybrid** | Fixed "core header" + variable "extended header" whose size is in the core. | Best of both. | Most complex. |

FFCV uses a Fixed Absolute Size (24 bytes). We'll use a 64-byte header, which provides room for future expansion and is a nice power-of-2 for alignment.

## Design Principle 5: Alignment

Data should be placed at addresses that are multiples of their size:
- 2-byte fields at even addresses.
- 4-byte fields at addresses divisible by 4.
- 8-byte fields at addresses divisible by 8.

This is crucial for:
1.  **CPU efficiency**: Aligned loads are faster (single bus transaction).
2.  **Memory mapping**: `mmap` requires aligned access on some systems.
3.  **NumPy compatibility**: `np.frombuffer` on an unaligned slice can fail or copy.

### Padding

To achieve alignment, we insert **padding bytes** that are never read.

```python
# BAD: Unaligned
BadHeader = np.dtype([
    ('version_major', '<u2'),   # Offset 0 (aligned: 0 % 2 == 0) ✓
    ('version_minor', '<u2'),   # Offset 2 (aligned: 2 % 2 == 0) ✓
    ('flags', '<u4'),           # Offset 4 (aligned: 4 % 4 == 0) ✓
    ('num_samples', '<u8'),     # Offset 8 (aligned: 8 % 8 == 0) ✓
    ('num_fields', '<u2'),      # Offset 16 (aligned: 16 % 2 == 0) ✓
    ('page_size', '<u4'),       # Offset 18 (aligned: 18 % 4 == 2) ✗ BAD!
], align=False)

# GOOD: With explicit padding
GoodHeader = np.dtype([
    ('version_major', '<u2'),
    ('version_minor', '<u2'),
    ('flags', '<u4'),
    ('num_samples', '<u8'),
    ('num_fields', '<u2'),
    ('_pad1', '<u2'),           # 2 bytes padding to align next field
    ('page_size', '<u4'),       # Offset 20 (aligned: 20 % 4 == 0) ✓
], align=False)  # We managed padding manually

# BEST: Let NumPy handle it
BestHeader = np.dtype([
    ('version_major', '<u2'),
    ('version_minor', '<u2'),
    ('flags', '<u4'),
    ('num_samples', '<u8'),
    ('num_fields', '<u2'),
    ('page_size', '<u4'),
], align=True)  # NumPy automatically inserts padding
```

## Complete Header Implementation

```python
import numpy as np
from typing import Dict, Any

# === CONSTANTS ===

# Magic bytes following PNG-style robustness
MAGIC_BYTES = b'\x89MLDF\r\n\x1a\n'  # 8 bytes

# Version
VERSION_MAJOR = 1
VERSION_MINOR = 0

# Flag definitions (bitfield)
FLAG_LITTLE_ENDIAN = 0x0001   # All multi-byte values are little-endian
FLAG_COMPRESSED    = 0x0002   # Data region is compressed
FLAG_CHECKSUMMED   = 0x0004   # CRC32 checksums are present
FLAG_ENCRYPTED     = 0x0008   # Data region is encrypted

# === HEADER DTYPE ===

# This dtype defines the exact binary layout of our header.
# Total size: 64 bytes (power of 2, fits in one cache line)
HeaderType = np.dtype([
    # Version information
    ('version_major', '<u2'),   # Offset 0:  2 bytes
    ('version_minor', '<u2'),   # Offset 2:  2 bytes
    
    # Feature flags (bitfield)
    ('flags', '<u4'),           # Offset 4:  4 bytes
    
    # Dataset counts
    ('num_samples', '<u8'),     # Offset 8:  8 bytes (up to 18 quintillion samples)
    ('num_fields', '<u2'),      # Offset 16: 2 bytes (up to 65535 fields)
    
    # Alignment padding
    ('_pad1', '<u2'),           # Offset 18: 2 bytes (to align next field)
    
    # Memory configuration
    ('page_size', '<u4'),       # Offset 20: 4 bytes (typically 2MB = 0x200000)
    
    # Pointers (byte offsets) to file sections
    ('field_desc_ptr', '<u8'),  # Offset 24: 8 bytes -> where field descriptors start
    ('metadata_ptr', '<u8'),    # Offset 32: 8 bytes -> where metadata table starts
    ('data_ptr', '<u8'),        # Offset 40: 8 bytes -> where data region starts
    ('alloc_table_ptr', '<u8'), # Offset 48: 8 bytes -> where allocation table starts
    
    # Reserved for future use (maintains 64-byte size)
    ('_reserved', '8u1'),       # Offset 56: 8 bytes
    
], align=True)

# Assertion: Compile-time check that our header is exactly 64 bytes
assert HeaderType.itemsize == 64, f"Header must be 64 bytes, got {HeaderType.itemsize}"

# === FIELD DESCRIPTOR DTYPE ===

# Each field (like "image" or "label") has a descriptor that tells us:
# - Its name
# - Its type (integer ID)
# - Type-specific arguments (e.g., JPEG quality for images)
FieldDescType = np.dtype([
    ('name', 'S64'),             # 64-byte fixed string (null-padded)
    ('type_id', '<u2'),          # Identifier for the field type
    ('flags', '<u2'),            # Field-specific flags
    ('metadata_size', '<u4'),    # How many bytes in the metadata table for this field
    ('_pad', '<u4'),             # Padding
    ('arguments', '128u1'),      # Type-specific arguments (e.g., image quality, sample rate)
], align=True)

assert FieldDescType.itemsize == 208, f"FieldDesc should be 208 bytes, got {FieldDescType.itemsize}"

# === ALLOCATION TABLE ENTRY ===

AllocEntryType = np.dtype([
    ('ptr', '<u8'),   # Pointer (file offset) to the data
    ('size', '<u4'),  # Size of the data in bytes
    ('_pad', '<u4'),  # Padding for 16-byte alignment
], align=True)

assert AllocEntryType.itemsize == 16


# === TYPE IDs ===

class TypeID:
    """
    Registry of known field types.
    
    When you read a file, you look up the type_id to know how to decode.
    """
    INT          = 1
    FLOAT        = 2
    BYTES        = 3
    NDARRAY      = 4
    JSON         = 5
    
    # Images
    IMAGE_RAW    = 10
    IMAGE_JPEG   = 11
    IMAGE_PNG    = 12
    
    # Audio
    AUDIO_WAVEFORM     = 20
    AUDIO_COMPRESSED   = 21
    AUDIO_SPECTROGRAM  = 22
    AUDIO_CODEC_TOKENS = 23
    
    # Video
    VIDEO_FRAMES       = 30
    VIDEO_COMPRESSED   = 31
    VIDEO_OPTICAL_FLOW = 32
    
    # Text
    TEXT_RAW           = 40
    TEXT_TOKENIZED     = 41
    TEXT_PACKED        = 42
    TEXT_HIERARCHICAL  = 43
    
    # Multimodal
    MULTIMODAL_SAMPLE  = 50


# === HEADER CREATION ===

def create_header(
    num_samples: int,
    num_fields: int,
    page_size: int = 2 * 1024 * 1024,  # 2MB default
    flags: int = FLAG_LITTLE_ENDIAN,
    field_desc_ptr: int = 0,
    metadata_ptr: int = 0,
    data_ptr: int = 0,
    alloc_table_ptr: int = 0
) -> np.ndarray:
    """
    Create a new header structure.
    
    The pointers (field_desc_ptr, etc.) are typically filled in later,
    after we know where each section will be written.
    """
    header = np.zeros(1, dtype=HeaderType)[0]
    header['version_major'] = VERSION_MAJOR
    header['version_minor'] = VERSION_MINOR
    header['flags'] = flags
    header['num_samples'] = num_samples
    header['num_fields'] = num_fields
    header['page_size'] = page_size
    header['field_desc_ptr'] = field_desc_ptr
    header['metadata_ptr'] = metadata_ptr
    header['data_ptr'] = data_ptr
    header['alloc_table_ptr'] = alloc_table_ptr
    return header


# === FIELD DESCRIPTOR CREATION ===

def create_field_descriptor(
    name: str,
    type_id: int,
    metadata_size: int,
    arguments: bytes = b'',
    flags: int = 0
) -> np.ndarray:
    """
    Create a field descriptor.
    
    Arguments:
        name: Field name (e.g., "image", "label"). Max 63 chars (64th is null terminator).
        type_id: From the TypeID class.
        metadata_size: Size in bytes of this field's entry in the per-sample metadata table.
        arguments: Type-specific arguments (e.g., JPEG quality=90 encoded as bytes).
        flags: Field-specific flags.
    """
    desc = np.zeros(1, dtype=FieldDescType)[0]
    
    # Encode name as null-terminated UTF-8
    name_bytes = name.encode('utf-8')[:63]
    desc['name'] = name_bytes
    
    desc['type_id'] = type_id
    desc['flags'] = flags
    desc['metadata_size'] = metadata_size
    
    # Pack arguments into the fixed-size buffer
    if len(arguments) > 128:
        raise ValueError(f"Field arguments too large: {len(arguments)} > 128 bytes")
    desc['arguments'][:len(arguments)] = np.frombuffer(arguments, dtype='u1')
    
    return desc


# === FILE VALIDATION ===

def validate_file(file_path: str) -> Dict[str, Any]:
    """
    Open a file, validate its structure, and return a summary.
    """
    with open(file_path, 'rb') as f:
        # Step 1: Check magic bytes
        magic = f.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError(
                f"Invalid magic bytes: expected {MAGIC_BYTES!r}, got {magic!r}\n"
                "This file is not in the expected format or is corrupted."
            )
        
        # Step 2: Read header
        header_bytes = f.read(HeaderType.itemsize)
        if len(header_bytes) < HeaderType.itemsize:
            raise ValueError("File truncated: header incomplete")
        
        header = np.frombuffer(header_bytes, dtype=HeaderType)[0]
        
        # Step 3: Version check
        can_read, msg = can_read_version(header['version_major'], header['version_minor'])
        if not can_read:
            raise ValueError(msg)
        
        # Step 4: Read field descriptors
        f.seek(header['field_desc_ptr'])
        field_descs = np.fromfile(f, dtype=FieldDescType, count=header['num_fields'])
        
        # Step 5: Build summary
        fields = []
        for d in field_descs:
            # Decode name (strip null bytes)
            name = bytes(d['name']).rstrip(b'\x00').decode('utf-8')
            fields.append({
                'name': name,
                'type_id': int(d['type_id']),
                'metadata_size': int(d['metadata_size']),
            })
        
        return {
            'version': f"{header['version_major']}.{header['version_minor']}",
            'flags': int(header['flags']),
            'num_samples': int(header['num_samples']),
            'num_fields': int(header['num_fields']),
            'page_size': int(header['page_size']),
            'fields': fields,
            'pointers': {
                'field_desc': int(header['field_desc_ptr']),
                'metadata': int(header['metadata_ptr']),
                'data': int(header['data_ptr']),
                'alloc_table': int(header['alloc_table_ptr']),
            }
        }


def can_read_version(file_major: int, file_minor: int) -> tuple:
    """Check if this reader can handle the file version."""
    if file_major > VERSION_MAJOR:
        return False, f"File version {file_major}.{file_minor} is too new for this reader (max {VERSION_MAJOR}.{VERSION_MINOR})"
    if file_major < VERSION_MAJOR:
        return False, f"File version {file_major}.{file_minor} is too old for this reader (requires {VERSION_MAJOR}.x)"
    return True, "OK"
```

## The Metadata Table: Per-Sample Quick Access

The metadata table is a contiguous array where `table[sample_id]` gives you a fixed-size record containing:
- Direct values for small fields (e.g., labels).
- Pointers and sizes for large/variable fields (e.g., images).

### Structure

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ METADATA TABLE LAYOUT                                                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  row_dtype = np.dtype([                                                        │
│      ('image', ImageMetadata),  # e.g., 24 bytes: ptr, size, height, width    │
│      ('label', '<i8'),          # e.g., 8 bytes: direct value                 │
│      ('audio', AudioMetadata),  # e.g., 24 bytes: ptr, size, sample_rate, ... │
│  ], align=True)                                                                │
│                                                                                │
│  table = np.zeros(num_samples, dtype=row_dtype)                                │
│                                                                                │
│  Sample 0: [image_meta, label_value, audio_meta]                              │
│  Sample 1: [image_meta, label_value, audio_meta]                              │
│  Sample 2: [image_meta, label_value, audio_meta]                              │
│  ...                                                                           │
│  Sample N: [image_meta, label_value, audio_meta]                              │
│                                                                                │
│  Access: table[42]['image']['data_ptr']  -> Offset into data region           │
│  Access: table[42]['label']              -> Direct value (e.g., 7)            │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Common Metadata Sub-Dtypes

```python
# For scalar values stored directly in metadata (no pointer needed)
ScalarMeta = np.dtype([
    ('value', '<i8'),  # The value itself (integer, float, etc.)
])

# For variable-length bytes (pointer + size)
BytesMeta = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u4'),
    ('_pad', '<u4'),
], align=True)

# For images (pointer + size + dimensions)
ImageMeta = np.dtype([
    ('data_ptr', '<u8'),    # Where the image data starts
    ('data_size', '<u4'),   # Compressed size in bytes
    ('height', '<u2'),      # Image height
    ('width', '<u2'),       # Image width
    ('channels', '<u1'),    # 1=grayscale, 3=RGB, 4=RGBA
    ('mode', '<u1'),        # 0=RAW (uncompressed), 1=JPEG, 2=PNG
    ('_pad', '<u2'),
], align=True)

# For audio (pointer + size + metadata)
AudioMeta = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u4'),
    ('num_samples', '<u4'),    # Number of audio samples (not dataset samples!)
    ('sample_rate', '<u4'),    # Hz (e.g., 16000, 44100)
    ('num_channels', '<u2'),   # 1=mono, 2=stereo
    ('bit_depth', '<u1'),      # 8, 16, 24, or 32
    ('codec', '<u1'),          # 0=RAW (PCM), 1=FLAC, 2=AAC, 3=MP3
], align=True)

# For text
TextMeta = np.dtype([
    ('data_ptr', '<u8'),
    ('data_size', '<u4'),
    ('num_tokens', '<u2'),     # If tokenized
    ('encoding', '<u1'),       # 0=UTF-8 raw, 1=token IDs (int32), 2=token IDs (int16)
    ('_pad', '<u1'),
], align=True)
```

## The Allocation Table: Pointer-to-Size Lookup

### Why a Separate Table?

When reading variable-length data, you need two things:
1.  **Where** the data starts (the pointer).
2.  **How much** data to read (the size).

The pointer is stored with the sample's metadata. The size *could* also be there, but separating it allows:
- **Binary search**: If the allocation table is sorted by pointer, we can find any size in $O(\log N)$.
- **Generic reader**: The `read(ptr)` function doesn't need to know which field it's reading.

### Structure

```python
# The allocation table is an array of (pointer, size) pairs, sorted by pointer.

AllocEntryType = np.dtype([
    ('ptr', '<u8'),    # The byte offset in the file
    ('size', '<u4'),   # How many bytes at that offset
    ('_pad', '<u4'),   # Padding
], align=True)

class AllocationTable:
    def __init__(self):
        self.entries = []
    
    def add(self, ptr: int, size: int):
        """Record an allocation during writing."""
        self.entries.append((ptr, size))
    
    def to_array(self) -> np.ndarray:
        """Convert to sorted NumPy array for writing/reading."""
        if not self.entries:
            return np.array([], dtype=AllocEntryType)
        
        arr = np.zeros(len(self.entries), dtype=AllocEntryType)
        for i, (ptr, size) in enumerate(self.entries):
            arr[i]['ptr'] = ptr
            arr[i]['size'] = size
        
        # Sort by pointer for binary search
        return np.sort(arr, order='ptr')
    
    @staticmethod
    def lookup_size(sorted_table: np.ndarray, ptr: int) -> int:
        """Binary search for the size of an allocation."""
        idx = np.searchsorted(sorted_table['ptr'], ptr)
        if idx < len(sorted_table) and sorted_table[idx]['ptr'] == ptr:
            return sorted_table[idx]['size']
        raise KeyError(f"Pointer {ptr} not in allocation table")
```

## Putting It All Together: A Minimal Writer

```python
class SimpleWriter:
    def __init__(self, path: str, fields: dict, page_size: int = 2*1024*1024):
        self.path = path
        self.fields = fields  # {'name': FieldObject, ...}
        self.page_size = page_size
        self.samples = []
        self.alloc_table = AllocationTable()
        
        # Build the metadata row dtype from field metadata types
        self.metadata_dtype = np.dtype([
            (name, field.metadata_type)
            for name, field in fields.items()
        ], align=True)
    
    def write(self, sample: dict):
        """Buffer a sample for writing."""
        self.samples.append(sample)
    
    def close(self):
        """Finalize and write the file."""
        num_samples = len(self.samples)
        num_fields = len(self.fields)
        
        with open(self.path, 'wb') as f:
            # 1. Write magic bytes
            f.write(MAGIC_BYTES)
            
            # 2. Reserve space for header (we'll fill it later)
            header_offset = f.tell()
            f.write(b'\x00' * HeaderType.itemsize)
            
            # 3. Write field descriptors
            field_desc_ptr = f.tell()
            for name, field in self.fields.items():
                desc = create_field_descriptor(
                    name=name,
                    type_id=field.TYPE_ID,
                    metadata_size=field.metadata_type.itemsize,
                    arguments=field.to_binary(),
                )
                f.write(desc.tobytes())
            
            # 4. Align and write placeholder metadata table
            metadata_ptr = self._align(f.tell(), 64)
            f.seek(metadata_ptr)
            metadata_table = np.zeros(num_samples, dtype=self.metadata_dtype)
            metadata_placeholder_pos = f.tell()
            f.write(metadata_table.tobytes())
            
            # 5. Align and write data region
            data_ptr = self._align(f.tell(), self.page_size)
            f.seek(data_ptr)
            
            for i, sample in enumerate(self.samples):
                for name, field in self.fields.items():
                    value = sample.get(name)
                    if value is None:
                        continue
                    
                    # Encode value
                    meta, data_bytes = field.encode(value)
                    
                    # Record data location
                    if 'data_ptr' in meta.dtype.names and len(data_bytes) > 0:
                        current_pos = f.tell()
                        meta['data_ptr'] = current_pos
                        self.alloc_table.add(current_pos, len(data_bytes))
                    
                    # Write data
                    f.write(data_bytes)
                    
                    # Store metadata
                    metadata_table[i][name] = meta
            
            # 6. Write allocation table
            alloc_table_ptr = f.tell()
            alloc_array = self.alloc_table.to_array()
            f.write(alloc_array.tobytes())
            
            # 7. Go back and write final metadata table
            f.seek(metadata_placeholder_pos)
            f.write(metadata_table.tobytes())
            
            # 8. Go back and write final header
            header = create_header(
                num_samples=num_samples,
                num_fields=num_fields,
                page_size=self.page_size,
                field_desc_ptr=field_desc_ptr,
                metadata_ptr=metadata_ptr,
                data_ptr=data_ptr,
                alloc_table_ptr=alloc_table_ptr,
            )
            f.seek(header_offset)
            f.write(header.tobytes())
    
    def _align(self, offset: int, boundary: int) -> int:
        if offset % boundary == 0:
            return offset
        return offset + (boundary - offset % boundary)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

## Exercises

1.  **Extend the Header**: Add a `checksum` field (CRC32 of the data region). Calculate and write it after the data, then verify on read.

2.  **Streaming Writes**: Modify `SimpleWriter` to not require knowing `num_samples` upfront. Hint: Write samples to a temporary buffer, then construct the header at the end.

3.  **Partial Field Reads**: Implement a reader that, given a sample ID and a specific field name, reads *only* that field's data without loading others.

4.  **Format Comparator**: Write a tool that opens two format files and compares their headers, reporting any differences.
