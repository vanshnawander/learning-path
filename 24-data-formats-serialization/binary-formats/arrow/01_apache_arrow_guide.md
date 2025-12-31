# Apache Arrow: In-Memory Columnar Format

## What is Apache Arrow?

Apache Arrow is a **language-independent columnar memory format** for flat and hierarchical data. It enables:

- **Zero-copy data sharing** between processes
- **Efficient analytics** on columnar data
- **Interoperability** between systems (Spark, Pandas, DuckDB, etc.)

## Why Columnar Format?

```
Row-oriented (traditional):
┌─────────┬─────┬──────────┐
│ name    │ age │ salary   │
├─────────┼─────┼──────────┤
│ "Alice" │ 30  │ 100000.0 │  Row 1
│ "Bob"   │ 25  │ 80000.0  │  Row 2
│ "Carol" │ 35  │ 120000.0 │  Row 3
└─────────┴─────┴──────────┘

Memory: [Alice,30,100000][Bob,25,80000][Carol,35,120000]

Column-oriented (Arrow):
┌─────────────────────────────┐
│ name:    ["Alice","Bob","Carol"]  │
│ age:     [30, 25, 35]             │
│ salary:  [100000.0, 80000.0, 120000.0] │
└─────────────────────────────┘

Memory: [Alice,Bob,Carol][30,25,35][100000,80000,120000]
```

### Benefits of Columnar

1. **Better compression**: Similar values together
2. **SIMD-friendly**: Process 4/8/16 values at once
3. **Cache-efficient**: Only load needed columns
4. **Skip irrelevant data**: Column pruning

## Arrow Memory Layout

### Primitive Arrays

```
Int64 Array: [1, null, 2, 3, null]

┌─────────────────────────────────────────────┐
│ Validity Bitmap: [1, 0, 1, 1, 0]            │
│   (1 byte per 8 elements)                   │
├─────────────────────────────────────────────┤
│ Values Buffer:                              │
│   [1, ?, 2, 3, ?]                           │
│   (8 bytes per element for int64)           │
│   Note: ? values are undefined for nulls    │
└─────────────────────────────────────────────┘
```

### Variable-Length Arrays (Strings)

```
String Array: ["hello", "world", null, "!"]

┌─────────────────────────────────────────────┐
│ Validity Bitmap: [1, 1, 0, 1]               │
├─────────────────────────────────────────────┤
│ Offsets Buffer: [0, 5, 10, 10, 11]          │
│   (n+1 offsets for n strings)               │
├─────────────────────────────────────────────┤
│ Data Buffer: "helloworld!"                  │
│   (concatenated string data)                │
└─────────────────────────────────────────────┘

String[0] = data[offsets[0]:offsets[1]] = data[0:5] = "hello"
String[1] = data[offsets[1]:offsets[2]] = data[5:10] = "world"
String[2] = null (validity bit = 0)
String[3] = data[offsets[3]:offsets[4]] = data[10:11] = "!"
```

### Nested Types (List, Struct)

```
List<Int32> Array: [[1, 2], [3], null, [4, 5, 6]]

┌─────────────────────────────────────────────┐
│ Validity Bitmap: [1, 1, 0, 1]               │
├─────────────────────────────────────────────┤
│ Offsets: [0, 2, 3, 3, 6]                    │
├─────────────────────────────────────────────┤
│ Child Int32 Array:                          │
│   Values: [1, 2, 3, 4, 5, 6]               │
└─────────────────────────────────────────────┘
```

## Zero-Copy Sharing

Arrow enables sharing data without copying:

```
┌──────────────┐     ┌──────────────┐
│   Process A  │     │   Process B  │
│              │     │              │
│  ┌────────┐  │     │  ┌────────┐  │
│  │ Arrow  │  │     │  │ Arrow  │  │
│  │ Array  │──┼─────┼──│ Array  │  │
│  └────────┘  │     │  └────────┘  │
│      │       │     │      │       │
└──────┼───────┘     └──────┼───────┘
       │                    │
       ▼                    ▼
  ┌─────────────────────────────────┐
  │    Shared Memory / mmap         │
  │    (Arrow buffers)              │
  └─────────────────────────────────┘
```

### IPC (Inter-Process Communication)

```python
import pyarrow as pa

# Write to shared memory
table = pa.table({'col': [1, 2, 3]})
sink = pa.BufferOutputStream()
writer = pa.ipc.new_stream(sink, table.schema)
writer.write_table(table)
writer.close()

# Read without copy
buffer = sink.getvalue()
reader = pa.ipc.open_stream(buffer)
received_table = reader.read_all()  # Zero-copy!
```

## Arrow Flight: High-Speed Data Transfer

Arrow Flight is a gRPC-based protocol for Arrow data:

```
┌──────────┐     Arrow Flight      ┌──────────┐
│  Client  │ ◄──────────────────► │  Server  │
│          │   (gRPC + Arrow IPC)  │          │
└──────────┘                       └──────────┘

Benefits:
- Parallel streams
- Authentication
- Encryption
- Metadata
- ~10GB/s throughput possible
```

## Python PyArrow Usage

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Create table
table = pa.table({
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [30, 25, 35],
    'salary': [100000.0, 80000.0, 120000.0]
})

# Compute operations (vectorized)
result = pc.add(table['age'], 1)
filtered = pc.filter(table, pc.greater(table['age'], 26))

# Memory-map Parquet (zero-copy read)
table = pq.read_table('data.parquet', memory_map=True)

# Convert to/from Pandas (zero-copy when possible)
df = table.to_pandas()  # Zero-copy for compatible types
table = pa.Table.from_pandas(df)

# Chunked operations for large data
batches = table.to_batches(max_chunksize=10000)
for batch in batches:
    process(batch)
```

## Arrow + GPU (cuDF/RAPIDS)

```python
import cudf
import pyarrow as pa

# Arrow to GPU
arrow_table = pa.table({'x': [1, 2, 3]})
gpu_df = cudf.DataFrame.from_arrow(arrow_table)

# GPU operations
result = gpu_df['x'] * 2

# Back to Arrow (no CPU copy!)
arrow_result = result.to_arrow()
```

## Performance Characteristics

| Operation | Arrow | Row-Format |
|-----------|-------|------------|
| Single column scan | O(n) | O(n * cols) |
| Aggregation | SIMD-friendly | Cache misses |
| Filter | Vectorized | Row-by-row |
| Serialization | Zero-copy | Copy required |

### Benchmark: Column Sum

```
Data: 100M rows, 10 columns

Row-oriented (iterate rows):
  Time: 2500 ms
  Bandwidth: 4 GB/s

Arrow columnar:
  Time: 50 ms
  Bandwidth: 200 GB/s (SIMD + cache)
  
Speedup: 50x
```

## When to Use Arrow

**Good fit:**
- Analytics / OLAP workloads
- Data exchange between systems
- Memory-mapped large datasets
- GPU data processing
- Streaming data

**Not ideal for:**
- OLTP (many small updates)
- Point lookups by row
- Small datasets (< 1MB)
- Highly nested JSON-like data

## Ecosystem

| System | Arrow Support |
|--------|--------------|
| Pandas | Native (PyArrow backend) |
| Spark | Native (PySpark Arrow) |
| DuckDB | Native |
| Polars | Built on Arrow |
| RAPIDS | cuDF Arrow-based |
| DataFusion | Rust Arrow query engine |
| Substrait | Query plan format |

## References

- Apache Arrow specification: https://arrow.apache.org/docs/format/
- PyArrow documentation
- "Apache Arrow: A Cross-Language Platform for Columnar Memory"
