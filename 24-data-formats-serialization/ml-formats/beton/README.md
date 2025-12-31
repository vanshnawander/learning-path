# .beton Format (FFCV)

FFCV's custom format for ultra-fast training data loading.

## Why .beton?

Traditional data loading bottlenecks:
1. Random disk access (ImageNet: millions of small files)
2. Decoding overhead (JPEG decode on CPU)
3. Python GIL limitations
4. Memory copies

## Design Principles

### Memory Mapping
- File is directly memory-mapped
- No data copying to read
- OS handles caching automatically

### Quasi-Random Sampling
- Not fully random but close
- Sequential disk access patterns
- Much faster than true random

### Pre-processing
- Can store decoded images
- Or store JPEG for smaller size
- Flexible field types

## File Structure

```
.beton file:
├── Header (metadata)
├── Field definitions
├── Sample offsets (index)
└── Sample data (contiguous)
```

## Creating .beton Files

```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

writer = DatasetWriter(
    "train.beton",
    {
        'image': RGBImageField(),
        'label': IntField()
    }
)
writer.from_indexed_dataset(dataset)
```

## Reference
- See `ffcv-main/ffcv/writer.py`
- Paper: "FFCV: Accelerating Training by Removing Data Bottlenecks"
