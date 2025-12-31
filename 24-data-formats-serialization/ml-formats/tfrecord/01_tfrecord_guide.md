# TFRecord: TensorFlow's Data Format

## Overview

TFRecord is TensorFlow's native binary format optimized for sequential reading.

```
TFRecord file structure:
┌──────────────────────────────────────────────────────────────┐
│ Record 0                                                      │
│   Length (8 bytes) + CRC (4 bytes) + Data + CRC (4 bytes)   │
├──────────────────────────────────────────────────────────────┤
│ Record 1                                                      │
│   Length (8 bytes) + CRC (4 bytes) + Data + CRC (4 bytes)   │
├──────────────────────────────────────────────────────────────┤
│ ...                                                          │
└──────────────────────────────────────────────────────────────┘
```

## Creating TFRecords

### Basic Example

```python
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Create TFRecord writer
with tf.io.TFRecordWriter('data.tfrecord') as writer:
    for i in range(1000):
        # Read image
        image_data = open(f'images/{i}.jpg', 'rb').read()
        
        # Create feature dict
        feature = {
            'image': _bytes_feature(image_data),
            'label': _int64_feature(labels[i]),
            'height': _int64_feature(224),
            'width': _int64_feature(224),
        }
        
        # Create Example protobuf
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Write serialized example
        writer.write(example.SerializeToString())
```

### With Compression

```python
options = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter('data.tfrecord.gz', options=options) as writer:
    # Write records
    pass
```

## Reading TFRecords

### Basic Reading

```python
# Define feature description
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(serialized):
    example = tf.io.parse_single_example(serialized, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, example['label']

# Create dataset
dataset = tf.data.TFRecordDataset('data.tfrecord')
dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

for images, labels in dataset:
    # Training step
    pass
```

### Sharded Reading

```python
# Read multiple shards
files = tf.data.Dataset.list_files('shards/train-*.tfrecord')
dataset = files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=4,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

## Performance Optimization

```python
# Optimized pipeline
dataset = (
    tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP',
        num_parallel_reads=tf.data.AUTOTUNE
    )
    .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # Cache in memory after first epoch
    .shuffle(buffer_size=10000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
```

## TFRecord vs Other Formats

| Feature | TFRecord | WebDataset | FFCV |
|---------|----------|------------|------|
| Framework | TensorFlow | PyTorch | PyTorch |
| Random Access | No | No | Yes |
| Compression | Yes | Yes | Yes |
| Schema | Protobuf | File extension | Custom |
| Cloud Support | Good | Excellent | Limited |

## Best Practices

1. **Shard large datasets** - 100MB-1GB per file
2. **Use compression for images** - GZIP for smaller files
3. **Prefetch and parallel reads** - Use AUTOTUNE
4. **Cache if dataset fits in memory**
5. **Use interleave for multiple shards**

## References

- TensorFlow Data Guide: https://www.tensorflow.org/guide/data
- TFRecord documentation: https://www.tensorflow.org/tutorials/load_data/tfrecord
