# Binary Representation and Bit Manipulation

The absolute foundation - how data is represented at the hardware level.

## Why This Matters for ML Systems

- **Quantization**: FP32 → FP16 → INT8 → INT4 requires understanding bit layouts
- **Memory optimization**: Packing data efficiently
- **CUDA kernels**: Bit manipulation for masks, indices
- **Data formats**: Understanding .beton, Arrow, etc.

## Learning Objectives

- [ ] Understand two's complement for signed integers
- [ ] Know IEEE 754 floating-point format (FP32, FP16, BF16)
- [ ] Perform bit manipulation operations
- [ ] Understand endianness
- [ ] Pack/unpack data efficiently

## Files in This Directory

| File | Description |
|------|-------------|
| `01_binary_basics.c` | Integer representation, two's complement |
| `02_floating_point.c` | IEEE 754, FP32/FP16/BF16 internals |
| `03_bit_operations.c` | Bitwise operations, masks, shifts |
| `04_endianness.c` | Little vs big endian |
| `05_data_packing.c` | Efficient data storage |
| `binary_deep_dive.ipynb` | Interactive exploration |
