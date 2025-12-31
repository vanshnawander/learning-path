# Hardware Fundamentals for ML Engineers

Understanding hardware is essential for building efficient ML systems.

## Why Hardware Knowledge Matters

```
Software optimization without hardware knowledge:
  "Make the code faster" → Random changes

Software optimization WITH hardware knowledge:
  "Memory bound → reduce data movement"
  "Cache miss → improve locality"
  "PCIe bottleneck → batch transfers"
```

## Directory Structure

```
06-hardware-fundamentals/
├── 01-system-architecture/
│   ├── README.md               # System overview
│   ├── 01_latency_numbers.c    # Measure real latencies
│   └── 02_pcie_deep_dive.md    # PCIe for GPU communication
├── 02-memory-hierarchy-deep/
│   ├── README.md               # Memory hierarchy overview
│   └── 01_cache_line_effects.c # Cache line demonstration
└── README.md
```

## The Numbers That Matter

| Component | Latency | Bandwidth | Implication |
|-----------|---------|-----------|-------------|
| L1 Cache | 1 ns | ~1 TB/s | Keep hot data here |
| L2 Cache | 4 ns | ~500 GB/s | Working set |
| L3 Cache | 10-20 ns | ~200 GB/s | Shared data |
| DRAM | 100 ns | 50-100 GB/s | Model weights |
| NVMe SSD | 10 µs | 7 GB/s | Dataset |
| PCIe 4.0 | 1-5 µs | 32 GB/s | CPU↔GPU |
| GPU HBM | 300 ns | 2-3 TB/s | Tensor operations |

## Key Concepts Covered

### 1. PCIe and Data Transfer
- PCIe generations and bandwidth
- Pinned vs pageable memory
- DMA and async transfers
- Multi-GPU topology

### 2. Memory Hierarchy
- Cache line effects
- Cache associativity
- TLB and page tables
- NUMA considerations

### 3. Storage I/O
- Block devices
- NVMe internals
- I/O schedulers
- Direct I/O vs buffered

## Learning Path

1. **Week 1**: System architecture overview
2. **Week 2**: Memory hierarchy deep dive
3. **Week 3**: PCIe and GPU communication
4. **Week 4**: Storage and I/O optimization

## Connection to ML Systems

| Hardware Concept | ML Application |
|------------------|----------------|
| Cache locality | Tensor tiling |
| PCIe bandwidth | Data loading |
| Memory bandwidth | Flash Attention |
| NUMA | Multi-socket training |
| NVMe | Large dataset access |
