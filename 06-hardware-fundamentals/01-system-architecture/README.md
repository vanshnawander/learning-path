# System Architecture for ML Engineers

Understanding hardware is essential for optimizing ML pipelines.

## The Data Path: From Storage to GPU

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐      │
│   │   SSD   │────▶│  DRAM   │────▶│   CPU   │────▶│   GPU   │      │
│   │ NVMe    │     │  DDR5   │     │ + Cache │     │  HBM    │      │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘      │
│       │               │               │               │             │
│    7 GB/s          100 GB/s       ~1 TB/s         3 TB/s           │
│   (PCIe 4.0)      (8 channels)   (L3 cache)      (HBM3)           │
│                                                                      │
│   Latency:        Latency:        Latency:        Latency:          │
│   ~10-100 µs      ~100 ns         ~10 ns          ~300 ns           │
└─────────────────────────────────────────────────────────────────────┘
```

## Critical Numbers Every ML Engineer Should Know

| Component | Bandwidth | Latency | Size |
|-----------|-----------|---------|------|
| L1 Cache | ~1 TB/s | ~1 ns | 32-64 KB |
| L2 Cache | ~500 GB/s | ~4 ns | 256 KB-1 MB |
| L3 Cache | ~200 GB/s | ~10-20 ns | 8-64 MB |
| DRAM (DDR5) | 50-100 GB/s | ~100 ns | 32-512 GB |
| NVMe SSD | 7 GB/s | ~10 µs | 1-8 TB |
| HDD | 200 MB/s | ~10 ms | 1-20 TB |
| PCIe 4.0 x16 | 32 GB/s | ~1-5 µs | N/A |
| PCIe 5.0 x16 | 64 GB/s | ~1-5 µs | N/A |
| NVLink 4.0 | 900 GB/s | <1 µs | N/A |
| GPU HBM3 | 3+ TB/s | ~300 ns | 40-80 GB |

## Files in This Directory

| File | Description |
|------|-------------|
| `01_latency_numbers.c` | Measure real latencies |
| `02_bandwidth_test.c` | Measure real bandwidths |
| `03_pcie_basics.md` | PCIe architecture deep dive |
| `04_dma_transfers.c` | Direct Memory Access |
| `05_numa_topology.c` | Non-Uniform Memory Access |

## Why This Matters for ML

### Data Loading Bottleneck
```
Training bottleneck = min(
    Storage read speed,      # 7 GB/s NVMe
    CPU preprocessing,       # Variable
    PCIe transfer to GPU,    # 32 GB/s
    GPU compute              # 300+ TFLOPS
)
```

### The GPU Starvation Problem
```
GPU compute: 300 TFLOPS = 300 trillion ops/sec
PCIe bandwidth: 32 GB/s

If each op needs 4 bytes: 32 GB/s ÷ 4 = 8 billion values/sec
GPU can process: 300 trillion ops/sec

GPU is 37,500x faster than PCIe can feed it!
```

**Solution**: Keep data on GPU, minimize transfers!
