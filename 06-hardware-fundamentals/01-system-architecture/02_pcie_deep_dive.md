# PCIe Deep Dive: The GPU-CPU Highway

PCIe is the bottleneck between CPU and GPU. Understanding it is critical.

## PCIe Generations

| Gen | Per Lane | x16 Slot | Encoding | Year |
|-----|----------|----------|----------|------|
| 3.0 | 1 GB/s | 16 GB/s | 128b/130b | 2010 |
| 4.0 | 2 GB/s | 32 GB/s | 128b/130b | 2017 |
| 5.0 | 4 GB/s | 64 GB/s | 128b/130b | 2019 |
| 6.0 | 8 GB/s | 128 GB/s | PAM4 | 2022 |

**Note**: These are theoretical maximums. Real-world is 85-95%.

## PCIe Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CPU                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Root Complex                        │    │
│  │    (Integrated into CPU or chipset)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│           │              │              │                    │
│       x16 lanes      x4 lanes       x4 lanes                │
│           │              │              │                    │
│      ┌────┴────┐    ┌────┴────┐   ┌────┴────┐              │
│      │   GPU   │    │  NVMe   │   │  NIC    │              │
│      │  32GB/s │    │  8GB/s  │   │ 100Gbps │              │
│      └─────────┘    └─────────┘   └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Data Transfer Types

### 1. Host to Device (H2D)
```
CPU Memory ──PCIe──▶ GPU Memory

cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
```

### 2. Device to Host (D2H)
```
GPU Memory ──PCIe──▶ CPU Memory

cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
```

### 3. Peer to Peer (P2P) - GPUs directly
```
GPU 0 ──NVLink/PCIe──▶ GPU 1

cudaMemcpyPeer(d_data1, dev1, d_data0, dev0, size);
```

## PCIe Latency Breakdown

```
Total PCIe transfer time = Setup + Transmission + Completion

Setup latency:    ~1-2 µs (TLP header, routing)
Transmission:     Size / Bandwidth
Completion:       ~0.5-1 µs (acknowledgment)

For 1 MB transfer at PCIe 4.0 x16:
  Setup:          ~1.5 µs
  Transmission:   1 MB / 32 GB/s = 31 µs
  Completion:     ~0.5 µs
  Total:          ~33 µs
```

## Pinned vs Pageable Memory

### Pageable Memory (Default)
```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Pageable │───▶│  Pinned  │───▶│   GPU    │
│  Memory  │    │  Buffer  │    │  Memory  │
└──────────┘    └──────────┘    └──────────┘
    CPU              CPU            GPU
              (extra copy!)
```

### Pinned Memory (cudaMallocHost)
```
┌──────────┐    ┌──────────┐
│  Pinned  │───▶│   GPU    │
│  Memory  │    │  Memory  │
└──────────┘    └──────────┘
    CPU            GPU
         (DMA direct!)
```

**Performance**: Pinned memory is 2-3x faster!

## Code Example: Pinned Memory

```cpp
// SLOW: Pageable memory
float* h_data = (float*)malloc(size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// FAST: Pinned memory
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // Pinned!
cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);

// PyTorch equivalent
tensor = torch.randn(1000, 1000, pin_memory=True)
tensor.cuda()  # Fast transfer!
```

## Async Transfers & Overlap

```cpp
// Create CUDA stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async copy while GPU computes
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(d_input, d_output);
cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream);
```

## PCIe Bandwidth Optimization

### DO:
- Use pinned memory
- Batch transfers (one large > many small)
- Overlap transfers with compute
- Use async copies

### DON'T:
- Transfer on every iteration
- Use small transfers
- Block on synchronous copies
- Transfer data that could stay on GPU

## Multi-GPU PCIe Topology

```
            ┌─────────────────────────┐
            │         CPU 0           │
            │    (PCIe Root Complex)  │
            └────────────┬────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │  GPU 0  │    │  GPU 1  │    │  GPU 2  │
    │         │◄──►│         │◄──►│         │
    │         │    │         │    │         │
    └─────────┘    └─────────┘    └─────────┘
         NVLink connections (if available)
```

**Check topology**: `nvidia-smi topo -m`

## ML Training Implications

| Scenario | Bottleneck | Solution |
|----------|------------|----------|
| Data loading | CPU→GPU transfer | Prefetch, pin_memory |
| Gradient sync | GPU→GPU via CPU | NVLink, NCCL |
| Model checkpointing | GPU→CPU→Disk | Async save, fp16 |
| Inference batch | CPU→GPU per batch | Batch on GPU |
