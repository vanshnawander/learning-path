# Device I/O: Data Acquisition Fundamentals

How data flows from physical world to your ML model.

## The Data Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Physical World                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐                                                │
│  │   SENSOR    │  Camera, Microphone, Keyboard, etc.            │
│  │  (Analog)   │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │     ADC     │  Analog-to-Digital Converter                   │
│  │ (Digitize)  │  Sampling + Quantization                       │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │  INTERFACE  │  USB, PCIe, Ethernet, I2C, SPI                 │
│  │ (Transfer)  │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │   DRIVER    │  OS kernel driver                              │
│  │  (Kernel)   │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │   BUFFER    │  Ring buffers, DMA                             │
│  │  (Memory)   │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │ USER SPACE  │  Your Python/C++ code                          │
│  │   (API)     │  OpenCV, PyAudio, etc.                         │
│  └─────────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Common Interfaces and Bandwidth

| Interface | Max Bandwidth | Latency | Use Case |
|-----------|---------------|---------|----------|
| USB 2.0 | 480 Mbps | ~1 ms | Webcam, mic |
| USB 3.0 | 5 Gbps | ~125 µs | HD camera |
| USB 3.1/3.2 | 10-20 Gbps | ~125 µs | 4K camera |
| USB4 | 40 Gbps | <100 µs | High-speed |
| PCIe 4.0 x16 | 256 Gbps | ~1 µs | GPU, NVMe |
| Gigabit Ethernet | 1 Gbps | ~1 ms | Network |
| 10GbE | 10 Gbps | ~100 µs | Data center |
| I2C | 3.4 Mbps | ~1 ms | Sensors |
| SPI | 100+ Mbps | ~1 µs | High-speed sensors |

## DMA: Direct Memory Access

```
WITHOUT DMA (CPU copies each byte):
┌─────┐   ┌─────┐   ┌─────┐
│Device│──▶│ CPU │──▶│ RAM │
└─────┘   └─────┘   └─────┘
  CPU busy copying, can't do other work!

WITH DMA (Hardware copies directly):
┌─────┐           ┌─────┐
│Device│─────────▶│ RAM │
└─────┘    DMA    └─────┘
           ↓
        ┌─────┐
        │ CPU │ Free to do other work!
        └─────┘

DMA is essential for:
- GPU data transfer (cudaMemcpy with pinned memory)
- NVMe SSD reads
- Network packet reception
- Camera frame capture
```

## 📁 Files in This Module

| File | Description | Language |
|------|-------------|----------|
| `01_storage_interfaces.md` | NVMe, mmap, io_uring, I/O patterns | Markdown |
| `02_camera_sensor_input.md` | V4L2, camera pipelines, zero-copy | Markdown |
| `03_audio_input_profiled.c` | Audio capture with timing (ALSA) | C |

## 🎯 Learning Path

```
1. Storage Interfaces (01_storage_interfaces.md)
   ├── NVMe architecture and queue depth
   ├── Memory mapping (mmap) for ML
   ├── io_uring for async I/O
   └── I/O patterns: sequential vs random

2. Camera/Sensor Input (02_camera_sensor_input.md)  
   ├── V4L2 capture flow
   ├── Frame rate vs resolution
   ├── Zero-copy with CUDA
   └── Multi-camera sync

3. Audio Input (03_audio_input_profiled.c)
   ├── Ring buffers for streaming
   ├── Buffer size vs latency tradeoff
   └── Sample rate considerations
```

## Key Concepts

### Ring Buffers
```
Producer (device) writes, Consumer (CPU) reads

    Write ──▶
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ D │ D │ D │   │   │   │ D │ D │
  └───┴───┴───┴───┴───┴───┴───┴───┘
                ◀── Read

Benefits:
- No memory allocation during operation
- Lock-free with single producer/consumer
- Fixed memory usage
```

### Memory-Mapped I/O
```
Device registers appear as memory addresses
CPU writes to address → Device receives command

volatile uint32_t* device_ctrl = (uint32_t*)0xFEDC0000;
*device_ctrl = START_CAPTURE;  // Write to device!
```

### Interrupt vs Polling
```
INTERRUPT:
  Device: "I have data!" → CPU handles
  + Low latency
  - Overhead for each interrupt

POLLING:
  CPU: "Do you have data?" (loop)
  + No interrupt overhead
  - Wastes CPU cycles
  
Best: Interrupt for rare events, polling for streaming
```

## ML Data Acquisition Patterns

### Real-time Inference
```python
# Camera → Model → Output (low latency)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()          # ~10ms
    tensor = preprocess(frame)        # ~1ms
    result = model(tensor)            # ~5ms (GPU)
    display(result)                   # ~1ms
    # Total: ~17ms = ~60 FPS possible
```

### Batch Collection
```python
# Collect data, then process
frames = []
for _ in range(1000):
    ret, frame = cap.read()
    frames.append(frame)

# Process in batch (more efficient)
batch = torch.stack([preprocess(f) for f in frames])
results = model(batch)
```

### Async Pipeline
```python
# Overlap capture with processing
import queue
import threading

frame_queue = queue.Queue(maxsize=10)

def capture_thread():
    while running:
        ret, frame = cap.read()
        frame_queue.put(frame)

def process_thread():
    while running:
        frame = frame_queue.get()
        result = model(preprocess(frame))
```
