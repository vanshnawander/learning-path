# Camera and Sensor Input for ML

How visual data flows from sensors to your training pipeline.

## Camera Pipeline Overview

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    CAMERA TO GPU PIPELINE                              ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ┌─────────────┐                                                      ║
║  │   Sensor    │  Photons → Electrons → Digital                       ║
║  │  (CMOS/CCD) │  Bayer pattern: RGGB                                 ║
║  └──────┬──────┘                                                      ║
║         │ Raw sensor data (10-14 bit)                                 ║
║  ┌──────▼──────┐                                                      ║
║  │     ISP     │  Image Signal Processor                              ║
║  │  (On-chip)  │  Demosaic, denoise, white balance                   ║
║  └──────┬──────┘                                                      ║
║         │ RGB/YUV data                                                ║
║  ┌──────▼──────┐                                                      ║
║  │   Encoder   │  H.264/H.265/MJPEG compression                      ║
║  │  (Optional) │  Reduces bandwidth for storage/transfer             ║
║  └──────┬──────┘                                                      ║
║         │ USB/PCIe/MIPI                                               ║
║  ┌──────▼──────┐                                                      ║
║  │    Host     │  CPU memory (DMA transfer)                          ║
║  │   Memory    │  Ring buffer for frames                             ║
║  └──────┬──────┘                                                      ║
║         │ Decode (if compressed)                                      ║
║  ┌──────▼──────┐                                                      ║
║  │    GPU      │  For training/inference                             ║
║  │   Memory    │  cudaMemcpy or zero-copy                            ║
║  └─────────────┘                                                      ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## Camera Interfaces

| Interface | Bandwidth | Latency | Use Case |
|-----------|-----------|---------|----------|
| USB 2.0 | 480 Mbps | ~1 ms | Webcams |
| USB 3.0 | 5 Gbps | 125 µs | HD cameras |
| GigE Vision | 1 Gbps | ~1 ms | Industrial |
| Camera Link | 6.8 Gbps | <1 µs | Machine vision |
| MIPI CSI-2 | 2.5 Gbps/lane | <1 µs | Embedded (Jetson) |
| PCIe | 32 GB/s | ~1 µs | High-speed capture |

## Frame Rate vs Resolution

```
Bandwidth = Width × Height × BytesPerPixel × FPS

Examples:
1080p @ 30fps, RGB:    1920 × 1080 × 3 × 30 = 186 MB/s
4K @ 30fps, RGB:       3840 × 2160 × 3 × 30 = 746 MB/s
4K @ 60fps, RGB:       3840 × 2160 × 3 × 60 = 1.49 GB/s

Interface limits:
USB 3.0 (5 Gbps):      ~500 MB/s practical → 4K@20fps max
USB 3.2 (20 Gbps):     ~2 GB/s practical → 4K@60fps possible
```

## V4L2: Linux Video Capture

### Basic V4L2 Capture Flow
```c
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

// 1. Open device
int fd = open("/dev/video0", O_RDWR);

// 2. Query capabilities
struct v4l2_capability cap;
ioctl(fd, VIDIOC_QUERYCAP, &cap);

// 3. Set format
struct v4l2_format fmt = {0};
fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
fmt.fmt.pix.width = 1920;
fmt.fmt.pix.height = 1080;
fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
ioctl(fd, VIDIOC_S_FMT, &fmt);

// 4. Request buffers (mmap for zero-copy)
struct v4l2_requestbuffers req = {0};
req.count = 4;  // Ring buffer
req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
req.memory = V4L2_MEMORY_MMAP;
ioctl(fd, VIDIOC_REQBUFS, &req);

// 5. Map buffers
struct buffer {
    void* start;
    size_t length;
} buffers[4];

for (int i = 0; i < 4; i++) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    ioctl(fd, VIDIOC_QUERYBUF, &buf);
    
    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buf.m.offset);
}

// 6. Queue buffers and start streaming
for (int i = 0; i < 4; i++) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    ioctl(fd, VIDIOC_QBUF, &buf);
}

enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
ioctl(fd, VIDIOC_STREAMON, &type);

// 7. Capture loop
while (running) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    // Dequeue filled buffer
    ioctl(fd, VIDIOC_DQBUF, &buf);
    
    // Process frame at buffers[buf.index].start
    process_frame(buffers[buf.index].start, buf.bytesused);
    
    // Re-queue buffer
    ioctl(fd, VIDIOC_QBUF, &buf);
}
```

## OpenCV Capture (Simple Python)

```python
import cv2
import time

# Open camera
cap = cv2.VideoCapture(0)  # /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# Measure actual performance
frame_times = []
for i in range(100):
    start = time.perf_counter()
    ret, frame = cap.read()
    frame_times.append(time.perf_counter() - start)

avg_time = sum(frame_times) / len(frame_times) * 1000
print(f"Average frame time: {avg_time:.2f} ms")
print(f"Effective FPS: {1000/avg_time:.1f}")
```

## NVIDIA Hardware Decode (NVDEC)

```python
# Using NVIDIA's hardware decoder for video files
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def
def video_pipeline(video_files):
    # GPU-accelerated video decode
    video = fn.readers.video(
        device="gpu",
        filenames=video_files,
        sequence_length=16,
        stride=1,
        shard_id=0,
        num_shards=1
    )
    return video

# Performance: 1000+ FPS for 1080p decode!
```

## Zero-Copy Camera to GPU (CUDA)

```cpp
// For NVIDIA GPUs with RDMA support
#include <cuda_runtime.h>

// Register camera buffer with CUDA
void* camera_buffer;  // From V4L2 mmap
cudaHostRegister(camera_buffer, buffer_size, 
                 cudaHostRegisterDefault);

// Now DMA directly to GPU
cudaMemcpyAsync(gpu_buffer, camera_buffer, buffer_size,
                cudaMemcpyHostToDevice, stream);

// Or use mapped memory (zero-copy)
cudaHostRegister(camera_buffer, buffer_size,
                 cudaHostRegisterMapped);
void* gpu_ptr;
cudaHostGetDevicePointer(&gpu_ptr, camera_buffer, 0);
// gpu_ptr can be used directly in kernels!
```

## Timing: Camera Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│              CAMERA LATENCY BREAKDOWN (1080p@30fps)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Exposure time:           5-33 ms (depends on lighting)         │
│  Sensor readout:          ~10 ms                                │
│  ISP processing:          ~2 ms                                 │
│  Encoding (if used):      ~5 ms                                 │
│  USB transfer:            ~5 ms                                 │
│  Driver/buffer:           ~2 ms                                 │
│  ─────────────────────────────────                              │
│  Total sensor-to-RAM:     30-60 ms                              │
│                                                                  │
│  For real-time ML:                                              │
│  + JPEG decode:           ~5 ms (CPU) or ~0.5 ms (GPU)         │
│  + Resize/preprocess:     ~2 ms (CPU) or ~0.2 ms (GPU)         │
│  + CPU→GPU transfer:      ~1 ms                                 │
│  + Model inference:       ~5-50 ms                              │
│  ─────────────────────────────────                              │
│  Total end-to-end:        50-120 ms                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Camera Sync

```python
# Synchronizing multiple cameras for multi-view ML
import cv2
import threading
import queue

class SyncedCameraCapture:
    def __init__(self, camera_ids):
        self.cameras = [cv2.VideoCapture(i) for i in camera_ids]
        self.frame_queues = [queue.Queue(maxsize=2) for _ in camera_ids]
        self.threads = []
        
    def _capture_thread(self, cam_idx):
        while self.running:
            ret, frame = self.cameras[cam_idx].read()
            if ret:
                try:
                    self.frame_queues[cam_idx].put_nowait(frame)
                except queue.Full:
                    self.frame_queues[cam_idx].get()  # Drop oldest
                    self.frame_queues[cam_idx].put(frame)
    
    def get_synced_frames(self, timeout=0.1):
        """Get frames from all cameras (approximately synced)"""
        frames = []
        for q in self.frame_queues:
            try:
                frames.append(q.get(timeout=timeout))
            except queue.Empty:
                return None
        return frames
```
