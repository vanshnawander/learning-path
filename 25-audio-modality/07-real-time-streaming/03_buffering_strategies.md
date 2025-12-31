# Buffering Strategies for Real-Time Audio

Comprehensive guide to buffering techniques for low-latency audio systems. Critical for production streaming applications.

## Table of Contents
1. [Why Buffering?](#why-buffering)
2. [Buffer Size Selection](#buffer-size-selection)
3. [Ring Buffers](#ring-buffers)
4. [Double Buffering](#double-buffering)
5. [Adaptive Buffering](#adaptive-buffering)
6. [Jitter Management](#jitter-management)
7. [Implementation Examples](#implementation-examples)
8. [Production Considerations](#production-considerations)

---

## Why Buffering?

### The Timing Problem

```
Audio processing has timing variability:

┌─────────────────────────────────────────────────────────────┐
│  Ideal (no buffering):                                      │
│                                                              │
│  Input:  [chunk] → Process → [output]                       │
│  Time:   10ms      10ms       10ms                          │
│                                                              │
│  Reality:                                                    │
│                                                              │
│  Input:  [chunk] → Process → [output]                       │
│  Time:   10ms      8-15ms     10ms  ← Variable!            │
│                                                              │
│  Without buffer: Glitches when processing > 10ms            │
│  With buffer: Smooth output, slightly higher latency        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Sources of Jitter

```
1. PROCESSING TIME VARIANCE
   ├── CPU scheduling
   ├── GPU kernel launch overhead
   ├── Memory allocation
   └── Cache misses

2. NETWORK JITTER (if applicable)
   ├── Packet arrival time variance
   ├── Network congestion
   └── Routing changes

3. SYSTEM LOAD
   ├── Other processes
   ├── Interrupts
   └── Background tasks

Buffers absorb this variance!
```

---

## Buffer Size Selection

### The Trade-off

```
SMALL BUFFER:
✓ Low latency
✓ Responsive
✗ Frequent underruns (glitches)
✗ High CPU usage

LARGE BUFFER:
✓ Smooth playback
✓ Tolerates jitter
✗ High latency
✗ Poor for interactive apps

OPTIMAL:
- Just large enough to handle jitter
- Minimize latency
- Depends on system and requirements
```

### Calculating Buffer Size

```python
def calculate_buffer_size(
    target_latency_ms: float,
    sample_rate: int,
    safety_factor: float = 1.5
) -> int:
    """
    Calculate buffer size based on target latency.
    
    Args:
        target_latency_ms: Desired latency in milliseconds
        sample_rate: Audio sample rate
        safety_factor: Multiplier for safety margin
    
    Returns:
        Buffer size in samples
    """
    # Base buffer size
    base_samples = int(target_latency_ms * sample_rate / 1000)
    
    # Add safety margin
    buffer_size = int(base_samples * safety_factor)
    
    # Round to power of 2 for efficiency
    buffer_size = 2 ** int(np.ceil(np.log2(buffer_size)))
    
    return buffer_size


# Example
buffer_size = calculate_buffer_size(
    target_latency_ms=20,  # 20ms target
    sample_rate=24000,
    safety_factor=1.5
)
print(f"Buffer size: {buffer_size} samples")
print(f"Actual latency: {buffer_size / 24000 * 1000:.1f}ms")
```

---

## Ring Buffers

### Circular Buffer Design

```
Ring buffer (circular buffer):

Write pointer ────────────────────┐
                                  ▼
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
              ▲
Read pointer ─┘

Properties:
├── Fixed size (no reallocation)
├── Constant-time operations
├── Wraps around at end
└── Efficient for streaming
```

### Implementation

```python
class RingBuffer:
    """
    Lock-free ring buffer for audio samples.
    
    Thread-safe for single producer, single consumer.
    """
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
    
    def write(self, data: np.ndarray) -> int:
        """
        Write data to buffer.
        
        Returns:
            Number of samples written (may be less than len(data) if full)
        """
        available = self.available_write()
        to_write = min(len(data), available)
        
        if to_write == 0:
            return 0  # Buffer full
        
        # Handle wrap-around
        end_pos = self.write_pos + to_write
        
        if end_pos <= self.size:
            # No wrap
            self.buffer[self.write_pos:end_pos] = data[:to_write]
        else:
            # Wrap around
            first_part = self.size - self.write_pos
            self.buffer[self.write_pos:] = data[:first_part]
            self.buffer[:to_write - first_part] = data[first_part:to_write]
        
        self.write_pos = (self.write_pos + to_write) % self.size
        
        return to_write
    
    def read(self, num_samples: int) -> np.ndarray:
        """
        Read data from buffer.
        
        Returns:
            Samples read (may be less than requested if empty)
        """
        available = self.available_read()
        to_read = min(num_samples, available)
        
        if to_read == 0:
            return np.array([])  # Buffer empty
        
        # Handle wrap-around
        end_pos = self.read_pos + to_read
        
        if end_pos <= self.size:
            # No wrap
            data = self.buffer[self.read_pos:end_pos].copy()
        else:
            # Wrap around
            first_part = self.size - self.read_pos
            data = np.concatenate([
                self.buffer[self.read_pos:],
                self.buffer[:to_read - first_part]
            ])
        
        self.read_pos = (self.read_pos + to_read) % self.size
        
        return data
    
    def available_read(self) -> int:
        """Number of samples available to read."""
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        else:
            return self.size - self.read_pos + self.write_pos
    
    def available_write(self) -> int:
        """Number of samples that can be written."""
        return self.size - self.available_read() - 1  # Keep 1 sample gap
```

---

## Double Buffering

### Concept

```
Two buffers: one for reading, one for writing

While processing buffer A, fill buffer B:

Time →
┌─────────────────────────────────────────┐
│ Fill A    │ Process A │ Fill A    │ ... │
│           │ Fill B    │ Process B │     │
└─────────────────────────────────────────┘

Advantages:
├── Simple to implement
├── Predictable latency
├── No race conditions

Disadvantages:
├── Fixed latency (2× buffer size)
├── Less flexible than ring buffer
└── Memory overhead
```

### Implementation

```python
class DoubleBuffer:
    """
    Double buffering for audio processing.
    """
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer_a = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_b = np.zeros(buffer_size, dtype=np.float32)
        
        self.active_buffer = 'A'  # Which buffer is being filled
        self.fill_pos = 0
    
    def add_samples(self, samples: np.ndarray):
        """Add samples to current buffer."""
        active = self.buffer_a if self.active_buffer == 'A' else self.buffer_b
        
        # Add to buffer
        space = self.buffer_size - self.fill_pos
        to_add = min(len(samples), space)
        active[self.fill_pos:self.fill_pos + to_add] = samples[:to_add]
        self.fill_pos += to_add
        
        # Return overflow
        return samples[to_add:]
    
    def swap_buffers(self) -> np.ndarray:
        """
        Swap buffers and return filled buffer for processing.
        """
        # Get filled buffer
        filled = self.buffer_a.copy() if self.active_buffer == 'A' else self.buffer_b.copy()
        
        # Swap
        self.active_buffer = 'B' if self.active_buffer == 'A' else 'A'
        self.fill_pos = 0
        
        return filled
```

---

## Adaptive Buffering

### Dynamic Buffer Adjustment

```python
class AdaptiveBuffer:
    """
    Adaptive buffering that adjusts to network/processing conditions.
    
    Increases buffer when underruns occur.
    Decreases buffer when stable.
    """
    def __init__(
        self,
        initial_size: int,
        min_size: int,
        max_size: int,
        sample_rate: int
    ):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.sample_rate = sample_rate
        
        self.buffer = RingBuffer(max_size)
        
        # Statistics
        self.underrun_count = 0
        self.overrun_count = 0
        self.stable_count = 0
        
        # Adjustment parameters
        self.check_interval = 1000  # Check every 1000 samples
        self.samples_since_check = 0
    
    def add_samples(self, samples: np.ndarray):
        """Add samples and check for adjustment."""
        written = self.buffer.write(samples)
        
        if written < len(samples):
            self.overrun_count += 1
        
        self.samples_since_check += written
        
        # Periodic adjustment
        if self.samples_since_check >= self.check_interval:
            self._adjust_buffer_size()
            self.samples_since_check = 0
    
    def get_samples(self, num_samples: int) -> np.ndarray:
        """Get samples and check for underrun."""
        samples = self.buffer.read(num_samples)
        
        if len(samples) < num_samples:
            self.underrun_count += 1
            # Pad with zeros (silence)
            samples = np.pad(samples, (0, num_samples - len(samples)))
        else:
            self.stable_count += 1
        
        return samples
    
    def _adjust_buffer_size(self):
        """Adjust buffer size based on statistics."""
        # Increase if underruns
        if self.underrun_count > 0:
            new_size = min(int(self.current_size * 1.5), self.max_size)
            print(f"Buffer underrun detected. Increasing: {self.current_size} → {new_size}")
            self.current_size = new_size
            self.underrun_count = 0
        
        # Decrease if very stable
        elif self.stable_count > 10 and self.current_size > self.min_size:
            new_size = max(int(self.current_size * 0.9), self.min_size)
            print(f"Buffer stable. Decreasing: {self.current_size} → {new_size}")
            self.current_size = new_size
        
        # Reset counters
        self.stable_count = 0
        self.overrun_count = 0
```

---

## Jitter Management

### Measuring Jitter

```python
class JitterAnalyzer:
    """
    Analyze timing jitter in audio stream.
    """
    def __init__(self, expected_interval_ms: float):
        self.expected_interval_ms = expected_interval_ms
        self.last_timestamp = None
        self.intervals = []
    
    def record_packet(self, timestamp_ms: float):
        """Record packet arrival time."""
        if self.last_timestamp is not None:
            interval = timestamp_ms - self.last_timestamp
            self.intervals.append(interval)
        
        self.last_timestamp = timestamp_ms
    
    def get_statistics(self):
        """Get jitter statistics."""
        if len(self.intervals) < 2:
            return None
        
        intervals = np.array(self.intervals)
        
        return {
            'mean_interval': intervals.mean(),
            'std_interval': intervals.std(),
            'jitter': intervals.std(),  # Standard definition
            'min_interval': intervals.min(),
            'max_interval': intervals.max(),
            'expected': self.expected_interval_ms,
        }
    
    def recommended_buffer_size(self, sample_rate: int, safety_factor: float = 3.0):
        """
        Recommend buffer size based on observed jitter.
        """
        stats = self.get_statistics()
        if stats is None:
            return None
        
        # Buffer should handle mean + safety_factor * jitter
        buffer_ms = stats['mean_interval'] + safety_factor * stats['jitter']
        buffer_samples = int(buffer_ms * sample_rate / 1000)
        
        return buffer_samples
```

### Jitter Buffer

```python
class JitterBuffer:
    """
    Jitter buffer for packet-based audio (e.g., VoIP).
    
    Reorders packets and compensates for timing variance.
    """
    def __init__(self, buffer_size: int, sample_rate: int):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.buffer = {}  # sequence_num -> samples
        self.next_seq = 0
        self.playout_buffer = []
    
    def add_packet(self, sequence_num: int, samples: np.ndarray):
        """
        Add packet to jitter buffer.
        """
        self.buffer[sequence_num] = samples
        
        # Remove old packets
        while len(self.buffer) > self.buffer_size:
            oldest = min(self.buffer.keys())
            del self.buffer[oldest]
    
    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Get samples for playback.
        
        Handles packet loss with concealment.
        """
        output = []
        
        while len(output) < num_samples:
            # Check if next packet available
            if self.next_seq in self.buffer:
                # Packet available
                packet = self.buffer.pop(self.next_seq)
                output.extend(packet)
                self.last_good_packet = packet
            else:
                # Packet lost - use concealment
                if hasattr(self, 'last_good_packet'):
                    # Repeat last packet (simple concealment)
                    output.extend(self.last_good_packet)
                else:
                    # Silence
                    output.extend(np.zeros(num_samples - len(output)))
            
            self.next_seq += 1
        
        return np.array(output[:num_samples])
```

---

## Implementation Examples

### Moshi's Buffering Strategy

```python
class MoshiAudioBuffer:
    """
    Moshi's buffering for full-duplex dialogue.
    
    Handles both input (user) and output (system) streams.
    """
    def __init__(self, frame_duration_ms=80, sample_rate=24000):
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Input buffer (user audio)
        self.input_buffer = RingBuffer(self.frame_size * 10)
        
        # Output buffer (system audio)
        self.output_buffer = RingBuffer(self.frame_size * 10)
        
        # Timing
        self.last_process_time = time.time()
    
    def add_input(self, samples: np.ndarray):
        """Add user audio samples."""
        self.input_buffer.write(samples)
    
    def get_input_frame(self) -> Optional[np.ndarray]:
        """Get one frame of user audio if available."""
        if self.input_buffer.available_read() >= self.frame_size:
            return self.input_buffer.read(self.frame_size)
        return None
    
    def add_output(self, samples: np.ndarray):
        """Add system audio samples."""
        self.output_buffer.write(samples)
    
    def get_output(self, num_samples: int) -> np.ndarray:
        """Get system audio for playback."""
        return self.output_buffer.read(num_samples)
    
    def process_frame(self, model):
        """
        Process one frame if ready.
        
        Returns True if processed, False if waiting.
        """
        # Check if input frame ready
        input_frame = self.get_input_frame()
        if input_frame is None:
            return False
        
        # Process with model
        output_frame = model.process(input_frame)
        
        # Add to output buffer
        self.add_output(output_frame)
        
        # Track timing
        now = time.time()
        elapsed_ms = (now - self.last_process_time) * 1000
        self.last_process_time = now
        
        # Warn if too slow
        if elapsed_ms > self.frame_duration_ms:
            print(f"⚠️  Processing took {elapsed_ms:.1f}ms (budget: {self.frame_duration_ms}ms)")
        
        return True
```

### WebRTC-Style Buffering

```python
class WebRTCAudioBuffer:
    """
    WebRTC-style adaptive jitter buffer.
    
    Used in real-world VoIP applications.
    """
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        
        # Dynamic buffer size
        self.target_delay_ms = 20  # Initial target
        self.min_delay_ms = 10
        self.max_delay_ms = 200
        
        # Packet buffer
        self.packets = {}
        self.next_seq = 0
        
        # Statistics
        self.arrival_times = []
        self.jitter_estimate = 0
    
    def add_packet(self, seq: int, samples: np.ndarray, arrival_time: float):
        """Add packet with timing info."""
        self.packets[seq] = samples
        self.arrival_times.append((seq, arrival_time))
        
        # Update jitter estimate
        if len(self.arrival_times) > 10:
            self._update_jitter_estimate()
            self._adjust_target_delay()
    
    def _update_jitter_estimate(self):
        """Estimate network jitter from arrival times."""
        if len(self.arrival_times) < 2:
            return
        
        # Compute inter-arrival time variance
        intervals = []
        for i in range(1, len(self.arrival_times)):
            seq_diff = self.arrival_times[i][0] - self.arrival_times[i-1][0]
            time_diff = self.arrival_times[i][1] - self.arrival_times[i-1][1]
            
            if seq_diff == 1:  # Consecutive packets
                intervals.append(time_diff)
        
        if intervals:
            self.jitter_estimate = np.std(intervals) * 1000  # Convert to ms
    
    def _adjust_target_delay(self):
        """Adjust target delay based on jitter."""
        # Target: mean + 3× jitter
        new_target = 20 + 3 * self.jitter_estimate
        
        # Clamp to limits
        new_target = max(self.min_delay_ms, min(new_target, self.max_delay_ms))
        
        # Smooth adjustment
        alpha = 0.1
        self.target_delay_ms = (1 - alpha) * self.target_delay_ms + alpha * new_target
```

---

## Production Considerations

### Underrun Handling

```python
def handle_underrun(buffer, requested_samples):
    """
    Handle buffer underrun gracefully.
    
    Strategies:
    1. Return silence (simple)
    2. Repeat last samples (better)
    3. Time-stretch last samples (best)
    """
    available = buffer.available_read()
    
    if available >= requested_samples:
        # Normal case
        return buffer.read(requested_samples)
    
    # Underrun!
    samples = buffer.read(available)
    shortage = requested_samples - available
    
    # Strategy 1: Silence
    # return np.concatenate([samples, np.zeros(shortage)])
    
    # Strategy 2: Repeat last
    if len(samples) > 0:
        repeat = np.tile(samples[-min(100, len(samples)):], 
                        (shortage // min(100, len(samples)) + 1))
        return np.concatenate([samples, repeat[:shortage]])
    
    # Strategy 3: Time-stretch (requires librosa or similar)
    # stretched = time_stretch(samples, ratio=requested_samples/available)
    # return stretched
```

### Monitoring and Metrics

```python
class BufferMonitor:
    """
    Monitor buffer health for production systems.
    """
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.underruns = 0
        self.overruns = 0
        self.fill_levels = []
    
    def record_fill_level(self, current_fill: int):
        """Record current buffer fill level."""
        self.fill_levels.append(current_fill)
        
        # Keep last 1000 measurements
        if len(self.fill_levels) > 1000:
            self.fill_levels = self.fill_levels[-1000:]
    
    def get_metrics(self):
        """Get buffer health metrics."""
        if not self.fill_levels:
            return None
        
        fill_array = np.array(self.fill_levels)
        fill_pct = fill_array / self.buffer_size * 100
        
        return {
            'underruns': self.underruns,
            'overruns': self.overruns,
            'mean_fill_pct': fill_pct.mean(),
            'min_fill_pct': fill_pct.min(),
            'max_fill_pct': fill_pct.max(),
            'std_fill_pct': fill_pct.std(),
        }
    
    def print_report(self):
        """Print buffer health report."""
        metrics = self.get_metrics()
        if metrics is None:
            print("No data yet")
            return
        
        print("Buffer Health Report:")
        print(f"  Underruns: {metrics['underruns']}")
        print(f"  Overruns: {metrics['overruns']}")
        print(f"  Mean fill: {metrics['mean_fill_pct']:.1f}%")
        print(f"  Fill range: {metrics['min_fill_pct']:.1f}% - {metrics['max_fill_pct']:.1f}%")
        print(f"  Fill std: {metrics['std_fill_pct']:.1f}%")
        
        # Health assessment
        if metrics['underruns'] > 0:
            print("  ⚠️  Underruns detected - increase buffer size")
        elif metrics['mean_fill_pct'] > 80:
            print("  ⚠️  Buffer often full - may have latency issues")
        elif metrics['mean_fill_pct'] < 20:
            print("  ℹ️  Buffer often empty - can decrease size")
        else:
            print("  ✓ Buffer health good")
```

---

## Key Takeaways

```
1. BUFFERING IS ESSENTIAL
   - Absorbs timing variance
   - Prevents glitches
   - Enables smooth playback

2. SIZE IS CRITICAL
   - Too small: Underruns
   - Too large: Latency
   - Adaptive is best

3. RING BUFFERS ARE STANDARD
   - Efficient, constant-time
   - Lock-free possible
   - Production-proven

4. MONITOR BUFFER HEALTH
   - Track underruns/overruns
   - Adjust dynamically
   - Alert on issues

5. PRODUCTION REQUIRES ROBUSTNESS
   - Handle underruns gracefully
   - Adaptive adjustment
   - Comprehensive monitoring
```

---

## Further Reading

- `01_streaming_constraints.md` - Latency budgets
- `02_causal_architectures.md` - Causal models
- `04_latency_profiling.py` - Profiling tools
