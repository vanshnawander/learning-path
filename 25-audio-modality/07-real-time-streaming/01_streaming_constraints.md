# Real-Time Streaming Constraints

Understanding latency, buffering, and architectural constraints for real-time audio systems.

## Table of Contents
1. [Latency Budget](#latency-budget)
2. [Sources of Latency](#sources-of-latency)
3. [Causal vs Non-Causal](#causal-vs-non-causal)
4. [Buffering Strategies](#buffering-strategies)
5. [Streaming Architecture Patterns](#streaming-architecture-patterns)
6. [Profiling Streaming Systems](#profiling-streaming-systems)

---

## Latency Budget

### Human Perception Thresholds

```
Latency Type                Threshold    Effect
─────────────────────────────────────────────────────
Imperceptible              < 20ms       Perfect sync
Noticeable                 20-100ms     Slight delay
Annoying                   100-300ms    Conversational lag
Unusable                   > 300ms      Cannot have dialogue

For real-time dialogue (like Moshi):
├── Target: < 200ms end-to-end
├── Acceptable: < 500ms
└── Unusable: > 1000ms
```

### Latency Budget Breakdown

```
Typical voice assistant pipeline:

Component               Latency
─────────────────────────────────
Microphone buffer       10-20ms
Network (if cloud)      50-100ms
ASR processing          100-200ms
LLM inference           200-500ms
TTS synthesis           100-200ms
Speaker buffer          10-20ms
─────────────────────────────────
TOTAL                   470-1040ms  ← Too slow!

End-to-end system (Moshi):

Component               Latency
─────────────────────────────────
Microphone buffer       10ms
Mimi encode             5ms
Transformer step        30ms
Mimi decode             5ms
Speaker buffer          10ms
Algorithmic (Mimi)      80ms
─────────────────────────────────
TOTAL                   ~140ms  ← Achievable!
```

---

## Sources of Latency

### 1. Algorithmic Latency

```
Inherent to the model architecture, cannot be reduced.

Codec stride (samples per frame):
├── EnCodec: 320 samples = 13.3ms @ 24kHz
├── Mimi: 1920 samples = 80ms @ 24kHz

Lookahead (non-causal models):
├── Bidirectional attention: needs future frames
├── Non-causal convolutions: needs future samples
└── Solution: Use causal architectures
```

### 2. Compute Latency

```
Time to process one frame:

Factors:
├── Model size (parameters)
├── Hardware (GPU, CPU)
├── Batch size
├── Optimization (quantization, fusion)

Example Moshi on A100:
├── Temporal transformer: ~20ms per step
├── Depth transformer: ~5ms per step
├── Mimi encode/decode: ~10ms
└── Total: ~35ms compute (< 80ms frame time)
```

### 3. I/O Latency

```
Microphone input:
├── Buffer size (typically 10-20ms)
├── USB latency (1-10ms)
└── Driver overhead

Speaker output:
├── Buffer size (typically 10-20ms)
├── DAC latency (1-5ms)
└── Bluetooth adds 40-200ms!

Network (if applicable):
├── Same datacenter: 1-5ms
├── Same region: 10-50ms
├── Cross-region: 50-200ms
└── Mobile network: 50-300ms
```

### 4. Buffering Latency

```
Buffers add latency but prevent dropouts:

Too small buffer:
├── Frequent underruns (audio glitches)
├── High CPU usage
└── Choppy output

Too large buffer:
├── High latency
├── Smooth but delayed
└── Poor conversational feel

Optimal: Just enough to handle jitter
```

---

## Causal vs Non-Causal

### Causal Systems

```
Output depends only on past and current input.

y[t] = f(x[t], x[t-1], x[t-2], ...)

Properties:
├── Can process in real-time (streaming)
├── No lookahead needed
├── May have lower quality (less context)

Examples:
├── Mimi encoder/decoder
├── Moshi transformer
├── Streaming ASR
```

### Non-Causal Systems

```
Output depends on future input too.

y[t] = f(..., x[t-1], x[t], x[t+1], ...)

Properties:
├── Needs to wait for future frames
├── Higher quality (more context)
├── Not suitable for real-time

Examples:
├── Whisper (processes 30s chunks)
├── Bidirectional transformers
├── Many ASR systems
```

### Making Models Causal

```python
# Non-causal attention (sees everything)
attention_mask = None  # Full attention

# Causal attention (sees only past)
attention_mask = torch.triu(
    torch.ones(seq_len, seq_len) * float('-inf'),
    diagonal=1
)

# Causal convolution
# Standard: output[t] uses input[t-k:t+k]
# Causal: output[t] uses input[t-2k:t]
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Pad on LEFT only
        return self.conv(x)
```

---

## Buffering Strategies

### Double Buffering

```
While processing buffer A, fill buffer B:

Time →
┌─────────────────────────────────────────┐
│ Fill A    │ Process A │ Fill A    │ ... │
│           │ Fill B    │ Process B │     │
└─────────────────────────────────────────┘

Latency: 1 buffer duration
Pros: Simple, reliable
Cons: Fixed latency
```

### Ring Buffer

```
Continuous circular buffer:

Write pointer ─────────────────────┐
                                   ▼
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
              ▲
Read pointer ─┘

Latency: Variable (write - read distance)
Pros: Handles jitter, flexible
Cons: More complex
```

### Adaptive Buffering

```python
class AdaptiveBuffer:
    def __init__(self, target_latency_ms: float, sample_rate: int):
        self.target_samples = int(target_latency_ms * sample_rate / 1000)
        self.buffer = []
        self.underrun_count = 0
        self.overflow_count = 0
    
    def add_samples(self, samples):
        self.buffer.extend(samples)
        
        # Overflow: too much buffered
        if len(self.buffer) > self.target_samples * 2:
            excess = len(self.buffer) - self.target_samples
            self.buffer = self.buffer[excess:]
            self.overflow_count += 1
    
    def get_samples(self, n: int):
        if len(self.buffer) < n:
            # Underrun: not enough samples
            self.underrun_count += 1
            # Return zeros (silence)
            return [0] * n
        
        samples = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return samples
```

---

## Streaming Architecture Patterns

### Pattern 1: Chunk-Based Processing

```
Process fixed-size chunks:

Audio stream → [Chunk 1][Chunk 2][Chunk 3] → Process → Output

Pros:
├── Simple implementation
├── Predictable latency
└── Easy batching

Cons:
├── Latency = chunk size
├── Boundary artifacts
└── Waste compute on silence
```

### Pattern 2: Frame-Based Streaming

```
Process one frame at a time:

Audio stream → Frame → Process → Output → Frame → ...

Pros:
├── Minimum latency
├── True streaming
└── Natural for audio

Cons:
├── No batching benefit
├── Higher overhead
└── Complex state management
```

### Pattern 3: Overlapped Processing

```
Overlap chunks for smooth transitions:

Chunk 1: [========]
Chunk 2:     [========]
Chunk 3:         [========]
Output:  [==][====][====][==]
         fade  full  full fade

Used in: Vocoders, audio effects
Handles: Boundary artifacts
```

### Moshi's Streaming Architecture

```python
class MoshiStreaming:
    def __init__(self, model, mimi):
        self.model = model
        self.mimi = mimi
        self.kv_cache = None
        self.frame_duration_ms = 80  # Mimi frame
        
    def process_frame(self, user_audio_frame):
        """Process one 80ms frame, return response audio."""
        
        # 1. Encode user audio (5ms)
        user_tokens = self.mimi.encode(user_audio_frame)
        
        # 2. Run transformer with KV cache (30ms)
        # Only compute attention for new tokens
        moshi_tokens, text_token = self.model.generate_step(
            user_tokens,
            kv_cache=self.kv_cache
        )
        self.kv_cache = self.model.get_kv_cache()
        
        # 3. Decode to audio (5ms)
        moshi_audio = self.mimi.decode(moshi_tokens)
        
        return moshi_audio  # Return within 80ms budget
```

---

## Profiling Streaming Systems

### Key Metrics

```python
class StreamingProfiler:
    def __init__(self):
        self.frame_times = []
        self.underruns = 0
        self.overruns = 0
        
    def log_frame(self, process_time_ms: float, frame_duration_ms: float):
        self.frame_times.append(process_time_ms)
        
        if process_time_ms > frame_duration_ms:
            self.overruns += 1  # Missed deadline!
    
    def report(self):
        times = np.array(self.frame_times)
        print(f"Frame processing time:")
        print(f"  Mean: {times.mean():.2f} ms")
        print(f"  P50:  {np.percentile(times, 50):.2f} ms")
        print(f"  P95:  {np.percentile(times, 95):.2f} ms")
        print(f"  P99:  {np.percentile(times, 99):.2f} ms")
        print(f"  Max:  {times.max():.2f} ms")
        print(f"Deadline misses: {self.overruns} ({100*self.overruns/len(times):.1f}%)")
```

### Real-Time Factor (RTF)

```
RTF = Processing Time / Audio Duration

RTF < 1.0: Faster than real-time (good!)
RTF = 1.0: Exactly real-time (dangerous)
RTF > 1.0: Cannot keep up (unusable)

Target: RTF < 0.5 for safety margin

Example:
├── 80ms frame processed in 40ms: RTF = 0.5 ✓
├── 80ms frame processed in 90ms: RTF = 1.125 ✗
```

### Latency Breakdown Tool

```python
import time
from contextlib import contextmanager

class LatencyTracker:
    def __init__(self):
        self.timings = {}
    
    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
    
    def report(self):
        print("\nLatency Breakdown:")
        total = 0
        for name, times in self.timings.items():
            mean = np.mean(times)
            total += mean
            print(f"  {name}: {mean:.2f} ms")
        print(f"  TOTAL: {total:.2f} ms")

# Usage
tracker = LatencyTracker()

with tracker.track("mimi_encode"):
    tokens = mimi.encode(audio)
    
with tracker.track("transformer"):
    output = model(tokens)
    
with tracker.track("mimi_decode"):
    audio = mimi.decode(output)

tracker.report()
```

---

## Key Takeaways

```
1. LATENCY BUDGET is tight for dialogue
   - Target < 200ms total
   - Every millisecond counts

2. CAUSAL architectures are required
   - No future lookahead
   - Trade quality for latency

3. ALGORITHMIC LATENCY is fixed
   - Mimi: 80ms (high compression)
   - EnCodec: 13ms (lower compression)

4. BUFFERING is a balancing act
   - Too small: glitches
   - Too large: latency

5. PROFILE EVERYTHING
   - P99 latency matters more than mean
   - One slow frame causes audible glitch
```

---

## Next Steps

- `02_causal_vs_noncausal.md` - Deep dive into causal architectures
- `03_buffering_strategies.md` - Advanced buffering techniques
- `04_latency_profiling.py` - Profiling tools and benchmarks
