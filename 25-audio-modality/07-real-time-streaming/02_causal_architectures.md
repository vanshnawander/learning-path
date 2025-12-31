# Causal Architectures for Real-Time Audio

Deep dive into causal neural network architectures required for streaming and real-time audio processing. Essential for low-latency systems.

## Table of Contents
1. [Causality Requirement](#causality-requirement)
2. [Causal Convolutions](#causal-convolutions)
3. [Causal Attention](#causal-attention)
4. [Causal vs Non-Causal Trade-offs](#causal-vs-non-causal-trade-offs)
5. [Streaming State Management](#streaming-state-management)
6. [Latency Analysis](#latency-analysis)
7. [Implementation Patterns](#implementation-patterns)
8. [Production Examples](#production-examples)

---

## Causality Requirement

### Definition

```
Causal system: Output at time t depends ONLY on inputs at times ≤ t

y[t] = f(x[t], x[t-1], x[t-2], ..., x[0])

Non-causal: Output can depend on future inputs

y[t] = f(..., x[t-1], x[t], x[t+1], x[t+2], ...)
```

### Why Causality Matters

```
Real-time streaming requires causality:

Non-causal system:
├── Must wait for future frames
├── Introduces latency
├── Cannot process live audio
└── Example: Whisper (30s chunks)

Causal system:
├── Process frame-by-frame
├── Minimal latency
├── Suitable for live audio
└── Example: Mimi encoder/decoder
```

### Latency Impact

```
Non-causal with 5-frame lookahead @ 100 Hz:
├── Must wait: 5 frames = 50ms
├── Plus processing time
├── Total: 50ms + compute
└── Unacceptable for dialogue

Causal with no lookahead:
├── Wait: 0ms (process immediately)
├── Plus processing time
├── Total: 0ms + compute
└── Suitable for real-time
```

---

## Causal Convolutions

### Standard vs Causal Convolution

```
Standard Conv1d (kernel_size=3):
Input:  [..., x[t-1], x[t], x[t+1], ...]
                  ↓
Output:           y[t]

Uses: x[t-1], x[t], x[t+1]  ← FUTURE!

Causal Conv1d (kernel_size=3):
Input:  [..., x[t-2], x[t-1], x[t], ...]
                           ↓
Output:                    y[t]

Uses: x[t-2], x[t-1], x[t]  ← PAST ONLY
```

### Implementation

```python
class CausalConv1d(nn.Module):
    """
    1D causal convolution.
    
    Key: Pad on LEFT only, not symmetrically.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1):
        super().__init__()
        
        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, padding=0
        )
    
    def forward(self, x):
        # Pad on LEFT (past) only
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# Verify causality
def test_causality():
    conv = CausalConv1d(1, 1, kernel_size=3)
    
    # Set weights to simple average
    with torch.no_grad():
        conv.conv.weight.fill_(1/3)
        conv.conv.bias.fill_(0)
    
    # Input: impulse at position 5
    x = torch.zeros(1, 1, 10)
    x[0, 0, 5] = 1.0
    
    y = conv(x)
    
    print("Input: ", x[0, 0].tolist())
    print("Output:", [f"{v:.2f}" for v in y[0, 0].tolist()])
    
    # Verify: output before position 5 should be 0
    assert torch.allclose(y[0, 0, :5], torch.zeros(5)), "Causality violated!"
    print("✓ Causality verified")

test_causality()
```

### Causal Transposed Convolution

```python
class CausalConvTranspose1d(nn.Module):
    """
    Causal transposed convolution for upsampling.
    
    Trickier than regular causal conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0
        )
        
        # Amount to trim from output
        self.trim = kernel_size - stride
    
    def forward(self, x):
        x = self.conv(x)
        
        # Trim from RIGHT (future)
        if self.trim > 0:
            x = x[..., :-self.trim]
        
        return x
```

---

## Causal Attention

### Causal Self-Attention

```python
class CausalSelfAttention(nn.Module):
    """
    Self-attention with causal masking.
    
    Each position can only attend to itself and previous positions.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(1024, 1024), diagonal=1).bool()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        
        # Attention scores
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        
        # Softmax and apply to values
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        
        return self.out_proj(out)
```

### Streaming Attention with KV Cache

```python
class StreamingAttention(nn.Module):
    """
    Causal attention optimized for streaming.
    
    Caches key/value tensors to avoid recomputation.
    """
    def __init__(self, d_model, num_heads, max_cache_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # KV cache
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, use_cache=True):
        """
        Args:
            x: (batch, 1, d_model) - single new token
        """
        B, T, D = x.shape
        
        # Compute Q, K, V for new token
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if use_cache and self.k_cache is not None:
            # Concatenate with cache
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)
            
            # Trim if exceeds max length
            if k.shape[2] > self.max_cache_len:
                k = k[:, :, -self.max_cache_len:]
                v = v[:, :, -self.max_cache_len:]
        
        # Update cache
        if use_cache:
            self.k_cache = k
            self.v_cache = v
        
        # Attention (only for new token)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        
        return self.out_proj(out)
    
    def reset_cache(self):
        """Clear cache for new sequence."""
        self.k_cache = None
        self.v_cache = None
```

---

## Causal vs Non-Causal Trade-offs

### Quality Comparison

```
Empirical results on speech tasks:

Task                  | Causal WER | Non-causal WER | Gap
----------------------|------------|----------------|-----
Clean speech (LibriSpeech) | 2.5%  | 2.1%          | 0.4%
Noisy speech (CHiME)  | 8.2%       | 7.1%          | 1.1%
Conversational (SWBD) | 12.5%      | 11.2%         | 1.3%

Observation: Gap increases with task difficulty
Non-causal has more context, helps in hard cases
```

### Latency vs Quality

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Quality                                                     │
│    ▲                                                         │
│    │                                                         │
│    │         ●  Non-causal (bidirectional)                  │
│    │                                                         │
│    │                                                         │
│    │              ●  Causal with lookahead                  │
│    │                                                         │
│    │                       ●  Pure causal                   │
│    │                                                         │
│    └────────────────────────────────────────────────▶       │
│                                            Latency           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Trade-off is fundamental:
- More context = better quality
- More context = higher latency
- Choose based on application
```

---

## Streaming State Management

### Stateful Processing

```python
class StreamingProcessor:
    """
    Manage state for streaming audio processing.
    
    Handles:
    - Input buffering
    - Model state (LSTM, attention cache)
    - Output buffering
    """
    def __init__(self, model, chunk_size=320):
        self.model = model
        self.chunk_size = chunk_size
        
        # Input buffer
        self.input_buffer = []
        
        # Model state
        self.lstm_state = None
        self.attn_cache = None
        
        # Output buffer
        self.output_buffer = []
    
    def process_chunk(self, audio_chunk):
        """
        Process one chunk of audio.
        
        Args:
            audio_chunk: New audio samples
        Returns:
            output: Processed audio (may be None if buffering)
        """
        # Add to buffer
        self.input_buffer.extend(audio_chunk)
        
        # Process if buffer is full
        if len(self.input_buffer) >= self.chunk_size:
            # Extract chunk
            chunk = torch.tensor(self.input_buffer[:self.chunk_size])
            self.input_buffer = self.input_buffer[self.chunk_size:]
            
            # Process with model
            with torch.no_grad():
                output, self.lstm_state, self.attn_cache = self.model(
                    chunk,
                    lstm_state=self.lstm_state,
                    attn_cache=self.attn_cache
                )
            
            return output
        
        return None
    
    def reset(self):
        """Reset state for new stream."""
        self.input_buffer = []
        self.lstm_state = None
        self.attn_cache = None
        self.output_buffer = []
```

### LSTM State Management

```python
class StreamingLSTM(nn.Module):
    """
    LSTM with explicit state management for streaming.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, state=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            state: (h, c) tuple or None
        Returns:
            output: (batch, seq_len, hidden_size)
            new_state: (h, c) tuple
        """
        if state is None:
            # Initialize state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            state = (h0, c0)
        
        output, new_state = self.lstm(x, state)
        
        return output, new_state
```

---

## Latency Analysis

### Algorithmic Latency

```
Inherent to architecture, cannot be reduced:

Codec stride:
├── EnCodec: 320 samples @ 24kHz = 13.3ms
├── Mimi: 1920 samples @ 24kHz = 80ms
└── Trade-off: compression vs latency

Lookahead:
├── 0 frames: Pure causal (0ms)
├── 1 frame: 10-20ms lookahead
├── 5 frames: 50-100ms lookahead
└── Each frame adds latency
```

### Compute Latency

```python
def profile_streaming_latency(model, chunk_size=320, num_chunks=100):
    """
    Profile per-chunk processing latency.
    """
    device = next(model.parameters()).device
    
    # Generate test chunks
    chunks = [torch.randn(1, 1, chunk_size, device=device) for _ in range(num_chunks)]
    
    model.eval()
    latencies = []
    
    # Reset state
    state = None
    
    for chunk in chunks:
        start = time.perf_counter()
        
        with torch.no_grad():
            output, state = model(chunk, state=state)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    latencies = np.array(latencies)
    
    print(f"Streaming Latency Analysis:")
    print(f"  Mean: {latencies.mean():.2f}ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f}ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f}ms")
    print(f"  Max:  {latencies.max():.2f}ms")
    
    # Check if real-time capable
    chunk_duration_ms = chunk_size / 24000 * 1000  # Assuming 24kHz
    rtf = latencies.mean() / chunk_duration_ms
    
    print(f"\n  Chunk duration: {chunk_duration_ms:.2f}ms")
    print(f"  RTF: {rtf:.4f}")
    print(f"  Real-time: {'✓ Yes' if rtf < 1.0 else '✗ No'}")
```

---

## Implementation Patterns

### Pattern 1: Frame-by-Frame Processing

```python
class FrameByFrameModel(nn.Module):
    """
    Process one frame at a time with state.
    
    Lowest latency, suitable for real-time.
    """
    def __init__(self, frame_size=320):
        super().__init__()
        self.frame_size = frame_size
        
        # Causal encoder
        self.encoder = nn.Sequential(
            CausalConv1d(1, 32, 7),
            CausalConv1d(32, 64, 7),
            CausalConv1d(64, 128, 7),
        )
        
        # Stateful LSTM
        self.lstm = StreamingLSTM(128, 256, num_layers=2)
        
        # Decoder
        self.decoder = nn.Sequential(
            CausalConvTranspose1d(256, 128, 7, 1),
            CausalConvTranspose1d(128, 64, 7, 1),
            CausalConvTranspose1d(64, 1, 7, 1),
        )
    
    def forward(self, frame, lstm_state=None):
        """
        Args:
            frame: (batch, 1, frame_size)
            lstm_state: Previous LSTM state
        """
        # Encode
        x = self.encoder(frame)
        
        # LSTM
        x = x.transpose(1, 2)
        x, new_lstm_state = self.lstm(x, lstm_state)
        x = x.transpose(1, 2)
        
        # Decode
        output = self.decoder(x)
        
        return output, new_lstm_state
```

### Pattern 2: Overlapped Processing

```python
class OverlappedStreamingModel(nn.Module):
    """
    Process with overlap for smoother transitions.
    
    Slightly higher latency but better quality.
    """
    def __init__(self, chunk_size=1024, overlap=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hop_size = chunk_size - overlap
        
        # Model (can be non-causal within chunk)
        self.model = SomeNonCausalModel()
    
    def process_stream(self, audio_stream):
        """
        Process stream with overlap-add.
        """
        buffer = []
        output = []
        
        for chunk in audio_stream:
            buffer.extend(chunk)
            
            # Process when buffer is full
            while len(buffer) >= self.chunk_size:
                # Extract chunk
                chunk_audio = torch.tensor(buffer[:self.chunk_size])
                
                # Process
                chunk_output = self.model(chunk_audio)
                
                # Overlap-add
                if len(output) > 0:
                    # Crossfade overlap region
                    fade_out = torch.linspace(1, 0, self.overlap)
                    fade_in = torch.linspace(0, 1, self.overlap)
                    
                    output[-self.overlap:] = (
                        output[-self.overlap:] * fade_out +
                        chunk_output[:self.overlap] * fade_in
                    )
                    output.extend(chunk_output[self.overlap:])
                else:
                    output.extend(chunk_output)
                
                # Advance buffer
                buffer = buffer[self.hop_size:]
        
        return torch.tensor(output)
```

---

## Production Examples

### Moshi's Streaming Architecture

```python
class MoshiStreamingInference:
    """
    Moshi's real-time streaming inference.
    
    Processes 80ms frames with full-duplex capability.
    """
    def __init__(self, model, mimi_codec):
        self.model = model  # Helium 7B + depth transformer
        self.mimi = mimi_codec
        
        # State
        self.kv_cache = None
        self.frame_buffer = []
        
        # Timing
        self.frame_duration_ms = 80
    
    def process_frame(self, user_audio_frame):
        """
        Process one 80ms frame.
        
        Must complete within 80ms for real-time!
        """
        start = time.perf_counter()
        
        # 1. Encode user audio (5ms)
        user_tokens = self.mimi.encode(user_audio_frame)
        
        # 2. Run transformer with KV cache (30ms)
        moshi_tokens, text_token = self.model.generate_step(
            user_tokens,
            kv_cache=self.kv_cache
        )
        
        # Update cache
        self.kv_cache = self.model.get_kv_cache()
        
        # 3. Decode to audio (5ms)
        moshi_audio = self.mimi.decode(moshi_tokens)
        
        # Check timing
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > self.frame_duration_ms:
            print(f"⚠️  Frame processing took {elapsed_ms:.1f}ms (budget: {self.frame_duration_ms}ms)")
        
        return moshi_audio
```

### EnCodec Streaming

```python
class StreamingEnCodec:
    """
    Streaming version of EnCodec.
    
    Processes audio in chunks with minimal latency.
    """
    def __init__(self, encodec_model, chunk_size=320):
        self.model = encodec_model
        self.chunk_size = chunk_size  # 13.3ms @ 24kHz
        
        # Encoder state (for LSTM)
        self.encoder_state = None
        
        # Decoder state
        self.decoder_state = None
    
    def encode_chunk(self, audio_chunk):
        """
        Encode one chunk to tokens.
        """
        with torch.no_grad():
            # Encode with state
            tokens, self.encoder_state = self.model.encoder(
                audio_chunk,
                state=self.encoder_state
            )
            
            # Quantize
            codes = self.model.quantizer.encode(tokens)
        
        return codes
    
    def decode_chunk(self, codes):
        """
        Decode tokens to audio chunk.
        """
        with torch.no_grad():
            # Dequantize
            tokens = self.model.quantizer.decode(codes)
            
            # Decode with state
            audio, self.decoder_state = self.model.decoder(
                tokens,
                state=self.decoder_state
            )
        
        return audio
```

---

## Key Takeaways

```
1. CAUSALITY IS REQUIRED FOR STREAMING
   - No future lookahead
   - Process frame-by-frame
   - Minimal latency

2. CAUSAL CONVOLUTIONS
   - Pad on left only
   - Preserve temporal order
   - Essential building block

3. CAUSAL ATTENTION
   - Mask future positions
   - KV cache for efficiency
   - Enables transformer streaming

4. STATE MANAGEMENT
   - LSTM states
   - Attention caches
   - Critical for quality

5. LATENCY BUDGET
   - Every millisecond counts
   - Profile everything
   - P99 latency matters
```

---

## Further Reading

- `01_streaming_constraints.md` - Latency budgets
- `03_buffering_strategies.md` - Buffer management
- Moshi paper for production example
