# Moshi: Full-Duplex Speech LLM Architecture

**Paper**: [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) (Kyutai, 2024)

Moshi is the first real-time, full-duplex speech-text foundation model. It understands and generates speech natively while maintaining an "inner monologue" in text.

## Table of Contents
1. [Why Moshi?](#why-moshi)
2. [Multi-Stream Architecture](#multi-stream-architecture)
3. [Helium: The Base LLM](#helium-the-base-llm)
4. [Audio Tokenization with Mimi](#audio-tokenization-with-mimi)
5. [Inner Monologue](#inner-monologue)
6. [Full-Duplex Dialogue](#full-duplex-dialogue)
7. [Training Pipeline](#training-pipeline)
8. [Inference and Streaming](#inference-and-streaming)
9. [Profiling Considerations](#profiling-considerations)

---

## Why Moshi?

### The Pipeline Problem

```
Traditional voice assistants:

User speaks → ASR → Text → LLM → Text → TTS → Audio output
              ↓      ↓      ↓      ↓      ↓
            200ms  50ms  500ms  50ms  200ms  = ~1 second latency!

Problems:
├── High latency (user waits)
├── Information loss (prosody, emotion discarded)
├── No interruption handling
├── Unnatural turn-taking
└── Each component is a point of failure
```

### Moshi's Solution

```
End-to-end speech-to-speech:

User audio ──┐
             ├──▶ Single Model ──▶ System audio
System audio ┘

Benefits:
├── Low latency (~200ms theoretical)
├── Preserves audio information
├── Natural interruptions (full-duplex)
├── Single model, end-to-end training
└── Native speech understanding
```

---

## Multi-Stream Architecture

### Three Parallel Streams

```
┌─────────────────────────────────────────────────────────────────┐
│                    MOSHI ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Depth Transformer                     │    │
│  │   (Predicts all RVQ levels for each timestep)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ▲                                     │
│                            │                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Temporal Transformer                   │    │
│  │              (Helium - 7B parameter LLM)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│         ▲              ▲              ▲                          │
│         │              │              │                          │
│  ┌──────┴──────┐ ┌─────┴─────┐ ┌─────┴─────┐                    │
│  │   Stream 1  │ │  Stream 2 │ │  Stream 3 │                    │
│  │  User Audio │ │ Moshi     │ │   Text    │                    │
│  │   (Mimi)    │ │  Audio    │ │ (Inner    │                    │
│  │             │ │  (Mimi)   │ │ Monologue)│                    │
│  └─────────────┘ └───────────┘ └───────────┘                    │
│                                                                  │
│  @ 12.5 Hz       @ 12.5 Hz      @ ~3 Hz                         │
│  (8 tokens/step) (8 tokens/step) (1 token/step)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stream Details

```
Stream 1: User Audio
├── Input: User's speech via Mimi codec
├── Rate: 12.5 Hz (80ms per frame)
├── Tokens: 8 RVQ levels per frame
└── Role: What the user is saying

Stream 2: Moshi Audio  
├── Output: Moshi's speech via Mimi codec
├── Rate: 12.5 Hz (80ms per frame)
├── Tokens: 8 RVQ levels per frame
└── Role: What Moshi is saying

Stream 3: Text (Inner Monologue)
├── Both input and output
├── Rate: ~3 Hz (one token every ~4 audio frames)
├── Tokens: Standard text vocabulary
└── Role: Moshi's "thinking" - guides audio generation
```

---

## Helium: The Base LLM

### Architecture

```
Helium specifications:
├── Parameters: 7 billion
├── Architecture: Transformer decoder-only
├── Layers: 32
├── Hidden dim: 4096
├── Heads: 32
├── Context length: 4096 tokens
└── Trained on: Text data (books, web, etc.)

Pre-training:
├── Standard next-token prediction on text
├── 2 trillion tokens
├── Foundation for language understanding
```

### Adaptation for Multi-Stream

```python
# Standard LLM processes single sequence
class StandardLLM(nn.Module):
    def forward(self, tokens):
        # tokens: (batch, seq_len)
        return self.transformer(tokens)

# Moshi processes multiple interleaved streams
class MoshiLLM(nn.Module):
    def forward(self, user_audio, moshi_audio, text):
        # Interleave streams at each timestep
        # user_audio: (batch, time, 8)  - 8 RVQ levels
        # moshi_audio: (batch, time, 8)
        # text: (batch, time // 4)  - slower rate
        
        combined = self.interleave_streams(
            user_audio, moshi_audio, text
        )
        return self.transformer(combined)
```

---

## Audio Tokenization with Mimi

### Mimi Integration

```
Moshi uses Mimi codec for audio tokenization:

Input audio → Mimi Encoder → 8 tokens per 80ms frame
                                ↓
                           Transformer
                                ↓
8 tokens per frame → Mimi Decoder → Output audio

Key design choice:
├── Semantic token (level 0) helps LLM understand content
├── Acoustic tokens (levels 1-7) preserve audio quality
└── 12.5 Hz rate keeps context manageable
```

### Token Interleaving

```
At each timestep (80ms), process:

Time t:
├── User audio: [u₀, u₁, u₂, u₃, u₄, u₅, u₆, u₇]  (8 tokens)
├── Moshi audio: [m₀, m₁, m₂, m₃, m₄, m₅, m₆, m₇]  (8 tokens)
└── Text: [txt] (if this is a text timestep)

Token sequence fed to transformer:
[u₀][u₁]...[u₇][m₀][m₁]...[m₇][txt]  (17 tokens per 80ms)

For 10 seconds of audio:
├── 125 timesteps × 17 tokens = 2125 tokens
├── Fits in 4096 context window
└── Leaves room for longer conversations
```

---

## Inner Monologue

### Why Text Stream?

```
Problem: Pure audio-to-audio models struggle with reasoning

Audio-only model:
├── Must encode all reasoning in audio tokens
├── Hard to learn: audio is complex
├── Limited to short-term dependencies
└── Poor factual recall

With text stream:
├── Text captures semantic content explicitly
├── Easier for transformer to process
├── Better factual knowledge (from text pretraining)
└── Text guides audio generation
```

### How It Works

```
Example dialogue:

User: "What's the capital of France?"

Moshi internal state:
├── User audio: [mimi tokens for "What's the capital..."]
├── Text (inner monologue): "The capital of France is Paris."
├── Moshi audio: [mimi tokens for "The capital of France is Paris"]

The text stream:
1. Is predicted BEFORE the audio
2. Guides what Moshi will say
3. Can be used for reasoning steps
4. Is not spoken (inner monologue)
```

### Text-Audio Alignment

```
Text tokens are slower than audio tokens:

Audio:  [A][A][A][A][A][A][A][A][A][A][A][A]  (12.5 Hz)
Text:   [T]      [T]      [T]      [T]        (~3 Hz)
         ↑        ↑        ↑        ↑
        One text token every ~4 audio frames

This matches natural speech rate:
├── ~3 words per second
├── Each text token ≈ one word
└── Audio provides prosody and timing
```

---

## Full-Duplex Dialogue

### Simultaneous Input/Output

```
Traditional (half-duplex):
User speaks → System listens → System speaks → User listens
              (can't interrupt)

Moshi (full-duplex):
User speaks ←→ System speaks (simultaneously!)
              ↑
        Can interrupt, overlap, back-channel

This is like human conversation:
├── "Uh-huh" while listening
├── Interrupting when disagreeing
├── Finishing each other's sentences
```

### Handling Interruptions

```python
def process_timestep(user_audio, moshi_state):
    """
    At each 80ms timestep, process both streams.
    """
    # Encode user audio
    user_tokens = mimi.encode(user_audio)
    
    # Get Moshi's planned output
    moshi_tokens, text_tokens = model.generate_step(
        user_tokens, moshi_state
    )
    
    # If user is speaking loudly → Moshi should quiet down
    # This is learned behavior, not hard-coded
    if user_is_speaking(user_audio):
        # Model naturally learns to:
        # - Reduce volume
        # - Stop mid-sentence
        # - Wait for user to finish
        pass
    
    return mimi.decode(moshi_tokens)
```

---

## Training Pipeline

### Stage 1: Text Pre-training (Helium)

```
Standard LLM training:
├── Dataset: 2T tokens (web, books, etc.)
├── Objective: Next token prediction
├── Duration: Several weeks on cluster
└── Result: Strong language understanding
```

### Stage 2: Audio Pre-training

```
Extend to audio understanding:

Dataset:
├── 7M hours of audio (speech + music + sounds)
├── Transcripts where available
└── Synthetic TTS data for coverage

Objective:
├── Predict next audio tokens (all streams)
├── Predict next text tokens
├── Joint audio-text modeling

Key insight: Audio-text alignment from ASR data
```

### Stage 3: Dialogue Fine-tuning

```
Create conversational ability:

Dataset:
├── Synthetic dialogues (text → TTS)
├── Real conversation transcripts
├── Multi-turn interactions

Objective:
├── Response quality
├── Natural turn-taking
├── Appropriate latency
└── Voice consistency
```

### Loss Function

```python
def moshi_loss(user_audio, moshi_audio, text, model):
    """
    Multi-stream loss with different weights.
    """
    # Predict next tokens for each stream
    user_pred = model.predict_user_audio(...)
    moshi_pred = model.predict_moshi_audio(...)
    text_pred = model.predict_text(...)
    
    # Cross-entropy for each
    user_loss = F.cross_entropy(user_pred, user_audio_target)
    moshi_loss = F.cross_entropy(moshi_pred, moshi_audio_target)
    text_loss = F.cross_entropy(text_pred, text_target)
    
    # Semantic tokens weighted higher (100x in paper!)
    semantic_weight = 100.0
    
    return (
        user_loss + 
        moshi_loss * semantic_weight +  # For level 0
        text_loss
    )
```

---

## Inference and Streaming

### Real-Time Generation

```
Latency breakdown:

Audio capture:     ~10ms (microphone buffer)
Mimi encode:       ~5ms
Transformer step:  ~20ms (one forward pass)
Mimi decode:       ~5ms
Audio output:      ~10ms (speaker buffer)
─────────────────────────
Total:             ~50ms

Plus one 80ms frame buffer for streaming:
Theoretical minimum: ~130ms

Practical (with safety margin): ~200ms
```

### Streaming Architecture

```python
class MoshiStreaming:
    def __init__(self, model, mimi):
        self.model = model
        self.mimi = mimi
        self.kv_cache = None
        self.audio_buffer = []
    
    def process_chunk(self, user_audio_chunk):
        """
        Process 80ms of user audio, generate 80ms of response.
        """
        # Encode user audio
        user_tokens = self.mimi.encode(user_audio_chunk)
        
        # Generate with KV cache (no recomputation)
        moshi_tokens, text_token = self.model.generate_step(
            user_tokens, 
            kv_cache=self.kv_cache
        )
        
        # Update cache
        self.kv_cache = self.model.get_kv_cache()
        
        # Decode to audio
        moshi_audio = self.mimi.decode(moshi_tokens)
        
        return moshi_audio
```

### Depth Transformer

```
Problem: 8 RVQ levels per timestep = 8 tokens to generate

Naive: Generate sequentially (8x slower)
Moshi: Use DEPTH transformer

Temporal transformer: Predicts semantic token + context
Depth transformer: Predicts all 8 RVQ levels in parallel(ish)

┌───────────────────────────────────────┐
│         Temporal Transformer          │
│  (processes sequence, outputs z)      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│          Depth Transformer            │
│  Input: z                             │
│  Output: [level_0, level_1, ..., level_7]  │
│  (small model, 8 positions)           │
└───────────────────────────────────────┘

Depth transformer is small: ~300M parameters
Runs in parallel for all levels
Much faster than sequential generation
```

---

## Profiling Considerations

### Memory Requirements

```
Moshi model:
├── Helium (temporal): 7B params → ~14 GB (bf16)
├── Depth transformer: 300M params → ~600 MB
├── Mimi codec: 300M params → ~600 MB
├── KV cache (4096 context): ~4 GB
└── Total: ~20 GB GPU memory

For inference:
├── Single A100 (40GB): ✓ Comfortable
├── Single A10 (24GB): ✓ Tight but works
├── Single RTX 3090 (24GB): ✓ With optimizations
└── Single RTX 4090 (24GB): ✓ Best consumer option
```

### Throughput Analysis

```
Per-step compute:

Temporal transformer (7B):
├── Attention: O(seq_len × dim²)
├── FFN: O(dim × 4dim)
└── With KV cache: O(dim²) per token

Depth transformer (300M):
├── 8 positions only
├── Very fast

Mimi:
├── Encode: ~5ms
├── Decode: ~5ms

Total per 80ms step: ~30-50ms
Real-time factor: 0.5-0.7 (faster than real-time)
```

### Optimization Strategies

```python
# Key optimizations for Moshi inference:

1. KV Cache
   # Don't recompute attention for past tokens
   # Store key/value tensors, append new ones
   
2. Batched Audio Processing
   # Process audio encode/decode on GPU
   # Overlap with transformer compute
   
3. Speculative Decoding (optional)
   # Small model proposes, large model verifies
   # Can reduce latency further

4. Quantization
   # INT8 for weights: 2x memory reduction
   # Minimal quality loss for dialogue
   
5. Continuous Batching
   # For serving multiple users
   # Share compute across conversations
```

---

## Key Takeaways

```
1. MULTI-STREAM architecture handles audio + text
   - User audio, Moshi audio, text (inner monologue)
   - Interleaved at each timestep

2. INNER MONOLOGUE improves reasoning
   - Text guides audio generation
   - Leverages text pretraining

3. FULL-DUPLEX enables natural conversation
   - Simultaneous speaking/listening
   - Interruptions handled naturally

4. MIMI codec provides semantic tokens
   - 12.5 Hz rate keeps context short
   - Level 0 encodes meaning

5. REAL-TIME capable
   - ~200ms latency achievable
   - Runs on single GPU
```

---

## Next Steps

- `03_step_audio_analysis.md` - Step Audio 2 architecture comparison
- `04_continuous_audio_models.md` - Kyutai's CALM paper
- `../07-real-time-streaming/` - Streaming implementation details
