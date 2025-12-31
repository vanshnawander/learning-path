# SoundStorm: Parallel Audio Generation

**Paper**: [SoundStorm: Efficient Parallel Audio Generation](https://arxiv.org/abs/2305.09636) (Google, 2023)

SoundStorm revolutionized audio generation by achieving 100x speedup over autoregressive methods while maintaining quality. Essential reading for understanding modern audio LLMs.

## Table of Contents
1. [The Autoregressive Bottleneck](#the-autoregressive-bottleneck)
2. [MaskGIT for Audio](#maskgit-for-audio)
3. [Architecture](#architecture)
4. [Confidence-Based Parallel Decoding](#confidence-based-parallel-decoding)
5. [Training Procedure](#training-procedure)
6. [Dialogue Synthesis](#dialogue-synthesis)
7. [Performance Analysis](#performance-analysis)
8. [Implementation](#implementation)

---

## The Autoregressive Bottleneck

### AudioLM's Sequential Generation

```
AudioLM generates audio tokens sequentially:

For 10 seconds of audio @ 50 Hz with 12 RVQ levels:
├── Total tokens: 10 × 50 × 12 = 6,000 tokens
├── Sequential generation: 6,000 forward passes
├── Time on TPU-v4: ~50 seconds
└── Real-time factor: 0.2 (5x slower than real-time)

Problem: Cannot parallelize due to autoregressive dependency
Each token depends on all previous tokens
```

### Why Autoregressive is Slow

```python
# Autoregressive generation (AudioLM style)
def generate_autoregressive(semantic_tokens, model, num_acoustic_levels=12):
    """
    Generate acoustic tokens one at a time.
    """
    batch_size, seq_len = semantic_tokens.shape
    
    # Initialize with semantic tokens
    tokens = semantic_tokens.clone()
    
    # Generate each RVQ level sequentially
    for level in range(num_acoustic_levels):
        # For each position in sequence
        for t in range(seq_len):
            # Predict next token given all previous
            logits = model(tokens[:, :t+1])
            next_token = logits[:, -1, :].argmax(dim=-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
    
    return tokens
    # Total iterations: seq_len × num_levels
    # Cannot parallelize!
```

---

## MaskGIT for Audio

### Inspiration from Image Generation

```
MaskGIT (2022) for images:
├── Start with all tokens masked
├── Predict all tokens simultaneously
├── Keep high-confidence predictions
├── Re-mask low-confidence tokens
├── Repeat until all unmasked

Key insight: Bidirectional attention
Can see context from both directions
Unlike autoregressive (only left context)
```

### Adapting to Audio

```
Audio has special structure:
├── RVQ hierarchy: Level 0 → Level 1 → ... → Level N
├── Temporal dependency: Frame t depends on frame t-1
├── Semantic conditioning: Acoustic depends on semantic

SoundStorm adaptation:
├── Condition on semantic tokens (from AudioLM)
├── Generate RVQ levels in PARALLEL per frame
├── Use bidirectional attention within each level
├── Iterative refinement with confidence masking
```

---

## Architecture

### Model Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOUNDSTORM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Semantic tokens (from AudioLM w2v-BERT)                 │
│         Shape: (batch, time)                                    │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Semantic Token Embedding      │                            │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   Conformer Encoder             │                            │
│  │   - Bidirectional attention     │                            │
│  │   - Relative position encoding  │                            │
│  │   - 12 layers, 512 dim          │                            │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │   RVQ Level Embeddings          │                            │
│  │   Separate head per level       │                            │
│  └─────────────┬───────────────────┘                            │
│                │                                                 │
│                ▼                                                 │
│  Output: Logits for each RVQ level                              │
│          Shape: (batch, time, num_levels, codebook_size)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Choices

```python
class SoundStormModel(nn.Module):
    """
    SoundStorm architecture (simplified).
    """
    def __init__(
        self,
        semantic_vocab_size: int = 1024,
        acoustic_vocab_size: int = 1024,
        num_rvq_levels: int = 12,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
    ):
        super().__init__()
        
        # Semantic token embedding
        self.semantic_embed = nn.Embedding(semantic_vocab_size, d_model)
        
        # Conformer encoder (bidirectional)
        self.encoder = ConformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # Separate prediction head for each RVQ level
        self.rvq_heads = nn.ModuleList([
            nn.Linear(d_model, acoustic_vocab_size)
            for _ in range(num_rvq_levels)
        ])
        
        # Acoustic token embeddings (for conditioning)
        self.acoustic_embeds = nn.ModuleList([
            nn.Embedding(acoustic_vocab_size, d_model)
            for _ in range(num_rvq_levels)
        ])
    
    def forward(self, semantic_tokens, acoustic_tokens=None, mask=None):
        """
        Args:
            semantic_tokens: (batch, time)
            acoustic_tokens: (batch, time, num_levels) - partially filled
            mask: (batch, time, num_levels) - which tokens to predict
        """
        # Embed semantic tokens
        x = self.semantic_embed(semantic_tokens)
        
        # Add acoustic token embeddings if provided
        if acoustic_tokens is not None:
            for level in range(acoustic_tokens.shape[2]):
                # Only add where not masked
                acoustic_emb = self.acoustic_embeds[level](acoustic_tokens[:, :, level])
                x = x + acoustic_emb * (1 - mask[:, :, level].unsqueeze(-1))
        
        # Encode with bidirectional attention
        x = self.encoder(x)
        
        # Predict each RVQ level
        logits = []
        for level, head in enumerate(self.rvq_heads):
            logits.append(head(x))
        
        # Stack: (batch, time, num_levels, vocab_size)
        logits = torch.stack(logits, dim=2)
        
        return logits
```

---

## Confidence-Based Parallel Decoding

### The Iterative Refinement Process

```
Iteration 1: All tokens masked
├── Predict all tokens simultaneously
├── Compute confidence (softmax probability)
├── Keep top 50% most confident
└── Re-mask bottom 50%

Iteration 2: 50% masked
├── Predict masked tokens (conditioned on unmasked)
├── Compute confidence
├── Keep top 50% of remaining
└── Re-mask bottom 50%

...continue until all unmasked

Typical: 8-12 iterations to complete
Much faster than 6000 sequential steps!
```

### Confidence Calculation

```python
def confidence_based_decode(
    model,
    semantic_tokens,
    num_iterations: int = 12,
    num_rvq_levels: int = 12,
    codebook_size: int = 1024,
):
    """
    Parallel decoding with confidence-based masking.
    """
    batch_size, seq_len = semantic_tokens.shape
    device = semantic_tokens.device
    
    # Initialize: all acoustic tokens masked
    acoustic_tokens = torch.zeros(
        batch_size, seq_len, num_rvq_levels, dtype=torch.long, device=device
    )
    mask = torch.ones(
        batch_size, seq_len, num_rvq_levels, dtype=torch.bool, device=device
    )
    
    # Iterative refinement
    for iteration in range(num_iterations):
        # Predict all masked positions
        with torch.no_grad():
            logits = model(semantic_tokens, acoustic_tokens, mask)
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        # Confidence: max probability
        confidence = probs.max(dim=-1).values
        
        # Only consider masked positions
        confidence = confidence * mask.float()
        
        # Determine how many to unmask this iteration
        num_masked = mask.sum()
        num_to_unmask = int(num_masked * (1 - (iteration / num_iterations)))
        
        # Find top-k most confident predictions
        confidence_flat = confidence.view(-1)
        _, top_indices = confidence_flat.topk(num_to_unmask)
        
        # Unmask top-k
        mask_flat = mask.view(-1)
        mask_flat[top_indices] = False
        
        # Update tokens
        acoustic_tokens_flat = acoustic_tokens.view(-1)
        predictions_flat = predictions.view(-1)
        acoustic_tokens_flat[top_indices] = predictions_flat[top_indices]
        
        # Reshape back
        mask = mask_flat.view(batch_size, seq_len, num_rvq_levels)
        acoustic_tokens = acoustic_tokens_flat.view(batch_size, seq_len, num_rvq_levels)
    
    return acoustic_tokens
```

### Scheduling Strategies

```
Different unmasking schedules:

1. LINEAR
   Unmask same fraction each iteration
   Schedule: [0.1, 0.2, 0.3, ..., 1.0]

2. COSINE
   Unmask more at beginning and end
   Schedule: cos(π * t / T)
   
3. EXPONENTIAL
   Unmask slowly at first, quickly at end
   Schedule: exp(t / T) - 1

SoundStorm uses COSINE:
- Better quality
- More stable
- Matches natural coarse-to-fine generation
```

---

## Training Procedure

### Masked Token Prediction

```python
def train_soundstorm(model, dataloader, optimizer, num_epochs):
    """
    Train SoundStorm with masked token prediction.
    """
    for epoch in range(num_epochs):
        for batch in dataloader:
            semantic_tokens = batch['semantic']
            acoustic_tokens = batch['acoustic']  # Ground truth
            
            # Random masking
            mask_prob = torch.rand(acoustic_tokens.shape)
            mask = mask_prob < 0.5  # Mask 50% on average
            
            # Masked tokens
            masked_acoustic = acoustic_tokens.clone()
            masked_acoustic[mask] = 0  # Or special mask token
            
            # Forward pass
            logits = model(semantic_tokens, masked_acoustic, mask)
            
            # Loss only on masked positions
            loss = F.cross_entropy(
                logits[mask].view(-1, logits.shape[-1]),
                acoustic_tokens[mask].view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Data Preparation

```
Training data:
├── Audio files (speech, music, etc.)
├── Extract semantic tokens (w2v-BERT)
├── Extract acoustic tokens (SoundStream)
└── Pair: (semantic, acoustic)

Augmentation:
├── Random masking ratios (0.3-0.7)
├── Different mask patterns
└── Curriculum: Easy → Hard
```

---

## Dialogue Synthesis

### Multi-Speaker Capability

```
SoundStorm excels at dialogue:

Input: Transcript with speaker turns
       "Speaker A: Hello, how are you?"
       "Speaker B: I'm doing great, thanks!"

Process:
1. Convert transcript to semantic tokens
2. Add speaker embeddings
3. Generate acoustic tokens with SoundStorm
4. Decode to audio

Result: Natural multi-speaker dialogue
- Proper turn-taking
- Consistent voices
- Natural prosody
```

### Speaker Conditioning

```python
class SoundStormDialogue(nn.Module):
    """
    SoundStorm with speaker conditioning for dialogue.
    """
    def __init__(self, num_speakers: int = 10, **kwargs):
        super().__init__()
        self.base_model = SoundStormModel(**kwargs)
        
        # Speaker embeddings
        self.speaker_embed = nn.Embedding(num_speakers, kwargs['d_model'])
    
    def forward(self, semantic_tokens, speaker_ids, **kwargs):
        """
        Args:
            semantic_tokens: (batch, time)
            speaker_ids: (batch, time) - speaker at each frame
        """
        # Get speaker embeddings
        speaker_emb = self.speaker_embed(speaker_ids)
        
        # Add to semantic embeddings
        semantic_emb = self.base_model.semantic_embed(semantic_tokens)
        combined_emb = semantic_emb + speaker_emb
        
        # Continue with standard forward pass
        # ... (rest of model)
```

---

## Performance Analysis

### Speed Comparison

```
Task: Generate 30 seconds of audio

AudioLM (Autoregressive):
├── Method: Sequential token generation
├── Time: 15 minutes on TPU-v4
├── RTF: 0.033 (30x slower than real-time)

SoundStorm (Parallel):
├── Method: Iterative parallel decoding
├── Time: 0.5 seconds on TPU-v4
├── RTF: 60 (60x faster than real-time)

Speedup: 1800x faster than AudioLM!
```

### Quality Comparison

```
Subjective evaluation (MOS):

Model          | Naturalness | Consistency | Overall
---------------|-------------|-------------|--------
Ground Truth   | 4.5         | 4.5         | 4.5
AudioLM        | 4.1         | 4.0         | 4.0
SoundStorm     | 4.1         | 4.2         | 4.1

Key finding: SoundStorm matches or exceeds AudioLM quality
- Better voice consistency (bidirectional context)
- Same naturalness
- 100x faster
```

### Ablation Studies

```
Impact of design choices:

1. NUMBER OF ITERATIONS
   8 iterations:  MOS 3.9, 0.3s
   12 iterations: MOS 4.1, 0.5s  ← Sweet spot
   16 iterations: MOS 4.1, 0.7s  (diminishing returns)

2. MASKING SCHEDULE
   Linear:      MOS 3.8
   Exponential: MOS 3.9
   Cosine:      MOS 4.1  ← Best

3. BIDIRECTIONAL vs CAUSAL
   Causal only:      MOS 3.7
   Bidirectional:    MOS 4.1  ← Much better
```

---

## Implementation

### Complete Generation Pipeline

```python
class SoundStormPipeline:
    """
    Complete pipeline for SoundStorm generation.
    """
    def __init__(self, semantic_model, soundstorm_model, codec):
        self.semantic_model = semantic_model  # w2v-BERT
        self.soundstorm_model = soundstorm_model
        self.codec = codec  # SoundStream
    
    def generate(self, text, speaker_prompt_audio=None):
        """
        Generate audio from text.
        
        Args:
            text: Input text
            speaker_prompt_audio: Optional audio for voice cloning
        """
        # Step 1: Text → Semantic tokens
        # (In practice, use AudioLM's semantic model)
        semantic_tokens = self.text_to_semantic(text)
        
        # Step 2: Semantic → Acoustic (SoundStorm)
        acoustic_tokens = confidence_based_decode(
            self.soundstorm_model,
            semantic_tokens,
            num_iterations=12
        )
        
        # Step 3: Acoustic tokens → Audio
        audio = self.codec.decode(acoustic_tokens)
        
        return audio
    
    def generate_dialogue(self, transcript, speaker_prompts):
        """
        Generate multi-speaker dialogue.
        
        Args:
            transcript: List of (speaker_id, text) tuples
            speaker_prompts: Dict of speaker_id → prompt audio
        """
        all_audio = []
        
        for speaker_id, text in transcript:
            # Generate with speaker conditioning
            audio = self.generate(
                text,
                speaker_prompt_audio=speaker_prompts.get(speaker_id)
            )
            all_audio.append(audio)
        
        # Concatenate with small gaps
        return torch.cat(all_audio, dim=-1)
```

### Optimization Tips

```python
# 1. Batch processing
def generate_batch(texts, model, batch_size=8):
    """Process multiple texts in parallel."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results.extend(model.generate(batch))
    return results

# 2. KV caching for transformer
class CachedSoundStorm(nn.Module):
    def forward(self, x, cache=None):
        # Reuse key/value from previous iterations
        # Only compute for newly unmasked tokens
        pass

# 3. Mixed precision
model = model.half()  # FP16
# 2x speedup, minimal quality loss

# 4. Compile with torch.compile (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')
# Additional 1.5-2x speedup
```

---

## Key Takeaways

```
1. PARALLEL GENERATION IS POSSIBLE
   - MaskGIT-style iterative refinement
   - 100x faster than autoregressive
   - Same or better quality

2. BIDIRECTIONAL CONTEXT HELPS
   - Better consistency
   - More natural output
   - Especially for dialogue

3. CONFIDENCE-BASED MASKING
   - Coarse-to-fine generation
   - Stable training
   - Efficient inference

4. PRACTICAL IMPACT
   - Enables real-time applications
   - Better for interactive systems
   - Influenced Moshi and others

5. TRADE-OFFS
   - Requires pre-trained semantic model
   - More complex than autoregressive
   - But worth it for speed
```

---

## Further Reading

- AudioLM paper for semantic tokens
- MaskGIT paper for masking strategy
- Moshi architecture (uses similar ideas)
- `02_moshi_architecture.md` for production system
