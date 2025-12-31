# Audio Tokenization for LLMs: Complete Guide

Understanding how to tokenize audio for language model processing is fundamental to building audio LLMs. This guide covers all approaches from discrete codecs to continuous representations.

## Table of Contents
1. [Why Tokenize Audio?](#why-tokenize-audio)
2. [Discrete Tokenization Approaches](#discrete-tokenization-approaches)
3. [Semantic vs Acoustic Tokens](#semantic-vs-acoustic-tokens)
4. [Hierarchical Tokenization](#hierarchical-tokenization)
5. [Token Rate Considerations](#token-rate-considerations)
6. [Vocabulary Design](#vocabulary-design)
7. [Context Length Management](#context-length-management)
8. [Latest Research (2024-2025)](#latest-research-2024-2025)

---

## Why Tokenize Audio?

### The LLM Compatibility Problem

```
LLMs are designed for DISCRETE tokens:
├── Text: BPE/SentencePiece tokenization
├── Vocabulary: 32k-100k tokens
├── Each token: integer index
└── Softmax over vocabulary for prediction

Raw audio is CONTINUOUS:
├── Waveform: float values in [-1, 1]
├── 16,000 samples per second
├── Cannot directly apply cross-entropy loss
└── Incompatible with standard LLM architectures

Solution: DISCRETIZE audio into tokens
```

### Benefits of Audio Tokenization

```
1. LLM COMPATIBILITY
   - Standard transformer architectures work
   - Cross-entropy loss applicable
   - Existing LLM infrastructure reusable

2. COMPRESSION
   - Raw 16kHz audio: 256 kbps (16-bit)
   - Tokenized: 1.5-6 kbps (50-170x compression)
   - Enables longer context windows

3. UNIFIED MULTIMODAL MODELS
   - Text tokens + Audio tokens + Image tokens
   - Single vocabulary space
   - Joint training possible

4. GENERATION EFFICIENCY
   - Autoregressive token generation
   - Caching mechanisms work
   - Beam search applicable
```

---

## Discrete Tokenization Approaches

### 1. Neural Codec Tokenization

```
Most common approach: Use neural audio codec

Pipeline:
Audio waveform → Encoder → Quantizer → Token indices
Token indices → Decoder → Reconstructed audio

Popular codecs:
├── SoundStream (Google, 2021)
│   ├── 75 Hz frame rate
│   ├── 1024-size codebook
│   └── 3-12 kbps bitrate
│
├── EnCodec (Meta, 2022)
│   ├── 75 Hz frame rate
│   ├── 1024-size codebook
│   └── 1.5-24 kbps bitrate
│
└── Mimi (Kyutai, 2024)
    ├── 12.5 Hz frame rate
    ├── 2048-size codebook
    └── 1.1 kbps bitrate
```

### 2. Clustering-Based Tokenization

```
Alternative: Cluster audio features

Process:
1. Extract features (MFCC, mel spectrogram, learned)
2. Run k-means clustering
3. Assign cluster IDs as tokens

Used in:
├── HuBERT: k-means on MFCC or previous model features
├── WavLM: k-means on masked features
└── AudioLM: k-means on w2v-BERT features

Pros:
- No codec training needed
- Semantic focus

Cons:
- Cannot reconstruct audio
- Fixed vocabulary
```

### 3. VQ-VAE Tokenization

```
Learn discrete latent space end-to-end:

Architecture:
├── Encoder: Audio → Continuous latents
├── VQ: Continuous → Discrete (codebook lookup)
├── Decoder: Discrete → Audio
└── Training: Reconstruction + VQ loss

Used in:
├── Jukebox (OpenAI, 2020)
├── MusicLM (Google, 2023)
└── Various music generation models

Advantage: Optimized for reconstruction
```

---

## Semantic vs Acoustic Tokens

### The Two-Level Representation

```
Modern audio LLMs use HIERARCHICAL tokens:

┌─────────────────────────────────────────────────────────────┐
│                   SEMANTIC TOKENS                            │
├─────────────────────────────────────────────────────────────┤
│  What: Content, meaning, phonemes                           │
│  From: Self-supervised models (WavLM, w2v-BERT)             │
│  Rate: Low (12.5-50 Hz)                                     │
│  Use: LLM models this for language understanding            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ACOUSTIC TOKENS                            │
├─────────────────────────────────────────────────────────────┤
│  What: Speaker, prosody, acoustic details                   │
│  From: Neural codecs (SoundStream, EnCodec)                 │
│  Rate: High (75 Hz × 8 levels = 600 tokens/sec)            │
│  Use: Decode to actual audio waveform                       │
└─────────────────────────────────────────────────────────────┘
```

### Semantic Token Extraction

```python
# Extract semantic tokens using WavLM
from transformers import WavLMModel
import torch

def extract_semantic_tokens(audio, sample_rate=16000):
    """
    Extract semantic tokens from audio using WavLM.
    
    Returns discrete tokens representing content/meaning.
    """
    # Load WavLM
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model.eval()
    
    # Extract features
    with torch.no_grad():
        outputs = model(audio, output_hidden_states=True)
        # Use layer 7 (good balance of semantic/acoustic)
        features = outputs.hidden_states[7]
    
    # Quantize with k-means (pre-trained codebook)
    # In practice, use pre-computed k-means centers
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=1024)  # 1024 semantic tokens
    
    # Flatten features
    features_flat = features.reshape(-1, features.shape[-1])
    
    # Assign to clusters
    semantic_tokens = kmeans.predict(features_flat.cpu().numpy())
    
    return semantic_tokens.reshape(features.shape[0], features.shape[1])
```

### Acoustic Token Extraction

```python
# Extract acoustic tokens using EnCodec
from encodec import EncodecModel
import torch

def extract_acoustic_tokens(audio, sample_rate=24000, bandwidth=6.0):
    """
    Extract acoustic tokens using EnCodec.
    
    Returns RVQ tokens (multiple levels per frame).
    """
    # Load EnCodec
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()
    
    # Encode
    with torch.no_grad():
        encoded_frames = model.encode(audio)
    
    # Extract codes (list of tensors, one per RVQ level)
    codes = [frame[0] for frame in encoded_frames]
    
    # Shape: (num_levels, batch, time)
    acoustic_tokens = torch.stack(codes, dim=0)
    
    return acoustic_tokens
```

---

## Hierarchical Tokenization

### AudioLM Approach (Google, 2022)

```
Three-stage hierarchical generation:

Stage 1: SEMANTIC TOKENS
├── From: w2v-BERT (self-supervised)
├── Rate: 25 Hz
├── Vocabulary: 1024
├── Model: Transformer LM
└── Output: Semantic token sequence

Stage 2: COARSE ACOUSTIC TOKENS
├── From: SoundStream (first 4 RVQ levels)
├── Rate: 50 Hz × 4 = 200 tokens/sec
├── Conditioned on: Semantic tokens
├── Model: Transformer LM
└── Output: Coarse acoustic tokens

Stage 3: FINE ACOUSTIC TOKENS
├── From: SoundStream (remaining 8 RVQ levels)
├── Rate: 50 Hz × 8 = 400 tokens/sec
├── Conditioned on: Coarse tokens
├── Model: Transformer LM
└── Output: Fine acoustic tokens

Final: Decode all acoustic tokens → Audio
```

### SoundStorm Approach (Google, 2023)

```
Parallel generation of acoustic tokens:

Input: Semantic tokens from AudioLM
       ↓
┌──────────────────────────────────────────┐
│      SOUNDSTORM (Parallel Decoder)       │
│                                          │
│  Uses: Bidirectional attention          │
│  Method: Confidence-based parallel decode│
│  Speed: 100x faster than AudioLM        │
│                                          │
└──────────────────────────────────────────┘
       ↓
Output: All RVQ levels simultaneously

Key innovation: MaskGIT-style iterative refinement
- Start with all tokens masked
- Predict all tokens
- Keep high-confidence predictions
- Re-predict low-confidence tokens
- Repeat until all unmasked
```

### Mimi Approach (Kyutai, 2024)

```
Integrated semantic + acoustic:

Level 0: SEMANTIC TOKEN
├── Distilled from WavLM layer 7
├── Encodes: Content, phonemes
├── Invariant to: Speaker, prosody
└── Purpose: LLM understanding

Levels 1-7: ACOUSTIC TOKENS
├── Standard RVQ from codec
├── Encodes: Speaker, prosody, details
├── Conditioned on: Level 0
└── Purpose: Audio reconstruction

Advantage: Single unified codec
No separate semantic model needed
```

---

## Token Rate Considerations

### The Token Rate Dilemma

```
LOWER token rate:
✓ Longer context (more audio in same window)
✓ Faster LLM inference
✓ Less memory
✗ Harder to model fine details
✗ May lose information

HIGHER token rate:
✓ Better audio quality
✓ Easier to model
✗ Shorter context
✗ Slower inference
✗ More memory
```

### Codec Comparison

| Codec | Frame Rate | RVQ Levels | Tokens/sec | 10s Audio | Quality |
|-------|------------|------------|------------|-----------|---------|
| SoundStream | 75 Hz | 12 | 900 | 9,000 | High |
| EnCodec | 75 Hz | 8 | 600 | 6,000 | High |
| Mimi | 12.5 Hz | 8 | 100 | 1,000 | Good |

### Context Window Impact

```
Example: 4096 token context window

With EnCodec (600 tokens/sec):
├── 4096 / 600 = 6.8 seconds of audio
└── Very limited for dialogue

With Mimi (100 tokens/sec):
├── 4096 / 100 = 40.9 seconds of audio
└── Much better for conversation

This is why Moshi uses Mimi:
- 6x fewer tokens than EnCodec
- Enables longer conversations
- Still maintains speech quality
```

---

## Vocabulary Design

### Separate Vocabularies

```python
class MultimodalVocabulary:
    """
    Separate vocabularies for each modality.
    """
    def __init__(self):
        # Text vocabulary (BPE)
        self.text_vocab_size = 32000
        self.text_offset = 0
        
        # Audio vocabulary (per RVQ level)
        self.audio_codebook_size = 1024
        self.audio_levels = 8
        self.audio_offset = 32000
        
        # Special tokens
        self.pad_token = 0
        self.eos_token = 1
        self.audio_start = 32000
        self.audio_end = 32001
    
    def encode_text(self, text):
        # Standard BPE tokenization
        return tokenize(text)  # Returns [0, 32000)
    
    def encode_audio(self, audio_codes, level):
        # Offset by text vocab + level offset
        offset = self.audio_offset + level * self.audio_codebook_size
        return audio_codes + offset
    
    def total_vocab_size(self):
        return (self.text_vocab_size + 
                self.audio_levels * self.audio_codebook_size + 
                10)  # Special tokens
```

### Unified Vocabulary

```python
class UnifiedVocabulary:
    """
    Single shared vocabulary for all modalities.
    
    Used in models like GPT-4o (inferred).
    """
    def __init__(self, vocab_size=100000):
        self.vocab_size = vocab_size
        
        # Learned embeddings map all tokens to same space
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Different modalities use different ranges
        self.text_range = (0, 50000)
        self.audio_range = (50000, 90000)
        self.image_range = (90000, 100000)
    
    def forward(self, tokens, modality):
        # Same embedding layer for all
        return self.embedding(tokens)
```

---

## Context Length Management

### Strategies for Long Audio

```
1. CHUNKING
   Process audio in fixed chunks, aggregate results
   
   Audio → [Chunk 1][Chunk 2][Chunk 3]
   Each chunk: Independent processing
   Aggregate: Concatenate or vote

2. SLIDING WINDOW
   Overlapping windows with attention
   
   Window 1: [========]
   Window 2:     [========]
   Window 3:         [========]
   
   Overlap: Smooth transitions

3. HIERARCHICAL PROCESSING
   Coarse-to-fine multi-level
   
   Level 1: Process every 10th frame (low-res)
   Level 2: Process every frame (high-res)
   Condition level 2 on level 1

4. EFFICIENT ATTENTION
   Flash Attention, sparse attention
   
   Reduces O(n²) memory to O(n)
   Enables longer sequences
```

### Moshi's Approach

```
Multi-stream interleaving:

Time step t (80ms):
├── User audio: 8 tokens (Mimi RVQ)
├── System audio: 8 tokens (Mimi RVQ)
└── Text: 1 token (every ~4 steps)

Total per step: ~17 tokens
For 10 seconds: 125 steps × 17 = 2,125 tokens

Fits comfortably in 4096 context
Leaves room for longer conversations
```

---

## Latest Research (2024-2025)

### GPT-4o Voice Mode (OpenAI, 2024)

```
Key insights from recent evaluation (Feb 2025):

Capabilities:
├── Strong audio understanding
├── Multilingual speech recognition
├── Music analysis
├── Intent classification
└── Robust against hallucinations

Limitations:
├── Struggles with duration prediction
├── Instrument classification issues
├── Safety mechanisms block some tasks
└── Sensitive to input quality

Architecture (inferred):
├── Native multimodal (not pipeline)
├── Unified tokenization
├── End-to-end audio processing
└── Real-time capable
```

### Neural Codec Investigation (Microsoft, Dec 2024)

```
Recent findings on codec choice for speech LLMs:

Best practices:
├── Semantic tokens improve language modeling
├── Lower frame rate better for LLMs
├── RVQ levels: 4-8 optimal for speech
└── Codec quality matters less than thought

Key result: Mimi-style semantic tokens
significantly improve downstream tasks
```

### SoundStorm Advances (Google, 2023-2024)

```
Parallel audio generation breakthrough:

Speed: 100x faster than autoregressive
Quality: Matches AudioLM
Method: MaskGIT-style iterative decoding

Impact on field:
├── Enables real-time generation
├── Better for interactive applications
├── Adopted in production systems
└── Influenced Moshi architecture
```

---

## Practical Implementation

### Complete Tokenization Pipeline

```python
class AudioTokenizer:
    """
    Complete audio tokenization for LLM use.
    """
    def __init__(
        self,
        codec_type='mimi',
        extract_semantic=True,
        sample_rate=24000
    ):
        self.sample_rate = sample_rate
        self.extract_semantic = extract_semantic
        
        # Load codec
        if codec_type == 'mimi':
            from moshi import MimiCodec
            self.codec = MimiCodec.from_pretrained("kyutai/mimi")
        elif codec_type == 'encodec':
            from encodec import EncodecModel
            self.codec = EncodecModel.encodec_model_24khz()
        
        # Load semantic model if needed
        if extract_semantic:
            from transformers import WavLMModel
            self.semantic_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-large"
            )
    
    def tokenize(self, audio):
        """
        Tokenize audio to discrete tokens.
        
        Returns:
            semantic_tokens: (batch, time) if extract_semantic
            acoustic_tokens: (batch, levels, time)
        """
        results = {}
        
        # Extract semantic tokens
        if self.extract_semantic:
            with torch.no_grad():
                outputs = self.semantic_model(audio)
                features = outputs.hidden_states[7]
                # Quantize (simplified - use pre-trained kmeans)
                semantic_tokens = self.quantize_features(features)
                results['semantic'] = semantic_tokens
        
        # Extract acoustic tokens
        with torch.no_grad():
            if hasattr(self.codec, 'encode'):
                acoustic_tokens = self.codec.encode(audio)
            else:
                encoded = self.codec.encode(audio)
                acoustic_tokens = torch.stack([e[0] for e in encoded])
        
        results['acoustic'] = acoustic_tokens
        
        return results
    
    def detokenize(self, acoustic_tokens):
        """
        Convert tokens back to audio.
        """
        with torch.no_grad():
            audio = self.codec.decode(acoustic_tokens)
        return audio
```

---

## Key Takeaways

```
1. TOKENIZATION ENABLES LLM COMPATIBILITY
   - Discrete tokens required for transformers
   - Neural codecs provide compression + discretization

2. SEMANTIC + ACOUSTIC HIERARCHY
   - Semantic: Content understanding
   - Acoustic: Audio quality
   - Both needed for full capability

3. TOKEN RATE IS CRITICAL
   - Lower rate: Longer context
   - Higher rate: Better quality
   - Mimi's 12.5 Hz enables long conversations

4. LATEST RESEARCH TRENDS
   - Integrated semantic tokens (Mimi)
   - Parallel generation (SoundStorm)
   - Multimodal unification (GPT-4o)

5. PRACTICAL CONSIDERATIONS
   - Context length management
   - Vocabulary design
   - Efficient attention mechanisms
```

---

## Further Reading

- `02_moshi_architecture.md` - How Moshi uses tokens
- `../03-neural-audio-codecs/` - Codec details
- `../04-speech-representations/02_wavlm_architecture.md` - Semantic features
