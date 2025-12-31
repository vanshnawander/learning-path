# Multimodal Audio-Vision-Text Integration

Understanding how audio integrates with vision and text modalities in modern foundation models. Critical for building comprehensive multimodal systems.

## Table of Contents
1. [Multimodal Landscape](#multimodal-landscape)
2. [Tokenization Strategies](#tokenization-strategies)
3. [Fusion Architectures](#fusion-architectures)
4. [Audio-Text Models](#audio-text-models)
5. [Audio-Vision Models](#audio-vision-models)
6. [Unified Multimodal Models](#unified-multimodal-models)
7. [Training Strategies](#training-strategies)
8. [Practical Considerations](#practical-considerations)

---

## Multimodal Landscape

### The Convergence

```
2020-2024: Modality-specific models dominated
├── Text: GPT, BERT, LLaMA
├── Vision: ViT, CLIP, Stable Diffusion
├── Audio: Whisper, WavLM, EnCodec

2024-2025: Unified multimodal models emerging
├── GPT-4o: Text + Vision + Audio (native)
├── Gemini: Text + Vision + Audio + Video
├── Moshi: Text + Audio (native speech)
├── LLaVA-Audio: Vision + Audio + Text
└── Any-to-Any models: Generate any modality from any
```

### Why Multimodal?

```
Single modality limitations:
├── Text: No prosody, no visual context
├── Audio: No visual grounding, limited reasoning
├── Vision: No temporal audio, no language

Multimodal advantages:
├── Richer understanding (see + hear + read)
├── Cross-modal reasoning
├── More natural interaction
├── Real-world applicability
```

---

## Tokenization Strategies

### Modality-Specific Tokenizers

```
┌─────────────────────────────────────────────────────────────────┐
│                 TOKENIZATION BY MODALITY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TEXT:                                                           │
│  ├── BPE/SentencePiece tokenizer                                │
│  ├── Vocabulary: ~32k-100k tokens                               │
│  ├── ~4 chars per token (English)                               │
│  └── Example: "Hello" → [15496]                                 │
│                                                                  │
│  VISION:                                                         │
│  ├── Patch-based: 14×14 or 16×16 patches                       │
│  ├── ViT: 224×224 image → 196 tokens                           │
│  ├── VQ-VAE: Discrete visual tokens                             │
│  └── Example: 256×256 image → 256-1024 tokens                  │
│                                                                  │
│  AUDIO:                                                          │
│  ├── Neural codecs: EnCodec, Mimi, SoundStream                  │
│  ├── Frame rate: 12.5-75 Hz                                     │
│  ├── RVQ levels: 1-32 tokens per frame                          │
│  └── Example: 1 sec audio → 100-600 tokens                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Token Rate Comparison

```
Modality    │ Content        │ Tokens  │ Rate
────────────┼────────────────┼─────────┼──────────────
Text        │ 100 words      │ ~130    │ ~1.3 tok/word
Vision      │ 1 image        │ 256-576 │ Fixed per image
Audio       │ 1 second       │ 100-600 │ Varies by codec
Video       │ 1 sec @ 30fps  │ 7680+   │ 256 tok/frame

Key insight: Audio and video are TOKEN HUNGRY
             Managing context length is critical
```

### Unified Vocabulary Approaches

```python
# Approach 1: Separate vocabularies, concatenated
class SeparateVocab:
    def __init__(self):
        self.text_vocab = 32000      # 0 - 31999
        self.audio_vocab = 2048      # 32000 - 34047 (per codebook)
        self.image_vocab = 8192      # 34048 - 42239
    
    def encode_text(self, text):
        return tokenize(text)  # Returns tokens in [0, 32000)
    
    def encode_audio(self, audio_codes, codebook_idx):
        offset = 32000 + codebook_idx * 2048
        return audio_codes + offset

# Approach 2: Shared continuous embedding space
class SharedEmbedding:
    def __init__(self, d_model):
        self.text_embed = nn.Embedding(32000, d_model)
        self.audio_embed = nn.Embedding(2048, d_model)  # Per codebook
        self.image_embed = nn.Embedding(8192, d_model)
        # All map to same d_model dimension
```

---

## Fusion Architectures

### Early Fusion

```
Concatenate modality tokens before processing:

[TEXT_TOKENS] [SEP] [AUDIO_TOKENS] [SEP] [IMAGE_TOKENS]
                           │
                           ▼
              ┌────────────────────────┐
              │    Unified Transformer │
              │    (processes all)     │
              └────────────────────────┘

Pros: Deep cross-modal interaction
Cons: Quadratic attention cost, long contexts
Used by: GPT-4o, Gemini (with modifications)
```

### Late Fusion

```
Process each modality separately, then combine:

[TEXT]  ─────▶ Text Encoder  ─────┐
                                   │
[AUDIO] ─────▶ Audio Encoder ─────┼──▶ Fusion Layer ──▶ Output
                                   │
[IMAGE] ─────▶ Image Encoder ─────┘

Pros: Efficient, modular
Cons: Limited cross-modal interaction
Used by: CLIP (vision-text), early multimodal systems
```

### Cross-Attention Fusion

```
Use cross-attention between modalities:

┌─────────────────────────────────────────────────────────┐
│                                                          │
│  Audio Tokens ──┐                                       │
│                 │                                       │
│                 ▼                                       │
│  Text Tokens ──▶ Cross-Attention ──▶ Fused Output     │
│                 ▲                                       │
│                 │                                       │
│  Image Tokens ──┘                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Each modality attends to others
More efficient than full early fusion
Used by: Flamingo, LLaVA
```

### Moshi's Multi-Stream Approach

```
Parallel streams processed jointly:

Time step t:
├── User audio stream:   [u₀, u₁, ..., u₇]  (8 Mimi tokens)
├── System audio stream: [s₀, s₁, ..., s₇]  (8 Mimi tokens)  
└── Text stream:         [text_token]       (1 token)

All interleaved and processed together
Enables real-time cross-modal interaction
```

---

## Audio-Text Models

### Whisper (ASR/Translation)

```
Audio → Text only (encoder-decoder)

Architecture:
├── Encoder: Audio → Representations
├── Decoder: Representations → Text (autoregressive)
└── No text → audio capability

Use cases:
├── Transcription
├── Translation
└── Language identification
```

### AudioLM (Google, 2022)

```
First audio language model:

Stage 1: Semantic tokens from w2v-BERT
Stage 2: Coarse acoustic tokens from SoundStream
Stage 3: Fine acoustic tokens

Hierarchical generation:
Semantic → Coarse Acoustic → Fine Acoustic → Audio

Enables: Audio continuation, unconditional generation
```

### SpeechGPT / LLaSM

```
LLM with audio understanding + generation:

Input: [TEXT or AUDIO (tokenized)]
       ↓
    LLM Backbone
       ↓
Output: [TEXT or AUDIO tokens]
       ↓
    Audio Decoder (if audio output)

Enables: Spoken dialogue, audio QA
```

---

## Audio-Vision Models

### Audio-Visual Speech Recognition

```
AVSR: Use video of lips + audio for robust ASR

┌──────────────┐     ┌──────────────┐
│  Audio       │     │  Video       │
│  (mel spec)  │     │  (lip crops) │
└──────┬───────┘     └──────┬───────┘
       │                     │
       ▼                     ▼
┌──────────────┐     ┌──────────────┐
│ Audio        │     │ Visual       │
│ Encoder      │     │ Encoder      │
└──────┬───────┘     └──────┬───────┘
       │                     │
       └─────────┬───────────┘
                 │
                 ▼
         ┌──────────────┐
         │   Fusion     │
         │   Layer      │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Decoder    │
         │   (ASR)      │
         └──────────────┘

Benefit: Robust in noisy environments
         Can work with silent video (lip reading)
```

### Audio-Visual Correspondence

```
Learning: Which sounds go with which visuals?

Training objective:
├── Positive: Matching audio-video pairs
├── Negative: Mismatched audio-video pairs
└── Contrastive loss (like CLIP but audio-video)

Applications:
├── Sound source localization
├── Audio-visual retrieval
├── Video generation with audio
```

### Text-to-Audio-Visual Generation

```
Generate video WITH audio from text:

"A dog barking in a park"
        │
        ▼
┌──────────────────────────────────┐
│     Multimodal Generator         │
├──────────────────────────────────┤
│ Text Encoder                     │
│      │                           │
│      ├──▶ Video Generator        │
│      │                           │
│      └──▶ Audio Generator        │
│           (synchronized)         │
└──────────────────────────────────┘
        │
        ▼
Video + Audio (temporally aligned)
```

---

## Unified Multimodal Models

### GPT-4o Architecture (Inferred)

```
Native multimodal (not pipeline):

Input:  Text | Image | Audio (any combination)
        ↓
   Unified Tokenization
        ↓
   Single Transformer
   (processes all modalities jointly)
        ↓
Output: Text | Image | Audio (any combination)

Key innovations:
├── Native audio understanding (not ASR wrapper)
├── Real-time voice capability
├── Cross-modal reasoning
└── End-to-end training on all modalities
```

### Gemini Architecture (Inferred)

```
Similar to GPT-4o but Google-scale:

Modalities:
├── Text (multiple languages)
├── Images (high resolution)
├── Audio (speech, music, sounds)
├── Video (temporal understanding)
└── Code (programming languages)

Training:
├── Massive multimodal corpus
├── Cross-modal alignment objectives
├── Instruction tuning
└── RLHF across modalities
```

### Any-to-Any Models (Emerging)

```
CoDi, NExT-GPT, and similar:

         ┌─────────────────────────────────────┐
         │     ANY-TO-ANY GENERATION           │
         ├─────────────────────────────────────┤
Input:   │  Text | Image | Audio | Video      │
         │            │                        │
         │            ▼                        │
         │   ┌───────────────────┐             │
         │   │  Unified Encoder  │             │
         │   └─────────┬─────────┘             │
         │             │                       │
         │             ▼                       │
         │   ┌───────────────────┐             │
         │   │   LLM Backbone    │             │
         │   └─────────┬─────────┘             │
         │             │                       │
         │   ┌─────────┼─────────┐             │
         │   ▼         ▼         ▼             │
         │  Text    Image     Audio            │
         │  Dec.    Dec.      Dec.             │
Output:  │  Text | Image | Audio | Video      │
         └─────────────────────────────────────┘
```

---

## Training Strategies

### Stage-wise Training

```
Stage 1: Modality-specific pretraining
├── Text: Standard LLM training
├── Audio: EnCodec/Mimi training
├── Vision: CLIP/ViT training

Stage 2: Alignment pretraining
├── Audio-text pairs (ASR data)
├── Image-text pairs (CLIP-style)
├── Audio-image correspondence

Stage 3: Unified training
├── Interleaved multimodal data
├── Cross-modal generation tasks
├── Instruction following

Stage 4: Fine-tuning
├── Task-specific datasets
├── RLHF for quality
├── Safety alignment
```

### Modality Dropout

```python
# During training, randomly drop modalities
def forward_with_modality_dropout(text, audio, image, p_drop=0.2):
    # Randomly mask modalities
    if random.random() < p_drop:
        audio = None
    if random.random() < p_drop:
        image = None
    
    # Model learns to work with any subset
    return model(text, audio, image)
```

### Loss Balancing

```python
def multimodal_loss(outputs, targets, weights=None):
    """
    Balance losses across modalities.
    Different modalities have different scales.
    """
    if weights is None:
        weights = {
            'text': 1.0,
            'audio': 10.0,   # Audio prediction harder
            'image': 5.0,
        }
    
    total_loss = 0
    for modality in ['text', 'audio', 'image']:
        if modality in outputs:
            loss = compute_loss(outputs[modality], targets[modality])
            total_loss += weights[modality] * loss
    
    return total_loss
```

---

## Practical Considerations

### Context Length Management

```
Problem: Multimodal inputs are TOKEN HUNGRY

10 seconds audio @ 100 tok/sec = 1000 tokens
1 image @ 256 tokens = 256 tokens
100 words text = ~130 tokens
───────────────────────────────────
Total = ~1400 tokens for simple input

Solutions:
├── Efficient attention (Flash Attention)
├── Token pruning/merging
├── Hierarchical processing
├── Streaming for long content
└── Use efficient codecs (Mimi: 12.5 Hz)
```

### Compute Requirements

```
Multimodal training is EXPENSIVE:

Single modality LLM:
├── GPT-3: ~3M GPU hours
├── LLaMA 70B: ~1.7M GPU hours

Multimodal model:
├── Need more data (video, audio, images)
├── Longer sequences (more tokens)
├── Multiple encoders/decoders
├── Estimated: 3-10x single modality cost

Inference also expensive:
├── Audio processing (encode/decode)
├── Image encoding
├── Longer context
```

### Latency Considerations

```
For real-time multimodal (like Moshi):

Audio input: ~80ms frame (Mimi)
Processing: ~30ms per step
Audio output: ~80ms frame
───────────────────────────
Minimum: ~200ms latency

For non-real-time (like GPT-4o voice):
Can batch more, higher latency OK
Focus on quality over speed
```

---

## Key Takeaways

```
1. TOKENIZATION is key to multimodal integration
   - Each modality needs appropriate tokenizer
   - Token rates vary dramatically

2. FUSION strategy affects capability vs efficiency
   - Early fusion: Better interaction, expensive
   - Late fusion: Efficient, limited interaction

3. AUDIO is uniquely challenging
   - High token rate
   - Temporal alignment critical
   - Real-time requirements

4. UNIFIED models are emerging
   - GPT-4o, Gemini lead the way
   - True multimodal understanding

5. PRACTICAL constraints matter
   - Context length
   - Compute cost
   - Latency requirements
```

---

## Further Reading

- GPT-4o system card (OpenAI)
- Gemini Technical Report (Google)
- CoDi: Any-to-Any Generation (Microsoft)
- NExT-GPT (NUS)
- ImageBind (Meta)
