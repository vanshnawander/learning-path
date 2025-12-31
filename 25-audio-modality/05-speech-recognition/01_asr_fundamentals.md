# ASR Fundamentals: From Basics to Modern Systems

Comprehensive guide to Automatic Speech Recognition covering traditional approaches, neural methods, and latest developments through 2025.

## Table of Contents
1. [What is ASR?](#what-is-asr)
2. [The ASR Pipeline](#the-asr-pipeline)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Traditional ASR (HMM-GMM)](#traditional-asr-hmm-gmm)
5. [Neural ASR Evolution](#neural-asr-evolution)
6. [End-to-End ASR](#end-to-end-asr)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [State-of-the-Art (2024-2025)](#state-of-the-art-2024-2025)

---

## What is ASR?

### Definition

```
Automatic Speech Recognition (ASR):
The task of converting spoken language (audio) into written text.

Input:  Audio waveform (speech)
Output: Text transcription

Example:
Audio:  [waveform of "hello world"]
Output: "hello world"
```

### Applications

```
Consumer:
├── Voice assistants (Siri, Alexa, Google Assistant)
├── Voice typing (dictation)
├── Smart home control
└── Accessibility (captions, transcription)

Enterprise:
├── Call center analytics
├── Meeting transcription
├── Medical dictation
└── Legal transcription

Research:
├── Speech understanding
├── Multimodal AI
└── Human-computer interaction
```

---

## The ASR Pipeline

### Traditional Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRADITIONAL ASR PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Waveform                                                  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │ Feature          │  MFCC, PLP, or filterbank                 │
│  │ Extraction       │  Output: 13-40 dim features @ 100Hz       │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Acoustic Model   │  HMM-GMM or DNN-HMM                       │
│  │ (AM)             │  P(acoustic | phoneme)                    │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Pronunciation    │  Maps phonemes to words                   │
│  │ Lexicon          │  "hello" → /h ɛ l oʊ/                    │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Language Model   │  N-gram or neural LM                      │
│  │ (LM)             │  P(word sequence)                         │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Decoder          │  Search for best word sequence            │
│  │ (WFST)           │  Combines AM, lexicon, LM                 │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  Text Output                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Modern End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   END-TO-END ASR PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio Waveform                                                  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │ Feature          │  Mel spectrogram or learned               │
│  │ Extraction       │  80 mel bins @ 100Hz                      │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Encoder          │  CNN + Transformer or Conformer           │
│  │                  │  Learns acoustic representations          │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Decoder          │  Transformer decoder                      │
│  │                  │  Generates text autoregressively          │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  Text Output                                                     │
│                                                                  │
│  Advantages:                                                     │
│  ✓ Single model (no separate components)                        │
│  ✓ End-to-end optimization                                      │
│  ✓ Simpler training                                             │
│  ✓ Better performance                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

### Word Error Rate (WER)

```
Primary metric for ASR evaluation:

WER = (S + D + I) / N × 100%

Where:
- S: Substitutions (wrong word)
- D: Deletions (missing word)
- I: Insertions (extra word)
- N: Total words in reference

Example:
Reference:  "the cat sat on the mat"
Hypothesis: "the cat sit on mat"

Alignment:
the cat sat on the mat
the cat sit on     mat
        S      D

S=1, D=1, I=0, N=6
WER = (1+1+0)/6 = 33.3%
```

### Character Error Rate (CER)

```
Similar to WER but at character level:

CER = (S + D + I) / N × 100%

Useful for:
├── Languages without clear word boundaries (Chinese, Japanese)
├── More fine-grained error analysis
└── Evaluating spelling accuracy

Example:
Reference:  "hello"
Hypothesis: "helo"

S=0, D=1, I=0, N=5
CER = 1/5 = 20%
```

### Real-Time Factor (RTF)

```
Measures inference speed:

RTF = Processing Time / Audio Duration

RTF < 1.0: Faster than real-time (good!)
RTF = 1.0: Exactly real-time
RTF > 1.0: Slower than real-time (bad)

Example:
Process 10 seconds of audio in 2 seconds
RTF = 2/10 = 0.2 (5x faster than real-time)
```

---

## Traditional ASR (HMM-GMM)

### Hidden Markov Model Basics

```
HMM models speech as sequence of states:

For word "cat":
┌─────┐    ┌─────┐    ┌─────┐
│ /k/ │───▶│ /æ/ │───▶│ /t/ │
└──┬──┘    └──┬──┘    └──┬──┘
   │          │          │
   ▼          ▼          ▼
  o₁         o₂         o₃
(MFCC observations)

Each phoneme: 3-5 HMM states
Transitions: Left-to-right topology
Emissions: GMM (Gaussian Mixture Model)
```

### Training Process

```
1. DATA PREPARATION
   ├── Collect audio + transcriptions
   ├── Force-align audio to phonemes
   └── Extract MFCC features

2. ACOUSTIC MODEL TRAINING
   ├── Initialize HMM-GMM parameters
   ├── Expectation-Maximization (EM)
   └── Iterate until convergence

3. LANGUAGE MODEL TRAINING
   ├── Collect text corpus
   ├── Train n-gram model (3-gram or 5-gram)
   └── Smooth probabilities

4. LEXICON CREATION
   ├── Phonetic dictionary
   └── Handle out-of-vocabulary words
```

### Limitations

```
1. FEATURE ENGINEERING
   - Hand-designed MFCC features
   - May not be optimal
   - Fixed for all tasks

2. INDEPENDENCE ASSUMPTIONS
   - Frames assumed independent given state
   - Ignores long-range dependencies
   - Suboptimal modeling

3. SEPARATE TRAINING
   - AM and LM trained separately
   - Not jointly optimized
   - Potential mismatch

4. COMPLEXITY
   - Many components to tune
   - Difficult to maintain
   - Hard to improve
```

---

## Neural ASR Evolution

### DNN-HMM Hybrid (2012)

```
Replace GMM with Deep Neural Network:

Traditional: HMM-GMM
Hybrid:      HMM-DNN

Architecture:
Input (MFCC) → DNN → P(state | features) → HMM decoder

Benefits:
├── Better acoustic modeling
├── 20-30% relative WER reduction
└── Still uses HMM framework

Limitations:
├── Still requires alignment
├── Frame-level independence
└── Separate LM
```

### RNN-Based Models (2014-2016)

```
Use RNNs for temporal modeling:

LSTM/GRU advantages:
├── Model temporal dependencies
├── No independence assumption
├── Better long-range context

Architecture:
Input → LSTM layers → Output distribution

Still requires:
├── CTC loss or attention mechanism
├── Separate LM often
└── Careful training
```

---

## End-to-End ASR

### CTC (Connectionist Temporal Classification)

```
Key innovation: No alignment needed!

Problem: Audio and text different lengths
Solution: Allow blank token, collapse repeats

Example:
Audio frames: [h][h][_][e][e][l][l][l][_][o]
After collapse: "hello"

Loss: Sum over all valid alignments
Training: Standard backprop
Inference: Beam search with LM
```

### Attention-Based Encoder-Decoder

```
Listen, Attend and Spell (LAS):

Encoder: Processes entire audio
Decoder: Generates text with attention

┌────────────────────────────────────┐
│  Audio → Encoder → Hidden states   │
│                         ↓           │
│  <s> → Decoder ← Attention         │
│         ↓                           │
│       "the"                         │
│         ↓                           │
│  "the" → Decoder ← Attention       │
│         ↓                           │
│       "cat"                         │
│         ...                         │
└────────────────────────────────────┘

Advantages:
├── End-to-end optimization
├── No alignment needed
├── Implicit language model
└── State-of-the-art performance
```

### Transformer ASR

```
Replace RNNs with Transformers:

Benefits:
├── Parallel training (vs sequential RNN)
├── Better long-range dependencies
├── Scales to large datasets
└── Faster training

Architecture:
Audio → Conformer Encoder → Transformer Decoder → Text

Conformer: Convolution + Transformer
- Convolution: Local patterns
- Transformer: Global context
- Best of both worlds
```

---

## Challenges and Solutions

### Challenge 1: Noisy Environments

```
Problem: Background noise degrades performance

Solutions:
├── Data augmentation (add noise during training)
├── Robust features (WavLM, wav2vec 2.0)
├── Multi-condition training
└── Denoising preprocessing

WavLM approach:
- Train on noisy/overlapped speech
- Predict clean targets
- Much better robustness
```

### Challenge 2: Accents and Dialects

```
Problem: Models trained on standard accent fail on others

Solutions:
├── Multi-accent training data
├── Accent-specific fine-tuning
├── Self-supervised pretraining (wav2vec 2.0)
└── Massive multilingual training (Whisper)

Whisper approach:
- 680k hours from diverse sources
- 99 languages
- Robust to accents naturally
```

### Challenge 3: Domain Adaptation

```
Problem: Medical/legal/technical terms not in training

Solutions:
├── Domain-specific fine-tuning
├── Custom language models
├── Vocabulary expansion
└── Few-shot adaptation

Practical approach:
1. Start with pretrained model (Whisper)
2. Fine-tune on domain data (even 1-10 hours helps)
3. Use domain-specific LM for rescoring
```

### Challenge 4: Real-Time Processing

```
Problem: Need low latency for interactive applications

Solutions:
├── Streaming models (causal attention)
├── Chunked processing
├── Efficient architectures
└── Hardware optimization

Streaming Whisper (2025):
- Two-pass decoding
- First pass: Fast, low-quality
- Second pass: Refine with context
- Achieves real-time on CPU
```

---

## State-of-the-Art (2024-2025)

### Whisper (OpenAI, 2022-2024)

```
Current SOTA for general ASR:

Whisper Large-v3 (2024):
├── 1.5B parameters
├── Trained on 680k hours
├── 99 languages
├── WER: 1.4% on LibriSpeech test-clean
└── Robust to noise, accents, domains

Key innovations:
├── Massive scale
├── Weak supervision (web data)
├── Multitask training (transcription + translation)
└── Zero-shot generalization
```

### Streaming ASR Advances (2025)

```
Bloomberg's Streaming Whisper (Interspeech 2025):
├── Two-pass decoding
├── First pass: Streaming (low latency)
├── Second pass: Non-streaming (high quality)
└── Best of both worlds

Performance:
├── Latency: <500ms
├── WER: Near offline Whisper
└── Production-ready
```

### Self-Supervised + Fine-tuning

```
Best practice (2024-2025):

1. Pretrain: wav2vec 2.0 or WavLM
   - 60k-94k hours unlabeled audio
   - Self-supervised learning

2. Fine-tune: Small labeled dataset
   - 10 min - 100 hours
   - Task-specific

Results:
├── 10 min labels: WER 4.8%
├── 1 hour labels: WER 2.9%
└── Matches fully supervised with 100x less data!
```

### Multilingual Models

```
Trend: Single model for all languages

Whisper: 99 languages
MMS (Meta, 2023): 1100+ languages
USM (Google, 2023): 100+ languages

Benefits:
├── Low-resource languages benefit
├── Code-switching handled naturally
├── Transfer learning across languages
└── Simpler deployment
```

---

## Practical Implementation

### Using Whisper

```python
import whisper

# Load model
model = whisper.load_model("large-v3")

# Transcribe
result = model.transcribe("audio.mp3")
print(result["text"])

# With options
result = model.transcribe(
    "audio.mp3",
    language="en",
    task="transcribe",  # or "translate"
    fp16=True,  # GPU acceleration
)
```

### Fine-tuning for Custom Domain

```python
from transformers import WhisperForConditionalGeneration, Trainer

# Load pretrained
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3"
)

# Prepare domain-specific data
train_dataset = prepare_medical_dataset()

# Fine-tune
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    # ... training args
)

trainer.train()
```

### Streaming ASR

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class StreamingASR:
    def __init__(self, chunk_length_s=5.0):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.chunk_length_s = chunk_length_s
        
    def transcribe_stream(self, audio_stream):
        """
        Transcribe audio stream in chunks.
        """
        buffer = []
        transcripts = []
        
        for chunk in audio_stream:
            buffer.extend(chunk)
            
            # Process when buffer is full
            if len(buffer) >= self.chunk_length_s * 16000:
                audio = torch.tensor(buffer[:int(self.chunk_length_s * 16000)])
                
                # Transcribe chunk
                inputs = self.processor(audio, return_tensors="pt", sampling_rate=16000)
                with torch.no_grad():
                    predicted_ids = self.model.generate(inputs.input_features)
                
                transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcripts.append(transcript)
                
                # Keep overlap for context
                buffer = buffer[int(self.chunk_length_s * 16000 * 0.5):]
        
        return " ".join(transcripts)
```

---

## Key Takeaways

```
1. ASR HAS EVOLVED DRAMATICALLY
   - HMM-GMM → DNN-HMM → End-to-end
   - WER: 30% (2000) → 1.4% (2024) on clean speech

2. END-TO-END IS NOW STANDARD
   - Simpler than traditional pipeline
   - Better performance
   - Easier to train

3. SELF-SUPERVISED PRETRAINING
   - Unlocks massive unlabeled data
   - Enables low-resource scenarios
   - Foundation for modern ASR

4. WHISPER IS CURRENT SOTA
   - Robust, multilingual, zero-shot
   - Production-ready
   - Open source

5. FUTURE DIRECTIONS
   - Streaming with low latency
   - Better multilingual support
   - Integration with LLMs
```

---

## Further Reading

- `02_whisper_architecture.md` - Deep dive into Whisper
- `04_ctc_vs_attention.md` - Training objectives comparison
- `../04-speech-representations/01_wav2vec_evolution.md` - Self-supervised learning
