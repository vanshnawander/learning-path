# Evolution of Speech Recognition: HMMs to Transformers

A comprehensive history of automatic speech recognition (ASR), from statistical methods to modern deep learning. Understanding this evolution is essential for appreciating current architectures.

## Table of Contents
1. [Timeline Overview](#timeline-overview)
2. [The HMM Era (1970s-2010s)](#the-hmm-era-1970s-2010s)
3. [Gaussian Mixture Models (GMMs)](#gaussian-mixture-models-gmms)
4. [The HCLG Framework](#the-hclg-framework)
5. [RNN Revolution](#rnn-revolution)
6. [CTC Loss](#ctc-loss)
7. [Attention Mechanisms](#attention-mechanisms)
8. [Transformer Era](#transformer-era)
9. [Self-Supervised Learning](#self-supervised-learning)
10. [Current State & Future](#current-state--future)

---

## Timeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPEECH RECOGNITION EVOLUTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ 1950s    │ First speech recognizers (10 digits)                            │
│          │ Bell Labs "Audrey" - single speaker                              │
│          │                                                                   │
│ 1970s    │ Hidden Markov Models introduced                                  │
│          │ CMU "Harpy" - 1000 words                                         │
│          │                                                                   │
│ 1980s    │ HMM + GMM becomes standard                                       │
│          │ DARPA speech programs begin                                       │
│          │                                                                   │
│ 1990s    │ LVCSR (Large Vocabulary Continuous Speech Recognition)           │
│          │ WFST decoding formalized                                          │
│          │                                                                   │
│ 2000s    │ Discriminative training (MMI, MPE)                               │
│          │ First commercial systems (Dragon, Nuance)                         │
│          │                                                                   │
│ 2010s    │ Deep Learning revolution begins                                   │
│          │ 2012: DNN-HMM hybrid systems                                      │
│          │ 2014: Deep Speech (end-to-end)                                    │
│          │ 2015: Attention-based seq2seq                                     │
│          │ 2017: Transformer architecture                                    │
│          │                                                                   │
│ 2020s    │ Self-supervised pretraining (wav2vec, WavLM)                     │
│          │ 2022: Whisper (680k hours)                                        │
│          │ 2024: Audio LLMs (Moshi, GPT-4o Voice)                           │
│          │                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The HMM Era (1970s-2010s)

### Why Hidden Markov Models?

```
Speech has temporal structure:
- Phonemes occur in sequence
- Each phoneme has variable duration
- Transitions between phonemes follow patterns

HMM models this perfectly:
- Hidden states = phonetic units
- Observations = acoustic features
- Transitions = phoneme sequences
```

### HMM Fundamentals

```
An HMM is defined by:
- N: Number of hidden states (phoneme substates)
- A: Transition probability matrix (N × N)
- B: Emission probability (observation given state)
- π: Initial state distribution

For speech:
┌─────┐    ┌─────┐    ┌─────┐
│ /a/ │───▶│ /a/ │───▶│ /a/ │   (left-to-right topology)
│ s1  │    │ s2  │    │ s3  │
└──┬──┘    └──┬──┘    └──┬──┘
   │          │          │
   ▼          ▼          ▼
  o₁         o₂         o₃     (MFCC observations)

Each phoneme: 3-5 states (captures beginning/middle/end)
```

### HMM Training: Baum-Welch Algorithm

```python
# Expectation-Maximization for HMM
def baum_welch(observations, hmm, max_iter=100):
    """
    Train HMM parameters using EM algorithm.
    
    E-step: Compute forward-backward probabilities
    M-step: Re-estimate A, B, π
    """
    for iteration in range(max_iter):
        # E-step: Forward-backward
        alpha = forward(observations, hmm)   # P(o₁...oₜ, qₜ=i)
        beta = backward(observations, hmm)   # P(oₜ₊₁...oₜ | qₜ=i)
        
        # Compute γ (state occupation) and ξ (transition)
        gamma = compute_gamma(alpha, beta)
        xi = compute_xi(alpha, beta, observations, hmm)
        
        # M-step: Re-estimate parameters
        hmm.A = update_transitions(xi, gamma)
        hmm.B = update_emissions(gamma, observations)
        hmm.pi = gamma[0]
    
    return hmm
```

### HMM Decoding: Viterbi Algorithm

```python
def viterbi(observations, hmm):
    """
    Find most likely state sequence given observations.
    
    Dynamic programming: O(N² × T)
    """
    T = len(observations)
    N = hmm.num_states
    
    # Viterbi tables
    delta = np.zeros((T, N))  # Best score to state j at time t
    psi = np.zeros((T, N), dtype=int)  # Best predecessor
    
    # Initialization
    delta[0] = hmm.pi * hmm.B[:, observations[0]]
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1] * hmm.A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] * hmm.B[j, observations[t]]
    
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    
    return path
```

---

## Gaussian Mixture Models (GMMs)

### Why GMM for Emissions?

```
HMM needs P(observation | state)

Observations are continuous (MFCC vectors, 39-dim typically)
Discrete HMM can't handle continuous observations

Solution: Model P(o | s) as Gaussian Mixture

P(o | s) = Σₖ wₖ N(o; μₖ, Σₖ)

Where:
- wₖ: mixture weights (sum to 1)
- μₖ: mean vectors
- Σₖ: covariance matrices (often diagonal)
- K: number of mixture components (8-32 typical)
```

### GMM-HMM System

```
┌─────────────────────────────────────────────────────────────────┐
│                     GMM-HMM ACOUSTIC MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio waveform                                                  │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────┐                                               │
│  │ Feature Ext. │  MFCC: 13 static + 13 delta + 13 delta-delta │
│  │   (MFCC)     │  = 39-dimensional feature vector              │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │     GMM      │  For each HMM state:                          │
│  │   Scoring    │  Compute P(features | state)                  │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │   Viterbi    │  Find best state sequence                     │
│  │   Decoder    │  → Phoneme sequence → Words                   │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Limitations of GMM-HMM

```
1. FIXED FEATURES
   - MFCC designed by hand
   - May not be optimal
   - Same features for all tasks

2. DIAGONAL COVARIANCE
   - Assumes feature dimensions independent
   - Limits modeling capacity

3. FRAME-LEVEL INDEPENDENCE
   - Each frame scored independently given state
   - No learned temporal patterns within state

4. LIMITED CONTEXT
   - Triphone context helps but limited
   - Can't capture long-range dependencies
```

---

## The HCLG Framework

### Weighted Finite State Transducers (WFSTs)

```
HCLG combines all ASR components into single searchable graph:

H: HMM topology
   - Maps HMM states to context-dependent phones
   
C: Context dependency
   - Maps context-dependent phones to phones
   - Handles triphone context (left-phone-right)

L: Lexicon (pronunciation dictionary)
   - Maps phone sequences to words
   - Handles multiple pronunciations

G: Grammar (language model)
   - Maps word sequences to sentences
   - Provides prior on word sequences

HCLG = H ∘ C ∘ L ∘ G  (composition)
```

### WFST Composition

```
Composition combines two transducers:

T1: a:b/w1    (input a, output b, weight w1)
T2: b:c/w2    (input b, output c, weight w2)

T1 ∘ T2: a:c/(w1 ⊗ w2)

For ASR:
- Input: acoustic frames
- Output: word sequence
- Weight: negative log probability
```

### Why HCLG Matters

```
1. EFFICIENCY
   - Single graph search vs multiple passes
   - All knowledge sources combined

2. MODULARITY
   - Train components separately
   - Swap language model without retraining

3. THEORY
   - Well-understood mathematical framework
   - Optimal search algorithms exist

Still used today in Kaldi and hybrid systems!
```

---

## RNN Revolution

### Why RNNs for Speech?

```
Speech is inherently sequential:
- Meaning depends on context
- Long-range dependencies exist
- Variable length inputs

RNNs can model this:
- Hidden state carries history
- Theoretically infinite memory
- Natural sequence processing
```

### From RNN to LSTM/GRU

```
Vanilla RNN:
h_t = tanh(W_h h_{t-1} + W_x x_t + b)

Problem: Vanishing gradients over long sequences

LSTM (Long Short-Term Memory):
- Cell state c_t carries long-term memory
- Gates control information flow:
  - Forget gate: what to discard
  - Input gate: what to store
  - Output gate: what to output

┌─────────────────────────────────────────────┐
│               LSTM CELL                      │
│                                              │
│   c_{t-1} ─────[×]───────[+]────── c_t      │
│               ↑  ↑         ↑                 │
│              f_t  │       i_t × ĉ_t         │
│               │   │         │                │
│               └───┴─────────┘                │
│                   │                          │
│              [Forget] [Input] [Output]       │
│                   │      │       │           │
│   h_{t-1} ───────┴──────┴───────┴────── h_t │
│                                              │
└─────────────────────────────────────────────┘

GRU (Gated Recurrent Unit):
- Simplified version of LSTM
- Reset and update gates
- Comparable performance, fewer parameters
```

### Deep Speech (2014)

```python
# Deep Speech Architecture (simplified)
class DeepSpeech(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        # 3 fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Bidirectional RNN
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, 
                          bidirectional=True)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, time, features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x, _ = self.rnn(x)
        
        x = self.fc_out(x)
        return x  # (batch, time, num_classes)
```

---

## CTC Loss

### The Alignment Problem

```
Input:  [frame_1] [frame_2] [frame_3] ... [frame_100]
Output: "hello"

Problem: Which frames correspond to which characters?
         We don't have frame-level labels!

Traditional solution: Force alignment with HMM
CTC solution: Sum over ALL possible alignments
```

### CTC Algorithm

```
CTC introduces blank token (∅):

Valid CTC paths for "cat":
- c a t ∅ ∅ ∅ ∅ ∅ ∅ ∅
- ∅ c ∅ a ∅ t ∅ ∅ ∅ ∅
- c c c a a t t t ∅ ∅
- ∅ ∅ c c a a a t t ∅

All collapse to "cat" after removing blanks and dedup

P(y | x) = Σ P(π | x) for all π that collapse to y
         all valid paths

This sum is computed efficiently with dynamic programming!
```

### CTC Forward Algorithm

```python
def ctc_forward(log_probs, labels):
    """
    Compute CTC loss using forward algorithm.
    
    log_probs: (T, vocab_size) - network output
    labels: target sequence (without blanks)
    """
    T = log_probs.shape[0]
    L = len(labels)
    
    # Extended labels with blanks: ∅ l_1 ∅ l_2 ∅ ... ∅ l_L ∅
    ext_labels = [BLANK]
    for l in labels:
        ext_labels.extend([l, BLANK])
    S = len(ext_labels)  # 2L + 1
    
    # Forward variables
    alpha = np.full((T, S), -np.inf)
    
    # Initialization (t=0)
    alpha[0, 0] = log_probs[0, BLANK]
    alpha[0, 1] = log_probs[0, ext_labels[1]]
    
    # Recursion
    for t in range(1, T):
        for s in range(S):
            label = ext_labels[s]
            
            # Same state
            alpha[t, s] = alpha[t-1, s]
            
            # Previous state
            if s > 0:
                alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-1])
            
            # Skip blank (only if not blank and different from previous)
            if s > 1 and ext_labels[s] != BLANK and ext_labels[s] != ext_labels[s-2]:
                alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-2])
            
            # Add emission probability
            alpha[t, s] += log_probs[t, label]
    
    # Total probability
    return np.logaddexp(alpha[-1, -1], alpha[-1, -2])
```

---

## Attention Mechanisms

### Limitation of Encoder-Only Models

```
CTC assumes:
- Output shorter than input
- Monotonic alignment (roughly)
- No strong output dependencies

For some tasks (translation, summarization):
- Need encoder-decoder structure
- Need flexible alignment
- Need output conditioning on previous outputs
```

### Sequence-to-Sequence with Attention

```
                    ┌─────────────────────────────┐
                    │         DECODER             │
                    │                             │
                    │  ┌─────┐  ┌─────┐  ┌─────┐ │
                    │  │ <s> │→ │ h   │→ │ e   │ │
                    │  └──┬──┘  └──┬──┘  └──┬──┘ │
                    │     │       │       │      │
                    │     ▼       ▼       ▼      │
                    │  ┌──────────────────────┐  │
                    │  │ ATTENTION (weighted) │  │
                    │  └──────────┬───────────┘  │
                    └─────────────┼──────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                      │
            ▼                     ▼                      ▼
         ┌─────┐              ┌─────┐              ┌─────┐
         │ h_1 │              │ h_2 │              │ h_3 │
         └──┬──┘              └──┬──┘              └──┬──┘
            │                    │                    │
            │         ENCODER    │                    │
         ┌──┴──┐              ┌──┴──┐              ┌──┴──┐
         │ x_1 │              │ x_2 │              │ x_3 │
         └─────┘              └─────┘              └─────┘
         audio frames
```

### Attention Computation

```python
def attention(query, keys, values):
    """
    Scaled dot-product attention.
    
    query: decoder hidden state (1, d)
    keys: encoder outputs (T, d)
    values: encoder outputs (T, d)
    """
    # Attention scores
    scores = query @ keys.T / np.sqrt(d)  # (1, T)
    
    # Attention weights (softmax)
    weights = softmax(scores, dim=-1)  # (1, T)
    
    # Context vector (weighted sum)
    context = weights @ values  # (1, d)
    
    return context, weights
```

### Listen, Attend and Spell (LAS, 2015)

```
First successful attention-based ASR:

1. LISTENER (Encoder)
   - Pyramidal BLSTM
   - Reduces time resolution by 2x at each layer
   - Manages long audio sequences

2. ATTENTION
   - Content-based attention
   - Learns soft alignment

3. SPELLER (Decoder)
   - Character-level LSTM
   - Generates output autoregressively
```

---

## Transformer Era

### Self-Attention for Speech

```
Transformer advantages:
- Parallel computation (vs sequential RNN)
- Direct long-range connections
- Scales to massive data

Challenges for speech:
- Very long sequences (1000s of frames)
- Quadratic attention complexity O(n²)

Solutions:
- Downsampling (conv layers before transformer)
- Relative positional encoding
- Efficient attention variants
```

### Speech Transformer Architecture

```python
class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        # Subsampling (reduce sequence length)
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
        )
        
        # Linear projection
        self.proj = nn.Linear(32 * ((input_dim - 3) // 4), d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
    def forward(self, audio_features, text_tokens):
        # Subsample audio
        x = self.subsample(audio_features.unsqueeze(1))
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.pos_enc(x)
        
        # Encode
        memory = self.encoder(x)
        
        # Decode
        output = self.decoder(text_tokens, memory)
        
        return output
```

### Conformer (2020)

```
Combines Transformer attention with Convolution:

┌───────────────────────────────────────┐
│          CONFORMER BLOCK              │
├───────────────────────────────────────┤
│                                       │
│  Input x                              │
│     │                                 │
│     ▼                                 │
│  ┌─────────────────┐                  │
│  │ Feed Forward    │ ─── × 0.5       │
│  │ (half)          │                  │
│  └────────┬────────┘                  │
│           │                           │
│           ▼                           │
│  ┌─────────────────┐                  │
│  │ Multi-Head      │                  │
│  │ Self-Attention  │                  │
│  └────────┬────────┘                  │
│           │                           │
│           ▼                           │
│  ┌─────────────────┐                  │
│  │ Convolution     │ ← Local patterns │
│  │ Module          │                  │
│  └────────┬────────┘                  │
│           │                           │
│           ▼                           │
│  ┌─────────────────┐                  │
│  │ Feed Forward    │ ─── × 0.5       │
│  │ (half)          │                  │
│  └────────┬────────┘                  │
│           │                           │
│           ▼                           │
│  LayerNorm + Residual                 │
│                                       │
└───────────────────────────────────────┘

Best of both worlds:
- Attention: global context
- Convolution: local patterns
```

---

## Self-Supervised Learning

### The Data Problem

```
Supervised ASR requires:
- Audio + transcripts
- Expensive to collect at scale
- Limited to ~100k hours

Available unlabeled audio:
- YouTube: millions of hours
- Podcasts: millions of hours
- Audiobooks: hundreds of thousands of hours

Solution: Learn from unlabeled audio first!
```

### wav2vec 2.0 (2020)

```
Self-supervised pretraining for speech:

1. FEATURE ENCODER
   - CNN converts waveform to features
   - 20ms per feature frame

2. QUANTIZER
   - Discretizes features into tokens
   - Learned codebook (like VQ-VAE)

3. TRANSFORMER
   - Masked prediction (like BERT)
   - Predict quantized features of masked positions

4. CONTRASTIVE LOSS
   - True quantized vector vs distractors
   - Encourages meaningful representations
```

### WavLM Innovations

```
Beyond wav2vec 2.0:

1. DENOISING PRETRAINING
   - Add noise/overlap during training
   - Predict clean targets
   - Robust to real-world audio

2. GATED RELATIVE POSITION BIAS
   - Better positional information
   - Content-dependent position weighting

3. FULL-STACK REPRESENTATIONS
   - Works for ASR, speaker ID, emotion, etc.
   - Universal speech encoder
```

---

## Current State & Future

### 2024-2025 Landscape

```
PRODUCTION SYSTEMS:
├── Whisper (OpenAI) - Robust, multilingual
├── Universal Speech Model (Google)
├── Azure Speech (Microsoft)
└── Amazon Transcribe

RESEARCH FRONTIERS:
├── Audio LLMs (Moshi, GPT-4o Voice)
├── Multimodal integration
├── Real-time full-duplex
└── Low-resource languages

REMAINING CHALLENGES:
├── Accented speech
├── Code-switching
├── Noisy environments
├── Low-resource languages
├── Hallucinations
```

### Key Takeaways

```
1. HMMs provided mathematical foundation
   - Still relevant for hybrid systems
   
2. RNNs enabled end-to-end learning
   - Removed need for hand-designed features

3. Attention/Transformers improved quality
   - Global context, parallel training

4. Self-supervision unlocked scale
   - Billions of hours of audio available

5. Future: Multimodal, real-time, conversational
   - Audio as first-class modality in LLMs
```

---

## Exercises

1. **Implement Viterbi** (`exercises/01_viterbi.py`): Code the Viterbi algorithm for a simple HMM
2. **MFCC Extraction** (`exercises/02_mfcc.py`): Implement MFCC from scratch
3. **CTC Forward** (`exercises/03_ctc.py`): Implement CTC forward algorithm
4. **Attention Visualization** (`exercises/04_attention.py`): Visualize attention weights on speech
