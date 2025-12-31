# CTC vs Attention: ASR Training Objectives

Comprehensive comparison of the two main approaches for end-to-end ASR training. Understanding both is essential for modern speech systems.

## Table of Contents
1. [The Alignment Problem](#the-alignment-problem)
2. [CTC (Connectionist Temporal Classification)](#ctc-connectionist-temporal-classification)
3. [Attention-Based Seq2Seq](#attention-based-seq2seq)
4. [Hybrid CTC-Attention](#hybrid-ctc-attention)
5. [RNN-T (Transducer)](#rnn-t-transducer)
6. [Comparative Analysis](#comparative-analysis)
7. [Implementation Details](#implementation-details)
8. [When to Use What](#when-to-use-what)

---

## The Alignment Problem

### Sequence Length Mismatch

```
Audio features: 100 frames/second
Text output: ~3 words/second = ~15 characters/second

Example (1 second of audio):
Input:  [f₁][f₂][f₃]...[f₁₀₀]  (100 frames)
Output: "hello"                 (5 characters)

Problem: How to align 100 inputs to 5 outputs?
         Which frames correspond to which characters?

Traditional: Force alignment with HMM
Modern: Learn alignment automatically
```

### Two Main Approaches

```
1. CTC (Connectionist Temporal Classification)
   ├── Monotonic alignment (left-to-right)
   ├── Independent frame predictions
   ├── Blank token for repetition
   └── Sum over all valid alignments

2. ATTENTION
   ├── Flexible alignment (can attend anywhere)
   ├── Decoder generates output autoregressively
   ├── Attention weights show alignment
   └── Implicit language model
```

---

## CTC (Connectionist Temporal Classification)

### Core Concept

```
Allow model to output:
├── Target characters (a, b, c, ...)
├── Blank token (∅)
└── Repetitions

Collapse rule:
1. Remove consecutive duplicates
2. Remove blanks

Example alignments for "cat":
c c c a a a t t t ∅ ∅ → cat
∅ c ∅ a ∅ ∅ t ∅ ∅ ∅ → cat
c ∅ ∅ a ∅ t ∅ ∅ ∅ ∅ → cat

All valid! CTC sums probability over ALL valid paths.
```

### CTC Loss Computation

```python
def ctc_loss_explained(log_probs, targets, input_lengths, target_lengths):
    """
    CTC loss using dynamic programming.
    
    Args:
        log_probs: (T, batch, vocab_size) - network output
        targets: (batch, S) - target sequences
        input_lengths: (batch,) - length of each input
        target_lengths: (batch,) - length of each target
    
    Returns:
        loss: Negative log probability
    """
    T, batch, vocab = log_probs.shape
    
    total_loss = 0
    
    for b in range(batch):
        # Get this sample's data
        probs = log_probs[:input_lengths[b], b, :]  # (T, vocab)
        target = targets[b, :target_lengths[b]]  # (S,)
        
        # Extended target with blanks: ∅ t₁ ∅ t₂ ∅ ... ∅ tₛ ∅
        ext_target = [BLANK]
        for t in target:
            ext_target.extend([t, BLANK])
        L = len(ext_target)  # 2S + 1
        
        # Forward variables: α[t, s] = P(ext_target[:s+1] | input[:t+1])
        alpha = torch.full((T, L), float('-inf'))
        
        # Initialization (t=0)
        alpha[0, 0] = probs[0, BLANK]
        alpha[0, 1] = probs[0, ext_target[1]]
        
        # Forward pass
        for t in range(1, T):
            for s in range(L):
                label = ext_target[s]
                
                # Three possible transitions:
                # 1. Stay in same state
                alpha[t, s] = alpha[t-1, s]
                
                # 2. Move from previous state
                if s > 0:
                    alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t-1, s-1])
                
                # 3. Skip blank (if current is not blank and different from s-2)
                if s > 1 and ext_target[s] != BLANK and ext_target[s] != ext_target[s-2]:
                    alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t-1, s-2])
                
                # Add emission probability
                alpha[t, s] = alpha[t, s] + probs[t, label]
        
        # Total probability: sum of last two states
        log_prob = torch.logaddexp(alpha[-1, -1], alpha[-1, -2])
        total_loss += -log_prob
    
    return total_loss / batch
```

### CTC Decoding

```python
def ctc_greedy_decode(log_probs):
    """
    Greedy CTC decoding (fastest, not optimal).
    
    Take argmax at each timestep, then collapse.
    """
    # Get most likely token at each timestep
    predictions = log_probs.argmax(dim=-1)  # (T,)
    
    # Collapse: remove blanks and consecutive duplicates
    output = []
    prev = None
    for pred in predictions:
        if pred != BLANK and pred != prev:
            output.append(pred)
        prev = pred
    
    return output


def ctc_beam_search(log_probs, beam_width=10, lm=None):
    """
    Beam search CTC decoding with optional language model.
    
    Maintains top-k hypotheses at each timestep.
    """
    T, vocab_size = log_probs.shape
    
    # Initialize beam: (prefix, prob)
    beam = [("", 0.0)]
    
    for t in range(T):
        candidates = []
        
        for prefix, prefix_prob in beam:
            # Extend with each possible token
            for token in range(vocab_size):
                new_prob = prefix_prob + log_probs[t, token]
                
                if token == BLANK:
                    # Blank: don't extend prefix
                    new_prefix = prefix
                elif len(prefix) > 0 and token == prefix[-1]:
                    # Repeat: don't extend (CTC collapse)
                    new_prefix = prefix
                else:
                    # New character
                    new_prefix = prefix + chr(token)
                
                # Apply language model if available
                if lm is not None:
                    lm_score = lm.score(new_prefix)
                    new_prob += lm_score
                
                candidates.append((new_prefix, new_prob))
        
        # Keep top-k
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Return best hypothesis
    return beam[0][0]
```

### CTC Advantages

```
✓ MONOTONIC ALIGNMENT
  - Natural for speech (left-to-right)
  - Efficient decoding
  - Streaming-friendly

✓ SIMPLE TRAINING
  - Single loss function
  - No teacher forcing
  - Stable convergence

✓ NO EXPOSURE BIAS
  - Doesn't depend on previous predictions
  - More robust at inference

✓ PARALLELIZABLE
  - All frames processed independently
  - Fast training
```

### CTC Limitations

```
✗ CONDITIONAL INDEPENDENCE
  - Assumes frames independent given alignment
  - Ignores output dependencies
  - Weak language modeling

✗ MONOTONIC ONLY
  - Cannot handle non-monotonic alignments
  - Not suitable for translation

✗ REQUIRES EXTERNAL LM
  - No implicit language model
  - Need separate LM for best results
  - Two-stage optimization
```

---

## Attention-Based Seq2Seq

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              ATTENTION-BASED ENCODER-DECODER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ENCODER (processes entire input)                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Audio features → Transformer/LSTM → Hidden states       │   │
│  │  h₁, h₂, h₃, ..., hₜ                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           │                                      │
│  DECODER (generates output autoregressively)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Step 1: <start> → Attention(h₁...hₜ) → "the"           │   │
│  │  Step 2: "the" → Attention(h₁...hₜ) → "cat"             │   │
│  │  Step 3: "cat" → Attention(h₁...hₜ) → "sat"             │   │
│  │  ...                                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Attention weights show alignment (soft, learned)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Attention Mechanism

```python
class BahdanauAttention(nn.Module):
    """
    Additive attention (original seq2seq).
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: (batch, time, encoder_dim)
            decoder_hidden: (batch, decoder_dim)
        
        Returns:
            context: (batch, encoder_dim)
            attention_weights: (batch, time)
        """
        # Project encoder and decoder
        enc_proj = self.encoder_proj(encoder_outputs)  # (B, T, A)
        dec_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (B, 1, A)
        
        # Additive attention scores
        energy = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, T)
        
        # Attention weights
        attention_weights = F.softmax(energy, dim=1)
        
        # Context vector (weighted sum)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (B, encoder_dim)
        
        return context, attention_weights


class DotProductAttention(nn.Module):
    """
    Scaled dot-product attention (Transformer-style).
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
    
    def forward(self, query, keys, values):
        """
        Args:
            query: (batch, query_dim)
            keys: (batch, time, key_dim)
            values: (batch, time, value_dim)
        """
        # Attention scores
        scores = torch.bmm(
            query.unsqueeze(1),
            keys.transpose(1, 2)
        ).squeeze(1) * self.scale  # (batch, time)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            values
        ).squeeze(1)
        
        return context, attention_weights
```

### Training with Teacher Forcing

```python
def train_attention_asr(model, audio, text, teacher_forcing_ratio=1.0):
    """
    Train attention-based ASR with teacher forcing.
    
    Teacher forcing: Use ground truth as decoder input
    """
    # Encode audio
    encoder_outputs = model.encoder(audio)
    
    # Initialize decoder
    decoder_input = torch.tensor([SOS_TOKEN])  # Start of sequence
    decoder_hidden = model.init_decoder_hidden()
    
    loss = 0
    
    # Generate each output token
    for t in range(len(text)):
        # Decoder step
        output, decoder_hidden, attention = model.decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        
        # Loss
        loss += F.cross_entropy(output, text[t])
        
        # Teacher forcing: use ground truth as next input
        if torch.rand(1) < teacher_forcing_ratio:
            decoder_input = text[t]
        else:
            decoder_input = output.argmax()
    
    return loss / len(text)
```

### Attention Advantages

```
✓ FLEXIBLE ALIGNMENT
  - Can attend to any position
  - Handles non-monotonic cases
  - Better for translation

✓ IMPLICIT LANGUAGE MODEL
  - Decoder learns language structure
  - No separate LM needed
  - Better output coherence

✓ BETTER PERFORMANCE
  - Generally lower WER than CTC
  - Especially on complex tasks
  - State-of-the-art results

✓ INTERPRETABLE
  - Attention weights show alignment
  - Useful for debugging
  - Visualization helps
```

### Attention Limitations

```
✗ EXPOSURE BIAS
  - Training: sees ground truth
  - Inference: sees own predictions
  - Mismatch can cause errors

✗ SLOW INFERENCE
  - Autoregressive decoding
  - Cannot parallelize output
  - Slower than CTC

✗ NOT STREAMING-FRIENDLY
  - Needs full encoder output
  - Decoder is sequential
  - High latency

✗ ATTENTION COLLAPSE
  - Can attend to wrong positions
  - Especially at beginning of training
  - Requires careful initialization
```

---

## Hybrid CTC-Attention

### Best of Both Worlds

```
Combine CTC and attention in single model:

┌─────────────────────────────────────────────────────────────────┐
│                  HYBRID CTC-ATTENTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio → Encoder → Hidden states                                │
│                      │         │                                 │
│                      │         └──────────────┐                  │
│                      │                        │                  │
│                      ▼                        ▼                  │
│              ┌──────────────┐      ┌──────────────┐             │
│              │  CTC Head    │      │  Attention   │             │
│              │              │      │  Decoder     │             │
│              └──────┬───────┘      └──────┬───────┘             │
│                     │                     │                      │
│                     ▼                     ▼                      │
│              CTC Loss (L_ctc)    Attention Loss (L_att)         │
│                                                                  │
│  Total Loss: λ × L_ctc + (1-λ) × L_att                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class HybridCTCAttentionASR(nn.Module):
    """
    Hybrid model with both CTC and attention.
    """
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        
        # Shared encoder
        self.encoder = ConformerEncoder(d_model=d_model)
        
        # CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)
        
        # Attention decoder
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            d_model=d_model
        )
    
    def forward(self, audio, text=None, ctc_weight=0.3):
        """
        Args:
            audio: Input audio features
            text: Target text (for training)
            ctc_weight: Weight for CTC loss (0.3 typical)
        """
        # Encode
        encoder_out = self.encoder(audio)
        
        # CTC branch
        ctc_logits = self.ctc_head(encoder_out)
        
        if text is not None:
            # Training: compute both losses
            ctc_loss = F.ctc_loss(
                ctc_logits.log_softmax(dim=-1).transpose(0, 1),
                text,
                input_lengths=torch.full((audio.shape[0],), encoder_out.shape[1]),
                target_lengths=torch.full((text.shape[0],), text.shape[1])
            )
            
            # Attention branch
            att_logits = self.decoder(text, encoder_out)
            att_loss = F.cross_entropy(
                att_logits.view(-1, att_logits.shape[-1]),
                text.view(-1)
            )
            
            # Combined loss
            total_loss = ctc_weight * ctc_loss + (1 - ctc_weight) * att_loss
            
            return total_loss, ctc_logits, att_logits
        else:
            # Inference: use CTC to guide attention
            return ctc_logits
```

### Joint Decoding

```python
def joint_ctc_attention_decode(model, audio, beam_width=10, ctc_weight=0.3):
    """
    Beam search combining CTC and attention scores.
    """
    # Encode
    encoder_out = model.encoder(audio)
    ctc_logits = model.ctc_head(encoder_out)
    ctc_probs = F.softmax(ctc_logits, dim=-1)
    
    # Initialize beam
    beam = [Hypothesis(tokens=[], score=0.0)]
    
    for step in range(max_length):
        candidates = []
        
        for hyp in beam:
            # Attention score
            att_logits = model.decoder.step(hyp.tokens, encoder_out)
            att_probs = F.softmax(att_logits, dim=-1)
            
            # CTC score (prefix probability)
            ctc_score = compute_ctc_prefix_score(hyp.tokens, ctc_probs)
            
            # Combine scores
            for token in range(vocab_size):
                att_score = torch.log(att_probs[token])
                combined_score = (
                    ctc_weight * ctc_score +
                    (1 - ctc_weight) * att_score
                )
                
                new_hyp = Hypothesis(
                    tokens=hyp.tokens + [token],
                    score=hyp.score + combined_score
                )
                candidates.append(new_hyp)
        
        # Keep top-k
        beam = sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_width]
    
    return beam[0].tokens
```

### Benefits of Hybrid

```
✓ CTC provides alignment guidance
  - Helps attention focus on correct positions
  - Reduces attention collapse
  - Faster convergence

✓ Attention provides language modeling
  - Better output coherence
  - Handles context dependencies
  - Lower WER

✓ Robust decoding
  - CTC score prevents attention errors
  - More reliable than either alone
  - Production-ready

Used in: ESPnet, SpeechBrain, many production systems
```

---

## RNN-T (Transducer)

### Streaming-Friendly Alternative

```
RNN-Transducer (RNN-T):
├── Combines CTC-like alignment with attention-like modeling
├── Streaming-friendly (online decoding)
├── Used in Google Assistant, many production systems

Architecture:
┌────────────────────────────────────────┐
│  Audio Encoder (processes input)       │
│         │                               │
│         ▼                               │
│  Prediction Network (language model)   │
│         │                               │
│         ▼                               │
│  Joint Network (combines both)         │
│         │                               │
│         ▼                               │
│  Output distribution                    │
└────────────────────────────────────────┘
```

### RNN-T Advantages

```
✓ STREAMING
  - Online decoding possible
  - Low latency
  - Suitable for real-time

✓ IMPLICIT LM
  - Prediction network models language
  - No separate LM needed
  - Better than CTC

✓ FLEXIBLE ALIGNMENT
  - Can emit multiple tokens per frame
  - Or no tokens (blank)
  - More flexible than CTC
```

---

## Comparative Analysis

### Performance Comparison

```
On LibriSpeech test-clean:

Model Type          | WER   | RTF  | Streaming
--------------------|-------|------|----------
CTC (Conformer)     | 2.1%  | 0.05 | Yes
Attention (Transformer) | 1.8% | 0.15 | No
Hybrid CTC-Attention | 1.7% | 0.12 | Partial
RNN-T               | 1.9%  | 0.08 | Yes
Whisper Large-v3    | 1.4%  | 0.20 | No

Trends:
- Attention: Best quality, not streaming
- CTC: Fast, streaming, needs LM
- Hybrid: Good balance
- RNN-T: Production choice for streaming
```

### Training Stability

```
Most stable → Least stable:

1. CTC
   - Simple loss
   - No teacher forcing
   - Robust

2. Hybrid CTC-Attention
   - CTC guides attention
   - More stable than pure attention
   - Good convergence

3. RNN-T
   - More complex loss
   - Requires careful tuning
   - Can be unstable

4. Pure Attention
   - Exposure bias
   - Attention collapse possible
   - Needs tricks (label smoothing, etc.)
```

---

## Implementation Details

### Complete CTC Model

```python
class CTCModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=12):
        super().__init__()
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=num_layers
        )
        
        # CTC head
        self.ctc_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, audio_features):
        # Encode
        hidden = self.encoder(audio_features)
        
        # CTC logits
        logits = self.ctc_head(hidden)
        
        return logits
    
    def decode(self, audio_features, beam_width=1):
        logits = self.forward(audio_features)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if beam_width == 1:
            return ctc_greedy_decode(log_probs)
        else:
            return ctc_beam_search(log_probs, beam_width)
```

### Complete Attention Model

```python
class AttentionASR(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        
        # Encoder
        self.encoder = ConformerEncoder(d_model=d_model)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8),
            num_layers=6
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, audio_features, text):
        # Encode
        memory = self.encoder(audio_features)
        
        # Decode
        text_emb = self.decoder_embedding(text)
        decoder_out = self.decoder(text_emb, memory)
        logits = self.output_proj(decoder_out)
        
        return logits
    
    def generate(self, audio_features, max_length=100):
        memory = self.encoder(audio_features)
        
        tokens = [SOS_TOKEN]
        for _ in range(max_length):
            text_emb = self.decoder_embedding(torch.tensor(tokens))
            decoder_out = self.decoder(text_emb, memory)
            logits = self.output_proj(decoder_out[-1])
            
            next_token = logits.argmax().item()
            tokens.append(next_token)
            
            if next_token == EOS_TOKEN:
                break
        
        return tokens
```

---

## When to Use What

### Decision Matrix

```
Use CTC when:
├── Need streaming/low latency
├── Simple deployment
├── Have good external LM
└── Computational constraints

Use Attention when:
├── Quality is priority
├── Offline processing OK
├── Complex linguistic phenomena
└── Have GPU resources

Use Hybrid when:
├── Want best of both
├── Production deployment
├── Balanced requirements
└── Following best practices (ESPnet, SpeechBrain)

Use RNN-T when:
├── Streaming is critical
├── Need implicit LM
├── Production at scale (Google, etc.)
└── Can handle training complexity
```

---

## Key Takeaways

```
1. CTC: Simple, fast, streaming-friendly
   - Monotonic alignment
   - Needs external LM
   - Good for production

2. ATTENTION: Flexible, high-quality
   - Learns alignment
   - Implicit LM
   - Not streaming

3. HYBRID: Best balance
   - Combines advantages
   - Most popular in research
   - ESPnet default

4. RNN-T: Production streaming
   - Used by Google, others
   - Complex but effective
   - Industry standard for streaming

5. MODERN TREND: Attention-based
   - Whisper uses attention
   - Better with large data
   - Hybrid for production
```

---

## Further Reading

- `02_whisper_architecture.md` - Attention-based SOTA
- `../01-foundations/00_asr_history_hmm_to_transformers.md` - Historical context
- ESPnet documentation for hybrid models
