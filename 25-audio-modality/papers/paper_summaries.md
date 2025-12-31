# Audio Research Paper Summaries

Quick reference summaries for all key papers in the audio modality curriculum.

## Foundational Papers (2016-2020)

### WaveNet (2016)
**Paper**: [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
**Authors**: van den Oord et al. (DeepMind)

**Key Contributions**:
- First autoregressive model for raw audio waveforms
- Dilated causal convolutions for large receptive fields
- Gated activation units (tanh ⊙ sigmoid)
- μ-law quantization to 256 levels

**Architecture**: Stack of dilated conv layers with residual/skip connections
**Limitation**: Very slow (90 min to generate 1 sec)
**Impact**: Foundation for all neural audio generation

---

### Parallel WaveNet (2017)
**Paper**: [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/abs/1711.10433)
**Authors**: van den Oord et al. (DeepMind)

**Key Contributions**:
- Knowledge distillation from autoregressive WaveNet
- Inverse autoregressive flow (IAF) for parallel generation
- 1000x speedup over original WaveNet
- Real-time synthesis achieved

**Method**: Train student (parallel) to match teacher (autoregressive)
**Limitation**: Complex training, requires trained WaveNet teacher

---

### MelGAN (2019)
**Paper**: [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
**Authors**: Kumar et al.

**Key Contributions**:
- GAN-based vocoder (no autoregression)
- Multi-scale discriminator
- Feature matching loss
- 100x faster than WaveNet

**Architecture**: Fully convolutional generator + multi-scale discriminator
**Quality**: Slightly lower than WaveNet but much faster

---

### HiFi-GAN (2020)
**Paper**: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
**Authors**: Kong et al.

**Key Contributions**:
- Multi-period discriminator (MPD)
- Multi-scale discriminator (MSD)
- Multi-receptive field fusion (MRF)
- State-of-the-art quality + real-time speed

**Architecture**: Generator with MRF + MPD + MSD
**Impact**: Became default vocoder for TTS systems

---

## Neural Audio Codecs (2021-2022)

### SoundStream (2021)
**Paper**: [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)
**Authors**: Zeghidour et al. (Google)

**Key Contributions**:
- First end-to-end neural audio codec
- Residual Vector Quantization (RVQ) for audio
- Variable bitrate (3-18 kbps)
- Single model for speech, music, general audio

**Architecture**: Encoder → RVQ → Decoder with adversarial training
**Bitrate**: 75 Hz frame rate, 1024-size codebooks
**Impact**: Foundation for EnCodec, Mimi, AudioLM

---

### EnCodec (2022)
**Paper**: [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
**Authors**: Défossez et al. (Meta)

**Key Contributions**:
- Open-source neural audio codec
- Multi-scale STFT discriminator
- Balancer for loss weighting
- Streamable architecture
- Stereo support

**Architecture**: Similar to SoundStream + LSTM + better discriminator
**Bitrate**: 1.5-24 kbps configurable
**Code**: github.com/facebookresearch/encodec

---

### WavLM (2021) ⭐ MUST READ
**Paper**: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
**Authors**: Chen et al. (Microsoft)

**Key Contributions**:
- Universal speech representations
- Denoising pre-training (handles noisy audio)
- Gated relative position bias
- SOTA on ALL SUPERB tasks

**Architecture**: CNN encoder + 24-layer Transformer
**Training**: Masked prediction + denoising on 94k hours
**Impact**: Used in Mimi for semantic token distillation

---

## Speech Recognition (2022)

### Whisper (2022)
**Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
**Authors**: Radford et al. (OpenAI)

**Key Contributions**:
- Trained on 680k hours of labeled audio
- Multilingual (99 languages)
- Robust to noise, accents, domains
- Zero-shot performance rivals supervised models

**Architecture**: Encoder-decoder Transformer
**Input**: 80-bin log mel spectrogram
**Sizes**: Tiny (39M) to Large-v3 (1.5B)

---

## Audio Language Models (2024-2025)

### Moshi (2024)
**Paper**: [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
**Authors**: Défossez et al. (Kyutai)

**Key Contributions**:
- First full-duplex speech LLM
- Multi-stream architecture (user audio, system audio, text)
- Inner monologue for reasoning
- Mimi codec with semantic tokens
- Real-time capable (~200ms latency)

**Architecture**: Helium (7B LLM) + Depth Transformer + Mimi
**Innovation**: Simultaneous listen/speak like humans
**Code**: github.com/kyutai-labs/moshi

---

### Step Audio 2 (2025)
**Paper**: [Step Audio 2 Technical Report](https://arxiv.org/pdf/2507.16632)
**Authors**: Step AI

**Key Contributions**:
- Large-scale audio language model
- Improved audio understanding
- Better instruction following
- Multi-turn dialogue

**Focus**: Production-ready audio AI system

---

### CALM (Continuous Audio Language Models) (2025)
**Paper**: [Continuous Audio Language Models](https://arxiv.org/pdf/2509.06926)
**Authors**: Rouard et al. (Kyutai)

**Key Contributions**:
- Bypasses discrete tokenization entirely
- Uses diffusion/flow for continuous generation
- No VQ/RVQ needed
- Potentially better quality

**Innovation**: Alternative to discrete token approach
**Status**: Research direction, not yet widely adopted

---

## Quick Reference Table

| Paper | Year | Key Innovation | Use For |
|-------|------|----------------|---------|
| WaveNet | 2016 | Dilated causal conv | Understanding foundations |
| Parallel WaveNet | 2017 | Knowledge distillation | Real-time synthesis |
| MelGAN | 2019 | GAN vocoder | Fast TTS |
| HiFi-GAN | 2020 | Multi-period discriminator | High-quality TTS |
| SoundStream | 2021 | RVQ for audio | Neural codec design |
| WavLM | 2021 | Denoising SSL | Speech representations |
| EnCodec | 2022 | Open-source codec | Audio compression |
| Whisper | 2022 | Large-scale ASR | Speech recognition |
| Moshi | 2024 | Full-duplex dialogue | Speech LLM |
| CALM | 2025 | Continuous generation | Future research |

---

## Reading Order Recommendation

### Beginner Path
1. WaveNet (understand autoregressive audio)
2. HiFi-GAN (understand GAN vocoders)
3. Whisper (understand ASR)
4. EnCodec (understand neural codecs)

### Advanced Path
1. WavLM ⭐ (self-supervised speech)
2. SoundStream (RVQ details)
3. Moshi (speech LLM architecture)
4. CALM (future directions)

### Implementation Path
1. EnCodec (has open-source code)
2. HiFi-GAN (simple to implement)
3. Whisper (easy to use)
4. Moshi (full system)
