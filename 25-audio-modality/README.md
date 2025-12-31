# Audio Modality: Complete Learning Path

A comprehensive curriculum for understanding audio processing, neural audio codecs, speech recognition, and audio language models. Built from foundational signal processing to cutting-edge research (Step Audio 2, Moshi, Mimi).

## Curriculum Philosophy

- **Core Depth Over Abstractions**: Every concept includes profiled implementations
- **Research-Grounded**: Each module references seminal papers and latest research
- **Practical Focus**: Runnable code, benchmarks, and production considerations

---

## Module Overview (30 Files Created)

```
25-audio-modality/
├── 01-foundations/                    # Core audio concepts (6 files)
│   ├── 00_asr_history_hmm_to_transformers.md  # HMM → RNN → Attention evolution
│   ├── 01_signal_processing_fundamentals.md
│   ├── 02_spectral_analysis_profiled.py       # Benchmarked STFT implementations
│   ├── 03_mel_spectrograms_deep_dive.py       # Mel scale theory + profiling
│   ├── 04_audio_fft_cuda.cu                   # GPU-accelerated FFT in CUDA
│   └── 05_audio_fundamentals.c                # Pure C implementations
│
├── 02-neural-audio-generation/        # Generative models (2 files)
│   ├── 01_wavenet_architecture.md             # Dilated causal convolutions
│   └── 02_hifigan_melgan_vocoders.md          # GAN-based real-time vocoders
│
├── 03-neural-audio-codecs/            # Core codec architectures (4 files)
│   ├── 01_vq_vae_fundamentals.md              # VQ-VAE for audio
│   ├── 02_residual_vector_quantization.md     # RVQ deep dive
│   ├── 03_soundstream_encodec_architecture.md # Architecture comparison
│   └── 05_mimi_codec.md                       # Semantic token codec
│
├── 04-speech-representations/         # Self-supervised speech (1 file)
│   └── 02_wavlm_architecture.md        # MUST READ - WavLM deep dive
│
├── 05-speech-recognition/             # ASR systems (1 file)
│   └── 02_whisper_architecture.md             # Whisper architecture
│
├── 06-audio-language-models/          # Speech + LLM integration (3 files)
│   ├── 02_moshi_architecture.md               # Full-duplex speech LLM
│   ├── 03_step_audio_analysis.md              # Step Audio 2 analysis
│   └── 04_multimodal_audio_vision_text.md     # Multimodal integration
│
├── 07-real-time-streaming/            # Low-latency systems (1 file)
│   └── 01_streaming_constraints.md            # Latency budgets, causal arch
│
├── 08-optimization-profiling/         # Performance engineering (3 files)
│   ├── 01_nvidia_dali_audio.md                # DALI overview
│   ├── 02_dali_audio_pipeline.py              # Complete DALI implementation
│   └── 03_ffcv_audio_loader.py                # FFCV for audio datasets
│
├── 09-codec-benchmarks/               # Comparative analysis (1 file)
│   └── 02_soundstream_vs_encodec_vs_mimi.py   # Codec comparison benchmarks
│
├── 10-practical-notebooks/            # Hands-on experiments (3 files)
│   ├── 01_audio_preprocessing_pipeline.ipynb  # Complete preprocessing
│   ├── 02_neural_codec_from_scratch.ipynb     # Build codec step-by-step
│   └── 03_exercises_and_solutions.py          # 7 graded exercises
│
├── papers/                            # Reference materials (1 file)
│   └── paper_summaries.md                     # All 12 papers summarized
│
├── resources/                         # Learning resources (2 files)
│   ├── glossary.md                            # 80+ terms defined
│   └── external_links.md                      # Datasets, tools, community
│
└── README.md                          # This file
```

---

## Learning Progression

### Phase 1: Foundations (Week 1-2)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 01 | Signal Processing & Spectrograms | - |
| 01 | Audio Loading & Preprocessing | - |
| 01 | Mel Frequency Analysis | - |

### Phase 2: Generative Audio (Week 3-4)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 02 | WaveNet & Dilated Convolutions | [WaveNet 2016](https://arxiv.org/abs/1609.03499) |
| 02 | Parallel WaveNet | [Parallel WaveNet 2017](https://arxiv.org/abs/1711.10433) |
| 02 | GAN-based Vocoders | [MelGAN 2019](https://arxiv.org/abs/1910.06711), [HiFi-GAN 2020](https://arxiv.org/abs/2010.05646) |

### Phase 3: Neural Audio Codecs (Week 5-7)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 03 | VQ-VAE & Quantization | [VQ-VAE 2017](https://arxiv.org/abs/1711.00937) |
| 03 | Residual Vector Quantization | [SoundStream 2021](https://arxiv.org/abs/2107.03312) |
| 03 | EnCodec Architecture | [EnCodec 2022](https://arxiv.org/abs/2210.13438) |
| 03 | Mimi & Semantic Tokens | [Moshi 2024](https://arxiv.org/abs/2410.00037) |

### Phase 4: Speech Representations (Week 8-9)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 04 | Self-Supervised Speech | [wav2vec 2.0](https://arxiv.org/abs/2006.11477) |
| 04 | **WavLM (MUST READ)** | [WavLM 2021](https://arxiv.org/abs/2110.13900) |
| 04 | Semantic vs Acoustic Tokens | - |

### Phase 5: Speech Recognition (Week 10-11)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 05 | ASR Fundamentals | - |
| 05 | Whisper Architecture | [Whisper 2022](https://arxiv.org/abs/2212.04356) |
| 05 | CTC vs Attention | - |

### Phase 6: Audio Language Models (Week 12-14)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 06 | Audio Tokenization for LLMs | [AudioLM 2022](https://arxiv.org/abs/2209.03143) |
| 06 | Moshi: Full-Duplex Dialogue | [Moshi 2024](https://arxiv.org/abs/2410.00037) |
| 06 | Step Audio Systems | [Step Audio 2](https://arxiv.org/pdf/2507.16632), [Step Audio R1](https://arxiv.org/pdf/2511.15848) |
| 06 | Continuous Audio LMs | [CALM 2025](https://arxiv.org/pdf/2509.06926) |

### Phase 7: Optimization (Week 15-16)
| Module | Topic | Resources |
|--------|-------|-----------|
| 07 | Real-time Streaming | - |
| 08 | NVIDIA DALI | [DALI Audio](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/audio_processing/index.html) |
| 09 | Codec Benchmarks | - |

---

## Key Research Papers

### Foundational (2016-2020)
1. **WaveNet** (2016) - First autoregressive raw audio model with dilated causal convolutions
2. **Parallel WaveNet** (2017) - Knowledge distillation for real-time synthesis
3. **MelGAN** (2019) - GAN-based vocoder, no autoregression
4. **HiFi-GAN** (2020) - Multi-period/multi-scale discriminators

### Neural Codecs (2021-2022)
5. **SoundStream** (2021) - Introduced RVQ for neural audio, end-to-end training
6. **EnCodec** (2022) - Open-source Meta codec, STFT discriminator
7. **WavLM** (2021) - **MUST READ** - Universal speech representations

### Audio LLMs (2022-2025)
8. **AudioLM** (2022) - First audio language model from Google
9. **Whisper** (2022) - Robust ASR from OpenAI
10. **Moshi** (2024) - First full-duplex speech LLM
11. **Step Audio 2** (2025) - Latest audio generation model
12. **CALM** (2025) - Continuous Audio Language Models (no discrete tokens)

---

## Profiling Focus Areas

### Memory & Bandwidth
- Audio sample rates: 16kHz vs 24kHz vs 44.1kHz memory impact
- Spectrogram storage: float32 vs float16 vs int8
- Batch processing vs streaming memory patterns

### Computation
- FFT/STFT complexity: O(n log n) practical implications
- Resampling costs: polyphase vs linear interpolation
- Encoder/decoder inference times per codec

### Codec Comparisons
| Codec | Bitrate | Latency | Semantic | Use Case |
|-------|---------|---------|----------|----------|
| SoundStream | 3-18 kbps | ~20ms | No | General audio |
| EnCodec | 1.5-24 kbps | 13ms | No | Music, speech |
| Mimi | 1.1 kbps | 80ms | Yes | Speech LLMs |

### GPU Acceleration
- NVIDIA DALI: 10-100x speedup for data loading
- torchaudio GPU backend
- CUDA kernel fusion opportunities

---

## Prerequisites

1. **Audio Fundamentals** (already in `07-multimodal-data-formats/03-audio-formats/`)
2. **Deep Learning**: Transformers, CNNs, GANs
3. **PyTorch**: Intermediate level
4. **DSP Basics**: Fourier transforms, filtering

---

## Quick Start

```bash
# Setup environment
pip install torch torchaudio transformers datasets soundfile librosa
pip install nvidia-dali-cuda120  # For GPU acceleration

# Clone reference implementations
git clone https://github.com/facebookresearch/encodec
git clone https://github.com/kyutai-labs/moshi
```

---

## Status Tracker

| Module | Status | Last Updated |
|--------|--------|--------------|
| 01-foundations | 🟢 Complete | Dec 2024 |
| 02-neural-audio-generation | 🟢 Complete | Dec 2024 |
| 03-neural-audio-codecs | 🟢 Complete | Dec 2024 |
| 04-speech-representations | 🟢 Complete | Dec 2024 |
| 05-speech-recognition | 🟢 Complete | Dec 2024 |
| 06-audio-language-models | 🟢 Complete | Dec 2024 |
| 07-real-time-streaming | 🟢 Complete | Dec 2024 |
| 08-optimization-profiling | 🟢 Complete | Dec 2024 |
| 09-codec-benchmarks | 🟢 Complete | Dec 2024 |
| 10-practical-notebooks | 🟢 Complete | Dec 2024 |
