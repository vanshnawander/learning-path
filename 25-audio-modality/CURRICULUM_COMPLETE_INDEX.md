# Audio Modality Curriculum - Complete Index

**Total Files Created: 40+**
**Coverage: Foundations → Latest Research (Dec 2025)**
**Includes: Theory, C/CUDA, Python, Notebooks, Exercises**

---

## 📚 Complete File Listing by Module

### 01-foundations/ (6 files)
- `00_asr_history_hmm_to_transformers.md` - HMM → RNN → Attention → Transformers evolution
- `01_signal_processing_fundamentals.md` - Sampling, FFT, STFT, windowing
- `02_spectral_analysis_profiled.py` - Benchmarked implementations (NumPy, Librosa, PyTorch)
- `03_mel_spectrograms_deep_dive.py` - Mel scale theory, filterbank construction, profiling
- `04_audio_fft_cuda.cu` - **CUDA implementation** with cuFFT, mel filterbank kernels
- `05_audio_fundamentals.c` - **Pure C** DFT/FFT, window functions, MFCC extraction

### 02-neural-audio-generation/ (2 files)
- `01_wavenet_architecture.md` - Dilated causal convolutions, gated activations, conditioning
- `02_hifigan_melgan_vocoders.md` - GAN vocoders, MPD/MSD discriminators, MRF blocks

### 03-neural-audio-codecs/ (5 files)
- `01_vq_vae_fundamentals.md` - Vector quantization, straight-through estimator, codebook learning
- `02_residual_vector_quantization.md` - RVQ algorithm, bitrate control, hierarchical info
- `03_soundstream_encodec_architecture.md` - Architecture comparison, discriminators, training
- `04_encodec_implementation.py` - **Complete EnCodec** in PyTorch with Snake activation
- `05_mimi_codec.md` - Semantic token distillation, extreme compression (1.1 kbps)

### 04-speech-representations/ (2 files)
- `01_wav2vec_evolution.md` - wav2vec → vq-wav2vec → wav2vec 2.0 → HuBERT → WavLM
- `02_wavlm_architecture.md` - **MUST READ** - Denoising pretraining, gated position bias

### 05-speech-recognition/ (3 files)
- `01_asr_fundamentals.md` - Traditional vs neural ASR, evaluation metrics, SOTA 2024-2025
- `02_whisper_architecture.md` - Encoder-decoder, multitask format, training data
- `04_ctc_vs_attention.md` - CTC vs attention comparison, hybrid models, RNN-T

### 06-audio-language-models/ (5 files)
- `01_audio_tokenization_for_llms.md` - Discrete tokenization, semantic vs acoustic, latest research
- `02_moshi_architecture.md` - Full-duplex dialogue, multi-stream, Helium LLM, depth transformer
- `03_step_audio_analysis.md` - Step Audio 2 & R1 analysis, comparison with Moshi
- `04_multimodal_audio_vision_text.md` - Multimodal integration, fusion architectures, GPT-4o
- `05_soundstorm_parallel_generation.md` - MaskGIT for audio, 100x speedup, confidence-based decoding

### 07-real-time-streaming/ (3 files)
- `01_streaming_constraints.md` - Latency budgets, sources of latency, human perception thresholds
- `02_causal_architectures.md` - Causal conv/attention, KV cache, streaming state management
- `03_buffering_strategies.md` - Ring buffers, double buffering, adaptive buffering, jitter management

### 08-optimization-profiling/ (3 files)
- `01_nvidia_dali_audio.md` - DALI overview, 10-100x speedup, GPU acceleration
- `02_dali_audio_pipeline.py` - **Complete DALI implementation** with benchmarks
- `03_ffcv_audio_loader.py` - **FFCV for audio**, custom fields, performance comparison

### 09-codec-benchmarks/ (1 file)
- `02_soundstream_vs_encodec_vs_mimi.py` - Comprehensive codec comparison with profiling

### 10-practical-notebooks/ (5 files)
- `01_audio_preprocessing_pipeline.ipynb` - Loading, resampling, spectrogram extraction
- `02_neural_codec_from_scratch.ipynb` - Build encoder + RVQ + decoder step-by-step
- `03_exercises_and_solutions.py` - **7 graded exercises** (STFT, VQ, causal conv, attention)
- `04_whisper_finetuning_complete.ipynb` - End-to-end Whisper fine-tuning
- `05_complete_audio_llm_inference.ipynb` - Multi-stream tokenization, profiling

### 11-advanced-topics/ (3 files) **NEW**
- `01_music_generation_musicgen.md` - MusicGen, MAGNeT, delay pattern, AudioCraft ecosystem
- `02_audio_deepfake_detection.md` - Detection methods, datasets, latest research 2024-2025
- `03_valle_natural_speech_synthesis.md` - VALL-E, zero-shot voice cloning, NaturalSpeech 2

### papers/ (1 file)
- `paper_summaries.md` - All 12+ papers summarized with reading order recommendations

### resources/ (2 files)
- `glossary.md` - 80+ audio ML terms defined
- `external_links.md` - Datasets, tools, models, community resources

---

## 🎯 Learning Paths

### Beginner Path (4-6 weeks)
1. `01-foundations/01_signal_processing_fundamentals.md`
2. `01-foundations/00_asr_history_hmm_to_transformers.md`
3. `02-neural-audio-generation/01_wavenet_architecture.md`
4. `03-neural-audio-codecs/01_vq_vae_fundamentals.md`
5. `05-speech-recognition/01_asr_fundamentals.md`
6. **Practice**: `10-practical-notebooks/03_exercises_and_solutions.py`

### Intermediate Path (6-8 weeks)
1. `03-neural-audio-codecs/02_residual_vector_quantization.md`
2. `03-neural-audio-codecs/03_soundstream_encodec_architecture.md`
3. `04-speech-representations/01_wav2vec_evolution.md`
4. `04-speech-representations/02_wavlm_architecture.md` ⭐ MUST READ
5. `05-speech-recognition/04_ctc_vs_attention.md`
6. **Practice**: `10-practical-notebooks/02_neural_codec_from_scratch.ipynb`

### Advanced Path (8-12 weeks)
1. `03-neural-audio-codecs/05_mimi_codec.md`
2. `06-audio-language-models/01_audio_tokenization_for_llms.md`
3. `06-audio-language-models/02_moshi_architecture.md`
4. `06-audio-language-models/05_soundstorm_parallel_generation.md`
5. `07-real-time-streaming/` - All 3 files
6. `11-advanced-topics/` - All 3 files
7. **Practice**: `10-practical-notebooks/04_whisper_finetuning_complete.ipynb`

### Systems/Performance Path (4-6 weeks)
1. `08-optimization-profiling/01_nvidia_dali_audio.md`
2. `08-optimization-profiling/02_dali_audio_pipeline.py`
3. `08-optimization-profiling/03_ffcv_audio_loader.py`
4. `01-foundations/04_audio_fft_cuda.cu` - CUDA implementation
5. `01-foundations/05_audio_fundamentals.c` - C implementation
6. `09-codec-benchmarks/02_soundstream_vs_encodec_vs_mimi.py`

---

## 💻 Code Implementations

### Low-Level (C/CUDA)
- **C**: `01-foundations/05_audio_fundamentals.c` - FFT, mel filterbank, MFCC
- **CUDA**: `01-foundations/04_audio_fft_cuda.cu` - cuFFT, GPU kernels, profiling

### Python (PyTorch)
- **Profiled**: `01-foundations/02_spectral_analysis_profiled.py`
- **Codec**: `03-neural-audio-codecs/04_encodec_implementation.py`
- **DALI**: `08-optimization-profiling/02_dali_audio_pipeline.py`
- **FFCV**: `08-optimization-profiling/03_ffcv_audio_loader.py`
- **Benchmarks**: `09-codec-benchmarks/02_soundstream_vs_encodec_vs_mimi.py`
- **Exercises**: `10-practical-notebooks/03_exercises_and_solutions.py`

### Jupyter Notebooks
1. `01_audio_preprocessing_pipeline.ipynb` - Complete preprocessing
2. `02_neural_codec_from_scratch.ipynb` - Build codec from scratch
3. `04_whisper_finetuning_complete.ipynb` - Fine-tune Whisper
4. `05_complete_audio_llm_inference.ipynb` - Audio LLM inference

---

## 📊 Latest Research Coverage (2024-2025)

### Papers Covered
- ✅ SoundStorm (Google, 2023) - Parallel generation
- ✅ Moshi (Kyutai, 2024) - Full-duplex speech LLM
- ✅ Mimi (Kyutai, 2024) - Semantic token codec
- ✅ Step Audio 2 (Step AI, 2025) - Latest audio LLM
- ✅ GPT-4o Voice Mode (OpenAI, 2024) - Multimodal analysis
- ✅ VALL-E (Microsoft, 2023) - Zero-shot TTS
- ✅ MusicGen/MAGNeT (Meta, 2023-2024) - Music generation
- ✅ Whisper Large-v3 (OpenAI, 2024) - Latest ASR
- ✅ Streaming Whisper (Bloomberg, 2025) - Real-time ASR
- ✅ Audio Deepfake Detection (2024-2025) - Latest methods

### Web Resources Analyzed
- Kyutai Codec Explainer
- NVIDIA cuSignal documentation
- NVIDIA DALI audio examples
- AudioCraft documentation
- Latest arXiv papers (Dec 2025)

---

## 🔧 Practical Tools Covered

### Data Loading
- ✅ **NVIDIA DALI** - 10-100x speedup, GPU-accelerated
- ✅ **FFCV** - Memory-mapped datasets, fast random access
- ✅ PyTorch DataLoader optimization

### Profiling
- ✅ RTF (Real-Time Factor) calculations
- ✅ GPU profiling with CUDA events
- ✅ Memory bandwidth analysis
- ✅ Latency breakdown tools

### Libraries
- ✅ torchaudio - PyTorch audio
- ✅ librosa - Audio analysis
- ✅ transformers - Pretrained models
- ✅ encodec - Neural codec
- ✅ moshi - Speech LLM

---

## 🎓 Exercises and Hands-On

### Exercises (with solutions)
1. Implement STFT from scratch
2. Build mel filterbank
3. Vector quantization with STE
4. Causal convolution
5. Residual VQ
6. Audio self-attention
7. Mini audio encoder

### Notebooks
1. Audio preprocessing pipeline
2. Neural codec from scratch
3. Whisper fine-tuning
4. Audio LLM inference

### Compile and Run
```bash
# C implementation
gcc -O3 -o audio_fundamentals 01-foundations/05_audio_fundamentals.c -lm
./audio_fundamentals

# CUDA implementation
nvcc -O3 -o audio_fft_cuda 01-foundations/04_audio_fft_cuda.cu -lcufft
./audio_fft_cuda

# Python exercises
python 10-practical-notebooks/03_exercises_and_solutions.py
```

---

## 📈 Coverage Statistics

| Category | Files | Lines of Code | Markdown Pages |
|----------|-------|---------------|----------------|
| Foundations | 6 | 2,500+ | 50+ |
| Neural Generation | 2 | 500+ | 30+ |
| Neural Codecs | 5 | 1,500+ | 60+ |
| Speech Representations | 2 | 200+ | 40+ |
| Speech Recognition | 3 | 300+ | 50+ |
| Audio LLMs | 5 | 400+ | 70+ |
| Real-Time Streaming | 3 | 800+ | 45+ |
| Optimization | 3 | 1,200+ | 25+ |
| Benchmarks | 1 | 400+ | - |
| Notebooks | 5 | 1,500+ | - |
| Advanced Topics | 3 | 200+ | 45+ |
| Resources | 3 | - | 30+ |
| **TOTAL** | **41** | **9,500+** | **445+** |

---

## 🚀 Quick Start

### For Beginners
```bash
# Start here
cd 25-audio-modality
cat README.md

# Read foundations
cat 01-foundations/01_signal_processing_fundamentals.md

# Run exercises
python 10-practical-notebooks/03_exercises_and_solutions.py
```

### For Practitioners
```bash
# Codec comparison
python 09-codec-benchmarks/02_soundstream_vs_encodec_vs_mimi.py

# DALI pipeline
python 08-optimization-profiling/02_dali_audio_pipeline.py

# Build codec
jupyter notebook 10-practical-notebooks/02_neural_codec_from_scratch.ipynb
```

### For Researchers
```bash
# Latest research
cat papers/paper_summaries.md

# Advanced topics
cat 11-advanced-topics/*.md

# Multimodal
cat 06-audio-language-models/04_multimodal_audio_vision_text.md
```

---

## 🎯 Key Highlights

### Unique Content
- ✅ **CUDA audio processing** - Low-level GPU kernels
- ✅ **C implementations** - Understanding fundamentals
- ✅ **DALI pipelines** - Production data loading
- ✅ **FFCV for audio** - Fast dataset loading
- ✅ **Complete codec implementation** - EnCodec from scratch
- ✅ **7 graded exercises** - Hands-on learning
- ✅ **Latest research** - Through December 2025

### Profiling Focus
Every implementation includes:
- ⏱️ Timing measurements
- 📊 RTF calculations
- 💾 Memory analysis
- 🔄 CPU vs GPU comparisons
- 📈 Throughput benchmarks

### Research Coverage
- 📄 12+ papers summarized
- 🔬 Latest developments (Dec 2025)
- 🌐 Web resources analyzed
- 📚 80+ terms in glossary
- 🔗 External links curated

---

## 📖 Recommended Reading Order

### Week 1-2: Foundations
1. Signal processing fundamentals
2. ASR history (HMM to Transformers)
3. Run C/CUDA implementations
4. Complete exercises 1-3

### Week 3-4: Neural Audio
1. WaveNet architecture
2. VQ-VAE fundamentals
3. RVQ deep dive
4. Build codec from scratch (notebook)

### Week 5-6: Speech Understanding
1. wav2vec evolution
2. WavLM architecture ⭐
3. Whisper architecture
4. CTC vs Attention

### Week 7-8: Audio LLMs
1. Audio tokenization for LLMs
2. Moshi architecture
3. SoundStorm parallel generation
4. Multimodal integration

### Week 9-10: Production
1. Real-time streaming constraints
2. Causal architectures
3. DALI/FFCV data loading
4. Codec benchmarks

### Week 11-12: Advanced
1. Music generation (MusicGen)
2. Deepfake detection
3. VALL-E speech synthesis
4. Latest research papers

---

## 🔗 External Resources Integration

All content references:
- ✅ Official papers (arXiv links)
- ✅ GitHub repositories
- ✅ Hugging Face models
- ✅ NVIDIA documentation
- ✅ Blog posts and tutorials
- ✅ Datasets and benchmarks

---

## ✨ What Makes This Curriculum Unique

1. **No Abstractions** - Core depth with profiling at every level
2. **Multi-Language** - Python, C, CUDA implementations
3. **Latest Research** - Through December 2025
4. **Production-Ready** - DALI, FFCV, optimization guides
5. **Hands-On** - Notebooks, exercises, runnable code
6. **Comprehensive** - 40+ files, 9,500+ lines of code
7. **Research-Grounded** - Every claim referenced
8. **Multimodal** - Audio-vision-text integration

---

**Start learning**: `cat README.md`
**Get help**: `cat resources/glossary.md`
**Latest research**: `cat papers/paper_summaries.md`
