# External Resources and Links

Curated collection of resources for audio ML learning.

---

## Official Documentation

### Libraries
- **torchaudio**: [pytorch.org/audio](https://pytorch.org/audio/stable/index.html)
- **librosa**: [librosa.org/doc](https://librosa.org/doc/latest/index.html)
- **NVIDIA DALI**: [docs.nvidia.com/deeplearning/dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)
- **Hugging Face Audio**: [huggingface.co/docs/transformers/tasks/audio_classification](https://huggingface.co/docs/transformers/tasks/audio_classification)

### Models
- **Whisper**: [github.com/openai/whisper](https://github.com/openai/whisper)
- **EnCodec**: [github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- **Moshi**: [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- **WavLM**: [github.com/microsoft/unilm/tree/master/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)

---

## Research Papers (arXiv)

### Foundational
| Paper | Link |
|-------|------|
| WaveNet (2016) | [arxiv.org/abs/1609.03499](https://arxiv.org/abs/1609.03499) |
| Parallel WaveNet (2017) | [arxiv.org/abs/1711.10433](https://arxiv.org/abs/1711.10433) |
| VQ-VAE (2017) | [arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937) |
| MelGAN (2019) | [arxiv.org/abs/1910.06711](https://arxiv.org/abs/1910.06711) |
| HiFi-GAN (2020) | [arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646) |

### Neural Codecs
| Paper | Link |
|-------|------|
| SoundStream (2021) | [arxiv.org/abs/2107.03312](https://arxiv.org/abs/2107.03312) |
| EnCodec (2022) | [arxiv.org/abs/2210.13438](https://arxiv.org/abs/2210.13438) |

### Speech Representations
| Paper | Link |
|-------|------|
| wav2vec 2.0 (2020) | [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477) |
| HuBERT (2021) | [arxiv.org/abs/2106.07447](https://arxiv.org/abs/2106.07447) |
| WavLM (2021) | [arxiv.org/abs/2110.13900](https://arxiv.org/abs/2110.13900) |

### Speech Recognition
| Paper | Link |
|-------|------|
| Whisper (2022) | [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356) |

### Audio Language Models
| Paper | Link |
|-------|------|
| AudioLM (2022) | [arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143) |
| MusicGen (2023) | [arxiv.org/abs/2306.05284](https://arxiv.org/abs/2306.05284) |
| Moshi (2024) | [arxiv.org/abs/2410.00037](https://arxiv.org/abs/2410.00037) |
| CALM (2025) | [arxiv.org/pdf/2509.06926](https://arxiv.org/pdf/2509.06926) |
| Step Audio 2 (2025) | [arxiv.org/pdf/2507.16632](https://arxiv.org/pdf/2507.16632) |
| Step Audio R1 (2025) | [arxiv.org/pdf/2511.15848](https://arxiv.org/pdf/2511.15848) |

---

## Interactive Explainers

- **Kyutai Codec Explainer**: [kyutai.org/codec-explainer](https://kyutai.org/codec-explainer)
  - Excellent visual explanation of neural audio codecs
  - VQ-VAE, RVQ, and Mimi explained

- **WaveNet Blog**: [deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio](https://deepmind.google/discover/blog/wavenet-a-generative-model-for-raw-audio/)
  - Original DeepMind blog post with audio samples

---

## Datasets

### Speech
| Dataset | Size | Use Case | Link |
|---------|------|----------|------|
| LibriSpeech | 1000h | ASR benchmark | [openslr.org/12](https://www.openslr.org/12/) |
| LibriLight | 60kh | Self-supervised | [github.com/facebookresearch/libri-light](https://github.com/facebookresearch/libri-light) |
| VoxPopuli | 400kh | Multilingual | [github.com/facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli) |
| Common Voice | 20k+h | Multilingual | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/) |
| VCTK | 44h | Multi-speaker | [datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) |

### Music
| Dataset | Size | Use Case | Link |
|---------|------|----------|------|
| MusicCaps | 5.5kh | Music captioning | [kaggle.com/datasets/googleai/musiccaps](https://www.kaggle.com/datasets/googleai/musiccaps) |
| FMA | 100kh | Music classification | [github.com/mdeff/fma](https://github.com/mdeff/fma) |
| MUSDB18 | 10h | Source separation | [sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html) |

### General Audio
| Dataset | Size | Use Case | Link |
|---------|------|----------|------|
| AudioSet | 5k+h | Audio tagging | [research.google.com/audioset](https://research.google.com/audioset/) |
| ESC-50 | 2.8h | Environmental | [github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) |

---

## Tutorials and Courses

### Video Courses
- **Hugging Face Audio Course**: [huggingface.co/learn/audio-course](https://huggingface.co/learn/audio-course/chapter0/introduction)
- **Stanford CS224S (Spoken Language Processing)**: [web.stanford.edu/class/cs224s](https://web.stanford.edu/class/cs224s/)

### Blog Posts
- **Sander Dieleman - Generative Modelling in Latent Space**: [sander.ai/2025/04/15/latents.html](https://sander.ai/2025/04/15/latents.html)
- **Audio Loss Functions**: [soundsandwords.io/audio-loss-functions](https://www.soundsandwords.io/audio-loss-functions/)

---

## Hugging Face Models

### Codecs
- EnCodec: [huggingface.co/facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz)
- Mimi: [huggingface.co/kyutai/mimi](https://huggingface.co/kyutai/mimi)

### Speech Recognition
- Whisper: [huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)

### Speech Representations
- WavLM: [huggingface.co/microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- wav2vec 2.0: [huggingface.co/facebook/wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)

### TTS
- SpeechT5: [huggingface.co/microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)
- Bark: [huggingface.co/suno/bark](https://huggingface.co/suno/bark)

---

## Tools and Utilities

### Audio Processing
- **FFmpeg**: [ffmpeg.org](https://ffmpeg.org/) - Universal audio/video tool
- **SoX**: [sox.sourceforge.net](http://sox.sourceforge.net/) - Sound eXchange
- **Audacity**: [audacityteam.org](https://www.audacityteam.org/) - GUI audio editor

### Visualization
- **Weights & Biases Audio**: [docs.wandb.ai/guides/track/log/media#audio](https://docs.wandb.ai/guides/track/log/media#audio)
- **TensorBoard Audio**: [tensorflow.org/tensorboard/audio_tutorial](https://www.tensorflow.org/tensorboard/audio_tutorial)

### Profiling
- **PyTorch Profiler**: [pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- **NVIDIA Nsight**: [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)

---

## Community

### Discord/Slack
- Hugging Face Discord: Audio channels
- EleutherAI Discord: Audio research discussions

### Twitter/X Accounts to Follow
- @kyaboron (Kyutai)
- @alexandredefossez (EnCodec, Moshi author)
- @naboron (AudioCraft)
- @OriolVinyalsML (DeepMind audio)

### Conferences
- **ICASSP**: IEEE International Conference on Acoustics, Speech and Signal Processing
- **Interspeech**: Annual conference on speech communication
- **NeurIPS**: Neural Information Processing Systems (audio ML tracks)
- **ICLR**: International Conference on Learning Representations

---

## Benchmarks

- **SUPERB**: [superbbenchmark.org](https://superbbenchmark.org/) - Speech processing benchmark
- **ESPnet**: [github.com/espnet/espnet](https://github.com/espnet/espnet) - End-to-end speech toolkit
- **SpeechBrain**: [speechbrain.github.io](https://speechbrain.github.io/) - Speech toolkit with benchmarks
