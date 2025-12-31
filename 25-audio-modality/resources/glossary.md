# Audio ML Glossary

Quick reference for terminology used across the audio modality curriculum.

---

## A

**ASR (Automatic Speech Recognition)**: Converting speech audio to text. Examples: Whisper, wav2vec 2.0.

**Attention**: Mechanism allowing models to focus on relevant parts of input. Used in Transformers for audio (WavLM, Whisper).

**Autoregressive**: Generation approach where each output depends on previous outputs. WaveNet generates audio sample-by-sample autoregressively.

---

## B

**Bandwidth**: For audio, the range of frequencies contained. Human hearing: 20 Hz - 20 kHz.

**Bit Depth**: Number of bits per audio sample. CD quality = 16-bit (65,536 levels).

**Bitrate**: Data rate for encoded audio. Measured in kbps (kilobits per second). 
- MP3: 128-320 kbps
- EnCodec: 1.5-24 kbps
- Mimi: 1.1 kbps

---

## C

**Causal**: Operations that only use past/current information, not future. Required for real-time streaming.

**Codebook**: Lookup table of learned embeddings in VQ. Size typically 1024-8192 entries.

**Codec**: Encoder-decoder pair for compression. Neural codecs: SoundStream, EnCodec, Mimi.

**Commitment Loss**: VQ training loss that encourages encoder outputs to stay close to codebook entries.

**CTC (Connectionist Temporal Classification)**: Loss function for sequence-to-sequence with alignment. Used in some ASR systems.

---

## D

**dB (Decibel)**: Logarithmic scale for audio amplitude. dB = 20 * log10(amplitude).

**Dilated Convolution**: Convolution with gaps between kernel elements. Increases receptive field exponentially without increasing parameters.

**Distillation**: Training a smaller model to mimic a larger one. Mimi distills WavLM for semantic tokens.

**DFT/FFT**: Discrete/Fast Fourier Transform. Converts time domain to frequency domain. FFT is O(n log n) algorithm.

---

## E

**EMA (Exponential Moving Average)**: Update method for codebook learning. Smoothly updates codebook entries over training.

**Encoder-Decoder**: Architecture with separate encoding and decoding networks. Used in codecs and seq2seq models.

---

## F

**F0 (Fundamental Frequency)**: Base frequency of voiced speech. Male: ~85-180 Hz, Female: ~165-255 Hz.

**Feature Extraction**: Converting raw audio to model-friendly representation. Common: mel spectrogram, MFCC.

**FFN (Feed-Forward Network)**: Fully connected layers in Transformer blocks.

**Filterbank**: Set of bandpass filters. Mel filterbank converts spectrogram to mel scale.

**Formants**: Resonant frequencies of vocal tract. F1, F2, F3 encode vowel identity.

**Frame**: Fixed-length segment of audio for processing. Typical: 20-80ms.

**Frame Rate**: Frames per second. 
- EnCodec: 75 Hz
- Mimi: 12.5 Hz
- Whisper features: 100 Hz

**Full-Duplex**: Simultaneous bidirectional communication. Moshi can listen and speak at same time.

---

## G

**GAN (Generative Adversarial Network)**: Training with generator and discriminator. Used in HiFi-GAN, MelGAN.

**Gating**: Controlling information flow with learned gates. WaveNet uses tanh ⊙ sigmoid gating.

---

## H

**Hop Length**: Samples between consecutive frames. Common: n_fft / 4.

**Hz (Hertz)**: Frequency unit (cycles per second). Human hearing: 20 Hz - 20,000 Hz.

---

## I

**IAF (Inverse Autoregressive Flow)**: Generative model enabling parallel generation. Used in Parallel WaveNet.

---

## K

**KV Cache**: Storing key/value tensors from attention for efficient autoregressive generation. Critical for Moshi real-time inference.

---

## L

**Latency**: Delay between input and output. Mimi: 80ms, EnCodec: 13ms.

**LLM (Large Language Model)**: Large-scale autoregressive text model. Moshi uses Helium (7B).

**Log Mel Spectrogram**: Mel spectrogram with log compression. Standard input for speech models.

---

## M

**Masked Prediction**: Self-supervised objective. Mask parts of input, predict masked parts. Used in WavLM, HuBERT.

**Mel Scale**: Perceptual frequency scale. Linear below 1000 Hz, logarithmic above.

**MFCC (Mel-Frequency Cepstral Coefficients)**: Compact audio features from mel spectrogram DCT. Traditional ASR feature.

**MOS (Mean Opinion Score)**: Subjective quality rating 1-5. Used to evaluate TTS/codec quality.

**MPD (Multi-Period Discriminator)**: HiFi-GAN discriminator analyzing different periodic patterns.

**MSD (Multi-Scale Discriminator)**: Discriminator operating at multiple time scales.

**μ-law (Mu-law)**: Companding algorithm for 8-bit audio quantization. Used in WaveNet.

---

## N

**n_fft**: FFT window size for STFT. Common: 400-2048. Larger = better frequency resolution.

**n_mels**: Number of mel frequency bins. Common: 80 for speech, 128 for music.

**Nyquist Frequency**: Maximum frequency representable at given sample rate = sample_rate / 2.

---

## P

**PCM (Pulse Code Modulation)**: Standard uncompressed audio format. Raw samples stored directly.

**Phase**: Position in oscillation cycle. Often discarded in spectrograms (magnitude only).

**Phoneme**: Smallest unit of speech sound. English has ~44 phonemes.

**Pre-emphasis**: High-pass filter applied before analysis. Boosts high frequencies. Coefficient ~0.97.

**Prosody**: Rhythm, stress, intonation of speech. Includes pitch, duration, loudness patterns.

---

## Q

**Quantization**: Converting continuous to discrete values.
- μ-law: 256 levels for waveform
- VQ: Mapping to codebook entries
- RVQ: Residual quantization with multiple levels

---

## R

**Receptive Field**: Input range that affects one output. Dilated convolutions increase this exponentially.

**Residual Connection**: Skip connection adding input to output. Enables training of deep networks.

**RTF (Real-Time Factor)**: Processing time / audio duration. RTF < 1 means faster than real-time.

**RVQ (Residual Vector Quantization)**: Multi-level VQ where each level quantizes the residual from previous. Used in SoundStream, EnCodec, Mimi.

---

## S

**Sample Rate**: Samples per second. Common:
- 8 kHz: Telephone
- 16 kHz: Speech ML
- 24 kHz: Neural codecs
- 44.1 kHz: CD audio
- 48 kHz: Professional audio

**Semantic Tokens**: Tokens encoding meaning/content, not acoustic details. Mimi level 0 is semantic.

**Skip Connection**: Direct path bypassing intermediate layers. Used in WaveNet, ResNets.

**Spectrogram**: Time-frequency representation from STFT. Shows how frequency content changes over time.

**SSL (Self-Supervised Learning)**: Learning from unlabeled data. WavLM, wav2vec 2.0 are SSL models.

**STFT (Short-Time Fourier Transform)**: FFT applied to windowed segments. Creates spectrogram.

**Stride**: Step size in convolution/pooling. Total stride determines temporal compression ratio.

**Straight-Through Estimator (STE)**: Gradient trick for non-differentiable quantization. Copies gradient through argmin.

---

## T

**Temporal Transformer**: Transformer processing sequence over time. Main component of Moshi.

**Token**: Discrete unit for sequence modeling. Text tokens (~50k vocab) or audio tokens (codebook indices).

**Tokenization**: Converting continuous signal to discrete tokens. Neural codecs tokenize audio.

**TTS (Text-to-Speech)**: Converting text to audio. Modern: text → mel spectrogram → vocoder.

---

## V

**VAE (Variational Autoencoder)**: Autoencoder with probabilistic latent space. VQ-VAE uses discrete latents.

**Vocoder**: Converts acoustic features (mel spectrogram) to waveform. Examples: WaveNet, HiFi-GAN.

**VQ (Vector Quantization)**: Mapping continuous vectors to nearest codebook entry.

**VQ-VAE**: VAE with vector-quantized latent space. Foundation for neural audio codecs.

---

## W

**Waveform**: Time-domain audio signal. Amplitude vs time.

**Window Function**: Smooth taper applied before FFT. Common: Hann, Hamming. Reduces spectral leakage.

**WER (Word Error Rate)**: ASR metric. (Substitutions + Deletions + Insertions) / Total words.

---

## Numbers

**16 kHz**: Standard sample rate for speech ML. Captures up to 8 kHz (sufficient for speech).

**24 kHz**: Sample rate for neural codecs (SoundStream, EnCodec, Mimi).

**80 mel bins**: Common number of mel frequency bins for speech models (Whisper, etc.).

**1024**: Common codebook size (10 bits per token).

**2048**: Mimi codebook size (11 bits per token).

---

## Acronyms Quick Reference

| Acronym | Full Form |
|---------|-----------|
| ASR | Automatic Speech Recognition |
| CTC | Connectionist Temporal Classification |
| dB | Decibel |
| EMA | Exponential Moving Average |
| FFT | Fast Fourier Transform |
| GAN | Generative Adversarial Network |
| LLM | Large Language Model |
| MFCC | Mel-Frequency Cepstral Coefficients |
| MOS | Mean Opinion Score |
| PCM | Pulse Code Modulation |
| RTF | Real-Time Factor |
| RVQ | Residual Vector Quantization |
| SSL | Self-Supervised Learning |
| STFT | Short-Time Fourier Transform |
| STE | Straight-Through Estimator |
| TTS | Text-to-Speech |
| VQ | Vector Quantization |
| WER | Word Error Rate |
