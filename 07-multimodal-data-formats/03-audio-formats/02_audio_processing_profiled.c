/**
 * 02_audio_processing_profiled.c - Profile Audio Processing Operations
 * 
 * Every audio operation is timed. This is what happens inside
 * torchaudio, librosa, and speech recognition pipelines.
 * 
 * Compile: gcc -O3 -o audio_process 02_audio_processing_profiled.c -lrt -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// TIMING INFRASTRUCTURE
// ============================================================

typedef struct {
    struct timespec start;
    const char* name;
} Timer;

void timer_start(Timer* t, const char* name) {
    t->name = name;
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_stop_ms(Timer* t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = ((end.tv_sec - t->start.tv_sec) * 1000.0 + 
                         (end.tv_nsec - t->start.tv_nsec) / 1e6);
    printf("⏱️  %-35s %8.3f ms\n", t->name, elapsed_ms);
    return elapsed_ms;
}

// ============================================================
// AUDIO STRUCTURES
// ============================================================

typedef struct {
    int16_t* data;
    int sample_rate;
    int num_samples;
    int channels;
} AudioInt16;

typedef struct {
    float* data;
    int sample_rate;
    int num_samples;
    int channels;
} AudioFloat;

typedef struct {
    float* data;        // Magnitude spectrogram
    int num_frames;
    int num_bins;
} Spectrogram;

// ============================================================
// AUDIO OPERATIONS (ALL PROFILED)
// ============================================================

AudioInt16* create_audio_int16(int sample_rate, int num_samples, int channels) {
    AudioInt16* audio = malloc(sizeof(AudioInt16));
    audio->sample_rate = sample_rate;
    audio->num_samples = num_samples;
    audio->channels = channels;
    audio->data = aligned_alloc(64, num_samples * channels * sizeof(int16_t));
    return audio;
}

AudioFloat* create_audio_float(int sample_rate, int num_samples, int channels) {
    AudioFloat* audio = malloc(sizeof(AudioFloat));
    audio->sample_rate = sample_rate;
    audio->num_samples = num_samples;
    audio->channels = channels;
    audio->data = aligned_alloc(64, num_samples * channels * sizeof(float));
    return audio;
}

void free_audio_int16(AudioInt16* audio) {
    free(audio->data);
    free(audio);
}

void free_audio_float(AudioFloat* audio) {
    free(audio->data);
    free(audio);
}

// Simulate loading PCM data
void fill_random_audio(AudioInt16* audio) {
    Timer t;
    timer_start(&t, "Generate random audio");
    
    for (int i = 0; i < audio->num_samples * audio->channels; i++) {
        // Simulate sine wave with noise
        float phase = (float)i / audio->sample_rate * 440.0 * 2 * M_PI;
        audio->data[i] = (int16_t)(sin(phase) * 16000 + (rand() % 1000 - 500));
    }
    
    timer_stop_ms(&t);
}

// Convert int16 to float32 (normalization to [-1, 1])
AudioFloat* int16_to_float(AudioInt16* src) {
    Timer t;
    timer_start(&t, "int16 → float32 normalize");
    
    AudioFloat* dst = create_audio_float(src->sample_rate, src->num_samples, src->channels);
    
    for (int i = 0; i < src->num_samples * src->channels; i++) {
        dst->data[i] = (float)src->data[i] / 32768.0f;
    }
    
    timer_stop_ms(&t);
    return dst;
}

// Simple resampling (linear interpolation)
AudioFloat* resample(AudioFloat* src, int new_sample_rate) {
    Timer t;
    timer_start(&t, "Resample (linear interp)");
    
    float ratio = (float)new_sample_rate / src->sample_rate;
    int new_num_samples = (int)(src->num_samples * ratio);
    
    AudioFloat* dst = create_audio_float(new_sample_rate, new_num_samples, src->channels);
    
    for (int i = 0; i < new_num_samples; i++) {
        float src_pos = i / ratio;
        int src_idx = (int)src_pos;
        float frac = src_pos - src_idx;
        
        if (src_idx + 1 < src->num_samples) {
            dst->data[i] = src->data[src_idx] * (1 - frac) + 
                           src->data[src_idx + 1] * frac;
        } else {
            dst->data[i] = src->data[src_idx];
        }
    }
    
    timer_stop_ms(&t);
    return dst;
}

// Pre-emphasis filter (common in speech processing)
void pre_emphasis(AudioFloat* audio, float coef) {
    Timer t;
    timer_start(&t, "Pre-emphasis filter");
    
    for (int i = audio->num_samples - 1; i > 0; i--) {
        audio->data[i] = audio->data[i] - coef * audio->data[i - 1];
    }
    
    timer_stop_ms(&t);
}

// Hann window
float* create_hann_window(int size) {
    float* window = aligned_alloc(64, size * sizeof(float));
    for (int i = 0; i < size; i++) {
        window[i] = 0.5f * (1 - cos(2 * M_PI * i / (size - 1)));
    }
    return window;
}

// Simple DFT (not FFT - for demonstration only)
// In production, use FFTW or similar
void dft_magnitude(float* input, float* output, int n) {
    for (int k = 0; k < n / 2 + 1; k++) {
        float real = 0, imag = 0;
        for (int t = 0; t < n; t++) {
            float angle = 2 * M_PI * k * t / n;
            real += input[t] * cos(angle);
            imag -= input[t] * sin(angle);
        }
        output[k] = sqrt(real * real + imag * imag);
    }
}

// Compute spectrogram using STFT
Spectrogram* compute_spectrogram(AudioFloat* audio, int n_fft, int hop_length) {
    Timer t;
    timer_start(&t, "STFT spectrogram");
    
    int num_frames = (audio->num_samples - n_fft) / hop_length + 1;
    int num_bins = n_fft / 2 + 1;
    
    Spectrogram* spec = malloc(sizeof(Spectrogram));
    spec->num_frames = num_frames;
    spec->num_bins = num_bins;
    spec->data = aligned_alloc(64, num_frames * num_bins * sizeof(float));
    
    float* window = create_hann_window(n_fft);
    float* windowed = aligned_alloc(64, n_fft * sizeof(float));
    
    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * hop_length;
        
        // Apply window
        for (int i = 0; i < n_fft; i++) {
            windowed[i] = audio->data[start + i] * window[i];
        }
        
        // Compute DFT magnitude
        dft_magnitude(windowed, &spec->data[frame * num_bins], n_fft);
    }
    
    free(window);
    free(windowed);
    
    timer_stop_ms(&t);
    return spec;
}

// Convert to log scale (decibels)
void spectrogram_to_db(Spectrogram* spec) {
    Timer t;
    timer_start(&t, "Magnitude → dB");
    
    float ref = 1.0f;
    float min_db = -80.0f;
    
    for (int i = 0; i < spec->num_frames * spec->num_bins; i++) {
        float db = 20 * log10(spec->data[i] / ref + 1e-10);
        spec->data[i] = fmax(db, min_db);
    }
    
    timer_stop_ms(&t);
}

// Simple mel filterbank application (simplified)
float* apply_mel_filterbank(Spectrogram* spec, int n_mels) {
    Timer t;
    timer_start(&t, "Apply mel filterbank");
    
    float* mel_spec = aligned_alloc(64, spec->num_frames * n_mels * sizeof(float));
    
    // Simplified: just downsample frequency bins
    // Real implementation uses triangular filters
    int bins_per_mel = spec->num_bins / n_mels;
    
    for (int frame = 0; frame < spec->num_frames; frame++) {
        for (int mel = 0; mel < n_mels; mel++) {
            float sum = 0;
            int start_bin = mel * bins_per_mel;
            for (int b = 0; b < bins_per_mel; b++) {
                sum += spec->data[frame * spec->num_bins + start_bin + b];
            }
            mel_spec[frame * n_mels + mel] = sum / bins_per_mel;
        }
    }
    
    timer_stop_ms(&t);
    return mel_spec;
}

void free_spectrogram(Spectrogram* spec) {
    free(spec->data);
    free(spec);
}

// ============================================================
// FULL PIPELINE BENCHMARK
// ============================================================

void run_full_pipeline() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  FULL AUDIO PREPROCESSING PIPELINE (Profiled)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Simulate 10 seconds of 44.1kHz audio (CD quality)
    int sample_rate = 44100;
    int duration = 10;
    int num_samples = sample_rate * duration;
    
    printf("Input: %d seconds @ %d Hz (%.2f MB)\n", 
           duration, sample_rate, 
           num_samples * sizeof(int16_t) / 1024.0 / 1024.0);
    printf("Target: 16kHz, 80-bin mel spectrogram\n\n");
    
    Timer total;
    timer_start(&total, "TOTAL PIPELINE");
    
    printf("─── Pipeline Steps ───\n\n");
    
    // Step 1: Create/load audio
    AudioInt16* raw = create_audio_int16(sample_rate, num_samples, 1);
    fill_random_audio(raw);
    
    // Step 2: Convert to float
    AudioFloat* audio_float = int16_to_float(raw);
    
    // Step 3: Resample to 16kHz
    AudioFloat* resampled = resample(audio_float, 16000);
    printf("  After resample: %d samples\n", resampled->num_samples);
    
    // Step 4: Pre-emphasis
    pre_emphasis(resampled, 0.97f);
    
    // Step 5: STFT (using small n_fft for demo speed)
    int n_fft = 256;  // Use 512 or 1024 in production
    int hop_length = 160;
    Spectrogram* spec = compute_spectrogram(resampled, n_fft, hop_length);
    printf("  Spectrogram: %d frames × %d bins\n", spec->num_frames, spec->num_bins);
    
    // Step 6: Convert to dB
    spectrogram_to_db(spec);
    
    // Step 7: Apply mel filterbank
    int n_mels = 80;
    float* mel_spec = apply_mel_filterbank(spec, n_mels);
    printf("  Mel spectrogram: %d frames × %d mels\n", spec->num_frames, n_mels);
    
    printf("\n");
    double total_time = timer_stop_ms(&total);
    
    printf("\n─── Memory Usage ───\n");
    printf("  Raw audio (int16): %.2f MB\n", 
           num_samples * sizeof(int16_t) / 1024.0 / 1024.0);
    printf("  Float audio: %.2f MB\n", 
           resampled->num_samples * sizeof(float) / 1024.0 / 1024.0);
    printf("  Spectrogram: %.2f MB\n", 
           spec->num_frames * spec->num_bins * sizeof(float) / 1024.0 / 1024.0);
    printf("  Mel spectrogram: %.2f MB\n", 
           spec->num_frames * n_mels * sizeof(float) / 1024.0 / 1024.0);
    
    printf("\n─── Throughput ───\n");
    printf("  Real-time factor: %.2fx\n", duration * 1000.0 / total_time);
    printf("  (>1.0 means faster than real-time)\n");
    
    // Cleanup
    free_audio_int16(raw);
    free_audio_float(audio_float);
    free_audio_float(resampled);
    free_spectrogram(spec);
    free(mel_spec);
}

// ============================================================
// COMPARE DIFFERENT SAMPLE RATES
// ============================================================

void benchmark_sample_rates() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  PROCESSING TIME BY SAMPLE RATE (10 sec audio)               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int sample_rates[] = {8000, 16000, 22050, 44100, 48000};
    int num_rates = sizeof(sample_rates) / sizeof(sample_rates[0]);
    int duration = 10;
    
    printf("%-12s %-12s %-12s %-15s\n", 
           "Rate (Hz)", "Samples", "Size (MB)", "Process (ms)");
    printf("────────────────────────────────────────────────────────────────\n");
    
    for (int i = 0; i < num_rates; i++) {
        int sr = sample_rates[i];
        int num_samples = sr * duration;
        
        AudioInt16* raw = create_audio_int16(sr, num_samples, 1);
        for (int j = 0; j < num_samples; j++) {
            raw->data[j] = rand() % 65536 - 32768;
        }
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Process
        AudioFloat* audio_float = int16_to_float(raw);
        pre_emphasis(audio_float, 0.97f);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + 
                         (end.tv_nsec - start.tv_nsec) / 1e6;
        
        printf("%-12d %-12d %-12.2f %-15.2f\n",
               sr, num_samples,
               num_samples * sizeof(int16_t) / 1024.0 / 1024.0,
               elapsed);
        
        free_audio_int16(raw);
        free_audio_float(audio_float);
    }
    
    printf("\n💡 For speech ML, 16kHz is often sufficient (faster processing!)\n");
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  AUDIO PROCESSING WITH PROFILING                             █\n");
    printf("█  Every operation timed - understand the real costs!          █\n");
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    run_full_pipeline();
    benchmark_sample_rates();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAYS FOR MULTIMODAL TRAINING                       ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. STFT/FFT dominates processing time                       ║\n");
    printf("║     → Use optimized FFT libraries (FFTW, MKL)                ║\n");
    printf("║                                                              ║\n");
    printf("║  2. Resampling is expensive                                  ║\n");
    printf("║     → Store audio at target sample rate                      ║\n");
    printf("║                                                              ║\n");
    printf("║  3. 16kHz is sufficient for most speech tasks                ║\n");
    printf("║     → 3x less data than 48kHz                                ║\n");
    printf("║                                                              ║\n");
    printf("║  4. Pre-compute mel spectrograms for training                ║\n");
    printf("║     → Save to disk, load directly                            ║\n");
    printf("║                                                              ║\n");
    printf("║  5. GPU can accelerate STFT significantly                    ║\n");
    printf("║     → Use torchaudio with CUDA                               ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
