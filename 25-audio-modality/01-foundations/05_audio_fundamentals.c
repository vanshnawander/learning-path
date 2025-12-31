/**
 * 05_audio_fundamentals.c - Low-Level Audio Processing in C
 * 
 * Pure C implementations of audio processing fundamentals.
 * Understanding these is essential before using high-level libraries.
 * 
 * Compile: gcc -O3 -o audio_fundamentals 05_audio_fundamentals.c -lm
 * Run: ./audio_fundamentals
 * 
 * Topics covered:
 * - PCM audio representation
 * - Sample rate conversion concepts
 * - Window functions
 * - DFT/FFT implementation
 * - Mel filterbank
 * - MFCC extraction
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// TIMING UTILITIES
// ============================================================

typedef struct {
    clock_t start;
    const char* name;
} Timer;

void timer_start(Timer* t, const char* name) {
    t->name = name;
    t->start = clock();
}

double timer_stop(Timer* t) {
    clock_t end = clock();
    double elapsed_ms = (double)(end - t->start) / CLOCKS_PER_SEC * 1000.0;
    printf("⏱️  %-40s %8.3f ms\n", t->name, elapsed_ms);
    return elapsed_ms;
}

// ============================================================
// PCM AUDIO BASICS
// ============================================================

/**
 * Convert 16-bit signed integer to float [-1, 1]
 * 
 * This is what happens when you load a WAV file.
 * int16 range: [-32768, 32767]
 */
void int16_to_float(const int16_t* input, float* output, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        output[i] = (float)input[i] / 32768.0f;
    }
}

/**
 * Convert float [-1, 1] to 16-bit signed integer
 */
void float_to_int16(const float* input, int16_t* output, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float val = input[i];
        // Clamp to [-1, 1]
        if (val > 1.0f) val = 1.0f;
        if (val < -1.0f) val = -1.0f;
        output[i] = (int16_t)(val * 32767.0f);
    }
}

/**
 * Demonstrate quantization effects
 */
void demo_quantization() {
    printf("\n--- Quantization Demo ---\n");
    
    // Original float value
    float original = 0.12345678f;
    
    // Quantize to different bit depths
    int bits[] = {8, 12, 16, 24};
    for (int i = 0; i < 4; i++) {
        int levels = 1 << bits[i];
        int quantized_int = (int)(original * (levels / 2));
        float reconstructed = (float)quantized_int / (levels / 2);
        float error = fabsf(original - reconstructed);
        printf("  %2d-bit: levels=%7d, reconstructed=%.8f, error=%.8f\n",
               bits[i], levels, reconstructed, error);
    }
}

// ============================================================
// WINDOW FUNCTIONS
// ============================================================

/**
 * Hann (Hanning) window
 * 
 * w(n) = 0.5 * (1 - cos(2π * n / (N-1)))
 * 
 * Properties:
 * - Smooth taper to zero at edges
 * - Good frequency resolution
 * - Moderate side lobe suppression (-31 dB)
 */
void hann_window(float* window, int size) {
    for (int i = 0; i < size; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
    }
}

/**
 * Hamming window
 * 
 * w(n) = 0.54 - 0.46 * cos(2π * n / (N-1))
 * 
 * Properties:
 * - Does NOT go to zero at edges
 * - Better side lobe suppression than Hann (-43 dB)
 * - Slightly worse frequency resolution
 */
void hamming_window(float* window, int size) {
    for (int i = 0; i < size; i++) {
        window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
    }
}

/**
 * Blackman window
 * 
 * Best side lobe suppression (-58 dB)
 * Worst frequency resolution
 */
void blackman_window(float* window, int size) {
    for (int i = 0; i < size; i++) {
        float x = 2.0f * M_PI * i / (size - 1);
        window[i] = 0.42f - 0.5f * cosf(x) + 0.08f * cosf(2.0f * x);
    }
}

/**
 * Apply window to signal (element-wise multiply)
 */
void apply_window(const float* signal, const float* window, 
                  float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = signal[i] * window[i];
    }
}

// ============================================================
// DFT IMPLEMENTATION (Educational - O(n²))
// ============================================================

typedef struct {
    float real;
    float imag;
} Complex;

/**
 * Discrete Fourier Transform (naive implementation)
 * 
 * X[k] = Σ x[n] * e^(-j*2π*k*n/N)
 *      = Σ x[n] * (cos(2π*k*n/N) - j*sin(2π*k*n/N))
 * 
 * Complexity: O(N²) - too slow for production!
 * Use FFT (O(N log N)) for real applications.
 */
void dft(const float* input, Complex* output, int n) {
    for (int k = 0; k < n; k++) {
        output[k].real = 0.0f;
        output[k].imag = 0.0f;
        
        for (int t = 0; t < n; t++) {
            float angle = 2.0f * M_PI * k * t / n;
            output[k].real += input[t] * cosf(angle);
            output[k].imag -= input[t] * sinf(angle);
        }
    }
}

/**
 * Compute magnitude spectrum
 * |X[k]| = sqrt(Re² + Im²)
 */
void magnitude_spectrum(const Complex* fft, float* magnitude, int n) {
    for (int k = 0; k < n; k++) {
        magnitude[k] = sqrtf(fft[k].real * fft[k].real + 
                            fft[k].imag * fft[k].imag);
    }
}

/**
 * Compute power spectrum
 * P[k] = |X[k]|² = Re² + Im²
 */
void power_spectrum(const Complex* fft, float* power, int n) {
    for (int k = 0; k < n; k++) {
        power[k] = fft[k].real * fft[k].real + 
                   fft[k].imag * fft[k].imag;
    }
}

// ============================================================
// FFT IMPLEMENTATION (Cooley-Tukey, Radix-2)
// ============================================================

/**
 * Bit-reversal permutation for FFT
 */
int bit_reverse(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * In-place Cooley-Tukey FFT (Radix-2)
 * 
 * Complexity: O(N log N)
 * Requirement: N must be power of 2
 */
void fft(Complex* x, int n) {
    // Compute log2(n)
    int log2n = 0;
    int temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }
    
    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = bit_reverse(i, log2n);
        if (j > i) {
            Complex tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
    
    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;  // 2^s
        int m2 = m >> 1; // m/2
        
        // Twiddle factor
        float angle = -2.0f * M_PI / m;
        Complex wm = {cosf(angle), sinf(angle)};
        
        for (int k = 0; k < n; k += m) {
            Complex w = {1.0f, 0.0f};
            
            for (int j = 0; j < m2; j++) {
                // Butterfly operation
                Complex t = {
                    w.real * x[k + j + m2].real - w.imag * x[k + j + m2].imag,
                    w.real * x[k + j + m2].imag + w.imag * x[k + j + m2].real
                };
                
                Complex u = x[k + j];
                
                x[k + j].real = u.real + t.real;
                x[k + j].imag = u.imag + t.imag;
                
                x[k + j + m2].real = u.real - t.real;
                x[k + j + m2].imag = u.imag - t.imag;
                
                // Update twiddle factor
                float tmp = w.real * wm.real - w.imag * wm.imag;
                w.imag = w.real * wm.imag + w.imag * wm.real;
                w.real = tmp;
            }
        }
    }
}

/**
 * Real-to-complex FFT wrapper
 */
void real_fft(const float* input, Complex* output, int n) {
    // Copy real input to complex array
    for (int i = 0; i < n; i++) {
        output[i].real = input[i];
        output[i].imag = 0.0f;
    }
    fft(output, n);
}

// ============================================================
// MEL SCALE CONVERSION
// ============================================================

/**
 * Hz to Mel conversion
 * m = 2595 * log10(1 + f/700)
 */
float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/**
 * Mel to Hz conversion
 * f = 700 * (10^(m/2595) - 1)
 */
float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * Create mel filterbank
 * 
 * Creates triangular filters spaced linearly in mel domain.
 */
void create_mel_filterbank(float* filterbank, int num_filters, 
                           int num_fft_bins, int sample_rate,
                           float f_min, float f_max) {
    // Mel points
    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);
    
    float* mel_points = (float*)malloc((num_filters + 2) * sizeof(float));
    float* hz_points = (float*)malloc((num_filters + 2) * sizeof(float));
    int* bin_points = (int*)malloc((num_filters + 2) * sizeof(int));
    
    // Linearly spaced in mel domain
    for (int i = 0; i < num_filters + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (num_filters + 1);
        mel_points[i] = mel;
        hz_points[i] = mel_to_hz(mel);
        bin_points[i] = (int)floorf((num_fft_bins * 2 - 1) * hz_points[i] / sample_rate);
    }
    
    // Create triangular filters
    memset(filterbank, 0, num_filters * num_fft_bins * sizeof(float));
    
    for (int m = 0; m < num_filters; m++) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];
        
        // Rising edge
        for (int k = left; k < center && k < num_fft_bins; k++) {
            if (center > left) {
                filterbank[m * num_fft_bins + k] = 
                    (float)(k - left) / (center - left);
            }
        }
        
        // Falling edge
        for (int k = center; k < right && k < num_fft_bins; k++) {
            if (right > center) {
                filterbank[m * num_fft_bins + k] = 
                    (float)(right - k) / (right - center);
            }
        }
    }
    
    free(mel_points);
    free(hz_points);
    free(bin_points);
}

/**
 * Apply mel filterbank to power spectrum
 */
void apply_mel_filterbank(const float* power_spec, const float* filterbank,
                          float* mel_spec, int num_fft_bins, int num_filters) {
    for (int m = 0; m < num_filters; m++) {
        float sum = 0.0f;
        for (int k = 0; k < num_fft_bins; k++) {
            sum += power_spec[k] * filterbank[m * num_fft_bins + k];
        }
        mel_spec[m] = sum;
    }
}

// ============================================================
// COMPLETE MEL SPECTROGRAM PIPELINE
// ============================================================

typedef struct {
    int sample_rate;
    int n_fft;
    int hop_length;
    int n_mels;
    int num_fft_bins;
    float* window;
    float* filterbank;
} MelSpectrogramConfig;

void mel_config_init(MelSpectrogramConfig* config, int sample_rate,
                     int n_fft, int hop_length, int n_mels) {
    config->sample_rate = sample_rate;
    config->n_fft = n_fft;
    config->hop_length = hop_length;
    config->n_mels = n_mels;
    config->num_fft_bins = n_fft / 2 + 1;
    
    // Allocate and initialize window
    config->window = (float*)malloc(n_fft * sizeof(float));
    hann_window(config->window, n_fft);
    
    // Allocate and initialize filterbank
    config->filterbank = (float*)malloc(n_mels * config->num_fft_bins * sizeof(float));
    create_mel_filterbank(config->filterbank, n_mels, config->num_fft_bins,
                          sample_rate, 0.0f, sample_rate / 2.0f);
}

void mel_config_free(MelSpectrogramConfig* config) {
    free(config->window);
    free(config->filterbank);
}

/**
 * Compute mel spectrogram for a single frame
 */
void compute_mel_frame(const float* audio_frame, MelSpectrogramConfig* config,
                       float* mel_output, Complex* fft_buffer, float* power_buffer) {
    // Apply window
    float* windowed = (float*)malloc(config->n_fft * sizeof(float));
    apply_window(audio_frame, config->window, windowed, config->n_fft);
    
    // FFT
    real_fft(windowed, fft_buffer, config->n_fft);
    
    // Power spectrum (only positive frequencies)
    for (int k = 0; k < config->num_fft_bins; k++) {
        power_buffer[k] = fft_buffer[k].real * fft_buffer[k].real +
                          fft_buffer[k].imag * fft_buffer[k].imag;
    }
    
    // Apply mel filterbank
    apply_mel_filterbank(power_buffer, config->filterbank, mel_output,
                         config->num_fft_bins, config->n_mels);
    
    // Log compression
    for (int m = 0; m < config->n_mels; m++) {
        mel_output[m] = logf(mel_output[m] + 1e-10f);
    }
    
    free(windowed);
}

/**
 * Compute full mel spectrogram
 */
void compute_mel_spectrogram(const float* audio, int num_samples,
                             MelSpectrogramConfig* config,
                             float* mel_spectrogram, int* num_frames) {
    *num_frames = (num_samples - config->n_fft) / config->hop_length + 1;
    
    // Allocate buffers
    Complex* fft_buffer = (Complex*)malloc(config->n_fft * sizeof(Complex));
    float* power_buffer = (float*)malloc(config->num_fft_bins * sizeof(float));
    
    // Process each frame
    for (int frame = 0; frame < *num_frames; frame++) {
        int start = frame * config->hop_length;
        compute_mel_frame(&audio[start], config,
                          &mel_spectrogram[frame * config->n_mels],
                          fft_buffer, power_buffer);
    }
    
    free(fft_buffer);
    free(power_buffer);
}

// ============================================================
// BENCHMARKS
// ============================================================

void run_benchmarks() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  AUDIO PROCESSING BENCHMARKS (Pure C)                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    Timer timer;
    int num_runs = 100;
    
    // Parameters
    int sample_rate = 16000;
    int duration_sec = 10;
    int num_samples = sample_rate * duration_sec;
    int n_fft = 512;
    int hop_length = 160;
    int n_mels = 80;
    
    printf("Configuration:\n");
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %d seconds (%d samples)\n", duration_sec, num_samples);
    printf("  n_fft: %d, hop: %d, mel bins: %d\n\n", n_fft, hop_length, n_mels);
    
    // Generate random audio
    float* audio = (float*)malloc(num_samples * sizeof(float));
    srand(42);
    for (int i = 0; i < num_samples; i++) {
        audio[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // Benchmark FFT
    printf("--- FFT Benchmarks ---\n");
    Complex* fft_buffer = (Complex*)malloc(n_fft * sizeof(Complex));
    
    timer_start(&timer, "FFT (512-point, 100 runs)");
    for (int r = 0; r < num_runs; r++) {
        real_fft(audio, fft_buffer, n_fft);
    }
    double fft_time = timer_stop(&timer);
    printf("  Per FFT: %.3f µs\n\n", fft_time / num_runs * 1000);
    
    // Benchmark mel spectrogram
    printf("--- Mel Spectrogram Benchmark ---\n");
    MelSpectrogramConfig config;
    mel_config_init(&config, sample_rate, n_fft, hop_length, n_mels);
    
    int num_frames;
    float* mel_spec = (float*)malloc(1000 * n_mels * sizeof(float));
    
    timer_start(&timer, "Mel spectrogram (10s audio)");
    compute_mel_spectrogram(audio, num_samples, &config, mel_spec, &num_frames);
    double mel_time = timer_stop(&timer);
    
    printf("  Frames computed: %d\n", num_frames);
    printf("  Time per frame: %.3f µs\n", mel_time * 1000 / num_frames);
    printf("  Real-time factor: %.4f\n", mel_time / 1000 / duration_sec);
    
    // Cleanup
    free(audio);
    free(fft_buffer);
    free(mel_spec);
    mel_config_free(&config);
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  AUDIO FUNDAMENTALS IN C                                     █\n");
    printf("█  Low-level audio processing implementations                  █\n");
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    // Demo quantization
    demo_quantization();
    
    // Run benchmarks
    run_benchmarks();
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY INSIGHTS                                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  1. FFT is O(N log N) - much faster than DFT O(N²)          ║\n");
    printf("║  2. Window functions reduce spectral leakage                 ║\n");
    printf("║  3. Mel scale matches human perception                       ║\n");
    printf("║  4. Log compression important for neural networks            ║\n");
    printf("║  5. These are what torchaudio/librosa do under the hood     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
