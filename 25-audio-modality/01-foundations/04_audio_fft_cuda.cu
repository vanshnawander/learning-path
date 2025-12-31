/**
 * 04_audio_fft_cuda.cu - GPU-Accelerated Audio Processing with CUDA
 * 
 * Low-level CUDA implementations of audio processing primitives:
 * - FFT using cuFFT
 * - Mel filterbank application
 * - Spectrogram computation
 * - Real-time streaming pipeline
 * 
 * Compile: nvcc -O3 -o audio_fft_cuda 04_audio_fft_cuda.cu -lcufft -lm
 * 
 * This is what happens inside torchaudio, librosa, and NVIDIA DALI
 * at the CUDA level.
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", \
                    __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================
// TIMING UTILITIES
// ============================================================

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    const char* name;
} GpuTimer;

void timer_create(GpuTimer* t, const char* name) {
    t->name = name;
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

void timer_start(GpuTimer* t) {
    cudaEventRecord(t->start, 0);
}

float timer_stop(GpuTimer* t) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    float ms;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    printf("⏱️  GPU %-30s %8.3f ms\n", t->name, ms);
    return ms;
}

void timer_destroy(GpuTimer* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

// ============================================================
// CUDA KERNELS
// ============================================================

/**
 * Apply Hann window to audio frames
 * 
 * Each thread handles one sample within a frame.
 * Grid: (num_frames, 1, 1)
 * Block: (frame_size, 1, 1) - up to 1024
 */
__global__ void apply_hann_window_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ window,
    int frame_size,
    int hop_length,
    int num_frames
) {
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if (frame_idx >= num_frames || sample_idx >= frame_size) return;
    
    int input_idx = frame_idx * hop_length + sample_idx;
    int output_idx = frame_idx * frame_size + sample_idx;
    
    output[output_idx] = input[input_idx] * window[sample_idx];
}

/**
 * Compute power spectrum from complex FFT output
 * 
 * |X|² = Re(X)² + Im(X)²
 */
__global__ void compute_power_spectrum_kernel(
    const cufftComplex* __restrict__ fft_output,
    float* __restrict__ power_spectrum,
    int num_bins,
    int num_frames
) {
    int frame_idx = blockIdx.x;
    int bin_idx = threadIdx.x;
    
    if (frame_idx >= num_frames || bin_idx >= num_bins) return;
    
    int idx = frame_idx * num_bins + bin_idx;
    cufftComplex val = fft_output[idx];
    
    power_spectrum[idx] = val.x * val.x + val.y * val.y;
}

/**
 * Apply mel filterbank to power spectrum
 * 
 * Each thread computes one mel bin for one frame.
 * Uses shared memory for filterbank weights.
 */
__global__ void apply_mel_filterbank_kernel(
    const float* __restrict__ power_spectrum,
    const float* __restrict__ mel_filterbank,
    float* __restrict__ mel_spectrum,
    int num_fft_bins,
    int num_mel_bins,
    int num_frames
) {
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x;
    
    if (frame_idx >= num_frames || mel_idx >= num_mel_bins) return;
    
    // Dot product: mel_spectrum[frame, mel] = sum(power[frame, :] * filterbank[mel, :])
    float sum = 0.0f;
    for (int bin = 0; bin < num_fft_bins; bin++) {
        sum += power_spectrum[frame_idx * num_fft_bins + bin] * 
               mel_filterbank[mel_idx * num_fft_bins + bin];
    }
    
    mel_spectrum[frame_idx * num_mel_bins + mel_idx] = sum;
}

/**
 * Optimized mel filterbank using shared memory
 */
__global__ void apply_mel_filterbank_shared_kernel(
    const float* __restrict__ power_spectrum,
    const float* __restrict__ mel_filterbank,
    float* __restrict__ mel_spectrum,
    int num_fft_bins,
    int num_mel_bins,
    int num_frames
) {
    extern __shared__ float shared_power[];
    
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x;
    
    if (frame_idx >= num_frames) return;
    
    // Cooperatively load power spectrum into shared memory
    for (int i = threadIdx.x; i < num_fft_bins; i += blockDim.x) {
        shared_power[i] = power_spectrum[frame_idx * num_fft_bins + i];
    }
    __syncthreads();
    
    if (mel_idx >= num_mel_bins) return;
    
    // Compute mel bin using shared memory
    float sum = 0.0f;
    for (int bin = 0; bin < num_fft_bins; bin++) {
        sum += shared_power[bin] * mel_filterbank[mel_idx * num_fft_bins + bin];
    }
    
    mel_spectrum[frame_idx * num_mel_bins + mel_idx] = sum;
}

/**
 * Log compression: log(x + epsilon)
 */
__global__ void log_compress_kernel(
    float* __restrict__ data,
    int size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = logf(data[idx] + epsilon);
    }
}

/**
 * Normalize mel spectrogram (per-utterance)
 * 
 * Two-pass: first compute mean/std, then normalize
 */
__global__ void compute_stats_kernel(
    const float* __restrict__ data,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int size
) {
    __shared__ float block_sum;
    __shared__ float block_sum_sq;
    
    if (threadIdx.x == 0) {
        block_sum = 0.0f;
        block_sum_sq = 0.0f;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < size) ? data[idx] : 0.0f;
    
    atomicAdd(&block_sum, val);
    atomicAdd(&block_sum_sq, val * val);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(sum, block_sum);
        atomicAdd(sum_sq, block_sum_sq);
    }
}

__global__ void normalize_kernel(
    float* __restrict__ data,
    int size,
    float mean,
    float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] - mean) / (std + 1e-8f);
    }
}

// ============================================================
// HOST FUNCTIONS
// ============================================================

/**
 * Create Hann window on GPU
 */
void create_hann_window(float* d_window, int size) {
    float* h_window = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
    }
    CUDA_CHECK(cudaMemcpy(d_window, h_window, size * sizeof(float), 
                          cudaMemcpyHostToDevice));
    free(h_window);
}

/**
 * Create mel filterbank on GPU
 * 
 * Simplified version - triangular filters linearly spaced in mel domain
 */
float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void create_mel_filterbank(float* d_filterbank, int sample_rate, 
                           int num_fft_bins, int num_mel_bins,
                           float f_min, float f_max) {
    int filterbank_size = num_mel_bins * num_fft_bins;
    float* h_filterbank = (float*)calloc(filterbank_size, sizeof(float));
    
    // Mel points
    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);
    float* mel_points = (float*)malloc((num_mel_bins + 2) * sizeof(float));
    
    for (int i = 0; i < num_mel_bins + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * i / (num_mel_bins + 1);
        mel_points[i] = mel_to_hz(mel);
    }
    
    // Convert to FFT bins
    int* bin_points = (int*)malloc((num_mel_bins + 2) * sizeof(int));
    for (int i = 0; i < num_mel_bins + 2; i++) {
        bin_points[i] = (int)floorf((num_fft_bins * 2 - 1) * mel_points[i] / sample_rate);
    }
    
    // Create triangular filters
    for (int mel = 0; mel < num_mel_bins; mel++) {
        int left = bin_points[mel];
        int center = bin_points[mel + 1];
        int right = bin_points[mel + 2];
        
        // Rising edge
        for (int bin = left; bin < center && bin < num_fft_bins; bin++) {
            if (center > left) {
                h_filterbank[mel * num_fft_bins + bin] = 
                    (float)(bin - left) / (center - left);
            }
        }
        
        // Falling edge
        for (int bin = center; bin < right && bin < num_fft_bins; bin++) {
            if (right > center) {
                h_filterbank[mel * num_fft_bins + bin] = 
                    (float)(right - bin) / (right - center);
            }
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_filterbank, h_filterbank, 
                          filterbank_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    free(h_filterbank);
    free(mel_points);
    free(bin_points);
}

// ============================================================
// COMPLETE MEL SPECTROGRAM PIPELINE
// ============================================================

typedef struct {
    // Parameters
    int sample_rate;
    int n_fft;
    int hop_length;
    int num_mel_bins;
    
    // Derived
    int num_fft_bins;
    
    // GPU resources
    float* d_window;
    float* d_filterbank;
    float* d_framed_audio;
    cufftComplex* d_fft_output;
    float* d_power_spectrum;
    float* d_mel_spectrum;
    cufftHandle fft_plan;
    
} MelSpectrogramPipeline;

void pipeline_init(MelSpectrogramPipeline* p, int sample_rate, 
                   int n_fft, int hop_length, int num_mel_bins,
                   int max_frames) {
    p->sample_rate = sample_rate;
    p->n_fft = n_fft;
    p->hop_length = hop_length;
    p->num_mel_bins = num_mel_bins;
    p->num_fft_bins = n_fft / 2 + 1;
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&p->d_window, n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->d_filterbank, 
                          num_mel_bins * p->num_fft_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->d_framed_audio, 
                          max_frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->d_fft_output, 
                          max_frames * p->num_fft_bins * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&p->d_power_spectrum, 
                          max_frames * p->num_fft_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p->d_mel_spectrum, 
                          max_frames * num_mel_bins * sizeof(float)));
    
    // Initialize window and filterbank
    create_hann_window(p->d_window, n_fft);
    create_mel_filterbank(p->d_filterbank, sample_rate, p->num_fft_bins,
                          num_mel_bins, 0.0f, sample_rate / 2.0f);
    
    // Create batched FFT plan
    CUFFT_CHECK(cufftPlan1d(&p->fft_plan, n_fft, CUFFT_R2C, max_frames));
}

void pipeline_destroy(MelSpectrogramPipeline* p) {
    cudaFree(p->d_window);
    cudaFree(p->d_filterbank);
    cudaFree(p->d_framed_audio);
    cudaFree(p->d_fft_output);
    cudaFree(p->d_power_spectrum);
    cudaFree(p->d_mel_spectrum);
    cufftDestroy(p->fft_plan);
}

void pipeline_process(MelSpectrogramPipeline* p, 
                      const float* d_audio, int num_samples,
                      float* d_output, int* out_num_frames) {
    int num_frames = (num_samples - p->n_fft) / p->hop_length + 1;
    *out_num_frames = num_frames;
    
    // Step 1: Apply window to create frames
    dim3 grid1(num_frames);
    dim3 block1(p->n_fft);
    apply_hann_window_kernel<<<grid1, block1>>>(
        d_audio, p->d_framed_audio, p->d_window,
        p->n_fft, p->hop_length, num_frames
    );
    
    // Step 2: FFT
    CUFFT_CHECK(cufftExecR2C(p->fft_plan, p->d_framed_audio, p->d_fft_output));
    
    // Step 3: Power spectrum
    dim3 grid3(num_frames);
    dim3 block3(p->num_fft_bins);
    compute_power_spectrum_kernel<<<grid3, block3>>>(
        p->d_fft_output, p->d_power_spectrum,
        p->num_fft_bins, num_frames
    );
    
    // Step 4: Mel filterbank
    dim3 grid4(num_frames);
    dim3 block4(p->num_mel_bins);
    size_t shared_mem = p->num_fft_bins * sizeof(float);
    apply_mel_filterbank_shared_kernel<<<grid4, block4, shared_mem>>>(
        p->d_power_spectrum, p->d_filterbank, p->d_mel_spectrum,
        p->num_fft_bins, p->num_mel_bins, num_frames
    );
    
    // Step 5: Log compression
    int total_size = num_frames * p->num_mel_bins;
    int block5 = 256;
    int grid5 = (total_size + block5 - 1) / block5;
    log_compress_kernel<<<grid5, block5>>>(p->d_mel_spectrum, total_size, 1e-10f);
    
    // Copy to output
    CUDA_CHECK(cudaMemcpy(d_output, p->d_mel_spectrum,
                          num_frames * p->num_mel_bins * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ============================================================
// BENCHMARKS
// ============================================================

void run_benchmarks() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA AUDIO PROCESSING BENCHMARKS                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Parameters
    int sample_rate = 16000;
    int n_fft = 512;
    int hop_length = 160;
    int num_mel_bins = 80;
    int duration_sec = 10;
    int num_samples = sample_rate * duration_sec;
    int max_frames = (num_samples - n_fft) / hop_length + 1;
    
    printf("Configuration:\n");
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %d seconds (%d samples)\n", duration_sec, num_samples);
    printf("  n_fft: %d, hop: %d, mel bins: %d\n", n_fft, hop_length, num_mel_bins);
    printf("  Expected frames: %d\n\n", max_frames);
    
    // Generate random audio on GPU
    float* d_audio;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_audio, num_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, max_frames * num_mel_bins * sizeof(float)));
    
    // Fill with random data
    float* h_audio = (float*)malloc(num_samples * sizeof(float));
    srand(42);
    for (int i = 0; i < num_samples; i++) {
        h_audio[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_audio, h_audio, num_samples * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // Initialize pipeline
    MelSpectrogramPipeline pipeline;
    pipeline_init(&pipeline, sample_rate, n_fft, hop_length, num_mel_bins, max_frames);
    
    // Warmup
    int out_frames;
    for (int i = 0; i < 5; i++) {
        pipeline_process(&pipeline, d_audio, num_samples, d_output, &out_frames);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    GpuTimer timer;
    timer_create(&timer, "Full mel spectrogram pipeline");
    
    int num_runs = 100;
    timer_start(&timer);
    for (int i = 0; i < num_runs; i++) {
        pipeline_process(&pipeline, d_audio, num_samples, d_output, &out_frames);
    }
    float total_ms = timer_stop(&timer);
    
    float avg_ms = total_ms / num_runs;
    float rtf = (avg_ms / 1000.0f) / duration_sec;
    
    printf("\n─── Results ───\n");
    printf("  Average time: %.3f ms\n", avg_ms);
    printf("  Real-time factor: %.4f (%.0fx faster than real-time)\n", rtf, 1.0f/rtf);
    printf("  Throughput: %.1f seconds of audio per second\n", duration_sec / (avg_ms / 1000.0f));
    
    // Cleanup
    timer_destroy(&timer);
    pipeline_destroy(&pipeline);
    cudaFree(d_audio);
    cudaFree(d_output);
    free(h_audio);
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    // Check CUDA device
    int device;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  CUDA AUDIO PROCESSING                                       █\n");
    printf("█  GPU: %-54s █\n", props.name);
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    run_benchmarks();
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY INSIGHTS                                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  1. cuFFT handles batched FFT efficiently                    ║\n");
    printf("║  2. Shared memory accelerates mel filterbank                 ║\n");
    printf("║  3. GPU overhead amortizes with batch processing             ║\n");
    printf("║  4. Memory bandwidth often the bottleneck                    ║\n");
    printf("║  5. For streaming: use pinned memory + streams               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
