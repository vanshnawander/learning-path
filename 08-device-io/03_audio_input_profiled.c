/**
 * 03_audio_input_profiled.c - Audio Device I/O with Profiling
 * 
 * Understanding audio capture for ML speech/audio models.
 * Every operation is timed.
 * 
 * Compile (Linux): gcc -O2 -o audio_io 03_audio_input_profiled.c -lasound -lrt
 * Note: Requires ALSA development libraries (libasound2-dev)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// Simulated ALSA types for cross-platform compilation
// On Linux, replace with actual ALSA headers
#ifndef __linux__
typedef void* snd_pcm_t;
typedef int snd_pcm_hw_params_t;
#define SND_PCM_FORMAT_S16_LE 0
#define SND_PCM_ACCESS_RW_INTERLEAVED 0
#endif

// ============================================================
// TIMING
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
    double ms = (end.tv_sec - t->start.tv_sec) * 1000.0 +
                (end.tv_nsec - t->start.tv_nsec) / 1e6;
    printf("⏱️  %-35s %8.3f ms\n", t->name, ms);
    return ms;
}

// ============================================================
// AUDIO BUFFER STRUCTURES
// ============================================================

typedef struct {
    int16_t* data;
    int sample_rate;
    int channels;
    int num_samples;
} AudioBuffer;

AudioBuffer* create_audio_buffer(int sample_rate, int channels, int num_samples) {
    AudioBuffer* buf = malloc(sizeof(AudioBuffer));
    buf->sample_rate = sample_rate;
    buf->channels = channels;
    buf->num_samples = num_samples;
    buf->data = aligned_alloc(64, num_samples * channels * sizeof(int16_t));
    return buf;
}

void free_audio_buffer(AudioBuffer* buf) {
    free(buf->data);
    free(buf);
}

// ============================================================
// SIMULATED AUDIO CAPTURE (Cross-platform)
// ============================================================

void simulate_audio_capture(AudioBuffer* buf, double duration_sec) {
    // Simulate microphone capture latency
    // Real latency depends on buffer size and sample rate
    
    // Buffer latency = buffer_size / sample_rate
    // e.g., 512 samples @ 16kHz = 32ms latency
    
    int buffer_size = 512;  // Typical ALSA buffer
    double buffer_latency_ms = (double)buffer_size / buf->sample_rate * 1000;
    
    printf("  Simulated capture: %.1f sec @ %d Hz\n", 
           duration_sec, buf->sample_rate);
    printf("  Buffer latency: %.1f ms (buffer_size=%d)\n", 
           buffer_latency_ms, buffer_size);
    
    // Fill with simulated audio (sine wave + noise)
    for (int i = 0; i < buf->num_samples; i++) {
        double t = (double)i / buf->sample_rate;
        double signal = sin(2 * M_PI * 440 * t) * 16000;  // 440 Hz tone
        signal += (rand() % 1000 - 500);  // Add noise
        buf->data[i] = (int16_t)signal;
    }
}

// ============================================================
// AUDIO PROCESSING OPERATIONS (All Profiled)
// ============================================================

// Convert int16 to float32 (normalize to [-1, 1])
float* int16_to_float32(AudioBuffer* buf) {
    Timer t;
    timer_start(&t, "int16 → float32 conversion");
    
    float* output = aligned_alloc(64, buf->num_samples * sizeof(float));
    
    for (int i = 0; i < buf->num_samples; i++) {
        output[i] = (float)buf->data[i] / 32768.0f;
    }
    
    timer_stop_ms(&t);
    return output;
}

// Simple resampling (linear interpolation)
float* resample(float* input, int input_samples, int input_rate, 
                int output_rate, int* output_samples) {
    Timer t;
    timer_start(&t, "Resample");
    
    double ratio = (double)output_rate / input_rate;
    *output_samples = (int)(input_samples * ratio);
    
    float* output = aligned_alloc(64, *output_samples * sizeof(float));
    
    for (int i = 0; i < *output_samples; i++) {
        double src_idx = i / ratio;
        int idx = (int)src_idx;
        double frac = src_idx - idx;
        
        if (idx + 1 < input_samples) {
            output[i] = input[idx] * (1 - frac) + input[idx + 1] * frac;
        } else {
            output[i] = input[idx];
        }
    }
    
    timer_stop_ms(&t);
    printf("  Resampled: %d → %d samples\n", input_samples, *output_samples);
    return output;
}

// Compute simple energy (for VAD)
float compute_energy(float* audio, int num_samples) {
    Timer t;
    timer_start(&t, "Compute energy (VAD)");
    
    float energy = 0;
    for (int i = 0; i < num_samples; i++) {
        energy += audio[i] * audio[i];
    }
    energy /= num_samples;
    
    timer_stop_ms(&t);
    return energy;
}

// Ring buffer for streaming audio
typedef struct {
    float* data;
    int capacity;
    int write_pos;
    int read_pos;
    int available;
} RingBuffer;

RingBuffer* create_ring_buffer(int capacity) {
    RingBuffer* rb = malloc(sizeof(RingBuffer));
    rb->data = aligned_alloc(64, capacity * sizeof(float));
    rb->capacity = capacity;
    rb->write_pos = 0;
    rb->read_pos = 0;
    rb->available = 0;
    return rb;
}

void ring_buffer_write(RingBuffer* rb, float* data, int count) {
    for (int i = 0; i < count; i++) {
        rb->data[rb->write_pos] = data[i];
        rb->write_pos = (rb->write_pos + 1) % rb->capacity;
        if (rb->available < rb->capacity) {
            rb->available++;
        }
    }
}

int ring_buffer_read(RingBuffer* rb, float* output, int count) {
    if (rb->available < count) return 0;
    
    for (int i = 0; i < count; i++) {
        output[i] = rb->data[rb->read_pos];
        rb->read_pos = (rb->read_pos + 1) % rb->capacity;
    }
    rb->available -= count;
    return count;
}

void free_ring_buffer(RingBuffer* rb) {
    free(rb->data);
    free(rb);
}

// ============================================================
// FULL AUDIO PIPELINE BENCHMARK
// ============================================================

void run_audio_capture_pipeline() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  AUDIO CAPTURE PIPELINE (Profiled)                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int capture_rate = 44100;
    int target_rate = 16000;  // Common for speech ML
    double duration = 5.0;    // 5 seconds
    
    printf("Configuration:\n");
    printf("  Capture rate: %d Hz\n", capture_rate);
    printf("  Target rate: %d Hz\n", target_rate);
    printf("  Duration: %.1f sec\n\n", duration);
    
    Timer total;
    timer_start(&total, "TOTAL PIPELINE");
    
    // 1. Capture audio
    printf("─── Stage 1: Capture ───\n");
    int num_samples = (int)(capture_rate * duration);
    AudioBuffer* raw = create_audio_buffer(capture_rate, 1, num_samples);
    
    Timer t;
    timer_start(&t, "Audio capture (simulated)");
    simulate_audio_capture(raw, duration);
    timer_stop_ms(&t);
    
    size_t raw_size = num_samples * sizeof(int16_t);
    printf("  Raw size: %.2f MB\n\n", raw_size / 1024.0 / 1024.0);
    
    // 2. Convert to float
    printf("─── Stage 2: Convert ───\n");
    float* audio_float = int16_to_float32(raw);
    
    // 3. Resample
    printf("\n─── Stage 3: Resample ───\n");
    int resampled_samples;
    float* resampled = resample(audio_float, num_samples, capture_rate,
                                 target_rate, &resampled_samples);
    
    // 4. Compute energy (simple processing)
    printf("\n─── Stage 4: Processing ───\n");
    float energy = compute_energy(resampled, resampled_samples);
    printf("  Average energy: %.6f\n", energy);
    
    printf("\n");
    timer_stop_ms(&total);
    
    // Memory summary
    printf("\n─── Memory Usage ───\n");
    printf("  Raw capture: %.2f MB\n", raw_size / 1024.0 / 1024.0);
    printf("  Float audio: %.2f MB\n", num_samples * sizeof(float) / 1024.0 / 1024.0);
    printf("  Resampled: %.2f MB\n", resampled_samples * sizeof(float) / 1024.0 / 1024.0);
    
    // Cleanup
    free_audio_buffer(raw);
    free(audio_float);
    free(resampled);
}

// ============================================================
// STREAMING AUDIO BENCHMARK
// ============================================================

void benchmark_streaming_audio() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  STREAMING AUDIO (Real-time Processing)                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int sample_rate = 16000;
    int chunk_size = 512;  // Process in 512-sample chunks (32ms)
    int buffer_capacity = sample_rate * 2;  // 2 seconds
    
    RingBuffer* rb = create_ring_buffer(buffer_capacity);
    float* chunk = aligned_alloc(64, chunk_size * sizeof(float));
    
    printf("Streaming configuration:\n");
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Chunk size: %d samples (%.1f ms)\n", 
           chunk_size, chunk_size * 1000.0 / sample_rate);
    printf("  Buffer capacity: %d samples (%.1f sec)\n\n",
           buffer_capacity, (double)buffer_capacity / sample_rate);
    
    // Simulate real-time streaming
    int num_chunks = 100;
    double total_write_time = 0;
    double total_read_time = 0;
    
    for (int i = 0; i < num_chunks; i++) {
        // Generate chunk (simulates capture)
        for (int j = 0; j < chunk_size; j++) {
            chunk[j] = (float)(rand() % 1000 - 500) / 500.0f;
        }
        
        // Write to ring buffer
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        ring_buffer_write(rb, chunk, chunk_size);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_write_time += (end.tv_sec - start.tv_sec) * 1e6 +
                           (end.tv_nsec - start.tv_nsec) / 1000.0;
        
        // Read from ring buffer
        float output[512];
        clock_gettime(CLOCK_MONOTONIC, &start);
        ring_buffer_read(rb, output, chunk_size);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_read_time += (end.tv_sec - start.tv_sec) * 1e6 +
                          (end.tv_nsec - start.tv_nsec) / 1000.0;
    }
    
    printf("Ring buffer performance (%d chunks):\n", num_chunks);
    printf("  Avg write: %.2f µs/chunk\n", total_write_time / num_chunks);
    printf("  Avg read:  %.2f µs/chunk\n", total_read_time / num_chunks);
    printf("  Overhead:  << 1%% of chunk duration (%.0f µs)\n",
           chunk_size * 1e6 / sample_rate);
    
    free_ring_buffer(rb);
    free(chunk);
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█  AUDIO DEVICE I/O FOR ML                                     █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    run_audio_capture_pipeline();
    benchmark_streaming_audio();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAYS FOR ML AUDIO                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. BUFFER SIZE AFFECTS LATENCY                              ║\n");
    printf("║     • 512 samples @ 16kHz = 32ms latency                    ║\n");
    printf("║     • Smaller = lower latency but more CPU overhead         ║\n");
    printf("║                                                              ║\n");
    printf("║  2. SAMPLE RATE AFFECTS PROCESSING                           ║\n");
    printf("║     • 16kHz is enough for speech (8kHz bandwidth)           ║\n");
    printf("║     • 44.1kHz for music, but 2.75x more data                ║\n");
    printf("║                                                              ║\n");
    printf("║  3. RING BUFFERS FOR STREAMING                               ║\n");
    printf("║     • Decouples capture from processing                     ║\n");
    printf("║     • Handles variable processing time                      ║\n");
    printf("║                                                              ║\n");
    printf("║  4. FORMAT CONVERSION IS FAST                                ║\n");
    printf("║     • int16 → float32: ~1ms for 5 sec audio                 ║\n");
    printf("║     • Not a bottleneck for most pipelines                   ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
