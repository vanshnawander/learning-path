/**
 * 02_image_decode_profiled.c - Profile Image Decoding Operations
 * 
 * Every image operation is timed to show real costs.
 * This is what happens inside FFCV, torchvision, PIL.
 * 
 * Compile: gcc -O3 -o image_decode 02_image_decode_profiled.c -lrt -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

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

double timer_stop(Timer* t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - t->start.tv_sec) + 
                     (end.tv_nsec - t->start.tv_nsec) * 1e-9;
    printf("⏱️  %-30s %8.3f ms\n", t->name, elapsed * 1000);
    return elapsed;
}

// ============================================================
// IMAGE STRUCTURES
// ============================================================

typedef struct {
    uint8_t* data;      // RGB interleaved
    int width;
    int height;
    int channels;
} Image;

typedef struct {
    float* data;        // CHW layout for ML
    int channels;
    int height;
    int width;
} Tensor;

// ============================================================
// IMAGE OPERATIONS (ALL PROFILED)
// ============================================================

Image* create_image(int width, int height, int channels) {
    Image* img = malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = aligned_alloc(64, width * height * channels);
    return img;
}

void free_image(Image* img) {
    free(img->data);
    free(img);
}

Tensor* create_tensor(int channels, int height, int width) {
    Tensor* t = malloc(sizeof(Tensor));
    t->channels = channels;
    t->height = height;
    t->width = width;
    t->data = aligned_alloc(64, channels * height * width * sizeof(float));
    return t;
}

void free_tensor(Tensor* t) {
    free(t->data);
    free(t);
}

// Simulate random image data (like decoded JPEG)
void fill_random_image(Image* img) {
    Timer t;
    timer_start(&t, "Fill random pixels");
    
    size_t size = img->width * img->height * img->channels;
    for (size_t i = 0; i < size; i++) {
        img->data[i] = rand() % 256;
    }
    
    timer_stop(&t);
}

// HWC (Height×Width×Channels) to CHW (Channels×Height×Width)
// This is what PyTorch's permute(2,0,1) does
void hwc_to_chw(Image* img, Tensor* tensor) {
    Timer t;
    timer_start(&t, "HWC → CHW permute");
    
    int H = img->height;
    int W = img->width;
    int C = img->channels;
    
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                // Source: HWC layout
                int src_idx = h * W * C + w * C + c;
                // Dest: CHW layout
                int dst_idx = c * H * W + h * W + w;
                
                tensor->data[dst_idx] = (float)img->data[src_idx];
            }
        }
    }
    
    timer_stop(&t);
}

// Normalize to [0, 1]
void normalize_0_1(Tensor* t) {
    Timer timer;
    timer_start(&timer, "Normalize /255");
    
    size_t size = t->channels * t->height * t->width;
    for (size_t i = 0; i < size; i++) {
        t->data[i] /= 255.0f;
    }
    
    timer_stop(&timer);
}

// ImageNet normalization
void imagenet_normalize(Tensor* t) {
    Timer timer;
    timer_start(&timer, "ImageNet normalize");
    
    // ImageNet mean and std per channel
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};
    
    int plane_size = t->height * t->width;
    
    for (int c = 0; c < t->channels; c++) {
        float* channel_data = t->data + c * plane_size;
        for (int i = 0; i < plane_size; i++) {
            channel_data[i] = (channel_data[i] - mean[c]) / std[c];
        }
    }
    
    timer_stop(&timer);
}

// Bilinear resize
Image* resize_bilinear(Image* src, int new_width, int new_height) {
    Timer t;
    timer_start(&t, "Bilinear resize");
    
    Image* dst = create_image(new_width, new_height, src->channels);
    
    float x_ratio = (float)(src->width - 1) / new_width;
    float y_ratio = (float)(src->height - 1) / new_height;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float gx = x * x_ratio;
            float gy = y * y_ratio;
            int gxi = (int)gx;
            int gyi = (int)gy;
            float dx = gx - gxi;
            float dy = gy - gyi;
            
            for (int c = 0; c < src->channels; c++) {
                // Get 4 neighbors
                uint8_t* p = src->data;
                int idx00 = (gyi * src->width + gxi) * src->channels + c;
                int idx10 = (gyi * src->width + (gxi + 1)) * src->channels + c;
                int idx01 = ((gyi + 1) * src->width + gxi) * src->channels + c;
                int idx11 = ((gyi + 1) * src->width + (gxi + 1)) * src->channels + c;
                
                // Bilinear interpolation
                float value = p[idx00] * (1 - dx) * (1 - dy) +
                              p[idx10] * dx * (1 - dy) +
                              p[idx01] * (1 - dx) * dy +
                              p[idx11] * dx * dy;
                
                int dst_idx = (y * new_width + x) * dst->channels + c;
                dst->data[dst_idx] = (uint8_t)value;
            }
        }
    }
    
    timer_stop(&t);
    return dst;
}

// Center crop
Image* center_crop(Image* src, int crop_size) {
    Timer t;
    timer_start(&t, "Center crop");
    
    Image* dst = create_image(crop_size, crop_size, src->channels);
    
    int x_offset = (src->width - crop_size) / 2;
    int y_offset = (src->height - crop_size) / 2;
    
    for (int y = 0; y < crop_size; y++) {
        for (int x = 0; x < crop_size; x++) {
            for (int c = 0; c < src->channels; c++) {
                int src_idx = ((y + y_offset) * src->width + (x + x_offset)) 
                              * src->channels + c;
                int dst_idx = (y * crop_size + x) * src->channels + c;
                dst->data[dst_idx] = src->data[src_idx];
            }
        }
    }
    
    timer_stop(&t);
    return dst;
}

// Horizontal flip (data augmentation)
void horizontal_flip(Image* img) {
    Timer t;
    timer_start(&t, "Horizontal flip");
    
    int W = img->width;
    int H = img->height;
    int C = img->channels;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W / 2; x++) {
            for (int c = 0; c < C; c++) {
                int left_idx = (y * W + x) * C + c;
                int right_idx = (y * W + (W - 1 - x)) * C + c;
                
                uint8_t temp = img->data[left_idx];
                img->data[left_idx] = img->data[right_idx];
                img->data[right_idx] = temp;
            }
        }
    }
    
    timer_stop(&t);
}

// ============================================================
// FULL PIPELINE BENCHMARK
// ============================================================

void run_full_pipeline() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  FULL IMAGE PREPROCESSING PIPELINE (Profiled)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Simulate a 1920x1080 decoded image (like from JPEG)
    printf("Input: 1920×1080 RGB image (simulated decoded JPEG)\n");
    printf("Output: 224×224 normalized tensor (CHW format)\n\n");
    
    Timer total;
    timer_start(&total, "TOTAL PIPELINE");
    
    printf("─── Pipeline Steps ───\n\n");
    
    // Step 1: Create/decode image
    Image* original = create_image(1920, 1080, 3);
    fill_random_image(original);
    printf("  Original size: %.2f MB\n\n", 
           original->width * original->height * original->channels / 1024.0 / 1024.0);
    
    // Step 2: Resize to 256x256
    Image* resized = resize_bilinear(original, 256, 256);
    
    // Step 3: Center crop to 224x224
    Image* cropped = center_crop(resized, 224);
    
    // Step 4: Random horizontal flip
    horizontal_flip(cropped);
    
    // Step 5: Convert to tensor (HWC → CHW)
    Tensor* tensor = create_tensor(3, 224, 224);
    hwc_to_chw(cropped, tensor);
    
    // Step 6: Normalize to [0, 1]
    normalize_0_1(tensor);
    
    // Step 7: ImageNet normalization
    imagenet_normalize(tensor);
    
    printf("\n");
    double total_time = timer_stop(&total);
    
    printf("\n─── Memory Usage ───\n");
    printf("  Original:  %.2f MB\n", 1920 * 1080 * 3 / 1024.0 / 1024.0);
    printf("  Resized:   %.2f MB\n", 256 * 256 * 3 / 1024.0 / 1024.0);
    printf("  Tensor:    %.2f MB\n", 224 * 224 * 3 * 4 / 1024.0 / 1024.0);
    
    printf("\n─── Throughput ───\n");
    printf("  Images/sec: %.1f\n", 1000.0 / (total_time * 1000));
    printf("  For batch of 32: %.1f ms\n", total_time * 1000 * 32);
    
    // Cleanup
    free_image(original);
    free_image(resized);
    free_image(cropped);
    free_tensor(tensor);
}

// ============================================================
// BATCH PROCESSING BENCHMARK
// ============================================================

void benchmark_batch_processing() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  BATCH PROCESSING COMPARISON                                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int batch_sizes[] = {1, 8, 16, 32, 64};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    printf("%-10s %-15s %-15s %-15s\n", 
           "Batch", "Total (ms)", "Per Image (ms)", "Images/sec");
    printf("────────────────────────────────────────────────────────────────\n");
    
    for (int b = 0; b < num_batches; b++) {
        int batch_size = batch_sizes[b];
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Process batch
        for (int i = 0; i < batch_size; i++) {
            Image* img = create_image(224, 224, 3);
            // Simulate minimal processing
            for (int j = 0; j < 224 * 224 * 3; j++) {
                img->data[j] = rand() % 256;
            }
            Tensor* t = create_tensor(3, 224, 224);
            hwc_to_chw(img, t);
            free_tensor(t);
            free_image(img);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) * 1e-9;
        
        printf("%-10d %-15.2f %-15.3f %-15.1f\n",
               batch_size,
               elapsed * 1000,
               elapsed * 1000 / batch_size,
               batch_size / elapsed);
    }
    
    printf("\n💡 Note: Real GPU batching would show different scaling!\n");
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  IMAGE PROCESSING WITH PROFILING                             █\n");
    printf("█  Every operation timed - understand the real costs!          █\n");
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    run_full_pipeline();
    benchmark_batch_processing();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAYS FOR MULTIMODAL TRAINING                       ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. Resize is expensive - pre-resize your dataset!           ║\n");
    printf("║                                                              ║\n");
    printf("║  2. Memory layout conversion (HWC→CHW) has overhead          ║\n");
    printf("║                                                              ║\n");
    printf("║  3. Total preprocessing can take 5-20ms per image            ║\n");
    printf("║                                                              ║\n");
    printf("║  4. For 32 images at 10ms each = 320ms (significant!)        ║\n");
    printf("║                                                              ║\n");
    printf("║  5. This is why FFCV pre-processes and memory-maps data      ║\n");
    printf("║                                                              ║\n");
    printf("║  6. GPU decode (NVJPEG) can be 10x faster for JPEG           ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
