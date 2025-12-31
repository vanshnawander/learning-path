/**
 * 03_video_decode_profiled.c - Profile Video Decoding Operations
 * 
 * Every video operation is timed. Understand why video loading
 * is the #1 bottleneck in multimodal training.
 * 
 * Compile: gcc -O3 -o video_decode 03_video_decode_profiled.c -lrt -lm
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

double timer_stop_ms(Timer* t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = ((end.tv_sec - t->start.tv_sec) * 1000.0 + 
                         (end.tv_nsec - t->start.tv_nsec) / 1e6);
    printf("⏱️  %-40s %8.3f ms\n", t->name, elapsed_ms);
    return elapsed_ms;
}

// ============================================================
// VIDEO STRUCTURES
// ============================================================

typedef struct {
    uint8_t* y;     // Luma plane
    uint8_t* u;     // Chroma U plane (half resolution)
    uint8_t* v;     // Chroma V plane (half resolution)
    int width;
    int height;
} YUVFrame;

typedef struct {
    uint8_t* data;  // RGB interleaved
    int width;
    int height;
} RGBFrame;

typedef struct {
    float* data;    // TCHW format for ML
    int num_frames;
    int channels;
    int height;
    int width;
} VideoTensor;

// ============================================================
// FRAME OPERATIONS (ALL PROFILED)
// ============================================================

YUVFrame* create_yuv_frame(int width, int height) {
    YUVFrame* frame = malloc(sizeof(YUVFrame));
    frame->width = width;
    frame->height = height;
    frame->y = aligned_alloc(64, width * height);
    frame->u = aligned_alloc(64, (width/2) * (height/2));
    frame->v = aligned_alloc(64, (width/2) * (height/2));
    return frame;
}

void free_yuv_frame(YUVFrame* frame) {
    free(frame->y);
    free(frame->u);
    free(frame->v);
    free(frame);
}

RGBFrame* create_rgb_frame(int width, int height) {
    RGBFrame* frame = malloc(sizeof(RGBFrame));
    frame->width = width;
    frame->height = height;
    frame->data = aligned_alloc(64, width * height * 3);
    return frame;
}

void free_rgb_frame(RGBFrame* frame) {
    free(frame->data);
    free(frame);
}

VideoTensor* create_video_tensor(int num_frames, int channels, int height, int width) {
    VideoTensor* tensor = malloc(sizeof(VideoTensor));
    tensor->num_frames = num_frames;
    tensor->channels = channels;
    tensor->height = height;
    tensor->width = width;
    tensor->data = aligned_alloc(64, num_frames * channels * height * width * sizeof(float));
    return tensor;
}

void free_video_tensor(VideoTensor* tensor) {
    free(tensor->data);
    free(tensor);
}

// Simulate H.264/H.265 decode (just fills with data)
void simulate_video_decode(YUVFrame* frame) {
    // In reality, this would be the codec decode step
    // which is VERY expensive without hardware acceleration
    
    // Fill Y plane
    for (int i = 0; i < frame->width * frame->height; i++) {
        frame->y[i] = rand() % 256;
    }
    
    // Fill U, V planes (quarter size due to 4:2:0)
    int uv_size = (frame->width/2) * (frame->height/2);
    for (int i = 0; i < uv_size; i++) {
        frame->u[i] = rand() % 256;
        frame->v[i] = rand() % 256;
    }
}

// YUV420 to RGB conversion (BT.601)
void yuv_to_rgb(YUVFrame* yuv, RGBFrame* rgb) {
    Timer t;
    timer_start(&t, "YUV420 → RGB conversion");
    
    for (int j = 0; j < yuv->height; j++) {
        for (int i = 0; i < yuv->width; i++) {
            int y_idx = j * yuv->width + i;
            int uv_idx = (j/2) * (yuv->width/2) + (i/2);
            
            int y = yuv->y[y_idx];
            int u = yuv->u[uv_idx] - 128;
            int v = yuv->v[uv_idx] - 128;
            
            // BT.601 conversion
            int r = y + (int)(1.402 * v);
            int g = y - (int)(0.344 * u + 0.714 * v);
            int b = y + (int)(1.772 * u);
            
            // Clamp
            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);
            
            int rgb_idx = (j * yuv->width + i) * 3;
            rgb->data[rgb_idx + 0] = r;
            rgb->data[rgb_idx + 1] = g;
            rgb->data[rgb_idx + 2] = b;
        }
    }
    
    timer_stop_ms(&t);
}

// Bilinear resize frame
RGBFrame* resize_frame(RGBFrame* src, int new_width, int new_height) {
    Timer t;
    timer_start(&t, "Bilinear resize frame");
    
    RGBFrame* dst = create_rgb_frame(new_width, new_height);
    
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
            
            for (int c = 0; c < 3; c++) {
                int idx00 = (gyi * src->width + gxi) * 3 + c;
                int idx10 = (gyi * src->width + (gxi + 1)) * 3 + c;
                int idx01 = ((gyi + 1) * src->width + gxi) * 3 + c;
                int idx11 = ((gyi + 1) * src->width + (gxi + 1)) * 3 + c;
                
                float value = src->data[idx00] * (1 - dx) * (1 - dy) +
                              src->data[idx10] * dx * (1 - dy) +
                              src->data[idx01] * (1 - dx) * dy +
                              src->data[idx11] * dx * dy;
                
                int dst_idx = (y * new_width + x) * 3 + c;
                dst->data[dst_idx] = (uint8_t)value;
            }
        }
    }
    
    timer_stop_ms(&t);
    return dst;
}

// Convert frame to tensor (normalize and change layout)
void frame_to_tensor(RGBFrame* frame, float* tensor_data, int frame_idx, int H, int W) {
    // Output layout: TCHW
    // tensor_data points to start of this frame's data
    
    int plane_size = H * W;
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int src_idx = (h * W + w) * 3 + c;
                int dst_idx = c * plane_size + h * W + w;
                tensor_data[dst_idx] = frame->data[src_idx] / 255.0f;
            }
        }
    }
}

// ============================================================
// FULL VIDEO PIPELINE
// ============================================================

void run_video_pipeline() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  FULL VIDEO PREPROCESSING PIPELINE (Profiled)                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // Simulate 1080p video clip (16 frames @ 30fps = 0.5 sec)
    int src_width = 1920;
    int src_height = 1080;
    int num_frames = 16;
    int target_size = 224;
    
    printf("Input: %dx%d, %d frames (simulated H.264 decode)\n", 
           src_width, src_height, num_frames);
    printf("Output: %dx%d, %d frames, TCHW tensor\n\n", 
           target_size, target_size, num_frames);
    
    // Calculate sizes
    size_t yuv_frame_size = src_width * src_height + 
                            2 * (src_width/2) * (src_height/2);
    size_t rgb_frame_size = src_width * src_height * 3;
    size_t tensor_size = num_frames * 3 * target_size * target_size * sizeof(float);
    
    printf("─── Memory Sizes ───\n");
    printf("  YUV frame (4:2:0): %.2f MB\n", yuv_frame_size / 1024.0 / 1024.0);
    printf("  RGB frame: %.2f MB\n", rgb_frame_size / 1024.0 / 1024.0);
    printf("  Output tensor: %.2f MB\n\n", tensor_size / 1024.0 / 1024.0);
    
    // Create output tensor
    VideoTensor* output = create_video_tensor(num_frames, 3, target_size, target_size);
    
    Timer total;
    timer_start(&total, "TOTAL PIPELINE");
    
    printf("─── Per-Frame Pipeline ───\n\n");
    
    double decode_total = 0, yuv_rgb_total = 0, resize_total = 0, tensor_total = 0;
    
    for (int f = 0; f < num_frames; f++) {
        if (f == 0) printf("Frame 0 (detailed):\n");
        
        // Step 1: Decode (simulated)
        Timer t;
        timer_start(&t, "  Decode frame (simulated)");
        YUVFrame* yuv = create_yuv_frame(src_width, src_height);
        simulate_video_decode(yuv);
        double decode_time = timer_stop_ms(&t);
        if (f > 0) decode_total += decode_time;
        else decode_total = decode_time;
        
        // Step 2: YUV to RGB
        RGBFrame* rgb = create_rgb_frame(src_width, src_height);
        if (f == 0) {
            yuv_to_rgb(yuv, rgb);
        } else {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            // Inline conversion without print
            for (int j = 0; j < yuv->height; j++) {
                for (int i = 0; i < yuv->width; i++) {
                    int y_idx = j * yuv->width + i;
                    int uv_idx = (j/2) * (yuv->width/2) + (i/2);
                    int y = yuv->y[y_idx];
                    int u = yuv->u[uv_idx] - 128;
                    int v = yuv->v[uv_idx] - 128;
                    int r = y + (int)(1.402 * v);
                    int g = y - (int)(0.344 * u + 0.714 * v);
                    int b = y + (int)(1.772 * u);
                    r = r < 0 ? 0 : (r > 255 ? 255 : r);
                    g = g < 0 ? 0 : (g > 255 ? 255 : g);
                    b = b < 0 ? 0 : (b > 255 ? 255 : b);
                    int rgb_idx = (j * yuv->width + i) * 3;
                    rgb->data[rgb_idx + 0] = r;
                    rgb->data[rgb_idx + 1] = g;
                    rgb->data[rgb_idx + 2] = b;
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            yuv_rgb_total += (end.tv_sec - start.tv_sec) * 1000.0 + 
                             (end.tv_nsec - start.tv_nsec) / 1e6;
        }
        
        // Step 3: Resize
        RGBFrame* resized;
        if (f == 0) {
            resized = resize_frame(rgb, target_size, target_size);
        } else {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            resized = create_rgb_frame(target_size, target_size);
            float x_ratio = (float)(rgb->width - 1) / target_size;
            float y_ratio = (float)(rgb->height - 1) / target_size;
            for (int y = 0; y < target_size; y++) {
                for (int x = 0; x < target_size; x++) {
                    float gx = x * x_ratio;
                    float gy = y * y_ratio;
                    int gxi = (int)gx;
                    int gyi = (int)gy;
                    for (int c = 0; c < 3; c++) {
                        int idx = (gyi * rgb->width + gxi) * 3 + c;
                        resized->data[(y * target_size + x) * 3 + c] = rgb->data[idx];
                    }
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            resize_total += (end.tv_sec - start.tv_sec) * 1000.0 + 
                            (end.tv_nsec - start.tv_nsec) / 1e6;
        }
        
        // Step 4: To tensor
        if (f == 0) {
            timer_start(&t, "  Frame → Tensor (normalize, CHW)");
        }
        float* frame_tensor = output->data + f * 3 * target_size * target_size;
        frame_to_tensor(resized, frame_tensor, f, target_size, target_size);
        if (f == 0) {
            tensor_total = timer_stop_ms(&t);
        }
        
        // Cleanup
        free_yuv_frame(yuv);
        free_rgb_frame(rgb);
        free_rgb_frame(resized);
    }
    
    printf("\n");
    double total_time = timer_stop_ms(&total);
    
    printf("\n─── Timing Breakdown (%d frames) ───\n", num_frames);
    printf("  Decode:      %.1f ms total (%.1f ms/frame)\n", 
           decode_total, decode_total / num_frames);
    printf("  YUV→RGB:     %.1f ms total (%.1f ms/frame)\n",
           yuv_rgb_total, yuv_rgb_total / num_frames);
    printf("  Resize:      %.1f ms total (%.1f ms/frame)\n",
           resize_total, resize_total / num_frames);
    printf("  To tensor:   ~%.1f ms total\n", tensor_total * num_frames);
    
    printf("\n─── Performance ───\n");
    printf("  Total time: %.1f ms\n", total_time);
    printf("  Per frame: %.1f ms\n", total_time / num_frames);
    printf("  FPS: %.1f\n", num_frames / (total_time / 1000.0));
    
    free_video_tensor(output);
}

// ============================================================
// COMPARE RESOLUTIONS
// ============================================================

void benchmark_resolutions() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  VIDEO PROCESSING TIME BY RESOLUTION                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    struct { int w; int h; const char* name; } resolutions[] = {
        {640, 480, "480p"},
        {1280, 720, "720p"},
        {1920, 1080, "1080p"},
        {2560, 1440, "1440p"},
        {3840, 2160, "4K"},
    };
    int num_res = sizeof(resolutions) / sizeof(resolutions[0]);
    
    printf("%-10s %-12s %-15s %-12s\n", 
           "Resolution", "Pixels (M)", "YUV→RGB (ms)", "Total (ms)");
    printf("────────────────────────────────────────────────────────────────\n");
    
    for (int r = 0; r < num_res; r++) {
        int w = resolutions[r].w;
        int h = resolutions[r].h;
        
        YUVFrame* yuv = create_yuv_frame(w, h);
        simulate_video_decode(yuv);
        RGBFrame* rgb = create_rgb_frame(w, h);
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // YUV to RGB
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                int y_idx = j * w + i;
                int uv_idx = (j/2) * (w/2) + (i/2);
                int y = yuv->y[y_idx];
                int u = yuv->u[uv_idx] - 128;
                int v = yuv->v[uv_idx] - 128;
                int rv = y + (int)(1.402 * v);
                int gv = y - (int)(0.344 * u + 0.714 * v);
                int bv = y + (int)(1.772 * u);
                rv = rv < 0 ? 0 : (rv > 255 ? 255 : rv);
                gv = gv < 0 ? 0 : (gv > 255 ? 255 : gv);
                bv = bv < 0 ? 0 : (bv > 255 ? 255 : bv);
                int rgb_idx = (j * w + i) * 3;
                rgb->data[rgb_idx + 0] = rv;
                rgb->data[rgb_idx + 1] = gv;
                rgb->data[rgb_idx + 2] = bv;
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double yuv_rgb_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                              (end.tv_nsec - start.tv_nsec) / 1e6;
        
        // Total includes decode simulation
        clock_gettime(CLOCK_MONOTONIC, &start);
        simulate_video_decode(yuv);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double decode_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                             (end.tv_nsec - start.tv_nsec) / 1e6;
        
        printf("%-10s %-12.2f %-15.2f %-12.2f\n",
               resolutions[r].name,
               (w * h) / 1e6,
               yuv_rgb_time,
               yuv_rgb_time + decode_time);
        
        free_yuv_frame(yuv);
        free_rgb_frame(rgb);
    }
    
    printf("\n💡 4K is ~9x more pixels than 1080p = ~9x processing time!\n");
    printf("   Consider resizing videos before training.\n");
}

int main() {
    printf("\n");
    printf("████████████████████████████████████████████████████████████████\n");
    printf("█                                                              █\n");
    printf("█  VIDEO PROCESSING WITH PROFILING                             █\n");
    printf("█  Understanding why video is the hardest modality             █\n");
    printf("█                                                              █\n");
    printf("████████████████████████████████████████████████████████████████\n");
    
    run_video_pipeline();
    benchmark_resolutions();
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  KEY TAKEAWAYS FOR MULTIMODAL VIDEO TRAINING                 ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  1. Video decode is EXPENSIVE (10-50ms per frame on CPU)     ║\n");
    printf("║     → Use hardware decode: NVDEC, VAAPI, VideoToolbox        ║\n");
    printf("║                                                              ║\n");
    printf("║  2. YUV→RGB conversion adds ~5-20ms per 1080p frame          ║\n");
    printf("║     → Can be done on GPU during decode                       ║\n");
    printf("║                                                              ║\n");
    printf("║  3. Seeking to random frames requires decoding from I-frame  ║\n");
    printf("║     → Sample clips, not random frames                        ║\n");
    printf("║     → Pre-extract keyframes for random access                ║\n");
    printf("║                                                              ║\n");
    printf("║  4. Resolution dominates processing time                     ║\n");
    printf("║     → Pre-resize videos to training resolution               ║\n");
    printf("║                                                              ║\n");
    printf("║  5. 16 frames @ 1080p = ~100-500ms on CPU                    ║\n");
    printf("║     → This is often > forward pass time!                     ║\n");
    printf("║     → Video loading is THE bottleneck                        ║\n");
    printf("║                                                              ║\n");
    printf("║  SOLUTIONS:                                                  ║\n");
    printf("║  • NVIDIA DALI for GPU decode                                ║\n");
    printf("║  • decord library (optimized for ML)                         ║\n");
    printf("║  • Pre-extract frames to images                              ║\n");
    printf("║  • WebDataset with pre-extracted clips                       ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
