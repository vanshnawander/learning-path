/**
 * 02_color_spaces.c - Understanding Color Representations
 * 
 * Color space conversions are fundamental to video/image processing.
 * This is what happens inside video decoders.
 * 
 * Compile: gcc -O2 -o color_spaces 02_color_spaces.c -lm
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

// ============================================================
// RGB: The Display Color Space
// ============================================================

typedef struct {
    uint8_t r, g, b;
} RGB;

typedef struct {
    uint8_t y, u, v;
} YUV;

typedef struct {
    float h, s, v;  // Hue [0-360], Saturation [0-1], Value [0-1]
} HSV;

// ============================================================
// RGB ↔ YUV Conversion (BT.601 standard)
// ============================================================

YUV rgb_to_yuv(RGB rgb) {
    // BT.601 conversion (used in most video)
    // Y = 0.299R + 0.587G + 0.114B
    // U = -0.169R - 0.331G + 0.500B + 128
    // V = 0.500R - 0.419G - 0.081B + 128
    
    YUV yuv;
    yuv.y = (uint8_t)(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b);
    yuv.u = (uint8_t)(-0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 128);
    yuv.v = (uint8_t)(0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 128);
    
    return yuv;
}

RGB yuv_to_rgb(YUV yuv) {
    // Inverse conversion
    // R = Y + 1.402(V - 128)
    // G = Y - 0.344(U - 128) - 0.714(V - 128)
    // B = Y + 1.772(U - 128)
    
    RGB rgb;
    int r = yuv.y + 1.402 * (yuv.v - 128);
    int g = yuv.y - 0.344 * (yuv.u - 128) - 0.714 * (yuv.v - 128);
    int b = yuv.y + 1.772 * (yuv.u - 128);
    
    // Clamp to [0, 255]
    rgb.r = r < 0 ? 0 : (r > 255 ? 255 : r);
    rgb.g = g < 0 ? 0 : (g > 255 ? 255 : g);
    rgb.b = b < 0 ? 0 : (b > 255 ? 255 : b);
    
    return rgb;
}

// ============================================================
// RGB ↔ HSV Conversion
// ============================================================

HSV rgb_to_hsv(RGB rgb) {
    float r = rgb.r / 255.0f;
    float g = rgb.g / 255.0f;
    float b = rgb.b / 255.0f;
    
    float max = fmaxf(fmaxf(r, g), b);
    float min = fminf(fminf(r, g), b);
    float delta = max - min;
    
    HSV hsv;
    hsv.v = max;
    hsv.s = (max == 0) ? 0 : delta / max;
    
    if (delta == 0) {
        hsv.h = 0;
    } else if (max == r) {
        hsv.h = 60 * fmodf((g - b) / delta, 6);
    } else if (max == g) {
        hsv.h = 60 * ((b - r) / delta + 2);
    } else {
        hsv.h = 60 * ((r - g) / delta + 4);
    }
    
    if (hsv.h < 0) hsv.h += 360;
    
    return hsv;
}

// ============================================================
// Demonstrate conversions
// ============================================================

void print_rgb(const char* name, RGB rgb) {
    printf("%s: RGB(%3d, %3d, %3d)\n", name, rgb.r, rgb.g, rgb.b);
}

void print_yuv(const char* name, YUV yuv) {
    printf("%s: YUV(%3d, %3d, %3d)\n", name, yuv.y, yuv.u, yuv.v);
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║              COLOR SPACES FOR VIDEO/ML                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    // ================================================================
    // Test colors
    // ================================================================
    printf("=== RGB TO YUV CONVERSION ===\n\n");
    
    RGB colors[] = {
        {255, 0, 0},     // Red
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 255, 255}, // White
        {0, 0, 0},       // Black
        {128, 128, 128}, // Gray
        {255, 255, 0},   // Yellow
    };
    const char* names[] = {"Red", "Green", "Blue", "White", "Black", "Gray", "Yellow"};
    
    for (int i = 0; i < 7; i++) {
        YUV yuv = rgb_to_yuv(colors[i]);
        RGB back = yuv_to_rgb(yuv);
        
        printf("%-8s: RGB(%3d,%3d,%3d) → YUV(%3d,%3d,%3d) → RGB(%3d,%3d,%3d)\n",
               names[i],
               colors[i].r, colors[i].g, colors[i].b,
               yuv.y, yuv.u, yuv.v,
               back.r, back.g, back.b);
    }
    
    // ================================================================
    // Key observations
    // ================================================================
    printf("\n=== KEY OBSERVATIONS ===\n\n");
    
    printf("1. Y (Luminance) captures brightness:\n");
    printf("   White Y=255, Black Y=0, Gray Y=128\n\n");
    
    printf("2. U,V (Chrominance) center at 128:\n");
    printf("   Gray has U=V=128 (no color)\n\n");
    
    printf("3. Y has most visual information:\n");
    printf("   Can discard U,V detail (chroma subsampling)\n\n");
    
    // ================================================================
    // Chroma subsampling demonstration
    // ================================================================
    printf("=== CHROMA SUBSAMPLING ===\n\n");
    
    printf("4:4:4 - Full resolution:\n");
    printf("  Y: ████████ (8 samples)\n");
    printf("  U: ████████ (8 samples)\n");
    printf("  V: ████████ (8 samples)\n");
    printf("  Total: 24 samples\n\n");
    
    printf("4:2:2 - Half horizontal chroma:\n");
    printf("  Y: ████████ (8 samples)\n");
    printf("  U: ████     (4 samples)\n");
    printf("  V: ████     (4 samples)\n");
    printf("  Total: 16 samples (1.5x compression)\n\n");
    
    printf("4:2:0 - Quarter chroma (most common):\n");
    printf("  Y: ████████ (8 samples for 2 rows)\n");
    printf("  U: ██       (2 samples)\n");
    printf("  V: ██       (2 samples)\n");
    printf("  Total: 12 samples (2x compression)\n\n");
    
    // ================================================================
    // ML implications
    // ================================================================
    printf("=== ML IMPLICATIONS ===\n\n");
    
    printf("1. VIDEO DECODE:\n");
    printf("   Most video stored as YUV 4:2:0\n");
    printf("   Decoder outputs YUV, must convert to RGB\n");
    printf("   This conversion can be GPU-accelerated\n\n");
    
    printf("2. NORMALIZATION:\n");
    printf("   RGB [0,255] → float [-1,1] or [0,1]\n");
    printf("   ImageNet mean: [0.485, 0.456, 0.406]\n");
    printf("   ImageNet std:  [0.229, 0.224, 0.225]\n\n");
    
    printf("3. AUGMENTATION:\n");
    printf("   HSV space good for color jitter\n");
    printf("   Change H: shift hue\n");
    printf("   Change S: saturation\n");
    printf("   Change V: brightness\n\n");
    
    printf("4. GRAYSCALE:\n");
    printf("   Just use Y channel!\n");
    printf("   gray = 0.299*R + 0.587*G + 0.114*B\n");
    
    return 0;
}
