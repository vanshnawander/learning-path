/**
 * 02_floating_point.c - IEEE 754 Floating Point Deep Dive
 * 
 * This is ESSENTIAL for ML. Every weight, activation, gradient is a float.
 * Understanding IEEE 754 explains:
 * - Why FP16 training needs loss scaling
 * - Why BF16 is better for gradients than FP16
 * - What denormals are and why they slow down GPUs
 * - Numerical stability issues
 * 
 * Compile: gcc -o 02_floating_point 02_floating_point.c -lm
 * Run: ./02_floating_point
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Helper to print binary
void print_binary_32(uint32_t value) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (value >> i) & 1);
        if (i == 31 || i == 23) printf(" ");  // Separate sign, exponent, mantissa
    }
}

// View float as uint32_t (same bits, different interpretation)
uint32_t float_to_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return bits;
}

float bits_to_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// FP16 simulation
uint16_t float_to_fp16(float f) {
    uint32_t bits = float_to_bits(f);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  // Unbias FP32 exponent
    uint32_t mant = bits & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 128) {  // Inf or NaN
        return (sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0);
    }
    if (exp < -24) {  // Too small, flush to zero
        return sign << 15;
    }
    if (exp < -14) {  // Denormal
        mant |= 0x800000;  // Add implicit 1
        mant >>= (-14 - exp);
        return (sign << 15) | (mant >> 13);
    }
    if (exp > 15) {  // Too large, infinity
        return (sign << 15) | (0x1F << 10);
    }
    
    // Normal number
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

// BF16 simulation (much simpler - just truncate!)
uint16_t float_to_bf16(float f) {
    uint32_t bits = float_to_bits(f);
    return bits >> 16;  // Just take upper 16 bits!
}

void analyze_float(float f) {
    uint32_t bits = float_to_bits(f);
    
    // Extract components
    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    printf("Value: %g\n", f);
    printf("Bits:  ");
    print_binary_32(bits);
    printf("\n");
    printf("       S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM\n");
    printf("Sign:     %d (%s)\n", sign, sign ? "negative" : "positive");
    printf("Exponent: %u (biased), %d (actual = biased - 127)\n", 
           exponent, (int)exponent - 127);
    printf("Mantissa: 0x%06X\n", mantissa);
    
    // Calculate value manually
    if (exponent == 0) {
        if (mantissa == 0) {
            printf("Special: %sZero\n", sign ? "Negative " : "");
        } else {
            printf("Special: Denormal number\n");
        }
    } else if (exponent == 255) {
        if (mantissa == 0) {
            printf("Special: %sInfinity\n", sign ? "Negative " : "");
        } else {
            printf("Special: NaN\n");
        }
    } else {
        double computed = pow(-1, sign) * (1.0 + mantissa / 8388608.0) * pow(2, (int)exponent - 127);
        printf("Formula: (-1)^%d × (1 + %u/2^23) × 2^(%d-127) = %g\n",
               sign, mantissa, exponent, computed);
    }
    printf("\n");
}

int main() {
    printf("=== IEEE 754 FLOATING POINT: THE HEART OF ML ===\n\n");
    
    // =========================================================
    // SECTION 1: FP32 Format
    // =========================================================
    printf("--- SECTION 1: FP32 (float) FORMAT ---\n\n");
    
    printf("FP32 Layout (32 bits):\n");
    printf("┌─────┬──────────┬───────────────────────┐\n");
    printf("│Sign │ Exponent │       Mantissa        │\n");
    printf("│1 bit│  8 bits  │       23 bits         │\n");
    printf("└─────┴──────────┴───────────────────────┘\n\n");
    
    printf("Value = (-1)^sign × (1 + mantissa/2^23) × 2^(exponent-127)\n\n");
    
    // Analyze some important values
    printf("--- Analyzing key values ---\n\n");
    
    analyze_float(1.0f);
    analyze_float(-1.0f);
    analyze_float(0.5f);
    analyze_float(2.0f);
    analyze_float(0.1f);  // Cannot be exactly represented!
    
    // =========================================================
    // SECTION 2: Precision Limits
    // =========================================================
    printf("--- SECTION 2: PRECISION LIMITS ---\n\n");
    
    printf("FP32 Limits:\n");
    printf("  Max value:     %g\n", FLT_MAX);
    printf("  Min positive:  %g\n", FLT_MIN);
    printf("  Epsilon:       %g (smallest x where 1+x != 1)\n", FLT_EPSILON);
    printf("\n");
    
    printf("Precision demonstration:\n");
    float big = 1e7f;
    float small = 1.0f;
    float sum = big + small;
    printf("1e7 + 1 = %f (lost precision!)\n", sum);
    printf("Expected: 10000001, Got: %.0f\n", sum);
    printf("This is catastrophic cancellation in gradient accumulation!\n\n");
    
    // =========================================================
    // SECTION 3: FP16 vs BF16
    // =========================================================
    printf("--- SECTION 3: FP16 vs BF16 (CRITICAL FOR ML) ---\n\n");
    
    printf("FP16 Layout (16 bits):\n");
    printf("┌─────┬──────────┬───────────┐\n");
    printf("│Sign │ Exponent │  Mantissa │\n");
    printf("│1 bit│  5 bits  │  10 bits  │\n");
    printf("└─────┴──────────┴───────────┘\n");
    printf("Range: ±65504, Precision: ~3 decimal digits\n\n");
    
    printf("BF16 Layout (16 bits):\n");
    printf("┌─────┬──────────┬─────────┐\n");
    printf("│Sign │ Exponent │ Mantissa│\n");
    printf("│1 bit│  8 bits  │ 7 bits  │\n");
    printf("└─────┴──────────┴─────────┘\n");
    printf("Range: Same as FP32!, Precision: ~2 decimal digits\n\n");
    
    printf("Why BF16 is preferred for training:\n");
    printf("1. Same exponent range as FP32 → no overflow/underflow in gradients\n");
    printf("2. Simple conversion: just truncate lower 16 bits of FP32\n");
    printf("3. FP16 needs loss scaling to prevent gradient underflow\n\n");
    
    float test_val = 0.0001f;  // Small gradient
    printf("Small value (typical gradient): %g\n", test_val);
    printf("FP16: Can represent (but close to underflow range)\n");
    printf("BF16: Safe (same range as FP32)\n\n");
    
    float large_val = 100000.0f;  // Large loss
    printf("Large value: %g\n", large_val);
    printf("FP16: OVERFLOW! (max is 65504)\n");
    printf("BF16: Safe\n\n");
    
    // =========================================================
    // SECTION 4: Special Values
    // =========================================================
    printf("--- SECTION 4: SPECIAL VALUES ---\n\n");
    
    printf("Zero:\n");
    analyze_float(0.0f);
    
    printf("Infinity (from overflow):\n");
    analyze_float(INFINITY);
    
    printf("NaN (from 0/0):\n");
    analyze_float(NAN);
    
    printf("Denormal (very small, loses precision):\n");
    analyze_float(1e-40f);
    
    // =========================================================
    // SECTION 5: Why This Matters for ML
    // =========================================================
    printf("--- SECTION 5: ML IMPLICATIONS ---\n\n");
    
    printf("1. MIXED PRECISION TRAINING:\n");
    printf("   - Forward/backward in FP16/BF16 for speed\n");
    printf("   - Master weights in FP32 for accuracy\n");
    printf("   - Gradient accumulation in FP32 to prevent overflow\n\n");
    
    printf("2. LOSS SCALING (FP16 only):\n");
    printf("   - Multiply loss by large factor (1024-65536)\n");
    printf("   - Gradients are scaled up, avoiding underflow\n");
    printf("   - Unscale before weight update\n");
    printf("   - BF16 doesn't need this due to larger range!\n\n");
    
    printf("3. DENORMAL FLUSHING:\n");
    printf("   - Denormal numbers are SLOW on GPU (100x slower!)\n");
    printf("   - CUDA kernels flush denormals to zero\n");
    printf("   - set_flush_denormal(True) in PyTorch\n\n");
    
    printf("4. TENSOR CORE REQUIREMENTS:\n");
    printf("   - Tensor Cores require specific formats: FP16, BF16, TF32, FP8\n");
    printf("   - Memory alignment requirements\n");
    printf("   - Matrix dimensions must be multiples of 8 or 16\n\n");
    
    // =========================================================
    // EXERCISES
    // =========================================================
    printf("=== EXERCISES ===\n\n");
    printf("1. Why can't 0.1 be exactly represented in binary?\n");
    printf("2. What is the largest FP16 value? What happens if you exceed it?\n");
    printf("3. Why does 1e7 + 1 lose the 1 in FP32?\n");
    printf("4. Calculate the bits for 3.14159 in FP32 by hand.\n");
    printf("5. Why is BF16 a 'drop-in' replacement for FP32 in training?\n");
    
    return 0;
}
