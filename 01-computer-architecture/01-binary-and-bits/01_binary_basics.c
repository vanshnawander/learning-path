/**
 * 01_binary_basics.c - Understanding Binary Representation
 * 
 * This is the absolute foundation. Every tensor, every weight, every gradient
 * in your ML model is ultimately just bits. Understanding this is crucial for:
 * - Quantization (FP32 → INT8)
 * - Memory optimization
 * - Understanding overflow/underflow
 * - Debugging numerical issues
 * 
 * Compile: gcc -o 01_binary_basics 01_binary_basics.c
 * Run: ./01_binary_basics
 */

#include <stdio.h>
#include <stdint.h>
#include <limits.h>

/**
 * Print binary representation of any integer type
 * This is your microscope into the bit level
 */
void print_binary(uint64_t value, int bits) {
    printf("0b");
    for (int i = bits - 1; i >= 0; i--) {
        printf("%d", (int)((value >> i) & 1));
        if (i > 0 && i % 4 == 0) printf("_");  // Nibble separator
    }
}

/**
 * Print value in multiple formats
 */
void inspect_integer(int64_t value, int bits) {
    printf("Value: %lld\n", (long long)value);
    printf("Binary: ");
    print_binary((uint64_t)value, bits);
    printf("\n");
    printf("Hex: 0x%llX\n", (unsigned long long)(uint64_t)value);
    printf("\n");
}

int main() {
    printf("=== BINARY BASICS: THE FOUNDATION OF EVERYTHING ===\n\n");
    
    // =========================================================
    // SECTION 1: Unsigned Integers
    // =========================================================
    printf("--- SECTION 1: UNSIGNED INTEGERS ---\n\n");
    
    printf("An 8-bit unsigned integer can represent 0 to 255:\n");
    uint8_t u8_min = 0;
    uint8_t u8_max = 255;
    
    printf("Min (0):   ");
    print_binary(u8_min, 8);
    printf(" = %u\n", u8_min);
    
    printf("Max (255): ");
    print_binary(u8_max, 8);
    printf(" = %u\n", u8_max);
    
    printf("\nEach bit position has a value (powers of 2):\n");
    printf("Bit:    7    6    5    4    3    2    1    0\n");
    printf("Value: 128   64   32   16   8    4    2    1\n\n");
    
    printf("Example: 42 in binary\n");
    printf("42 = 32 + 8 + 2 = 2^5 + 2^3 + 2^1\n");
    printf("Binary: ");
    print_binary(42, 8);
    printf("\n\n");
    
    // =========================================================
    // SECTION 2: Signed Integers (Two's Complement)
    // =========================================================
    printf("--- SECTION 2: SIGNED INTEGERS (TWO'S COMPLEMENT) ---\n\n");
    
    printf("Two's complement is how computers represent negative numbers.\n");
    printf("The most significant bit (MSB) indicates sign: 0=positive, 1=negative\n\n");
    
    int8_t positive = 42;
    int8_t negative = -42;
    
    printf("+42: ");
    print_binary((uint8_t)positive, 8);
    printf("\n");
    
    printf("-42: ");
    print_binary((uint8_t)negative, 8);
    printf("\n\n");
    
    printf("To get -42 from +42:\n");
    printf("1. Invert all bits:  ");
    print_binary((uint8_t)~positive, 8);
    printf(" (this is ~42 = -43)\n");
    printf("2. Add 1:            ");
    print_binary((uint8_t)negative, 8);
    printf(" (this is -42)\n\n");
    
    printf("Why two's complement? Addition just works!\n");
    int8_t a = 10;
    int8_t b = -3;
    int8_t result = a + b;
    printf("10 + (-3) = %d\n", result);
    printf("  ");
    print_binary((uint8_t)a, 8);
    printf(" (10)\n");
    printf("+ ");
    print_binary((uint8_t)b, 8);
    printf(" (-3)\n");
    printf("= ");
    print_binary((uint8_t)result, 8);
    printf(" (7) - overflow bit discarded\n\n");
    
    // =========================================================
    // SECTION 3: Integer Overflow (CRITICAL for ML!)
    // =========================================================
    printf("--- SECTION 3: INTEGER OVERFLOW ---\n\n");
    
    printf("Overflow is when a value exceeds the type's range.\n");
    printf("This is CRITICAL in ML: gradient accumulation, loss scaling, etc.\n\n");
    
    // Unsigned overflow
    uint8_t u8 = 255;
    printf("uint8_t: 255 + 1 = %u (wraps to 0!)\n", (uint8_t)(u8 + 1));
    
    // Signed overflow (undefined behavior in C, but typically wraps)
    int8_t s8 = 127;
    printf("int8_t: 127 + 1 = %d (wraps to -128!)\n", (int8_t)(s8 + 1));
    
    printf("\nRange of common types:\n");
    printf("int8_t:  [%d, %d]\n", INT8_MIN, INT8_MAX);
    printf("int16_t: [%d, %d]\n", INT16_MIN, INT16_MAX);
    printf("int32_t: [%d, %d]\n", INT32_MIN, INT32_MAX);
    printf("int64_t: [%lld, %lld]\n", (long long)INT64_MIN, (long long)INT64_MAX);
    printf("\n");
    
    // =========================================================
    // SECTION 4: Why This Matters for ML
    // =========================================================
    printf("--- SECTION 4: ML IMPLICATIONS ---\n\n");
    
    printf("1. QUANTIZATION:\n");
    printf("   FP32 weights → INT8 means mapping floats to [-128, 127]\n");
    printf("   Overflow causes catastrophic errors!\n\n");
    
    printf("2. GRADIENT ACCUMULATION:\n");
    printf("   Summing many gradients can overflow intermediate values\n");
    printf("   This is why we use FP32 for accumulation even with FP16 training\n\n");
    
    printf("3. LOSS SCALING:\n");
    printf("   In mixed precision, we scale loss to prevent underflow\n");
    printf("   Then unscale gradients, watching for overflow\n\n");
    
    printf("4. TOKENIZERS:\n");
    printf("   Token IDs are integers - vocabulary size determines required bits\n");
    printf("   50k vocab needs at least 16 bits (uint16_t)\n\n");
    
    // =========================================================
    // EXERCISES
    // =========================================================
    printf("=== EXERCISES ===\n\n");
    printf("1. What is -1 in binary (8-bit)? Why?\n");
    printf("2. What happens when you add 1 to INT8_MAX?\n");
    printf("3. How many bits do you need for a vocabulary of 100,000 tokens?\n");
    printf("4. Why is two's complement better than sign-magnitude?\n");
    
    return 0;
}
