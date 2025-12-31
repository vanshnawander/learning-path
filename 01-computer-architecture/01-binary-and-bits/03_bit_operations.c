/**
 * 03_bit_operations.c - Bitwise Operations Deep Dive
 * 
 * Bit manipulation is everywhere in systems programming:
 * - CUDA thread/block indexing
 * - Memory alignment
 * - Hash functions
 * - Masks for data selection
 * - Efficient conditionals
 * 
 * Compile: gcc -O2 -o 03_bit_operations 03_bit_operations.c
 * Run: ./03_bit_operations
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

void print_binary(uint32_t value, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        printf("%d", (value >> i) & 1);
        if (i > 0 && i % 4 == 0) printf("_");
    }
}

void print_operation(const char* name, uint32_t a, uint32_t b, uint32_t result) {
    printf("%s:\n", name);
    printf("  a:      ");
    print_binary(a, 8);
    printf(" (%u)\n", a);
    printf("  b:      ");
    print_binary(b, 8);
    printf(" (%u)\n", b);
    printf("  result: ");
    print_binary(result, 8);
    printf(" (%u)\n\n", result);
}

int main() {
    printf("=== BITWISE OPERATIONS: THE BUILDING BLOCKS ===\n\n");
    
    // =========================================================
    // SECTION 1: Basic Operations
    // =========================================================
    printf("--- SECTION 1: BASIC BITWISE OPERATIONS ---\n\n");
    
    uint8_t a = 0b11001010;  // 202
    uint8_t b = 0b10110101;  // 181
    
    print_operation("AND (&) - Both bits must be 1", a, b, a & b);
    print_operation("OR (|) - Either bit can be 1", a, b, a | b);
    print_operation("XOR (^) - Bits must differ", a, b, a ^ b);
    
    printf("NOT (~) - Invert all bits:\n");
    printf("  a:      ");
    print_binary(a, 8);
    printf("\n");
    printf("  ~a:     ");
    print_binary((uint8_t)~a, 8);
    printf("\n\n");
    
    // =========================================================
    // SECTION 2: Shift Operations
    // =========================================================
    printf("--- SECTION 2: SHIFT OPERATIONS ---\n\n");
    
    uint8_t x = 0b00001100;  // 12
    
    printf("Left shift (<<) - Multiply by power of 2:\n");
    printf("  x:      ");
    print_binary(x, 8);
    printf(" (%u)\n", x);
    printf("  x << 1: ");
    print_binary(x << 1, 8);
    printf(" (%u) = x * 2\n", x << 1);
    printf("  x << 2: ");
    print_binary(x << 2, 8);
    printf(" (%u) = x * 4\n", x << 2);
    printf("\n");
    
    printf("Right shift (>>) - Divide by power of 2:\n");
    printf("  x:      ");
    print_binary(x, 8);
    printf(" (%u)\n", x);
    printf("  x >> 1: ");
    print_binary(x >> 1, 8);
    printf(" (%u) = x / 2\n", x >> 1);
    printf("  x >> 2: ");
    print_binary(x >> 2, 8);
    printf(" (%u) = x / 4\n\n", x >> 2);
    
    // =========================================================
    // SECTION 3: Common Bit Tricks
    // =========================================================
    printf("--- SECTION 3: COMMON BIT TRICKS ---\n\n");
    
    // Check if bit is set
    printf("1. CHECK IF BIT N IS SET: (x >> n) & 1\n");
    uint8_t val = 0b10101010;
    for (int i = 7; i >= 0; i--) {
        printf("   Bit %d of ", i);
        print_binary(val, 8);
        printf(": %d\n", (val >> i) & 1);
    }
    printf("\n");
    
    // Set a bit
    printf("2. SET BIT N: x | (1 << n)\n");
    uint8_t v = 0b00000000;
    printf("   Original: ");
    print_binary(v, 8);
    printf("\n");
    v = v | (1 << 3);  // Set bit 3
    printf("   Set bit 3: ");
    print_binary(v, 8);
    printf("\n\n");
    
    // Clear a bit
    printf("3. CLEAR BIT N: x & ~(1 << n)\n");
    v = 0b11111111;
    printf("   Original: ");
    print_binary(v, 8);
    printf("\n");
    v = v & ~(1 << 3);  // Clear bit 3
    printf("   Clear bit 3: ");
    print_binary(v, 8);
    printf("\n\n");
    
    // Toggle a bit
    printf("4. TOGGLE BIT N: x ^ (1 << n)\n");
    v = 0b00001000;
    printf("   Original: ");
    print_binary(v, 8);
    printf("\n");
    v = v ^ (1 << 3);  // Toggle bit 3
    printf("   Toggle bit 3: ");
    print_binary(v, 8);
    printf("\n");
    v = v ^ (1 << 3);  // Toggle again
    printf("   Toggle again: ");
    print_binary(v, 8);
    printf("\n\n");
    
    // Check power of 2
    printf("5. CHECK IF POWER OF 2: (x & (x-1)) == 0\n");
    uint32_t powers[] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t non_powers[] = {3, 5, 6, 7, 9, 10, 15, 100};
    printf("   Powers of 2: ");
    for (int i = 0; i < 8; i++) {
        printf("%d:%s ", powers[i], ((powers[i] & (powers[i]-1)) == 0) ? "✓" : "✗");
    }
    printf("\n");
    printf("   Non-powers:  ");
    for (int i = 0; i < 8; i++) {
        printf("%d:%s ", non_powers[i], ((non_powers[i] & (non_powers[i]-1)) == 0) ? "✓" : "✗");
    }
    printf("\n\n");
    
    // =========================================================
    // SECTION 4: Memory Alignment
    // =========================================================
    printf("--- SECTION 4: MEMORY ALIGNMENT (CRITICAL FOR GPU!) ---\n\n");
    
    printf("Alignment means address is divisible by N.\n");
    printf("GPU memory coalescing requires 128-byte (or 32-byte) alignment.\n\n");
    
    // Align up to power of 2
    printf("ALIGN UP: (addr + (align-1)) & ~(align-1)\n");
    uint64_t addr = 1000;
    uint64_t align = 128;
    uint64_t aligned = (addr + (align - 1)) & ~(align - 1);
    printf("  Address: %llu\n", (unsigned long long)addr);
    printf("  Align to: %llu bytes\n", (unsigned long long)align);
    printf("  Aligned:  %llu\n", (unsigned long long)aligned);
    printf("  Check: %llu %% %llu = %llu (should be 0)\n\n", 
           (unsigned long long)aligned, (unsigned long long)align, 
           (unsigned long long)(aligned % align));
    
    // Check alignment
    printf("CHECK ALIGNMENT: (addr & (align-1)) == 0\n");
    uint64_t addrs[] = {0, 64, 128, 256, 100, 200, 1024};
    for (int i = 0; i < 7; i++) {
        bool is_aligned = (addrs[i] & (128 - 1)) == 0;
        printf("  %4llu: %s\n", (unsigned long long)addrs[i], 
               is_aligned ? "128-byte aligned" : "NOT aligned");
    }
    printf("\n");
    
    // =========================================================
    // SECTION 5: Masking (Used in CUDA/Triton)
    // =========================================================
    printf("--- SECTION 5: MASKING FOR DATA SELECTION ---\n\n");
    
    printf("Masking selects which data to process (critical in Triton!)\n\n");
    
    // Simulate processing 8 elements where only 5 are valid
    int n_elements = 5;
    int block_size = 8;
    
    printf("Processing %d elements with block size %d:\n", n_elements, block_size);
    printf("Thread ID:  ");
    for (int i = 0; i < block_size; i++) printf("%d ", i);
    printf("\n");
    printf("Valid mask: ");
    for (int i = 0; i < block_size; i++) printf("%d ", i < n_elements ? 1 : 0);
    printf("\n");
    printf("Elements:   ");
    for (int i = 0; i < block_size; i++) {
        if (i < n_elements) printf("* ");  // Process
        else printf("- ");  // Skip
    }
    printf("\n\n");
    
    // Create mask efficiently
    printf("Create mask for first N bits: (1 << n) - 1\n");
    for (int n = 1; n <= 8; n++) {
        uint8_t mask = (1 << n) - 1;
        printf("  n=%d: ", n);
        print_binary(mask, 8);
        printf("\n");
    }
    printf("\n");
    
    // =========================================================
    // SECTION 6: Population Count and Leading Zeros
    // =========================================================
    printf("--- SECTION 6: POPCOUNT AND CLZ ---\n\n");
    
    printf("POPCOUNT: Count number of 1 bits\n");
    printf("Used in: sparse operations, hash functions\n");
    uint32_t vals[] = {0, 1, 7, 255, 0xAAAAAAAA};
    for (int i = 0; i < 5; i++) {
        printf("  popcount(0x%08X) = %d\n", vals[i], __builtin_popcount(vals[i]));
    }
    printf("\n");
    
    printf("CLZ (Count Leading Zeros): Find highest set bit\n");
    printf("Used in: finding log2, priority queues\n");
    uint32_t clz_vals[] = {1, 2, 4, 128, 1024, 0x80000000};
    for (int i = 0; i < 6; i++) {
        int clz = __builtin_clz(clz_vals[i]);
        int highest_bit = 31 - clz;
        printf("  clz(%u) = %d, highest bit position = %d\n", 
               clz_vals[i], clz, highest_bit);
    }
    printf("\n");
    
    // =========================================================
    // SECTION 7: ML Applications
    // =========================================================
    printf("--- SECTION 7: ML APPLICATIONS ---\n\n");
    
    printf("1. CUDA THREAD INDEXING:\n");
    printf("   Linear to 2D: row = idx / width, col = idx %% width\n");
    printf("   But if width is power of 2: col = idx & (width-1)\n");
    printf("   This is MUCH faster than modulo!\n\n");
    
    printf("2. TRITON MASKS:\n");
    printf("   offs = tl.arange(0, BLOCK_SIZE)\n");
    printf("   mask = offs < n_elements\n");
    printf("   tl.load(ptr + offs, mask=mask)\n\n");
    
    printf("3. QUANTIZATION BIT PACKING:\n");
    printf("   Pack 8 INT4 values into one uint32:\n");
    printf("   packed = (v0 << 0) | (v1 << 4) | ... | (v7 << 28)\n\n");
    
    printf("4. ATTENTION MASKS:\n");
    printf("   Causal mask: valid = (query_pos >= key_pos)\n");
    printf("   Packed as bits for memory efficiency\n\n");
    
    // =========================================================
    // EXERCISES
    // =========================================================
    printf("=== EXERCISES ===\n\n");
    printf("1. Write a function to count trailing zeros (CTZ)\n");
    printf("2. Implement swap without temporary variable using XOR\n");
    printf("3. Round up to next power of 2\n");
    printf("4. Extract bits [high:low] from a value\n");
    printf("5. Pack two FP16 values into one uint32\n");
    
    return 0;
}
