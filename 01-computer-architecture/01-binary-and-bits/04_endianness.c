/**
 * 04_endianness.c - Byte Ordering in Memory
 * 
 * Endianness determines how multi-byte values are stored.
 * Important for: file formats, network protocols, cross-platform data.
 * 
 * Compile: gcc -o 04_endianness 04_endianness.c
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

void print_bytes(void* ptr, int size) {
    uint8_t* bytes = (uint8_t*)ptr;
    printf("Bytes: ");
    for (int i = 0; i < size; i++) {
        printf("%02X ", bytes[i]);
    }
    printf("\n");
}

int main() {
    printf("=== ENDIANNESS: BYTE ORDERING ===\n\n");
    
    // Check system endianness
    uint32_t test = 0x01020304;
    uint8_t* bytes = (uint8_t*)&test;
    
    printf("Value: 0x%08X\n", test);
    print_bytes(&test, 4);
    
    if (bytes[0] == 0x04) {
        printf("System is LITTLE-ENDIAN (LSB first)\n");
        printf("Intel/AMD x86, most ARM, most GPUs\n");
    } else {
        printf("System is BIG-ENDIAN (MSB first)\n");
        printf("Network byte order, some PowerPC\n");
    }
    
    printf("\n--- MEMORY LAYOUT ---\n");
    printf("Little-endian: Address 0 has LEAST significant byte\n");
    printf("  0x01020304 stored as: [04] [03] [02] [01]\n");
    printf("                        low          high address\n\n");
    printf("Big-endian: Address 0 has MOST significant byte\n");
    printf("  0x01020304 stored as: [01] [02] [03] [04]\n");
    printf("                        low          high address\n");
    
    printf("\n--- FLOAT REPRESENTATION ---\n");
    float f = 3.14159f;
    printf("Float: %f\n", f);
    print_bytes(&f, 4);
    
    printf("\n=== ML DATA FORMAT IMPLICATIONS ===\n");
    printf("1. NumPy .npy files store endianness in header\n");
    printf("2. ONNX uses little-endian\n");
    printf("3. Network transfer may need byte swapping\n");
    printf("4. GPU and CPU usually match (little-endian)\n");
    
    return 0;
}
