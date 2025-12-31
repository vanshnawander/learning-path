/**
 * 01_unicode_utf8.c - Text Encoding Fundamentals
 * 
 * Understanding text encoding is essential for NLP and multimodal ML.
 * UTF-8 is the dominant encoding - you MUST understand it.
 * 
 * Compile: gcc -o unicode 01_unicode_utf8.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ============================================================
// SECTION 1: ASCII - The Foundation
// ============================================================

void demonstrate_ascii() {
    printf("=== ASCII (7 bits, 128 characters) ===\n\n");
    
    // ASCII uses 7 bits (0-127)
    printf("ASCII Table Highlights:\n");
    printf("  '0'-'9': %d-%d\n", '0', '9');
    printf("  'A'-'Z': %d-%d\n", 'A', 'Z');
    printf("  'a'-'z': %d-%d\n", 'a', 'z');
    printf("  Space:   %d\n", ' ');
    printf("  Newline: %d\n", '\n');
    
    char hello[] = "Hello";
    printf("\n\"Hello\" in bytes: ");
    for (int i = 0; i < strlen(hello); i++) {
        printf("%d ", (unsigned char)hello[i]);
    }
    printf("\n\n");
}

// ============================================================
// SECTION 2: UTF-8 Encoding
// ============================================================

void demonstrate_utf8() {
    printf("=== UTF-8 (Variable length, 1-4 bytes) ===\n\n");
    
    printf("UTF-8 Encoding Rules:\n");
    printf("┌─────────────────┬──────────────────────────────────┐\n");
    printf("│ Codepoint Range │ UTF-8 Byte Pattern               │\n");
    printf("├─────────────────┼──────────────────────────────────┤\n");
    printf("│ U+0000-U+007F   │ 0xxxxxxx (1 byte, ASCII)         │\n");
    printf("│ U+0080-U+07FF   │ 110xxxxx 10xxxxxx (2 bytes)      │\n");
    printf("│ U+0800-U+FFFF   │ 1110xxxx 10xxxxxx 10xxxxxx       │\n");
    printf("│ U+10000-U+10FFFF│ 11110xxx 10xx 10xx 10xx (4 bytes)│\n");
    printf("└─────────────────┴──────────────────────────────────┘\n\n");
    
    // Example: Euro sign €  (U+20AC)
    char euro[] = "€";
    printf("Euro sign '€' (U+20AC):\n");
    printf("  Bytes: ");
    for (int i = 0; euro[i]; i++) {
        printf("0x%02X ", (unsigned char)euro[i]);
    }
    printf("\n  Binary: ");
    for (int i = 0; euro[i]; i++) {
        unsigned char b = euro[i];
        for (int j = 7; j >= 0; j--) {
            printf("%d", (b >> j) & 1);
        }
        printf(" ");
    }
    printf("\n  Decoding: 1110_0010 10_000010 10_101100\n");
    printf("            ^^^^      ^^        ^^    \n");
    printf("  Codepoint: 0010 000010 101100 = 0x20AC ✓\n\n");
    
    // Chinese character: 中 (U+4E2D)
    char zhong[] = "中";
    printf("Chinese '中' (U+4E2D):\n");
    printf("  Bytes: ");
    for (int i = 0; zhong[i]; i++) {
        printf("0x%02X ", (unsigned char)zhong[i]);
    }
    printf("\n  Length in bytes: %zu\n\n", strlen(zhong));
    
    // Emoji: 😀 (U+1F600)
    char emoji[] = "😀";
    printf("Emoji '😀' (U+1F600):\n");
    printf("  Bytes: ");
    for (int i = 0; emoji[i]; i++) {
        printf("0x%02X ", (unsigned char)emoji[i]);
    }
    printf("\n  Length in bytes: %zu\n", strlen(emoji));
    printf("  Note: 4 bytes needed for codepoints > U+FFFF\n\n");
}

// ============================================================
// SECTION 3: Counting Characters vs Bytes
// ============================================================

int count_utf8_chars(const char* s) {
    int count = 0;
    while (*s) {
        // Check first byte to determine char length
        if ((*s & 0x80) == 0) {
            // ASCII: 0xxxxxxx
            s += 1;
        } else if ((*s & 0xE0) == 0xC0) {
            // 2-byte: 110xxxxx
            s += 2;
        } else if ((*s & 0xF0) == 0xE0) {
            // 3-byte: 1110xxxx
            s += 3;
        } else if ((*s & 0xF8) == 0xF0) {
            // 4-byte: 11110xxx
            s += 4;
        } else {
            s += 1;  // Invalid, skip
        }
        count++;
    }
    return count;
}

void demonstrate_counting() {
    printf("=== BYTES VS CHARACTERS ===\n\n");
    
    char* texts[] = {
        "Hello",           // ASCII only
        "Héllo",           // Latin with accent
        "你好",            // Chinese
        "Hello 世界 😀",   // Mixed
    };
    
    printf("String              | Bytes | Characters\n");
    printf("────────────────────┼───────┼───────────\n");
    
    for (int i = 0; i < 4; i++) {
        printf("%-20s| %5zu | %d\n", 
               texts[i], strlen(texts[i]), count_utf8_chars(texts[i]));
    }
    printf("\n");
}

// ============================================================
// SECTION 4: ML Implications
// ============================================================

void ml_implications() {
    printf("=== ML/NLP IMPLICATIONS ===\n\n");
    
    printf("1. TOKENIZATION:\n");
    printf("   - Byte-level: Each byte is a token (GPT-2)\n");
    printf("   - Character-level: Each UTF-8 char is a token\n");
    printf("   - Subword (BPE): Common sequences become tokens\n");
    printf("   - Word-level: Whitespace splitting\n\n");
    
    printf("2. VOCABULARY SIZE:\n");
    printf("   - Byte-level: 256 tokens (complete coverage)\n");
    printf("   - BPE: 30k-100k tokens (GPT: 50257)\n");
    printf("   - Character: ~100k+ (all Unicode)\n\n");
    
    printf("3. SEQUENCE LENGTH:\n");
    printf("   Text: 'Hello 世界' \n");
    printf("   - Bytes: 12\n");
    printf("   - Characters: 8\n");
    printf("   - BPE tokens: ~3-4 (varies by tokenizer)\n\n");
    
    printf("4. STORAGE EFFICIENCY:\n");
    printf("   - UTF-8 is efficient for ASCII-heavy text\n");
    printf("   - UTF-16 better for CJK-heavy text\n");
    printf("   - Tokenized IDs: 2-4 bytes per token\n\n");
    
    printf("5. COMMON PITFALLS:\n");
    printf("   - len(string) returns BYTES in C, chars in Python\n");
    printf("   - Slicing bytes can break UTF-8 sequences!\n");
    printf("   - Different normalizations (NFC vs NFD)\n");
}

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║            TEXT ENCODING FOR ML ENGINEERS                      ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    demonstrate_ascii();
    demonstrate_utf8();
    demonstrate_counting();
    ml_implications();
    
    return 0;
}
