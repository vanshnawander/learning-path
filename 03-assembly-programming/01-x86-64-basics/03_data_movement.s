# 03_data_movement.s - Complete Data Movement Guide
#
# This file covers all data movement instructions in x86-64 assembly
# with detailed examples for each addressing mode and instruction type.
#
# Compile with C wrapper:
#   gcc -o data_movement 03_data_movement.s data_movement_main.c

.section .text
    .global data_movement_demo
    .global memory_copy
    .global string_operations
    .global extended_moves

# ========================================================================
# SECTION 1: BASIC MOV INSTRUCTIONS
# ========================================================================

data_movement_demo:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 1.1 Immediate to Register
    # ----------------------------------------------------------------
    mov $42, %rax              # RAX = 42 (64-bit immediate)
    mov $0xDEADBEEF, %eax      # EAX = 0xDEADBEEF (upper 32 bits zeroed!)
    mov $0xFF, %al             # AL = 0xFF (8-bit)
    mov $0x12345678, %ax       # AX = 0x5678 (16-bit)

    # ----------------------------------------------------------------
    # 1.2 Register to Register
    # ----------------------------------------------------------------
    mov %rax, %rbx             # RBX = RAX
    mov %rdi, %rsi             # RSI = RDI
    mov %r8, %r9               # R9 = R8

    # ----------------------------------------------------------------
    # 1.3 Memory to Register (Load)
    # ----------------------------------------------------------------
    # Assume: array: .quad 100, 200, 300
    lea array(%rip), %rax      # RAX = address of array
    mov (%rax), %rbx           # RBX = *RAX (load 8 bytes)
    mov 8(%rax), %rcx          # RCX = *(RAX + 8)
    mov 16(%rax), %rdx         # RDX = *(RAX + 16)

    # ----------------------------------------------------------------
    # 1.4 Register to Memory (Store)
    # ----------------------------------------------------------------
    mov $999, (%rax)           # *RAX = 999
    mov $888, 8(%rax)          # *(RAX + 8) = 888

    # ----------------------------------------------------------------
    # 1.5 Sign and Zero Extension
    # ----------------------------------------------------------------
    movb $-1, %al              # AL = 0xFF
    movsbl %al, %ebx           # EBX = 0xFFFFFFFF (sign-extend)
    movzbl %al, %ecx           # ECX = 0x000000FF (zero-extend)
    movswl %ax, %edx           # EDX = sign-extend AX
    movzwl %ax, %esi           # ESI = zero-extend AX
    movslq %eax, %rdi          # RDI = sign-extend EAX to 64-bit

    # Special: cltq = movslq %eax, %rax (only for RAX)
    mov $0xFFFFFFFF, %eax
    cltq                       # RAX = sign-extend EAX

    pop %rbp
    ret

# ========================================================================
# SECTION 2: ADDRESSING MODES (All 7 forms)
# ========================================================================

memory_copy:
    push %rbp
    mov %rsp, %rbp

    lea array(%rip), %rdi      # RDI = base address

    # Mode 1: Direct (absolute address)
    movl $1, 0x604892          # *0x604892 = 1

    # Mode 2: Register Indirect
    mov %rdi, %rax
    movl $2, (%rax)            # *RAX = 2

    # Mode 3: Register Indirect with Displacement
    movl $3, -8(%rdi)          # *(RDI - 8) = 3
    movl $4, 16(%rdi)          # *(RDI + 16) = 4

    # Mode 4: Based with Displacement
    mov $8, %rcx
    movl $5, -24(%rbp)         # *(RBP - 24) = 5

    # Mode 5: Indexed with Scale and Displacement
    # Format: displacement(base, index, scale)
    # Address = displacement + base + index * scale
    mov $4, %rax               # index = 4
    movl $6, 8(%rsp, %rax, 4)  # *(RSP + 8 + RAX*4) = 6
    movl $7, (%rdi, %rcx, 8)   # *(RDI + RCX*8) = 7
    movl $8, 0x10(, %rdx, 4)   # *(RDX*4 + 0x10) = 8
    movl $9, 4(%rax, %rcx)     # *(RAX + RCX + 4) = 9 (scale=1 implied)

    # Mode 6: RIP-Relative (position independent!)
    lea array(%rip), %rax      # RAX = &array (PC-relative)
    mov (%rax), %rbx           # Load from array

    # Mode 7: RIP-Relative with Displacement
    lea msg(%rip), %rsi         # RSI = &msg
    movb $0, 100(%rip)         # 100 bytes past current IP

    pop %rbp
    ret

# ========================================================================
# SECTION 3: SPECIAL DATA MOVEMENT INSTRUCTIONS
# ========================================================================

string_operations:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 3.1 MOVS - Move String (Byte/Word/Dword/Qword)
    # ----------------------------------------------------------------
    # Moves data from DS:RSI to ES:RDI
    # DF flag controls direction (std = forward, cld = backward)
    cld                         # Clear direction flag (forward)

    lea src(%rip), %rsi         # Source pointer
    lea dst(%rip), %rdi         # Destination pointer
    mov $10, %rcx              # Move 10 bytes
    rep movsb                   # Repeat: copy RCX bytes from RSI to RDI

    # ----------------------------------------------------------------
    # 3.2 STOS - Store String
    # ----------------------------------------------------------------
    # Stores AL/AX/EAX/RAX to ES:RDI
    cld
    mov $0xAA, %al
    lea buffer(%rip), %rdi
    mov $20, %rcx
    rep stosb                   # Fill buffer with 0xAA

    # ----------------------------------------------------------------
    # 3.3 LODS - Load String
    # ----------------------------------------------------------------
    # Loads from DS:RSI to AL/AX/EAX/RAX
    cld
    lea src(%rip), %rsi
    lodsb                       # AL = *RSI, RSI++
    lodsw                       # AX = *RSI, RSI+=2
    lodsl                       # EAX = *RSI, RSI+=4
    lodsq                       # RAX = *RSI, RSI+=8

    # ----------------------------------------------------------------
    # 3.4 CMPS - Compare String
    # ----------------------------------------------------------------
    # Compares DS:RSI to ES:RDI
    cld
    lea str1(%rip), %rsi
    lea str2(%rip), %rdi
    mov $100, %rcx
    repe cmpsb                  # Compare until mismatch or RCX=0
    jne strings_differ          # Jump if ZF=0 (strings differ)

strings_differ:
    pop %rbp
    ret

# ========================================================================
# SECTION 4: CONDITIONAL MOVES (CMOV)
# ========================================================================

extended_moves:
    push %rbp
    mov %rsp, %rbp

    # CMOV instructions: move only if condition is met
    # Format: cmov<condition> src, dst

    mov $10, %rax
    mov $5, %rbx

    cmp %rax, %rbx              # Compare RBX to RAX
    cmovg %rax, %rbx             # RBX = RAX if RBX > RAX (signed)

    # Common conditions:
    # e  - equal (ZF=1)
    # ne - not equal (ZF=0)
    # l  - less (signed, SF!=OF)
    # le - less or equal (signed, SF!=OF or ZF=1)
    # g  - greater (signed, ZF=0 and SF==OF)
    # ge - greater or equal (signed, SF==OF)
    #
    # b  - below (unsigned, CF=1)
    # a  - above (unsigned, CF=0 and ZF=0)
    # be - below or equal (unsigned, CF=1 or ZF=1)
    # ae - above or equal (unsigned, CF=0)

    # Example: Find max of two values
    mov $100, %rdi
    mov $200, %rsi
    cmp %rdi, %rsi              # RSI vs RDI
    cmovl %rdi, %rsi            # If RSI < RDI, use RDI

    # Example: Absolute value
    mov $-50, %rax
    cmp $0, %rax
    cmovs %rax, %rbx             # If negative, copy to RBX
    neg %rbx                     # Negate to get absolute value

    pop %rbp
    ret

# ========================================================================
# SECTION 5: LEA (Load Effective Address)
# ========================================================================

    # LEA is an arithmetic instruction, NOT a memory load!
    # It computes the address and stores it, without dereferencing.

    # Basic usage: calculate addresses
    lea 16(%rsp), %rax          # RAX = RSP + 16
    lea (%rdi, %rsi, 4), %rax   # RAX = RDI + RSI*4

    # LEA for arithmetic (no memory access!)
    lea (%rdi, %rdi, 2), %rax  # RAX = RDI * 3 (RDI + RDI*2)
    lea (%rdi, %rdi, 4), %rax  # RAX = RDI * 5
    lea (%rdi, %rdi, 8), %rax  # RAX = RDI * 9
    lea -1(%rdi), %rax          # RAX = RDI - 1

    # LEA for complex arithmetic
    lea (%rsi, %rdi, 1), %rax   # RAX = RSI + RDI
    lea 3(%rsi, %rdi, 2), %rax # RAX = RSI + RDI*2 + 3

    ret

# ========================================================================
# DATA SECTIONS
# ========================================================================

.section .data
    array:      .quad 100, 200, 300, 400, 500
    src:        .asciz "Source string for operations"
    dst:        .space 64                     # 64 bytes reserved
    str1:       .asciz "Hello"
    str2:       .asciz "World"
    buffer:     .space 256
    msg:        .asciz "Message for RIP-relative addressing"

# ========================================================================
# INSTRUCTION REFERENCE
# ========================================================================
#
# MOV variants:
#   movb  - 8 bits
#   movw  - 16 bits
#   movl  - 32 bits
#   movq  - 64 bits
#
# Sign-extend (fill with sign bit):
#   movsbw, movsbl, movsbq - byte to word/dword/qword
#   movswl, movswq         - word to dword/qword
#   movslq                 - dword to qword
#
# Zero-extend (fill with zeros):
#   movzbw, movzbl, movzbq - byte to word/dword/qword
#   movzwl, movzwq         - word to dword/qword
#
# String instructions:
#   movsb  - copy byte DS:RSI -> ES:RDI
#   movsw  - copy word
#   movsl  - copy dword
#   movsq  - copy qword
#
# Conditional moves:
#   cmove, cmovne  - equal / not equal
#   cmovl, cmovle  - less / less or equal (signed)
#   cmovg, cmovge  - greater / greater or equal (signed)
#   cmovb, cmovbe  - below / below or equal (unsigned)
#   cmova, cmovae  - above / above or equal (unsigned)
#
# Key insight: mov to 32-bit register ZEROES upper 32 bits!
#   mov $0, %eax    # RAX = 0x0000000000000000
#   mov %eax, %ebx  # RBX = 0x00000000 (upper bits unchanged!)
