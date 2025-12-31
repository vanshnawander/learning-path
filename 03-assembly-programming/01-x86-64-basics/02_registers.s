# 02_registers.s - Understanding x86-64 Registers
#
# Registers are the fastest storage in the CPU.
# Understanding them is key to understanding performance.
#
# Compile with C wrapper:
#   gcc -o registers 02_registers.s registers_main.c

.section .text
    .global register_demo
    .global add_numbers

# Function: register_demo()
# Shows register usage patterns
register_demo:
    # Prologue - save callee-saved registers
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    
    # 64-bit operations
    mov $0xDEADBEEFCAFEBABE, %rax
    mov %rax, %rbx
    
    # 32-bit operations (zeros upper 32 bits!)
    mov $0x12345678, %eax    # RAX = 0x0000000012345678
    
    # 16-bit operations (preserves upper bits)
    mov $0xFFFF, %ax         # Only changes lower 16 bits
    
    # 8-bit operations
    mov $0x42, %al           # Only changes lowest 8 bits
    mov $0x43, %ah           # Changes bits 8-15
    
    # Epilogue - restore and return
    pop %r12
    pop %rbx
    pop %rbp
    ret

# Function: long add_numbers(long a, long b, long c, long d, long e, long f)
# Demonstrates calling convention
# Args come in: RDI, RSI, RDX, RCX, R8, R9
add_numbers:
    # No prologue needed - we don't use callee-saved registers
    
    mov %rdi, %rax      # result = a
    add %rsi, %rax      # result += b
    add %rdx, %rax      # result += c
    add %rcx, %rax      # result += d
    add %r8, %rax       # result += e
    add %r9, %rax       # result += f
    
    ret                 # Return value in RAX

# ============================================================
# REGISTER REFERENCE:
# ============================================================
#
# 64-bit  32-bit  16-bit  8-bit(L/H)  Purpose
# ------  ------  ------  ----------  -------
# RAX     EAX     AX      AL/AH       Accumulator, return value
# RBX     EBX     BX      BL/BH       Base (callee-saved)
# RCX     ECX     CX      CL/CH       Counter, 4th arg
# RDX     EDX     DX      DL/DH       Data, 3rd arg
# RSI     ESI     SI      SIL         Source, 2nd arg
# RDI     EDI     DI      DIL         Dest, 1st arg
# RBP     EBP     BP      BPL         Base pointer (callee-saved)
# RSP     ESP     SP      SPL         Stack pointer
# R8-R15  R8D-R15D R8W-R15W R8B-R15B  Extended registers
#
# IMPORTANT FOR ML:
# - XMM/YMM/ZMM registers hold SIMD vectors
# - One YMM can hold 8 floats for AVX operations
# - This is how llama.cpp gets fast inference!
