# 01_hello_world.s - First x86-64 Assembly Program
#
# This is raw machine code representation.
# Understanding this helps you:
# - Read compiler output
# - Debug at the lowest level
# - Appreciate what C/CUDA abstracts away
#
# Assemble: as -o hello.o 01_hello_world.s
# Link:     ld -o hello hello.o
# Run:      ./hello

.section .data
    message:    .ascii "Hello from Assembly!\n"
    msg_len = . - message    # Calculate length

.section .text
    .global _start

_start:
    # System call: write(1, message, msg_len)
    # 
    # Linux system call convention:
    #   RAX = syscall number
    #   RDI = 1st argument
    #   RSI = 2nd argument
    #   RDX = 3rd argument
    
    mov $1, %rax          # syscall 1 = write
    mov $1, %rdi          # fd 1 = stdout
    lea message(%rip), %rsi   # pointer to message
    mov $msg_len, %rdx    # length
    syscall               # invoke kernel
    
    # System call: exit(0)
    mov $60, %rax         # syscall 60 = exit
    xor %rdi, %rdi        # status = 0
    syscall

# ============================================================
# KEY CONCEPTS:
# ============================================================
#
# 1. SECTIONS:
#    .data  - Initialized data (variables)
#    .text  - Code (instructions)
#    .bss   - Uninitialized data
#
# 2. INSTRUCTION FORMAT:
#    operation source, destination  (AT&T syntax)
#    mov $42, %rax  →  RAX = 42
#
# 3. IMMEDIATES:
#    $42  - Immediate value (constant)
#    %rax - Register
#    (%rax) - Memory at address in RAX
#
# 4. SYSCALL:
#    Software interrupt to kernel
#    Much slower than regular instructions (~100ns vs ~1ns)
#    This is why mmap beats many read() calls!
