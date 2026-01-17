# 05_functions.s - Functions and Calling Conventions in x86-64
#
# Covers: function prologue/epilogue, System V AMD64 ABI, Windows x64 ABI
#
# Compile with C wrapper:
#   gcc -o functions 05_functions.s functions_main.c

.section .text
    .global function_demo
    .global sysv_calling_convention
    .global windows_calling_convention
    .global leaf_function
    .global nested_function

# ========================================================================
# SECTION 1: SYSTEM V AMD64 ABI (Linux/macOS/BSD)
# ========================================================================

sysv_calling_convention:
    push %rbp
    mov %rsp, %rbp

    # Arguments: RDI, RSI, RDX, RCX, R8, R9
    # Return: RAX
    # Callee-saved: RBX, RBP, R12, R13, R14, R15
    # Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11

    # Example: long add7(long a, long b, long c, long d, long e, long f, long g)
    # a=RD, b=RSI, c=RDX, d=RCX, e=R8, f=R9, g=stack

    push %r12                    # Save callee-saved register
    mov %rdi, %r12               # a in R12

    mov %rsi, %rax               # b
    add %rdx, %rax               # + c
    add %rcx, %rax               # + d
    add %r8, %rax                # + e
    add %r9, %rax                # + f

    # g is on stack (pushed by caller)
    mov 8(%rbp), %r12            # Load g from stack (after return addr)
    add %r12, %rax

    pop %r12                     # Restore callee-saved
    pop %rbp
    ret

# ========================================================================
# SECTION 2: WINDOWS X64 ABI
# ========================================================================

windows_calling_convention:
    # Arguments: RCX, RDX, R8, R9 (shadow space required!)
    # Return: RAX
    # Callee-saved: RBX, RBP, RSI, RDI, R12-R15
    # Caller-saved: RAX, RCX, RDX, R8-R11

    # Windows requires 32 bytes of shadow space on stack before call
    sub $32, %rsp

    mov %rcx, %rax               # arg1
    add %rdx, %rax               # + arg2
    add %r8, %rax                # + arg3
    add %r9, %rax                # + arg4

    add $32, %rsp                # Restore shadow space
    ret

# ========================================================================
# SECTION 3: FUNCTION PROLOGUE AND EPILOGUE
# ========================================================================

function_demo:
    # Standard prologue
    push %rbp                    # Save old base pointer
    mov %rsp, %rbp              # Set new base pointer

    # Optional: Save callee-saved registers if needed
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15

    # Function body here
    mov $0, %rax                 # Result = 0

    # Optional: Allocate local variables
    # sub $N, %rsp              # Reserve N bytes on stack

    # Restore callee-saved registers
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx

    # Standard epilogue
    pop %rbp                     # Restore base pointer
    ret                          # Return

# ========================================================================
# SECTION 4: LEAF FUNCTIONS (no function calls inside)
# ========================================================================

leaf_function:
    # Leaf functions don't need to save RSP/RBP if they don't call
    # But must maintain 16-byte stack alignment for calls

    # Simple leaf function: sum of array
    # long sum_array(long* arr, int n)
    # arr in RDI, n in ESI

    mov %rdi, %rax               # Pointer in RAX
    mov %esi, %ecx               # Count in ECX
    xor %edx, %edx               # Sum = 0
sum_loop:
    add (%rax, %rdx, 8), %rdx    # Add array[rdx]
    inc %rdx                     # Index++
    cmp %ecx, %rdx               # Check bounds
    jne sum_loop

    # Result in RDX (32-bit sum in EDX)
    mov %edx, %eax               # Return in RAX
    ret

# ========================================================================
# SECTION 5: NESTED FUNCTIONS (functions calling other functions)
# ========================================================================

nested_function:
    push %rbp
    mov %rsp, %rbp

    # Must maintain 16-byte stack alignment before calls
    # RSP must be 16-byte aligned when CALL is executed

    # Allocate frame if needed
    sub $16, %rsp

    # Call another function
    mov $10, %rdi
    mov $20, %rsi
    call add_values              # CALL pushes return address

    # Return value in RAX

    add $16, %rsp                # Clean up frame
    pop %rbp
    ret

add_values:
    # Simple function: long add_values(long a, long b)
    mov %rdi, %rax
    add %rsi, %rax
    ret

# ========================================================================
# SECTION 6: VARIADIC FUNCTIONS (printf-style)
# ========================================================================

variadic_function:
    push %rbp
    mov %rsp, %rbp

    # AL register holds number of vector registers used (for AVX)
    xor %eax, %eax               # No vector registers used

    # Arguments already in RDI, RSI, RDX, RCX, R8, R9
    # Stack arguments: 8(%rbp), 16(%rbp), etc.

    # Example: printf(fmt, arg1, arg2)
    lea fmt_str(%rip), %rdi
    mov $42, %rsi
    mov $100, %rdx
    xor %eax, %eax               # Clear AL for varargs
    call printf

    pop %rbp
    ret

# ========================================================================
# SECTION 7: STRUCTURES AND ALIGNMENT
# ========================================================================

struct_function:
    push %rbp
    mov %rsp, %rbp

    # Structure passed by pointer (RDI points to struct)
    # Assume struct Point { long x; long y; }

    mov 0(%rdi), %rax            # Load x (offset 0)
    add 8(%rdi), %rax            # Add y (offset 8)

    # Return sum
    pop %rbp
    ret

# ========================================================================
# SECTION 8: RED ZONE (System V only)
# ========================================================================

red_zone_demo:
    # In System V ABI, 128 bytes below RSP are "red zone"
    # Can be used for temporaries without adjusting RSP
    # NOT safe in signal handlers or if async stack changes!

    mov $1, (%rsp)               # Store at RSP (safe in System V)
    mov $2, -8(%rsp)             # Store at RSP-8
    mov $3, -128(%rsp)           # Store at RSP-128 (still safe!)
    mov $4, -136(%rsp)           # OVERFLOW! -136 < RSP-128

    ret

# ========================================================================
# DATA SECTIONS
# ========================================================================

.section .data
    fmt_str:    .asciz "Values: %ld, %ld\n"

# ========================================================================
# CALLING CONVENTION REFERENCE
# ========================================================================
#
# SYSTEM V AMD64 ABI (Linux, macOS, BSD):
#   Arguments: RDI, RSI, RDX, RCX, R8, R9, then stack (right to left)
#   Return: RAX (128-bit: RAX:RDX)
#   Callee-saved: RBX, RBP, R12-R15
#   Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
#   Stack alignment: 16 bytes before CALL
#   Red zone: 128 bytes below RSP (leaf functions only)
#
# WINDOWS X64 ABI:
#   Arguments: RCX, RDX, R8, R9, then stack
#   Return: RAX (XMM0 for floats)
#   Callee-saved: RBX, RBP, RSI, RDI, R12-R15
#   Caller-saved: RAX, RCX, RDX, R8-R11
#   Shadow space: 32 bytes reserved by caller
#   Stack alignment: 16 bytes
#
# REGISTER ROLES IN CALLING:
#   RAX - return value, also caller-saved scratch
#   RBX - callee-saved
#   RCX - 4th arg (SysV) / 1st arg (Windows)
#   RDX - 3rd arg (SysV) / 2nd arg (Windows)
#   RSI - 2nd arg (SysV)
#   RDI - 1st arg (SysV)
#   RBP - frame pointer (callee-saved)
#   RSP - stack pointer
#   R8-R11 - caller-saved (R8=5th, R9=6th in SysV)
#   R12-R15 - callee-saved
