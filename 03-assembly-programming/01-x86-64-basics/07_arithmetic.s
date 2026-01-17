# 07_arithmetic.s - Arithmetic and Bitwise Operations in x86-64
#
# Covers: ADD, SUB, MUL, DIV, shifts, logical operations, extended precision
#
# Compile with C wrapper:
#   gcc -o arithmetic 07_arithmetic.s arithmetic_main.c

.section .text
    .global arithmetic_demo
    .global integer_arithmetic
    .global shifts_and_rotates
    .global bitwise_operations
    .global extended_precision

# ========================================================================
# SECTION 1: BASIC ARITHMETIC
# ========================================================================

arithmetic_demo:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 1.1 ADD and SUB
    # ----------------------------------------------------------------
    mov $10, %rax               # RAX = 10
    add $5, %rax                # RAX = RAX + 5 = 15
    sub $3, %rax                # RAX = RAX - 3 = 12

    # ----------------------------------------------------------------
    # 1.2 INC and DEC (optimized, no immediate needed)
    # ----------------------------------------------------------------
    inc %rax                     # RAX = RAX + 1
    dec %rax                     # RAX = RAX - 1

    # ----------------------------------------------------------------
    # 1.3 NEG (arithmetic negation)
    # ----------------------------------------------------------------
    mov $42, %rax
    neg %rax                     # RAX = -42

    # ----------------------------------------------------------------
    # 1.4 Flags affected by arithmetic
    # After ADD/SUB/INC/DEC/NEG:
    #   ZF = result == 0
    #   SF = result < 0 (signed)
    #   CF = unsigned overflow (carry/borrow)
    #   OF = signed overflow

    mov $0x7FFFFFFF, %eax
    inc %eax                     # OF=1 (overflow!)
    jo overflow_caught

overflow_caught:
    pop %rbp
    ret

# ========================================================================
# SECTION 2: MULTIPLICATION
# ========================================================================

integer_arithmetic:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 2.1 IMUL - Signed Multiplication
    # ----------------------------------------------------------------

    # One operand form: RDX:RAX = RAX * src
    mov $10, %rax
    mov $20, %rbx
    imul %rbx                    # RDX:RAX = RAX * RBX
    # RAX = 200, RDX = 0

    # Two operand form: dst = dst * src
    mov $5, %rax
    imul $3, %rax                # RAX = RAX * 3 = 15

    # Three operand form: dst = src1 * src2 (immediate)
    mov $7, %rax
    imul $6, %rax, %rbx          # RBX = RAX * 6 = 42

    # ----------------------------------------------------------------
    # 2.2 MUL - Unsigned Multiplication
    # ----------------------------------------------------------------
    mov $10, %rax
    mov $20, %rbx
    mul %rbx                     # RDX:RAX = RAX * RBX
    # RAX = 200, RDX = 0

    # ----------------------------------------------------------------
    # 2.3 Large multiplication (128-bit result)
    # ----------------------------------------------------------------
    mov $0xFFFFFFFF, %rax       # 64-bit operand
    mov $0xFFFFFFFF, %rbx
    mul %rbx                     # RDX:RAX = 128-bit result
    # RAX = 0xFFFFFFFE00000001
    # RDX = 0x00000000FFFFFFFE

    # ----------------------------------------------------------------
    # 2.4 Division
    # ----------------------------------------------------------------

    # IDIV - Signed Division
    # Dividend in RDX:RAX, divisor in src
    # Quotient in RAX, remainder in RDX
    mov $100, %rax               # Dividend
    mov $7, %rbx                 # Divisor
    cqo                          # Sign-extend RAX into RDX
    idiv %rbx                    # RAX = 14, RDX = 2

    # DIV - Unsigned Division
    mov $100, %rax
    mov $7, %rbx
    xor %edx, %edx               # Clear RDX (upper dividend)
    div %rbx                     # RAX = 14, RDX = 2

    pop %rbp
    ret

# ========================================================================
# SECTION 3: SHIFTS AND ROTATES
# ========================================================================

shifts_and_rotates:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 3.1 Logical Shifts (shift in zeros)
    # ----------------------------------------------------------------

    # SHL/SAL - Shift Left (same operation)
    mov $0b1, %rax
    shl $1, %rax                 # RAX = 0b10 = 2
    shl $2, %rax                 # RAX = 0b1000 = 8

    # SHR - Shift Right (logical, unsigned)
    mov $0b1000, %rax
    shr $1, %rax                 # RAX = 0b0100 = 4

    # Shift by CL register
    mov $0b10000000, %rax
    mov $3, %cl
    shl %cl, %rax                # RAX = 0b10000000 << 3 = 0b10000000000

    # ----------------------------------------------------------------
    # 3.2 Arithmetic Shifts (preserve sign bit)
    # ----------------------------------------------------------------

    # SAR - Shift Right Arithmetic (signed)
    mov $0xFFFFFFFFFFFFFF00, %rax
    sar $8, %rax                 # Sign-extend: 0xFFFFFFFFFFFFFF00 >> 8 = 0xFFFFFFFFFFFFFFFF

    # SAR with positive number
    mov $0b1000, %rax
    sar $1, %rax                 # RAX = 0b0100 = 4 (MSB was 0)

    # ----------------------------------------------------------------
    # 3.3 Rotate
    # ----------------------------------------------------------------

    # ROL - Rotate Left
    mov $0x12345678, %rax
    rol $8, %rax                 # RAX = 0x34567812

    # ROR - Rotate Right
    mov $0x12345678, %rax
    ror $8, %rax                 # RAX = 0x78123456

    # RCL/RCR - Rotate Through Carry
    stc                          # Set CF = 1
    mov $0x00, %rax
    rcl $1, %rax                 # RAX = 0x01 (CF becomes 0)

    # ----------------------------------------------------------------
    # 3.4 Shift Applications
    # ----------------------------------------------------------------

    # Division by power of 2 (unsigned)
    mov $100, %rax
    shr $2, %rax                 # RAX = 25 (100 / 4)

    # Multiplication by power of 2
    mov $7, %rax
    shl $3, %rax                 # RAX = 56 (7 * 8)

    # Bit extraction
    mov $0b10101010, %rax
    shr $4, %rax                 # Shift right 4
    and $0xF, %rax               # Mask lower 4 bits

    pop %rbp
    ret

# ========================================================================
# SECTION 4: BITWISE OPERATIONS
# ========================================================================

bitwise_operations:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 4.1 AND, OR, XOR
    # ----------------------------------------------------------------

    # AND - Clear bits
    mov $0xFF, %rax
    and $0x0F, %rax              # RAX = 0x0F (clear upper nibble)

    # OR - Set bits
    mov $0x0F, %rax
    or $0xF0, %rax               # RAX = 0xFF (set upper nibble)

    # XOR - Toggle bits
    mov $0xFF, %rax
    xor $0x0F, %rax              # RAX = 0xF0 (toggle lower nibble)

    # XOR for zeroing (faster than MOV $0, %rax)
    xor %rax, %rax               # RAX = 0

    # ----------------------------------------------------------------
    # 4.2 NOT (bitwise complement)
    # ----------------------------------------------------------------
    mov $0x00, %rax
    not %rax                     # RAX = 0xFFFFFFFFFFFFFFFF

    # ----------------------------------------------------------------
    # 4.3 Bit Testing and Manipulation
    # ----------------------------------------------------------------

    # BT - Test Bit (sets CF to bit value)
    mov $0b1000, %rax
    bt $3, %rax                  # CF = 1 (bit 3 is set)
    bt $0, %rax                  # CF = 0 (bit 0 is not set)

    # BTS - Test and Set Bit
    mov $0, %rax
    bts $5, %rax                 # RAX = 0b100000, CF = 0

    # BTR - Test and Reset Bit
    mov $0b111111, %rax
    btr $3, %rax                 # RAX = 0b101111, CF = 1

    # BTC - Test and Complement Bit
    mov $0b101010, %rax
    btc $2, %rax                 # RAX = 0b101110, CF = 0

    # ----------------------------------------------------------------
    # 4.4 Bit Scanning
    # ----------------------------------------------------------------

    # BSF - Bit Scan Forward (find LSB set)
    mov $0b1000, %rax
    bsf %rax, %rbx               # RBX = 3 (bit 3 is set)

    # BSR - Bit Scan Reverse (find MSB set)
    mov $0b1000, %rax
    bsr %rax, %rbx               # RBX = 3

    # Handle zero case (ZF set if operand is zero)
    mov $0, %rax
    bsf %rax, %rbx               # ZF = 1, RBX unchanged
    jz was_zero

was_zero:
    pop %rbp
    ret

# ========================================================================
# SECTION 5: EXTENDED PRECISION ARITHMETIC
# ========================================================================

extended_precision:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 5.1 Multi-Precision Addition
    # ----------------------------------------------------------------
    # Add two 128-bit numbers: [RDX:RAX] + [RBX:RCX]

    # Clear carry
    clc
    add %rcx, %rax               # Add low parts
    adc %rbx, %rdx               # Add high parts with carry

    # ----------------------------------------------------------------
    # 5.2 Multi-Precision Subtraction
    # ----------------------------------------------------------------
    # Subtract two 128-bit numbers: [RDX:RAX] - [RBX:RCX]

    sub %rcx, %rax               # Subtract low parts
    sbb %rbx, %rdx               # Subtract high parts with borrow

    # ----------------------------------------------------------------
    # 5.3 Multi-Precision Comparison
    # ----------------------------------------------------------------
    # Compare [RDX:RAX] vs [RBX:RCX]

    cmp %rcx, %rax               # Compare low parts
    jne compare_done
    cmp %rdx, %rbx               # Compare high parts
compare_done:

    # ----------------------------------------------------------------
    # 5.4 Multi-Precision Negation
    # ----------------------------------------------------------------
    not %rax                     # Invert low
    not %rbx                     # Invert high
    add $1, %rax                 # Add 1 to low
    adc $0, %rbx                 # Propagate carry to high

    pop %rbp
    ret

# ========================================================================
# SECTION 6: COMMON ARITHMETIC PATTERNS
# ========================================================================

    # ----------------------------------------------------------------
    # Absolute value
    # ----------------------------------------------------------------
    mov $-42, %rax
    cmp $0, %rax
    cmovs %rax, %rbx
    neg %rbx                     # RBX = abs(RAX)

    # ----------------------------------------------------------------
    # Clamp to range [min, max]
    # ----------------------------------------------------------------
    mov $100, %rdi               # value
    mov $0, %rsi                 # min
    mov $255, %rdx               # max

    cmp %rsi, %rdi               # if value < min
    cmovl %rsi, %rdi             # value = min
    cmp %rdx, %rdi               # if value > max
    cmovg %rdx, %rdi             # value = max

    # ----------------------------------------------------------------
    # Average of two numbers (avoid overflow)
    # ----------------------------------------------------------------
    mov $0xFFFFFFFF, %rax        # a
    mov $0xFFFFFFFF, %rbx        # b
    mov %rax, %rcx
    and %rbx, %rcx               # min = a & b
    mov %rax, %rdx
    or %rbx, %rdx                # max = a | b
    shr $1, %rcx
    shr $1, %rdx
    add %rcx, %rdx               # (min >> 1) + (max >> 1)

    ret

# ========================================================================
# ARITHMETIC REFERENCE
# ========================================================================
#
# Basic arithmetic:
#   add src, dst    # dst += src
#   sub src, dst    # dst -= src
#   inc dst         # dst++
#   dec dst         # dst--
#   neg dst         # dst = -dst
#
# Multiplication:
#   imul src        # RDX:RAX = RAX * src (signed)
#   imul src, dst   # dst = dst * src (signed)
#   imul s1, s2, d  # d = s1 * s2 (signed, immediate)
#   mul src         # RDX:RAX = RAX * src (unsigned)
#
# Division:
#   idiv src        # RDX:RAX / src (signed)
#   div src         # RDX:RAX / src (unsigned)
#
# Shifts:
#   shl/sal count, dst   # dst <<= count (logical left)
#   shr count, dst      # dst >>= count (logical right)
#   sar count, dst      # dst >>= count (arithmetic right)
#
# Rotates:
#   rol count, dst   # Rotate left
#   ror count, dst   # Rotate right
#   rcl count, dst  # Rotate through CF left
#   rcr count, dst  # Rotate through CF right
#
# Bitwise:
#   and src, dst    # dst &= src
#   or src, dst     # dst |= src
#   xor src, dst    # dst ^= src
#   not dst         # dst = ~dst
#
# Bit tests:
#   bt pos, reg     # Set CF to bit at position
#   bts pos, reg    # Set bit, CF = old value
#   btr pos, reg    # Reset bit, CF = old value
#   btc pos, reg    # Complement bit, CF = old value
#   bsf/d reg, dst  # Find first set bit
