# 04_control_flow.s - Control Flow and Branching in x86-64
#
# Covers: comparisons, conditional jumps, loops, and jump tables
#
# Compile with C wrapper:
#   gcc -o control_flow 04_control_flow.s control_flow_main.c

.section .text
    .global control_flow_demo
    .global compare_and_jump
    .global loop_patterns
    .global jump_tables

# ========================================================================
# SECTION 1: FLAGS AND COMPARISONS
# ========================================================================

control_flow_demo:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 1.1 CMP - Compare (sets flags, doesn't store result)
    # ----------------------------------------------------------------
    mov $10, %rax
    mov $5, %rbx
    cmp %rbx, %rax              # RAX - RBX = 5 (sets flags)

    # After CMP RAX, RBX:
    #   ZF = 0 (result != 0)
    #   SF = 0 (result >= 0)
    #   CF = 0 (no borrow, RAX >= RBX)
    #   OF = 0 (no overflow)

    # ----------------------------------------------------------------
    # 1.2 TEST - Bitwise AND (for flag testing)
    # ----------------------------------------------------------------
    mov $0b1010, %rax
    test $0b1000, %rax          # RAX & 0x8 (check bit 3)
    jz bit_not_set              # Jump if ZF=1 (bit 3 is 0)
    jnz bit_is_set              # Jump if ZF=0 (bit 3 is 1)

bit_is_set:
    # Bit 3 is set
    jmp after_test

bit_not_set:
    # Bit 3 is not set
    nop

after_test:
    # ----------------------------------------------------------------
    # 1.3 Arithmetic sets flags too!
    # ----------------------------------------------------------------
    mov $-128, %al
    add $1, %al                 # AL = -127, OF=1 (signed overflow!)
    jo overflow_handled         # Jump if OF=1

overflow_handled:
    mov $0x7FFFFFFF, %eax
    add $1, %eax                # EAX = 0x80000000, OF=1 (unsigned overflow)
    jo signed_overflow

signed_overflow:
    pop %rbp
    ret

# ========================================================================
# SECTION 2: CONDITIONAL JUMPS
# ========================================================================

compare_and_jump:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 2.1 Signed comparisons (for signed integers)
    # ----------------------------------------------------------------
    mov $10, %rdi               # a = 10
    mov $20, %rsi               # b = 20

    cmp %rsi, %rdi              # Compare a vs b
    je equal                    # a == b
    jne not_equal               # a != b
    jl a_less_b                 # a < b (signed)
    jle a_less_equal_b          # a <= b (signed)
    jg a_greater_b              # a > b (signed)
    jge a_greater_equal_b       # a >= b (signed)

equal:
    mov $1, %rax
    jmp done_signed

not_equal:
    mov $2, %rax
    jmp done_signed

a_less_b:
    mov $3, %rax
    jmp done_signed

a_less_equal_b:
    mov $4, %rax
    jmp done_signed

a_greater_b:
    mov $5, %rax
    jmp done_signed

a_greater_equal_b:
    mov $6, %rax

done_signed:
    # ----------------------------------------------------------------
    # 2.2 Unsigned comparisons (for addresses, arrays)
    # ----------------------------------------------------------------
    mov $10, %rdi               # unsigned a = 10
    mov $20, %rsi               # unsigned b = 20

    cmp %rsi, %rdi
    je u_equal                  # a == b
    jne u_not_equal             # a != b
    jb a_below_b                # a < b (unsigned, below)
    jbe a_below_equal_b         # a <= b (unsigned)
    ja a_above_b                # a > b (unsigned, above)
    jae a_above_equal_b         # a >= b (unsigned)

u_equal:
    mov $11, %rax
    jmp done_unsigned

u_not_equal:
    mov $12, %rax
    jmp done_unsigned

a_below_b:
    mov $13, %rax
    jmp done_unsigned

a_below_equal_b:
    mov $14, %rax
    jmp done_unsigned

a_above_b:
    mov $15, %rax
    jmp done_unsigned

a_above_equal_b:
    mov $16, %rax

done_unsigned:
    # ----------------------------------------------------------------
    # 2.3 Carry flag jumps (for multi-precision arithmetic)
    # ----------------------------------------------------------------
    mov $0xFFFFFFFF, %rax
    add $1, %rax                # RAX = 0, CF=1
    jc carry_set                # Jump if CF=1
    jnc carry_clear             # Jump if CF=0

carry_set:
    mov $1, %rbx
    jmp after_carry

carry_clear:
    mov $0, %rbx

after_carry:
    # ----------------------------------------------------------------
    # 2.4 Parity and sign jumps
    # ----------------------------------------------------------------
    mov $0b10101010, %al
    test $1, %al                # Check LSB
    jpe parity_even             # Jump if even parity
    jpo parity_odd              # Jump if odd parity

parity_even:
    mov $1, %rcx
    jmp after_parity

parity_odd:
    mov $0, %rcx

after_parity:
    mov $-5, %rax
    js negative                 # Jump if SF=1 (negative)
    jns positive                # Jump if SF=0 (positive)

negative:
    mov $1, %rdx
    jmp after_sign

positive:
    mov $0, %rdx

after_sign:
    pop %rbp
    ret

# ========================================================================
# SECTION 3: LOOP PATTERNS
# ========================================================================

loop_patterns:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 3.1 Basic loop with LOOP instruction
    # ----------------------------------------------------------------
    mov $5, %rcx                # Counter
basic_loop:
    # Loop body here
    dec %rcx                    # Decrement counter
    jnz basic_loop             # Continue if RCX != 0

    # Or use LOOP instruction (decrements RCX, jumps if RCX != 0)
    mov $5, %rcx
simple_loop:
    # Body
    loop simple_loop            # RCX--, jump if RCX != 0

    # ----------------------------------------------------------------
    # 3.2 While loop pattern
    # ----------------------------------------------------------------
    mov $0, %rax                # sum = 0
    mov $1, %rdi                # i = 1
while_loop:
    cmp $10, %rdi               # while i < 10
    jge while_done
    add %rdi, %rax              # sum += i
    inc %rdi                    # i++
    jmp while_loop
while_done:

    # ----------------------------------------------------------------
    # 3.3 For loop pattern
    # ----------------------------------------------------------------
    mov $0, %rax                # sum = 0
    mov $0, %rdi                # i = 0
for_loop:
    cmp $10, %rdi               # i < 10?
    jge for_done
    add %rdi, %rax              # sum += i
    inc %rdi                    # i++
    jmp for_loop
for_done:

    # ----------------------------------------------------------------
    # 3.4 Do-while loop (always executes at least once)
    # ----------------------------------------------------------------
    mov $0, %rax
    mov $1, %rdi
do_while:
    add %rdi, %rax
    inc %rdi
    cmp $10, %rdi
    jne do_while                # Continue if RDI != 10

    # ----------------------------------------------------------------
    # 3.5 Array iteration with pointer
    # ----------------------------------------------------------------
    lea array(%rip), %rsi       # Pointer to array
    mov $5, %rcx                # Count
    mov $0, %rax                # Sum
array_loop:
    add (%rsi), %rax            # Add *RSI to sum
    add $8, %rsi               # Move to next element (8 bytes for quad)
    loop array_loop

    pop %rbp
    ret

# ========================================================================
# SECTION 4: JUMP TABLES (Switch statements)
# ========================================================================

jump_tables:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 4.1 Simple jump table
    # ----------------------------------------------------------------
    mov $3, %rdi                # switch (3)
    cmp $4, %rdi                # Check bounds
    jae default_case            # if >= 5, default

    lea jt(%rip), %rax          # Load address of jump table
    mov (%rax, %rdi, 8), %rax  # Get target address
    jmp *%rax                   # Jump to target

jt:
    .quad case_0                # jt[0] = &case_0
    .quad case_1                # jt[1] = &case_1
    .quad case_2                # jt[2] = &case_2
    .quad case_3                # jt[3] = &case_3
    .quad case_4                # jt[4] = &case_4

case_0:
    mov $0, %rax
    jmp after_switch

case_1:
    mov $1, %rax
    jmp after_switch

case_2:
    mov $2, %rax
    jmp after_switch

case_3:
    mov $3, %rax
    jmp after_switch

case_4:
    mov $4, %rax
    jmp after_switch

default_case:
    mov $-1, %rax

after_switch:
    # ----------------------------------------------------------------
    # 4.2 Range jump table (sparse values)
    # ----------------------------------------------------------------
    mov $50, %rdi               # switch (50)
    cmp $10, %rdi
    jb range_default
    cmp $60, %rdi
    ja range_default

    sub $10, %rdi               # Normalize: 50-10 = 40
    lea range_jt(%rip), %rax
    mov (%rax, %rdi, 8), %rax
    jmp *%rax

range_jt:
    .quad range_10
    .quad range_20
    .quad range_30
    .quad range_40
    .quad range_50
    .quad range_60

range_10:
    mov $10, %rax
    jmp after_range

range_20:
    mov $20, %rax
    jmp after_range

range_30:
    mov $30, %rax
    jmp after_range

range_40:
    mov $40, %rax
    jmp after_range

range_50:
    mov $50, %rax
    jmp after_range

range_60:
    mov $60, %rax
    jmp after_range

range_default:
    mov $-1, %rax

after_range:
    pop %rbp
    ret

# ========================================================================
# SECTION 5: FUNCTION CALLS AND RET
# ========================================================================

    # CALL pushes return address and jumps
    call target_function        # Push RIP, jump to target

    # RET pops return address and jumps back
    ret                         # Pop RIP from stack

    # Nested calls work correctly
    call outer                  # Push return addr, jump to outer
outer:
    call inner                  # Push return addr, jump to inner
inner:
    ret                         # Return to outer
outer_ret:
    ret                         # Return to caller

# ========================================================================
# DATA SECTIONS
# ========================================================================

.section .data
    array:      .quad 1, 2, 3, 4, 5

# ========================================================================
# CONDITIONAL JUMP REFERENCE
# ========================================================================
#
# Signed comparisons (use with signed integers):
#   je/jz   - equal / zero
#   jne/jnz - not equal / not zero
#   jl/jnge - less / not greater or equal
#   jle/jng - less or equal / not greater
#   jg/jnle - greater / not less or equal
#   jge/jnl - greater or equal / not less
#
# Unsigned comparisons (use with unsigned, addresses):
#   je/jz   - equal / zero
#   jne/jnz - not equal / not zero
#   jb/jnae - below / not above or equal (CF=1)
#   jbe/jna - below or equal / not above (CF=1 or ZF=1)
#   ja/jnbe - above / not below or equal (CF=0 and ZF=0)
#   jae/jnb - above or equal / not below (CF=0)
#
# Flag tests:
#   jc/jb   - carry flag set
#   jnc     - carry flag clear
#   jo      - overflow flag set
#   jno     - overflow flag clear
#   js      - sign flag set (negative)
#   jns     - sign flag clear (non-negative)
#   jpe/jp  - parity even
#   jpo/jnp - parity odd
