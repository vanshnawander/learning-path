# 06_optimization.s - x86-64 Assembly Optimization Techniques
#
# Covers: instruction selection, latency vs throughput, loop optimization
#
# Compile with C wrapper:
#   gcc -o optimization 06_optimization.s optimization_main.c

.section .text
    .global optimization_demo
    .global latency_vs_throughput
    .global loop_optimization
    .global instruction_selection

# ========================================================================
# SECTION 1: LATENCY VS THROUGHPUT
# ========================================================================

latency_vs_throughput:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 1.1 Understanding Latency and Throughput
    # ----------------------------------------------------------------
    # Latency: Cycles to complete ONE instruction
    # Throughput: Instructions that can start per cycle

    # Example: ADD has latency 1, throughput 1
    #          MUL has latency 3, throughput 1
    #          DIV has latency 15-40, throughput 1

    # ----------------------------------------------------------------
    # 1.2 Dependency Chains (Latency Bound)
    # ----------------------------------------------------------------
    # Long chains of dependent operations are latency-bound
    mov $0, %rax                # Chain length 4:
    add $1, %rax                #   Depends on RAX
    add $1, %rax                #   Depends on RAX
    add $1, %rax                #   Depends on RAX
    add $1, %rax                #   Depends on RAX

    # Better: Break dependencies with independent operations
    xor %eax, %eax              # Clear (breaks dependency)
    mov $4, %ecx                # Counter
dep_loop:
    add $1, %eax                # RAX = RAX + 1
    dec %ecx
    jnz dep_loop                # Still 4 cycles total

    # ----------------------------------------------------------------
    # 1.3 Instruction-Level Parallelism (Throughput Bound)
    # ----------------------------------------------------------------
    # Modern CPUs can execute multiple independent instructions/cycle
    # Haswell: up to 4 uops/cycle, 2 ALU ports

    # Independent operations execute in parallel:
    mov $10, %rax               # uop 1
    mov $20, %rbx               # uop 2 (independent, runs in parallel)
    add $30, %rcx               # uop 3 (runs in parallel)
    add $40, %rdx               # uop 4 (runs in parallel)

    # ----------------------------------------------------------------
    # 1.4 Using Multiple Accumulators
    # ----------------------------------------------------------------
    # Instead of one accumulator (dependency chain):
    mov $0, %rax                # sum1 = 0
    mov $0, %rbx                # sum2 = 0
    mov $0, %rcx                # sum3 = 0
    mov $0, %rdx                # sum4 = 0

    # Distribute work to break dependencies
    lea array(%rip), %rsi
    mov $100, %r8               # Count / 4
multi_acc:
    mov (%rsi), %r9
    add %r9, %rax               # Independent chain 1
    mov 8(%rsi), %r9
    add %r9, %rbx               # Independent chain 2
    mov 16(%rsi), %r9
    add %r9, %rcx               # Independent chain 3
    mov 24(%rsi), %r9
    add %r9, %rdx               # Independent chain 4
    add $32, %rsi
    dec %r8
    jnz multi_acc

    # Combine results
    add %rbx, %rax
    add %rcx, %rax
    add %rdx, %rax

    pop %rbp
    ret

# ========================================================================
# SECTION 2: LOOP OPTIMIZATION
# ========================================================================

loop_optimization:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 2.1 Loop Unrolling
    # ----------------------------------------------------------------
    # Reduces loop overhead and increases ILP
    # Trade-off: code size vs speed

    # 4x unrolled loop:
    mov $0, %rax                # Sum
    lea array(%rip), %rsi
    mov $100, %rcx              # Original count
    shr $2, %rcx                # Count / 4
unrolled_loop:
    add (%rsi), %rax            # Process 4 elements
    add 8(%rsi), %rax
    add 16(%rsi), %rax
    add 24(%rsi), %rax
    add $32, %rsi
    dec %rcx
    jnz unrolled_loop

    # Handle remainder (0-3 elements)
    mov $3, %rcx
remainder:
    add (%rsi), %rax
    add $8, %rsi
    dec %rcx
    jnz remainder

    # ----------------------------------------------------------------
    # 2.2 Loop Invariant Code Motion
    # ----------------------------------------------------------------
    # Move computations outside loop if they don't change

    # Bad: Recalculate inside loop
    mov $0, %rax
    mov $100, %rcx
calc_loop:
    mov $42, %rbx               # This is loop-invariant!
    add %rbx, %rax
    dec %rcx
    jnz calc_loop

    # Good: Move outside
    mov $42, %rbx               # Calculate once
    mov $0, %rax
    mov $100, %rcx
opt_calc_loop:
    add %rbx, %rax
    dec %rcx
    jnz opt_calc_loop

    # ----------------------------------------------------------------
    # 2.3 Strength Reduction
    # ----------------------------------------------------------------
    # Replace expensive operations with cheaper ones

    # Expensive: Multiplication in loop
    mov $0, %rax
    mov $0, %rbx
    mov $10, %rcx
mul_loop:
    add %rbx, %rax              # Sum += i
    add $1, %rbx               # i++
    dec %rcx
    jnz mul_loop               # Uses MUL implicitly

    # Better: Addition instead of multiplication
    mov $0, %rax
    mov $0, %rbx                # i = 0
    mov $10, %rcx
add_loop:
    add %rbx, %rax              # Sum += i
    add $1, %rbx                # i = i + 1
    dec %rcx
    jnz add_loop

    # ----------------------------------------------------------------
    # 2.4 Software Pipelining
    # ----------------------------------------------------------------
    # Overlap iterations to hide latency

    # Simple loop (latency-bound):
    # L1: load, add, store
    mov $0, %rax
    mov $0, %rbx
    mov $100, %rcx
simple_loop:
    add (%rsi), %rax
    add $8, %rsi
    dec %rcx
    jnz simple_loop

    # Software pipelined:
    # Process 2 iterations ahead
    add $8, %rsi                # Prefetch next
    mov $99, %rcx
pipeline_loop:
    add (%rsi), %rax
    add $8, %rsi
    dec %rcx
    jnz pipeline_loop

    pop %rbp
    ret

# ========================================================================
# SECTION 3: INSTRUCTION SELECTION
# ========================================================================

instruction_selection:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 3.1 Choose Fast Instructions
    # ----------------------------------------------------------------

    # SLOW: Division
    mov $100, %rax
    mov $7, %rbx
    div %rbx                    # ~20-40 cycles!

    # FAST: Shift for power of 2
    mov $100, %rax
    shr $1                      # / 2 (1 cycle)
    shl $2, %rbx                # * 4 (1 cycle)

    # ----------------------------------------------------------------
    # 3.2 Use LEA for Complex Arithmetic
    # ----------------------------------------------------------------

    # SLOW: Multiple ADD instructions
    mov %rdi, %rax
    add %rdi, %rax              # RAX = RDI * 2
    add %rdi, %rax              # RAX = RDI * 3

    # FAST: Single LEA
    lea (%rdi, %rdi, 2), %rax  # RAX = RDI * 3 (1 cycle)

    # SLOW: Addition chain
    mov %rdi, %rax
    add %rdi, %rax              # 2x
    add %rdi, %rax              # 3x
    add %rax, %rax              # 6x
    add %rdi, %rax              # 7x

    # FAST: Optimal LEA
    lea (%rdi, %rdi, 8), %rax  # RDI * 9
    sub %rdi, %rax              # RDI * 8 (9-1)

    # ----------------------------------------------------------------
    # 3.3 Use XOR for Zeroing
    # ----------------------------------------------------------------

    # SLOW: MOV
    mov $0, %rax                # 5 bytes

    # FAST: XOR (same result, smaller code)
    xor %rax, %rax              # 2 bytes, same effect

    # ----------------------------------------------------------------
    # 3.4 Use Conditional Moves (CMOV)
    # ----------------------------------------------------------------

    # SLOW: Branch (misprediction penalty ~10-20 cycles)
    mov $10, %rax
    mov $5, %rbx
    cmp %rax, %rbx
    jge skip
    mov %rax, %rbx
skip:

    # FAST: CMOV (no branch, no misprediction)
    cmp %rax, %rbx
    cmovl %rax, %rbx            # Conditional move

    # ----------------------------------------------------------------
    # 3.5 Use SET for Boolean Results
    # ----------------------------------------------------------------

    # Get boolean: (a < b) ? 1 : 0
    mov $10, %rax
    mov $5, %rbx
    cmp %rax, %rbx
    setl %al                    # AL = 1 if RBX < RAX, else 0
    movzx %al, %eax             # Zero-extend to 64-bit

    pop %rbp
    ret

# ========================================================================
# SECTION 4: MEMORY OPTIMIZATION
# ========================================================================

memory_optimization:
    push %rbp
    mov %rsp, %rbp

    # ----------------------------------------------------------------
    # 4.1 Cache-Friendly Access Patterns
    # ----------------------------------------------------------------

    # BAD: Strided access (cache misses)
    mov $0, %rax
    mov $1000, %rcx
strided_loop:
    add (%rsi, %rcx, 8), %rax   # Access with stride 1000!
    dec %rcx
    jnz strided_loop

    # GOOD: Sequential access (cache friendly)
    mov $0, %rax
    mov $1000, %rcx
seq_loop:
    add (%rsi), %rax
    add $8, %rsi
    dec %rcx
    jnz seq_loop

    # ----------------------------------------------------------------
    # 4.2 Prefetching
    # ----------------------------------------------------------------

    # Manual prefetch for large arrays
    mov $0, %rax
    lea array(%rip), %rsi
    mov $1000, %rcx
prefetch_loop:
    prefetcht0 256(%rsi)        # Prefetch 256 bytes ahead
    add (%rsi), %rax
    add $8, %rsi
    dec %rcx
    jnz prefetch_loop

    # ----------------------------------------------------------------
    # 4.3 Alignment
    # ----------------------------------------------------------------

    # Align critical data on cache line boundaries (64 bytes)
    # Use .p2align directive in data section

    pop %rbp
    ret

# ========================================================================
# DATA SECTIONS
# ========================================================================

.section .data
    array:      .quad 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

# ========================================================================
# OPTIMIZATION REFERENCE
# ========================================================================
#
# LATENCY (cycles, approximate):
#   MOV reg, reg          1
#   ADD/SUB               1
#   AND/OR/XOR            1
#   SHL/SHR               1
#   LEA                   1-3
#   MUL (64-bit)          3-5
#   DIV (64-bit)          15-40
#
# THROUGHPUT (instructions/cycle):
#   Haswell/Skylake: 4 uops/cycle, 2 ALU ports
#   AMD Zen: 5 uops/cycle, 3 ALU ports
#
# OPTIMIZATION TECHNIQUES:
#   1. Break dependency chains with XOR/independent ops
#   2. Unroll loops (4x or 8x) to reduce overhead
#   3. Use LEA for multiplication by constants
#   4. Use CMOV instead of branches for simple conditions
#   5. Use SET for boolean results
#   6. Sequential memory access for cache locality
#   7. Prefetch for large working sets
#   8. Align hot data on 64-byte boundaries
#
# COMMON MISTAKES:
#   - Long dependency chains (use multiple accumulators)
#   - Division in loops (use multiplication/shift)
#   - Branches for simple conditions (use CMOV)
#   - Non-sequential memory access (cache misses)
