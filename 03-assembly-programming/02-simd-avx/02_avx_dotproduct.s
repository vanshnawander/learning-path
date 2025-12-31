# 02_avx_dotproduct.s - Hand-Written AVX Dot Product
#
# This is what high-performance libraries do.
# Understanding this helps you read llama.cpp code.
#
# Function: float avx_dot(float* a, float* b, int n)
#
# Compile with C wrapper:
#   gcc -mavx2 -mfma -O3 -o dotprod 02_avx_dotproduct.s dotprod_main.c

.section .text
    .global avx_dot

# float avx_dot(float* a, float* b, int n)
# Arguments: RDI=a, RSI=b, EDX=n
# Return: XMM0 (scalar float)
avx_dot:
    # Prologue
    push %rbp
    mov %rsp, %rbp
    
    # Initialize sum to zero (8 floats)
    vxorps %ymm0, %ymm0, %ymm0    # sum = [0,0,0,0,0,0,0,0]
    
    # Check if n >= 8
    cmp $8, %edx
    jl .Lremainder
    
    # Calculate number of 8-element iterations
    mov %edx, %ecx
    shr $3, %ecx                  # ecx = n / 8
    
.Lmain_loop:
    # Load 8 floats from a and b
    vmovups (%rdi), %ymm1         # ymm1 = a[i:i+8]
    vmovups (%rsi), %ymm2         # ymm2 = b[i:i+8]
    
    # FMA: sum += a * b
    vfmadd231ps %ymm1, %ymm2, %ymm0
    
    # Advance pointers
    add $32, %rdi                 # 8 floats * 4 bytes
    add $32, %rsi
    
    # Decrement counter and loop
    dec %ecx
    jnz .Lmain_loop
    
    # Calculate remaining elements
    and $7, %edx                  # n % 8
    
.Lremainder:
    # Handle remaining elements (scalar)
    test %edx, %edx
    jz .Lhorizontal_sum
    
.Lscalar_loop:
    vmovss (%rdi), %xmm1
    vmovss (%rsi), %xmm2
    vfmadd231ss %xmm1, %xmm2, %xmm0
    add $4, %rdi
    add $4, %rsi
    dec %edx
    jnz .Lscalar_loop
    
.Lhorizontal_sum:
    # Reduce ymm0 to single float
    # ymm0 = [a,b,c,d,e,f,g,h]
    
    # Extract high 128 bits
    vextractf128 $1, %ymm0, %xmm1  # xmm1 = [e,f,g,h]
    
    # Add to low 128 bits
    vaddps %xmm1, %xmm0, %xmm0     # xmm0 = [a+e,b+f,c+g,d+h]
    
    # Horizontal add twice
    vhaddps %xmm0, %xmm0, %xmm0    # xmm0 = [a+e+b+f, c+g+d+h, ...]
    vhaddps %xmm0, %xmm0, %xmm0    # xmm0 = [sum, sum, ...]
    
    # Clear upper YMM bits (AVX-SSE transition penalty)
    vzeroupper
    
    # Epilogue
    pop %rbp
    ret

# ============================================================
# INSTRUCTION BREAKDOWN:
# ============================================================
#
# vxorps:       Zero a register (fastest way)
# vmovups:      Unaligned load (u = unaligned, ps = packed single)
# vfmadd231ps:  FMA - dst = src1 * src2 + dst (231 = operand order)
# vextractf128: Extract high/low 128 bits
# vhaddps:      Horizontal add (adjacent pairs)
# vzeroupper:   Clear upper YMM bits (avoid AVX-SSE penalty)
#
# WHY THIS IS FAST:
# 1. Loop processes 8 floats per iteration
# 2. FMA does multiply+add in one cycle
# 3. Memory access is sequential (prefetcher helps)
# 4. Modern CPUs execute ~2-3 FMAs per cycle
