# Reading Compiler Output

Learning to read assembly helps you:
1. Understand what your code actually does
2. See optimization opportunities
3. Debug performance issues
4. Verify compiler optimizations

## Tools

### Godbolt Compiler Explorer
https://godbolt.org/
- Interactive, shows assembly side-by-side with C
- Multiple compilers (GCC, Clang, MSVC, ICC)
- Color-coded matching

### objdump
```bash
# Disassemble binary
objdump -d program

# With source interleaved
gcc -g -o program program.c
objdump -S program

# Intel syntax (easier to read)
objdump -M intel -d program
```

### GCC output
```bash
# Generate assembly file
gcc -S -O2 program.c -o program.s

# With annotations
gcc -S -O2 -fverbose-asm program.c
```

## Reading x86-64 Assembly

### AT&T vs Intel Syntax

| AT&T | Intel | Meaning |
|------|-------|---------|
| `movl $42, %eax` | `mov eax, 42` | eax = 42 |
| `addl %ebx, %eax` | `add eax, ebx` | eax += ebx |
| `(%rax)` | `[rax]` | Memory at rax |

### Common Patterns

```asm
# Function prologue
push    rbp
mov     rbp, rsp
sub     rsp, 32          # Allocate stack space

# Function epilogue  
leave                    # mov rsp, rbp; pop rbp
ret

# Loop
.L2:
    # loop body
    add     rax, 1
    cmp     rax, rcx
    jne     .L2          # Jump if not equal

# Vectorized loop
.L3:
    vmovaps ymm0, [rdi+rax]
    vaddps  ymm0, ymm0, ymm1
    vmovaps [rsi+rax], ymm0
    add     rax, 32
    cmp     rax, rcx
    jb      .L3          # Jump if below
```

## Files in This Directory

| File | Description |
|------|-------------|
| `01_simple_functions.c` | See how C becomes assembly |
| `02_optimization_levels.c` | Compare -O0, -O2, -O3 |
| `03_vectorization.c` | See auto-vectorization |
