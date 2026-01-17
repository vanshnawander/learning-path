# x86-64 Assembly Basics

Understanding machine code - what the CPU actually executes.

## Why Learn Assembly for ML?

1. **Understanding compiler output** - See what your C/CUDA code becomes
2. **SIMD optimization** - AVX/AVX-512 for LLM inference (llama.cpp)
3. **Debugging** - Understand crashes at the instruction level
4. **Performance analysis** - Know instruction latency/throughput

## x86-64 Registers

### General Purpose (64-bit)
```
RAX, RBX, RCX, RDX  - Data registers
RSI, RDI            - Source/Destination index
RBP, RSP            - Base/Stack pointer
R8-R15              - Extended registers
```

### Subregisters
```
RAX (64-bit) → EAX (32-bit) → AX (16-bit) → AL/AH (8-bit)
```

### SIMD Registers
```
XMM0-XMM15   - 128-bit (SSE)
YMM0-YMM15   - 256-bit (AVX)
ZMM0-ZMM31   - 512-bit (AVX-512)
```

## Calling Convention (System V AMD64)

| Purpose | Register |
|---------|----------|
| Args 1-6 | RDI, RSI, RDX, RCX, R8, R9 |
| Return | RAX |
| Callee-saved | RBX, RBP, R12-R15 |
| Caller-saved | Everything else |

## Files in This Directory

| File | Description |
|------|-------------|
| `01_hello_world.s` | First assembly program |
| `02_registers.s` | Register operations and subregister access |
| `03_data_movement.s` | Complete MOV, LEA, string ops, CMOV |
| `04_control_flow.s` | CMP, TEST, jumps, loops, jump tables |
| `05_functions.s` | Calling conventions, prologue/epilogue |
| `06_optimization.s` | Latency, throughput, loop unrolling |
| `07_arithmetic.s` | ADD, SUB, MUL, DIV, shifts, bitwise |

### C Wrappers (compile with gcc)

| File | Description |
|------|-------------|
| `02_registers_main.c` | Test register operations |
| `data_movement_main.c` | Test data movement |
| `control_flow_main.c` | Test control flow |
| `functions_main.c` | Test calling conventions |
| `optimization_main.c` | Test optimization patterns |
| `arithmetic_main.c` | Test arithmetic operations |

## Week-by-Week Plan

1. **01_hello_world.s** - Start here, understand basic syntax
2. **02_registers.s** - Learn register naming and usage
3. **03_data_movement.s** - Master MOV, LEA, addressing modes
4. **04_control_flow.s** - Understand branching and loops
5. **05_functions.s** - Learn calling conventions
6. **06_optimization.s** - Advanced optimization techniques
7. **07_arithmetic.s** - Arithmetic, shifts, bitwise operations

## Key Concepts

### AT&T Syntax
```
operation source, destination
mov $42, %rax      # RAX = 42 (immediate)
mov %rax, %rbx     # RBX = RAX (register)
mov (%rax), %rcx   # RCX = *RAX (memory load)
mov %rcx, (%rax)   # *RAX = RCX (memory store)
```

### Addressing Modes
```
mov 8(%rsp, %rdi, 4), %rax   # RAX = *(RSP + RDI*4 + 8)
lea (%rdi, %rsi, 2), %rax    # RAX = RDI + RSI*2 (address calc)
```

### Important Behaviors
- **32-bit MOV** zero-extends to 64 bits automatically
- **LEA** calculates address without dereferencing
- **CMP** sets flags but doesn't store result
- **CMOV** moves conditionally based on flags

## Tools

```bash
# Assemble and link
as -o program.o program.s
ld -o program program.o

# Or with gcc
gcc -no-pie -o program program.s main.c

# Disassemble binary
objdump -d -M intel program

# Generate assembly from C
gcc -S -O2 -fverbose-asm program.c

# Interactive: Godbolt Compiler Explorer
# https://godbolt.org/
```

## Performance Reference

### Instruction Latency (cycles, approximate)
| Instruction | Latency | Throughput |
|-------------|---------|------------|
| MOV reg,reg | 1 | 1 |
| ADD/SUB | 1 | 1 |
| LEA | 1-3 | 1 |
| MUL 64-bit | 3-5 | 1 |
| DIV 64-bit | 15-40 | 15-40 |
| CMOV | 1 | 1 |

### Optimization Checklist
- [ ] Break dependency chains with XOR
- [ ] Use LEA for multiplication by constants
- [ ] Unroll loops (4x or 8x) for better ILP
- [ ] Sequential memory access for cache locality
- [ ] Use CMOV instead of branches for simple conditions
- [ ] Avoid division in loops (use shifts)
