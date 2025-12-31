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
| `02_registers.s` | Register operations |
| `03_memory.s` | Memory addressing modes |
| `04_control_flow.s` | Branches and loops |
| `05_functions.s` | Calling convention |

## Tools

```bash
# Assemble and link
as -o program.o program.s
ld -o program program.o

# Or with gcc
gcc -no-pie -o program program.s

# Disassemble binary
objdump -d program

# Interactive: Godbolt Compiler Explorer
# https://godbolt.org/
```
