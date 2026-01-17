# 03 - Assembly Programming

Low-level programming to understand exactly what the CPU executes.

## 📚 Topics Covered

### x86-64 Assembly Fundamentals
- **Registers**: General purpose (RAX, RBX, etc.), subregister access (EAX, AX, AL)
- **Addressing Modes**: Direct, indirect, indexed, RIP-relative
- **Data Movement**: MOV, LEA, MOVS*, STOS*, CMOV
- **Control Flow**: CMP, TEST, conditional jumps, loops, jump tables
- **Functions**: Prologue/epilogue, calling conventions (SysV AMD64, Windows x64)

### SIMD Programming (AVX/AVX-512)
- **Vector Registers**: XMM (128-bit), YMM (256-bit), ZMM (512-bit)
- **AVX Instructions**: vmovups, vaddps, vmulps, vfma
- **Mask Registers**: k0-k7 for conditional operations
- **ML Operations**: Dot product, matrix-vector multiply, quantization

### Assembly Optimization
- **Latency vs Throughput**: Understanding CPU pipelining
- **Dependency Chains**: Breaking them for better ILP
- **Loop Unrolling**: Reducing overhead, increasing parallelism
- **Instruction Selection**: LEA vs ADD, CMOV vs branches
- **Memory Optimization**: Cache locality, prefetching

### Debugging & Tools
- **Disassemblers**: objdump, IDA, Ghidra
- **Debuggers**: GDB, LLDB assembly mode
- **Compiler Output**: Understanding -S flag
- **Godbolt Compiler Explorer**: Online tool

## 🎯 Learning Objectives

- [ ] Read and write x86-64 assembly
- [ ] Understand all addressing modes and data movement
- [ ] Master calling conventions (SysV AMD64)
- [ ] Write optimized loops with unrolling and pipelining
- [ ] Use SIMD instructions for ML operations
- [ ] Debug at the instruction level

## 💻 Practical Exercises

1. Write "Hello World" in pure assembly
2. Implement strlen/strcpy in assembly
3. Write SIMD dot product (float32)
4. Implement INT8 quantized matrix multiply
5. Optimize a loop with unrolling and CMOV
6. Read and understand compiler-generated assembly

## 📁 Structure

```
03-assembly-programming/
├── 01-x86-64-basics/              # Fundamentals
│   ├── README.md                  # Complete guide with performance reference
│   ├── QUICK_REFERENCE.md          # Quick reference card
│   ├── 01_hello_world.s           # First assembly program
│   ├── 02_registers.s             # Register naming, subregisters, calling convention
│   ├── 02_registers_main.c
│   ├── 03_data_movement.s         # MOV, LEA, string ops, CMOV, all addressing modes
│   ├── data_movement_main.c
│   ├── 04_control_flow.s          # CMP, TEST, jumps, loops, jump tables
│   ├── control_flow_main.c
│   ├── 05_functions.s             # Calling conventions, prologue/epilogue, red zone
│   ├── functions_main.c
│   ├── 06_optimization.s          # Latency, throughput, loop unrolling, ILP
│   ├── optimization_main.c
│   └── 07_arithmetic.s           # ADD, SUB, MUL, DIV, shifts, bitwise
│       └── arithmetic_main.c
│
├── 02-simd-avx/                   # SIMD for ML
│   ├── README.md                  # AVX instruction reference
│   ├── 01_avx_basics.c            # AVX intrinsics introduction
│   ├── 02_avx_dotproduct.s        # Hand-written AVX assembly
│   └── 02_dotprod_main.c
│
├── 03-optimization-patterns/      # LLM inference patterns
│   ├── README.md
│   ├── 01_quantized_dot.c         # INT8/INT4 quantization
│   └── 02_prefetch_patterns.c     # Software prefetching
│
├── 04-reading-compiler-output/    # Understanding compilers
│   ├── README.md
│   └── 01_simple_functions.c      # See C → assembly
│
└── LEARNING_ORDER.md              # Recommended learning sequence
```

## 📖 Resources

### Books
- "Programming from the Ground Up" - Jonathan Bartlett (FREE)
- "x86-64 Assembly Language Programming with Ubuntu"

### Online
- [Godbolt Compiler Explorer](https://godbolt.org/)
- [Intel x86-64 Manual](https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html)
- [Agner Fog's Optimization Manuals](https://agner.org/optimize/)
- [Stanford CS107 Guide to x86-64](https://web.stanford.edu/class/cs107/guide/x86-64.html)
- [Brown CS033 x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)

### Essential Tools
```bash
# Assemble and link
as -o prog.o prog.s && ld -o prog prog.o

# With gcc (handles linking)
gcc -no-pie -o prog prog.s main.c

# Generate assembly from C
gcc -S -O2 -fverbose-asm prog.c

# Disassemble binary
objdump -d -M intel prog

# Debug with GDB
gdb ./prog
(gdb) disassemble
(gdb) info registers
(gdb) x/10i $rip
```

## ⏱️ Estimated Time: 4-6 weeks

## Week-by-Week Plan

| Week | Topic | Files | Goals |
|------|-------|-------|-------|
| 1 | Basics | 01_hello_world.s, 02_registers.s | Syntax, registers, first program |
| 2 | Data Movement | 03_data_movement.s | MOV, LEA, addressing modes |
| 3 | Control Flow | 04_control_flow.s | Jumps, loops, jump tables |
| 4 | Functions | 05_functions.s | Calling conventions, stack |
| 5 | Optimization | 06_optimization.s | Latency, ILP, unrolling |
| 6 | SIMD | 02-simd-avx/ | AVX for ML operations |
