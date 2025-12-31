# 03 - Assembly Programming

Low-level programming to understand exactly what the CPU executes.

## 📚 Topics Covered

### x86-64 Assembly
- **Registers**: General purpose, special purpose
- **Addressing Modes**: Immediate, register, memory
- **Instructions**: MOV, ADD, CMP, JMP, CALL
- **Calling Conventions**: System V ABI, Windows ABI
- **Stack Operations**: PUSH, POP, function calls

### ARM Assembly
- **ARM64/AArch64**: Modern ARM architecture
- **Register Set**: x0-x30, special registers
- **Conditional Execution**: Flags, branches
- **NEON SIMD**: Vector operations

### Assembly Optimization
- **Instruction Latency**: Cycles per instruction
- **Throughput**: Instructions per cycle
- **Micro-ops**: Instruction decoding
- **Loop Unrolling**: Manual optimization
- **SIMD in Assembly**: AVX intrinsics

### Debugging & Tools
- **Disassemblers**: objdump, IDA, Ghidra
- **Debuggers**: GDB, LLDB assembly mode
- **Compiler Output**: Understanding -S flag
- **Godbolt Compiler Explorer**: Online tool

## 🎯 Learning Objectives

- [ ] Read x86-64 disassembly
- [ ] Write simple assembly functions
- [ ] Understand calling conventions
- [ ] Use SIMD instructions directly
- [ ] Debug at assembly level

## 💻 Practical Exercises

1. Write "Hello World" in assembly
2. Implement strlen in assembly
3. Write SIMD dot product
4. Reverse engineer a simple binary

## 📖 Resources

### Books
- "Programming from the Ground Up" - Jonathan Bartlett (FREE)
- "x86-64 Assembly Language Programming with Ubuntu"

### Online
- Godbolt Compiler Explorer (godbolt.org)
- Intel x86-64 Manual
- ARM Architecture Reference Manual

## 📁 Structure

```
03-assembly-programming/
├── 01-x86-64-basics/           # Fundamentals
│   ├── README.md
│   ├── 01_hello_world.s
│   ├── 02_registers.s
│   └── 02_registers_main.c
├── 02-simd-avx/                # SIMD for ML
│   ├── README.md
│   ├── 01_avx_basics.c
│   ├── 02_avx_dotproduct.s
│   └── 02_dotprod_main.c
├── 03-optimization-patterns/   # LLM inference
│   ├── README.md
│   ├── 01_quantized_dot.c
│   └── 02_prefetch_patterns.c
├── 04-reading-compiler-output/ # Understanding compilers
│   ├── README.md
│   └── 01_simple_functions.c
└── LEARNING_ORDER.md
```

## ⏱️ Estimated Time: 3-4 weeks
