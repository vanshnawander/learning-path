# Assembly Programming Learning Order

Master low-level optimization for LLM inference.

## Prerequisites

- Complete `01-computer-architecture` module
- Basic C programming
- Understanding of binary/hex

## Week 1: x86-64 Fundamentals
**Folder**: `01-x86-64-basics/`

1. `01_hello_world.s` - First assembly program
2. `02_registers.s` - Register usage
3. `02_registers_main.c` - C wrapper

**Goal:** Understand registers, calling convention, syscalls.

## Week 2: SIMD/AVX
**Folder**: `02-simd-avx/`

1. `01_avx_basics.c` - AVX intrinsics
2. `02_avx_dotproduct.s` - Hand-written AVX
3. `02_dotprod_main.c` - Benchmark

**Goal:** Write vectorized code for ML operations.

## Week 3: Optimization Patterns
**Folder**: `03-optimization-patterns/`

1. `01_quantized_dot.c` - INT8/INT4 quantization
2. `02_prefetch_patterns.c` - Software prefetching

**Goal:** Apply techniques from llama.cpp.

## Week 4: Reading Compiler Output
**Folder**: `04-reading-compiler-output/`

1. `01_simple_functions.c` - See C → assembly
2. Use Godbolt Compiler Explorer

**Goal:** Understand what compiler generates.

## Key Tools

```bash
# Assemble
as -o prog.o prog.s
ld -o prog prog.o

# With C
gcc -no-pie -o prog prog.s main.c

# Generate assembly from C
gcc -S -O2 -fverbose-asm prog.c

# Disassemble
objdump -d -M intel prog

# Online: https://godbolt.org/
```

## LLM Inference Relevance

| Assembly Concept | LLM Use Case |
|------------------|--------------|
| AVX dot product | Matrix-vector multiply |
| INT8 SIMD | Quantized inference |
| Prefetching | KV cache access |
| FMA chains | GEMM kernels |
| VNNI | INT8 neural networks |

## Resources

- Intel Intrinsics Guide: https://intel.ly/3example
- llama.cpp ggml: Study `ggml.c` SIMD implementations
- Agner Fog's optimization manuals
