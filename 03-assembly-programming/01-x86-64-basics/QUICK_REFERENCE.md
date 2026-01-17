# x86-64 Assembly Quick Reference Card

## Register Naming Convention

### General Purpose Registers (64-bit)
| Register | 32-bit | 16-bit | 8-bit Low | 8-bit High |
|----------|--------|--------|-----------|------------|
| RAX | EAX | AX | AL | AH |
| RBX | EBX | BX | BL | BH |
| RCX | ECX | CX | CL | CH |
| RDX | EDX | DX | DL | DH |
| RSI | ESI | SI | SIL | - |
| RDI | EDI | DI | DIL | - |
| RBP | EBP | BP | BPL | - |
| RSP | ESP | SP | SPL | - |
| R8 | R8D | R8W | R8B | - |
| R9 | R9D | R9W | R9B | - |
| R10 | R10D | R10W | R10B | - |
| R11 | R11D | R11W | R11B | - |
| R12 | R12D | R12W | R12B | - |
| R13 | R13D | R13W | R13B | - |
| R14 | R14D | R14W | R14B | - |
| R15 | R15D | R15W | R15B | - |

### Special Registers
| Register | Purpose |
|----------|---------|
| RIP | Instruction pointer |
| RSP | Stack pointer |
| RBP | Frame pointer |
| RFLAGS | Flags (CF, PF, AF, ZF, SF, OF, etc.) |

### SIMD Registers
| Register | Width | Floats | Bytes |
|----------|-------|--------|-------|
| XMM0-15 | 128-bit | 4 | 16 |
| YMM0-15 | 256-bit | 8 | 32 |
| ZMM0-31 | 512-bit | 16 | 64 |

## System V AMD64 Calling Convention

### Argument Passing
| Position | Register |
|-----------|----------|
| 1st | RDI |
| 2nd | RSI |
| 3rd | RDX |
| 4th | RCX |
| 5th | R8 |
| 6th | R9 |
| 7th+ | Stack (right to left) |

### Register Roles
| Register | Purpose |
|----------|---------|
| RAX | Return value, caller-saved scratch |
| RBX | Callee-saved |
| RCX | 4th argument, caller-saved |
| RDX | 3rd argument, caller-saved |
| RSI | 2nd argument, caller-saved |
| RDI | 1st argument, caller-saved |
| RBP | Callee-saved frame pointer |
| RSP | Stack pointer |
| R8 | 5th argument, caller-saved |
| R9 | 6th argument, caller-saved |
| R10-R11 | Caller-saved |
| R12-R15 | Callee-saved |

## Data Movement

### MOV Variants
```asm
mov $imm, %reg          # Immediate to register
mov %reg1, %reg2        # Register to register
mov mem, %reg           # Memory to register
mov %reg, mem           # Register to memory
movb/w/l/q              # 8/16/32/64-bit move
```

### Sign/Zero Extension
```asm
movsbl %al, %ebx        # byte → dword, sign-extend
movzbl %al, %ebx        # byte → dword, zero-extend
movswl %ax, %edx        # word → dword, sign-extend
movzwl %ax, %edx        # word → dword, zero-extend
movslq %eax, %rax        # dword → qword, sign-extend
cltq                     # EAX → RAX, sign-extend (shortcut)
```

### LEA (Load Effective Address)
```asm
lea disp(%base), %rax   # RAX = base + displacement
lea (%rdi, %rsi, 2), %rax # RAX = RDI + RSI*2
lea (%rdi, %rsi, 4), %rax # RAX = RDI + RSI*4
lea -1(%rdi), %rax       # RAX = RDI - 1
```

### Conditional Moves
```asm
cmove %rax, %rbx        # Move if equal
cmovne %rax, %rbx       # Move if not equal
cmovl %rax, %rbx        # Move if less (signed)
cmovg %rax, %rbx        # Move if greater (signed)
cmovb %rax, %rbx        # Move if below (unsigned)
cmova %rax, %rbx        # Move if above (unsigned)
```

## Addressing Modes

```
disp(base, index, scale)

Examples:
(%rax)              # Base only
8(%rax)             # Base + displacement
-16(%rbp)           # Base + negative displacement
(%rax, %rcx, 4)     # Base + index*scale
8(%rsp, %rdi, 8)    # Base + index*scale + displacement
(%rax, %rcx)        # Base + index (scale=1)
0x10(, %rdx, 4)     # Index*scale + displacement (no base)
```

## Arithmetic Operations

### Basic Arithmetic
```asm
add $imm, %reg        # reg += imm
add %reg1, %reg2       # reg2 += reg1
sub $imm, %reg         # reg -= imm
inc %reg              # reg++
dec %reg              # reg--
neg %reg              # reg = -reg
```

### Multiplication
```asm
mul %reg               # RDX:RAX = RAX * reg (unsigned)
imul %reg              # RDX:RAX = RAX * reg (signed)
imul $imm, %reg        # reg *= imm
imul $imm, %reg1, %reg2 # reg2 = reg1 * imm
```

### Division
```asm
div %reg               # RDX:RAX / reg (unsigned)
idiv %reg              # RDX:RAX / reg (signed)
# Quotient in RAX, remainder in RDX
```

### Shifts
```asm
shl $n, %reg           # Logical left shift
shr $n, %reg           # Logical right shift (unsigned)
sar $n, %reg           # Arithmetic right shift (signed)
shl %cl, %reg          # Shift by CL
```

## Bitwise Operations
```asm
and $imm, %reg         # reg &= imm
or $imm, %reg          # reg |= imm
xor $imm, %reg         # reg ^= imm
not %reg               # reg = ~reg
```

## Control Flow

### Comparison
```asm
cmp %reg1, %reg2       # Sets flags (reg2 - reg1)
test %reg1, %reg2      # Sets flags (reg1 & reg2)
```

### Conditional Jumps (Signed)
```asm
je label               # Equal (ZF=1)
jne label              # Not equal (ZF=0)
jl label               # Less (SF≠OF)
jle label              # Less or equal (ZF=1 or SF≠OF)
jg label               # Greater (ZF=0 and SF=OF)
jge label              # Greater or equal (SF=OF)
```

### Conditional Jumps (Unsigned)
```asm
je label               # Equal
jne label              # Not equal
jb label               # Below (CF=1)
jbe label              # Below or equal (CF=1 or ZF=1)
ja label               # Above (CF=0 and ZF=0)
jae label              # Above or equal (CF=0)
```

### Flag Jumps
```asm
jc label               # Carry (CF=1)
jnc label              # No carry (CF=0)
jo label               # Overflow (OF=1)
jno label              # No overflow (OF=0)
js label               # Sign (SF=1)
jns label              # No sign (SF=0)
jp label               # Parity even
jnp label              # Parity odd
```

### Unconditional Jumps
```asm
jmp label              # Direct jump
jmp *%rax              # Indirect jump (register)
jmp *(%rax)            # Indirect jump (memory)
call label             # Call function
ret                    # Return from function
```

## String Operations
```asm
movsb/w/l/q            # Move string byte/word/dword/qword
stosb/w/l/q            # Store string
lodsb/w/l/q            # Load string
cmpsb/w/l/q            # Compare string
scasb/w/l/q            # Scan string
rep movsb              # Repeat while RCX != 0
repe cmpsb             # Repeat while equal
```

## Common Patterns

### Zero a Register
```asm
xor %rax, %rax         # Faster than mov $0, %rax
```

### Function Prologue
```asm
push %rbp
mov %rsp, %rbp
push %rbx
push %r12
# ... function body ...
pop %r12
pop %rbx
pop %rbp
ret
```

### Set if Condition
```asm
cmp %rax, %rbx
setl %al               # AL = 1 if RBX < RAX, else 0
movzx %al, %eax         # Zero-extend to 64-bit
```

### Absolute Value
```asm
mov $-42, %rax
cmp $0, %rax
cmovs %rax, %rbx
neg %rbx
```

### Clamp to Range
```asm
# Clamp RDI to [RSI, RDX]
cmp %rsi, %rdi
cmovl %rsi, %rdi        # if val < min, val = min
cmp %rdx, %rdi
cmovg %rdx, %rdi        # if val > max, val = max
```

## Instruction Latency Reference

| Instruction | Latency | Throughput |
|-------------|---------|------------|
| MOV | 1 | 1 |
| ADD/SUB | 1 | 1 |
| AND/OR/XOR | 1 | 1 |
| SHL/SHR | 1 | 1 |
| LEA | 1-3 | 1 |
| MUL (64-bit) | 3-5 | 1 |
| DIV (64-bit) | 15-40 | 15-40 |
| CMOV | 1 | 1 |
| FMA (AVX) | 4-5 | 1 |
| VADDPS (AVX) | 1 | 1 |
| VMULPS (AVX) | 4 | 1 |

## Flags Register (RFLAGS)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | CF | Carry flag |
| 2 | PF | Parity flag |
| 4 | AF | Auxiliary carry |
| 6 | ZF | Zero flag |
| 7 | SF | Sign flag |
| 8 | TF | Trap flag |
| 9 | IF | Interrupt enable |
| 10 | DF | Direction flag |
| 11 | OF | Overflow flag |
| 12-13 | IOPL | I/O privilege level |
| 14 | NT | Nested task |
| 16 | RF | Resume flag |
| 17 | VM | Virtual-8086 mode |
| 18 | AC | Alignment check |
| 19 | VIF | Virtual interrupt flag |
| 20 | VIP | Virtual interrupt pending |
| 21 | ID | CPUID available |

## Stack Operations
```asm
push %reg               # RSP -= 8; *RSP = reg
pop %reg                # reg = *RSP; RSP += 8
push $imm               # RSP -= 8; *RSP = imm
pusha                   # Push all general registers
popa                    # Pop all general registers
pushf                   # Push flags
popf                    # Pop flags
```

## Multi-Precision Arithmetic
```asm
# 128-bit addition: [RDX:RAX] + [RBX:RCX]
clc                     # Clear carry
add %rcx, %rax          # Add low parts
adc %rbx, %rdx          # Add high parts with carry

# 128-bit subtraction: [RDX:RAX] - [RBX:RCX]
sub %rcx, %rax          # Subtract low parts
sbb %rbx, %rdx          # Subtract high parts with borrow
```

## Useful Directives
```asm
.section .data         # Initialized data
.section .text         # Code
.section .bss          # Uninitialized data
.global name            # Export symbol
.local name             # Local symbol
.equ name, value        # Define constant
.set name, value        # Set symbol value
.align n               # Align to 2^n
.p2align n             # Align to 2^n
.ascii "str"           # String (no null)
.asciz "str"           # String with null
.byte 1,2,3            # Bytes
.word 1,2,3            # 16-bit words
.long 1,2,3            # 32-bit longs
.quad 1,2,3            # 64-bit quads
.float 1.0             # Single float
.double 1.0            # Double float
```
