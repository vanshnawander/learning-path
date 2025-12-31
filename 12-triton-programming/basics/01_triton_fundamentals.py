"""
01_triton_fundamentals.py - Triton Programming from First Principles

Triton is a high-level GPU programming language by OpenAI.
It's easier than CUDA but produces highly optimized code.

Why Triton?
- Write at block level (not thread level like CUDA)
- Automatic memory coalescing
- Automatic shared memory management
- Built-in auto-tuning
- Compiles to PTX via LLVM

For multimodal training:
- Custom fused kernels (GELU+bias, etc.)
- Flash Attention implementations
- Quantization kernels (INT8, FP8)
- Custom loss functions

Run: python 01_triton_fundamentals.py
Requirements: triton, torch
"""

import torch
import triton
import triton.language as tl
import time
from typing import Tuple, List

# ============================================================================
# PROFILING INFRASTRUCTURE
# ============================================================================

def profile_triton(func, warmup=25, iterations=100):
    """Profile a Triton/PyTorch operation."""
    if not torch.cuda.is_available():
        return 0.0
    
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations

# ============================================================================
# EXPERIMENT 1: YOUR FIRST TRITON KERNEL
# ============================================================================

@triton.jit
def add_kernel(
    x_ptr,      # Pointer to input tensor x
    y_ptr,      # Pointer to input tensor y
    output_ptr, # Pointer to output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    """
    Simple vector addition kernel.
    
    Key Triton concepts:
    - tl.program_id(0): Block index (like blockIdx.x in CUDA)
    - tl.arange(0, BLOCK_SIZE): Range of offsets within block
    - tl.load/tl.store: Memory operations with automatic coalescing
    - mask: Handle boundary conditions
    """
    # Get the program (block) ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (handles non-multiple-of-BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper function for the Triton kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Calculate grid size (number of blocks)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def experiment_first_kernel():
    """Demonstrate basic Triton kernel structure."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: YOUR FIRST TRITON KERNEL")
    print(" Vector addition - the 'Hello World' of GPU programming")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    TRITON KERNEL ANATOMY:
    
    @triton.jit                          # JIT compile decorator
    def kernel(x_ptr, y_ptr, out_ptr,    # Pointers to tensors
               n_elements,                # Size information
               BLOCK_SIZE: tl.constexpr): # Compile-time constants
        
        pid = tl.program_id(0)           # Which block am I?
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements      # Boundary handling
        
        x = tl.load(x_ptr + offsets, mask=mask)  # Load
        y = tl.load(y_ptr + offsets, mask=mask)
        
        out = x + y                      # Compute
        
        tl.store(out_ptr + offsets, out, mask=mask)  # Store
    """)
    
    # Test correctness
    size = 1_000_000
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    output_triton = triton_add(x, y)
    output_torch = x + y
    
    print(f"\n Correctness check:")
    print(f" Max difference: {torch.max(torch.abs(output_triton - output_torch)):.2e}")
    print(f" All close: {torch.allclose(output_triton, output_torch)}")
    
    # Performance comparison
    sizes = [1024, 65536, 1048576, 16777216]
    
    print(f"\n Performance comparison:")
    print(f"{'Size':<15} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Ratio'}")
    print("-" * 60)
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        time_triton = profile_triton(lambda: triton_add(x, y))
        time_torch = profile_triton(lambda: x + y)
        
        ratio = time_torch / time_triton
        print(f"{size:<15} {time_triton:<15.4f} {time_torch:<15.4f} {ratio:.2f}x")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Simple ops: PyTorch is already optimized")
    print(f" - Triton shines for fused/custom operations")
    print(f" - Compilation overhead amortized over many calls")

# ============================================================================
# EXPERIMENT 2: UNDERSTANDING PROGRAM IDs AND BLOCKS
# ============================================================================

@triton.jit
def row_sum_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    x_stride,  # Stride between rows
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sum each row of a 2D tensor.
    Each program handles one row.
    """
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate pointer to start of this row
    row_start = x_ptr + row_idx * x_stride
    
    # Accumulate sum over columns in blocks
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols
        
        values = tl.load(row_start + col_indices, mask=mask, other=0.0)
        acc += values
    
    # Reduce within the block
    row_sum = tl.sum(acc, axis=0)
    
    # Store result
    tl.store(output_ptr + row_idx, row_sum)


def triton_row_sum(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for row sum kernel."""
    n_rows, n_cols = x.shape
    output = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    grid = (n_rows,)  # One program per row
    
    row_sum_kernel[grid](
        x, output,
        n_rows, n_cols,
        x.stride(0),
        BLOCK_SIZE=1024,
    )
    
    return output


def experiment_program_ids():
    """Understanding how work is distributed across programs."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: PROGRAM IDs AND WORK DISTRIBUTION")
    print(" Each program (block) handles a chunk of work")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    WORK DISTRIBUTION:
    
    1D Grid (vector ops):
    ┌─────────────────────────────────────────────────────┐
    │ Program 0 │ Program 1 │ Program 2 │ ... │ Program N │
    │ [0:1024]  │ [1024:2048]│[2048:3072]│     │           │
    └─────────────────────────────────────────────────────┘
    
    2D Grid (matrix ops):
    ┌───────────────────────────────────────┐
    │ P(0,0) │ P(0,1) │ P(0,2) │ P(0,3) │
    │ P(1,0) │ P(1,1) │ P(1,2) │ P(1,3) │
    │ P(2,0) │ P(2,1) │ P(2,2) │ P(2,3) │
    └───────────────────────────────────────┘
    """)
    
    # Test row sum
    rows, cols = 4096, 4096
    x = torch.randn(rows, cols, device='cuda')
    
    output_triton = triton_row_sum(x)
    output_torch = x.sum(dim=1)
    
    print(f" Row sum test ({rows} x {cols}):")
    print(f" Max difference: {torch.max(torch.abs(output_triton - output_torch)):.2e}")
    
    # Performance
    time_triton = profile_triton(lambda: triton_row_sum(x))
    time_torch = profile_triton(lambda: x.sum(dim=1))
    
    print(f"\n Performance:")
    print(f" Triton: {time_triton:.3f} ms")
    print(f" PyTorch: {time_torch:.3f} ms")
    print(f" Ratio: {time_torch/time_triton:.2f}x")

# ============================================================================
# EXPERIMENT 3: MEMORY ACCESS PATTERNS
# ============================================================================

@triton.jit
def coalesced_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Coalesced memory access - consecutive threads access consecutive memory."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced: thread i accesses element i
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


@triton.jit
def strided_copy_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided memory access - demonstrates poor access pattern."""
    pid = tl.program_id(0)
    col_idx = pid  # Each program handles one column
    
    for row_offset in range(0, n_rows, BLOCK_SIZE):
        row_indices = row_offset + tl.arange(0, BLOCK_SIZE)
        mask = row_indices < n_rows
        
        # Strided access: jump by n_cols for each row
        offsets = row_indices * n_cols + col_idx
        
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)


def experiment_memory_patterns():
    """Demonstrate impact of memory access patterns."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: MEMORY ACCESS PATTERNS")
    print(" Coalesced vs strided access")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    MEMORY COALESCING IN TRITON:
    
    Triton automatically handles coalescing, but layout still matters!
    
    Row-major access (coalesced):
    Thread 0: data[0], Thread 1: data[1], Thread 2: data[2], ...
    Memory:   [████████████████] → Single transaction
    
    Column-major access (strided):  
    Thread 0: data[0], Thread 1: data[N], Thread 2: data[2N], ...
    Memory:   [█   █   █   █   ] → Multiple transactions!
    """)
    
    rows, cols = 4096, 4096
    x = torch.randn(rows, cols, device='cuda')
    
    # Row-wise sum (coalesced)
    time_row = profile_triton(lambda: x.sum(dim=1))
    
    # Column-wise sum (strided)
    time_col = profile_triton(lambda: x.sum(dim=0))
    
    print(f"\n PyTorch reduction comparison (4096 x 4096):")
    print(f" Row-wise sum (coalesced):    {time_row:.3f} ms")
    print(f" Column-wise sum (strided):   {time_col:.3f} ms")
    print(f" Strided is {time_col/time_row:.1f}x slower")
    
    # Transposed access
    x_t = x.t().contiguous()
    time_col_contig = profile_triton(lambda: x_t.sum(dim=1))
    print(f" Column sum via transpose:    {time_col_contig:.3f} ms")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Triton handles thread-level coalescing automatically")
    print(f" - But tensor layout (contiguous dimension) still matters!")
    print(f" - Always reduce over the last (contiguous) dimension when possible")

# ============================================================================
# EXPERIMENT 4: AUTO-TUNING
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],  # Re-tune when this changes
)
@triton.jit
def autotuned_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Vector addition with auto-tuning for block size."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def autotuned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for autotuned kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    autotuned_add_kernel[grid](x, y, output, n_elements)
    return output


def experiment_autotuning():
    """Demonstrate Triton's auto-tuning capability."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: AUTO-TUNING")
    print(" Triton automatically finds optimal configuration")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    TRITON AUTO-TUNING:
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 64}),
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
        ],
        key=['n_elements'],  # Re-tune when size changes
    )
    @triton.jit
    def kernel(..., BLOCK_SIZE: tl.constexpr):
        ...
    
    Triton benchmarks each config and picks the fastest!
    """)
    
    sizes = [1024, 65536, 1048576, 16777216]
    
    print(f"\n Auto-tuning in action:")
    print(f"{'Size':<15} {'Time (ms)':<15}")
    print("-" * 30)
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        # First call triggers auto-tuning
        _ = autotuned_add(x, y)
        torch.cuda.synchronize()
        
        # Profile
        time_ms = profile_triton(lambda: autotuned_add(x, y))
        print(f"{size:<15} {time_ms:<15.4f}")
    
    print(f"\n AUTO-TUNING PARAMETERS:")
    print(f" - BLOCK_SIZE: Number of elements per program")
    print(f" - num_warps: Number of warps per block")
    print(f" - num_stages: Software pipelining depth")
    print(f"\n WHEN TO USE:")
    print(f" - Kernels called many times with same shapes")
    print(f" - Development: Find optimal config, then hardcode")

# ============================================================================
# EXPERIMENT 5: FUSED OPERATIONS
# ============================================================================

@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused addition and ReLU - single memory pass."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Fused: add + relu in one kernel
    result = tl.maximum(x + y, 0.0)
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def fused_gelu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU activation.
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU computation - all fused in registers
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)  # sqrt(2/pi) ≈ 0.7978845608
    tanh_inner = tl.libdevice.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused GELU."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def experiment_fusion():
    """Demonstrate the power of kernel fusion."""
    print("\n" + "="*70)
    print(" EXPERIMENT 5: KERNEL FUSION")
    print(" Fusing operations = fewer memory trips = faster")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    size = 16_000_000  # 16M elements
    x = torch.randn(size, device='cuda')
    
    # Unfused GELU
    def unfused_gelu():
        x3 = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x3)
        tanh_inner = torch.tanh(inner)
        return 0.5 * x * (1.0 + tanh_inner)
    
    # PyTorch fused GELU
    def pytorch_gelu():
        return torch.nn.functional.gelu(x)
    
    # Triton fused GELU
    def triton_gelu():
        return triton_fused_gelu(x)
    
    # Verify correctness
    output_unfused = unfused_gelu()
    output_pytorch = pytorch_gelu()
    output_triton = triton_gelu()
    
    print(f"\n Correctness check:")
    print(f" Triton vs PyTorch: {torch.allclose(output_triton, output_pytorch, atol=1e-5)}")
    
    # Performance
    time_unfused = profile_triton(unfused_gelu)
    time_pytorch = profile_triton(pytorch_gelu)
    time_triton = profile_triton(triton_gelu)
    
    print(f"\n GELU Performance ({size/1e6:.0f}M elements):")
    print(f"{'Method':<30} {'Time (ms)':<15} {'Speedup vs Unfused'}")
    print("-" * 60)
    print(f"{'Unfused (manual)':<30} {time_unfused:<15.3f} 1.0x")
    print(f"{'PyTorch F.gelu':<30} {time_pytorch:<15.3f} {time_unfused/time_pytorch:.2f}x")
    print(f"{'Triton fused':<30} {time_triton:<15.3f} {time_unfused/time_triton:.2f}x")
    
    print(f"\n MEMORY TRAFFIC ANALYSIS:")
    print(f" Unfused: ~5 kernels, each reads/writes 64MB → 640 MB total")
    print(f" Fused:   1 kernel, reads 64MB, writes 64MB → 128 MB total")
    print(f" Theoretical speedup: ~5x (memory-bound)")

# ============================================================================
# EXPERIMENT 6: TRITON DEBUGGING
# ============================================================================

def experiment_debugging():
    """Tips for debugging Triton kernels."""
    print("\n" + "="*70)
    print(" EXPERIMENT 6: DEBUGGING TRITON KERNELS")
    print(" Tools and techniques for debugging")
    print("="*70)
    
    print("""
    DEBUGGING TECHNIQUES:
    
    1. PRINT DEBUGGING (in kernel):
       tl.device_print("value:", some_value)
       Note: Only prints for one thread, can slow down kernel
    
    2. INTERPRETER MODE:
       @triton.jit(interpret=True)  # Run on CPU, full debugging
       Or: os.environ["TRITON_INTERPRET"] = "1"
    
    3. ASSERT IN KERNEL:
       tl.device_assert(condition, "error message")
    
    4. CHECK SHAPES:
       Always verify input shapes match expectations
       print(f"x.shape={x.shape}, x.stride()={x.stride()}")
    
    5. NUMERICAL DEBUGGING:
       - Compare with PyTorch reference implementation
       - Check for NaN/Inf: torch.isnan(output).any()
       - Check relative error: (triton - torch).abs().max()
    
    6. TRITON-VIZ:
       from triton_viz import trace
       @trace
       @triton.jit
       def kernel(...):
           ...
       # Visualizes memory access patterns
    
    COMMON BUGS:
    
    1. Off-by-one in masks:
       mask = offsets < n_elements  # Correct
       mask = offsets <= n_elements  # Bug if offsets start at 0
    
    2. Wrong stride:
       ptr + row * stride(0) + col  # Assuming stride(1) == 1
       ptr + row * stride(0) + col * stride(1)  # Correct
    
    3. Missing mask in load/store:
       tl.load(ptr + offsets)  # Crashes if offsets > size
       tl.load(ptr + offsets, mask=mask, other=0.0)  # Safe
    
    4. Integer overflow:
       idx = row * n_cols + col  # Can overflow for large tensors
       idx = row.to(tl.int64) * n_cols + col  # Safe
    """)

# ============================================================================
# SUMMARY
# ============================================================================

def print_triton_summary():
    """Print comprehensive Triton fundamentals summary."""
    print("\n" + "="*70)
    print(" TRITON FUNDAMENTALS SUMMARY")
    print("="*70)
    
    print("""
    TRITON VS CUDA COMPARISON:
    
    ┌────────────────────────────────────────────────────────────────────┐
    │ Aspect          │ CUDA                  │ Triton                  │
    ├─────────────────┼───────────────────────┼─────────────────────────┤
    │ Level           │ Thread-level          │ Block-level             │
    │ Memory          │ Manual (shared mem)   │ Automatic               │
    │ Coalescing      │ Manual                │ Automatic               │
    │ Tuning          │ Manual                │ Auto-tuning             │
    │ Language        │ C++ extension         │ Python + @triton.jit    │
    │ Debugging       │ Harder (printf)       │ interpret=True mode     │
    │ Performance     │ Can be faster         │ Usually 90%+ of CUDA    │
    └────────────────────────────────────────────────────────────────────┘
    
    TRITON KERNEL TEMPLATE:
    
    @triton.jit
    def kernel(
        x_ptr,                    # Input pointer
        output_ptr,               # Output pointer
        n_elements,               # Size
        BLOCK_SIZE: tl.constexpr, # Compile-time constant
    ):
        # 1. Get program ID
        pid = tl.program_id(0)
        
        # 2. Calculate offsets
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # 3. Create mask
        mask = offsets < n_elements
        
        # 4. Load data
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # 5. Compute
        output = x * 2  # Your operation here
        
        # 6. Store result
        tl.store(output_ptr + offsets, output, mask=mask)
    
    KEY TRITON FUNCTIONS:
    
    - tl.program_id(axis)      : Get block index
    - tl.arange(start, end)    : Create range
    - tl.load(ptr, mask, other): Load with masking
    - tl.store(ptr, val, mask) : Store with masking
    - tl.dot(a, b)             : Matrix multiply
    - tl.sum(x, axis)          : Reduction
    - tl.max(x, axis)          : Reduction
    - tl.where(cond, x, y)     : Conditional
    - tl.libdevice.*           : Math functions (sin, cos, exp, etc.)
    
    WHEN TO USE TRITON:
    
    ✓ Custom fused operations
    ✓ Operations not in PyTorch
    ✓ Memory-bound operations that benefit from fusion
    ✓ Flash Attention and variants
    ✓ Quantization kernels
    
    ✗ Simple operations (PyTorch is already optimized)
    ✗ Very complex operations (CUDA may be better)
    ✗ When you need maximum control
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " TRITON PROGRAMMING FUNDAMENTALS ".center(68) + "║")
    print("║" + " High-level GPU programming made easy ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" Triton version: {triton.__version__}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_first_kernel()
    experiment_program_ids()
    experiment_memory_patterns()
    experiment_autotuning()
    experiment_fusion()
    experiment_debugging()
    print_triton_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Advanced patterns - matmul, softmax, attention")
    print("="*70)
