"""
01_softmax_kernel.py - Implementing Softmax in Triton

Softmax is a fundamental operation in transformers and neural networks.
It's also a great example of a memory-bound operation that benefits
from kernel fusion.

Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

The challenge:
1. Need to compute max over entire row (first pass)
2. Need to compute sum of exp (second pass)
3. Need to compute final result (third pass)

Standard: 3 memory passes
Fused: 1-2 memory passes (huge win for memory-bound ops!)

Run: python 01_softmax_kernel.py
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math

# ============================================================================
# PROFILING
# ============================================================================

def profile_triton(func, warmup=25, iterations=100):
    """Profile a function."""
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
# SOFTMAX KERNEL - SINGLE ROW PER PROGRAM
# ============================================================================

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel - one row per program.
    
    Algorithm:
    1. Load row into SRAM (registers)
    2. Compute max (for numerical stability)
    3. Compute exp(x - max)
    4. Compute sum
    5. Normalize
    6. Store
    
    All in one kernel = minimal memory traffic!
    """
    # Row index
    row_idx = tl.program_id(0)
    
    # Pointer to start of this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Column offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Pointers for this row
    input_ptrs = row_start_ptr + col_offsets
    
    # Mask for valid columns
    mask = col_offsets < n_cols
    
    # Load row (use -inf for masked positions so they don't affect max)
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exp
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)
    
    # Zero out masked positions for sum
    numerator = tl.where(mask, numerator, 0.0)
    
    # Compute sum
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for softmax kernel."""
    n_rows, n_cols = x.shape
    
    # Block size must be power of 2 and >= n_cols for this simple kernel
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Limit block size
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    
    output = torch.empty_like(x)
    
    # One program per row
    grid = (n_rows,)
    
    softmax_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# ============================================================================
# SOFTMAX WITH AUTO-TUNING
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_autotuned(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax with auto-tuning for different column sizes."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    numerator = tl.where(mask, numerator, 0.0)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax_autotuned(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for autotuned softmax."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    softmax_kernel_autotuned[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
    )
    
    return output

# ============================================================================
# ONLINE SOFTMAX (FOR LARGE SEQUENCES)
# ============================================================================

@triton.jit
def online_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax - handles sequences longer than BLOCK_SIZE.
    
    Uses the "online" algorithm:
    - Process chunks, maintain running max and sum
    - Correct for max changes using the recurrence
    
    This is how Flash Attention computes softmax!
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Initialize running statistics
    m_prev = -float('inf')  # Running max
    l_prev = 0.0            # Running sum of exp
    
    # Process in blocks
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load block
        block = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Block statistics
        m_curr = tl.max(block, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        
        # Correction factor for previous sum
        correction = tl.exp(m_prev - m_new)
        
        # Update sum: l_new = l_prev * correction + sum(exp(block - m_new))
        block_exp = tl.exp(block - m_new)
        block_exp = tl.where(mask, block_exp, 0.0)
        l_curr = tl.sum(block_exp, axis=0)
        l_new = l_prev * correction + l_curr
        
        # Update running stats
        m_prev = m_new
        l_prev = l_new
    
    # Second pass: compute normalized output
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        block = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Compute softmax
        softmax_block = tl.exp(block - m_prev) / l_prev
        
        # Store
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, softmax_block, mask=mask)


def triton_online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for online softmax."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Use smaller block size for online processing
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    grid = (n_rows,)
    online_softmax_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_softmax_correctness():
    """Verify softmax implementations are correct."""
    print("\n" + "="*70)
    print(" EXPERIMENT 1: SOFTMAX CORRECTNESS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Test various shapes
    shapes = [(32, 128), (128, 512), (1024, 1024), (4096, 4096)]
    
    print(f"\n{'Shape':<20} {'Max Error':<20} {'Correct?'}")
    print("-" * 55)
    
    for shape in shapes:
        x = torch.randn(shape, device='cuda')
        
        # PyTorch reference
        ref = F.softmax(x, dim=-1)
        
        # Triton implementation
        out = triton_softmax(x)
        
        max_error = torch.max(torch.abs(out - ref)).item()
        correct = torch.allclose(out, ref, atol=1e-5)
        
        print(f"{str(shape):<20} {max_error:<20.2e} {correct}")
    
    # Test online softmax on large sequence
    print(f"\n Online softmax (large sequences):")
    x_large = torch.randn(32, 8192, device='cuda')
    ref_large = F.softmax(x_large, dim=-1)
    out_large = triton_online_softmax(x_large)
    
    max_error = torch.max(torch.abs(out_large - ref_large)).item()
    print(f" Shape (32, 8192): Max error = {max_error:.2e}")


def experiment_softmax_performance():
    """Benchmark softmax implementations."""
    print("\n" + "="*70)
    print(" EXPERIMENT 2: SOFTMAX PERFORMANCE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # Test different sizes
    test_cases = [
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 1024),
        (1024, 4096),
        (8192, 2048),
    ]
    
    print(f"\n{'Shape':<18} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup'}")
    print("-" * 65)
    
    for shape in test_cases:
        x = torch.randn(shape, device='cuda')
        
        time_pytorch = profile_triton(lambda: F.softmax(x, dim=-1))
        time_triton = profile_triton(lambda: triton_softmax(x))
        
        speedup = time_pytorch / time_triton
        print(f"{str(shape):<18} {time_pytorch:<15.4f} {time_triton:<15.4f} {speedup:.2f}x")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Triton softmax can match or beat PyTorch")
    print(f" - Performance depends on shape and tuning")
    print(f" - PyTorch uses highly optimized cuDNN kernels")


def experiment_softmax_memory():
    """Analyze memory efficiency of fused softmax."""
    print("\n" + "="*70)
    print(" EXPERIMENT 3: SOFTMAX MEMORY EFFICIENCY")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    MEMORY ANALYSIS:
    
    Standard Softmax (unfused):
    1. max_val = x.max(dim=-1)     → Read x (N bytes)
    2. x_shifted = x - max_val    → Read x, Write temp (2N bytes)
    3. exp_x = exp(x_shifted)     → Read temp, Write temp (2N bytes)
    4. sum_exp = exp_x.sum()      → Read temp (N bytes)
    5. out = exp_x / sum_exp      → Read temp, Write out (2N bytes)
    
    Total: ~8N bytes memory traffic
    
    Fused Softmax (Triton):
    1. Load x, compute all in registers, store out
    
    Total: ~2N bytes memory traffic (4x reduction!)
    """)
    
    # Demonstrate with unfused vs fused
    n_rows, n_cols = 4096, 2048
    x = torch.randn(n_rows, n_cols, device='cuda')
    
    def unfused_softmax(x):
        max_val = x.max(dim=-1, keepdim=True).values
        x_shifted = x - max_val
        exp_x = torch.exp(x_shifted)
        sum_exp = exp_x.sum(dim=-1, keepdim=True)
        return exp_x / sum_exp
    
    time_unfused = profile_triton(lambda: unfused_softmax(x))
    time_fused = profile_triton(lambda: F.softmax(x, dim=-1))
    time_triton = profile_triton(lambda: triton_softmax(x))
    
    print(f" Performance ({n_rows} x {n_cols}):")
    print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup vs Unfused'}")
    print("-" * 55)
    print(f"{'Unfused (manual)':<25} {time_unfused:<15.3f} 1.0x")
    print(f"{'PyTorch F.softmax':<25} {time_fused:<15.3f} {time_unfused/time_fused:.2f}x")
    print(f"{'Triton fused':<25} {time_triton:<15.3f} {time_unfused/time_triton:.2f}x")
    
    # Memory bandwidth calculation
    bytes_per_elem = 4  # float32
    total_bytes = n_rows * n_cols * bytes_per_elem
    
    bw_unfused = (8 * total_bytes) / (time_unfused / 1000) / 1e9
    bw_fused = (2 * total_bytes) / (time_fused / 1000) / 1e9
    
    print(f"\n Memory bandwidth utilization:")
    print(f" Unfused: ~{bw_unfused:.0f} GB/s (8 passes)")
    print(f" Fused:   ~{bw_fused:.0f} GB/s (2 passes)")


def experiment_online_softmax():
    """Demonstrate online softmax algorithm."""
    print("\n" + "="*70)
    print(" EXPERIMENT 4: ONLINE SOFTMAX ALGORITHM")
    print(" Key technique used in Flash Attention")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    print("""
    ONLINE SOFTMAX ALGORITHM:
    
    Problem: Standard softmax needs max over entire row first.
    For attention with sequence length S, this means:
    - Cannot fuse QK^T computation with softmax
    - Must materialize O(S²) attention matrix
    
    Solution: Online (streaming) algorithm
    
    For each new block of values:
    1. Compute local max: m_curr = max(block)
    2. Update global max: m_new = max(m_prev, m_curr)
    3. Correct previous sum: l_prev * exp(m_prev - m_new)
    4. Add new contribution: sum(exp(block - m_new))
    
    This allows single-pass softmax!
    """)
    
    # Test online softmax on long sequences
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    
    print(f"\n Long sequence softmax performance:")
    print(f"{'Seq Len':<12} {'Standard (ms)':<18} {'Online (ms)':<18} {'Max Error'}")
    print("-" * 70)
    
    for seq_len in seq_lens:
        try:
            x = torch.randn(32, seq_len, device='cuda')
            
            time_standard = profile_triton(lambda: triton_softmax(x), iterations=50)
            time_online = profile_triton(lambda: triton_online_softmax(x), iterations=50)
            
            ref = F.softmax(x, dim=-1)
            out = triton_online_softmax(x)
            max_error = torch.max(torch.abs(out - ref)).item()
            
            print(f"{seq_len:<12} {time_standard:<18.3f} {time_online:<18.3f} {max_error:.2e}")
        except Exception as e:
            print(f"{seq_len:<12} Error: {str(e)[:40]}")
    
    print(f"\n FLASH ATTENTION CONNECTION:")
    print(f" Flash Attention uses online softmax to:")
    print(f" 1. Compute attention in tiles (block by block)")
    print(f" 2. Never materialize full S×S attention matrix")
    print(f" 3. Achieve O(S) memory instead of O(S²)")


def print_softmax_summary():
    """Print summary of softmax implementations."""
    print("\n" + "="*70)
    print(" SOFTMAX IMPLEMENTATION SUMMARY")
    print("="*70)
    
    print("""
    SOFTMAX ALGORITHM:
    
    softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    
    Subtracting max is crucial for numerical stability!
    exp(1000) = inf, but exp(1000 - 1000) = 1
    
    IMPLEMENTATION STRATEGIES:
    
    1. STANDARD (3-PASS):
       - Pass 1: Find max
       - Pass 2: Compute exp(x - max) and sum
       - Pass 3: Divide by sum
       Memory: O(3N) passes
    
    2. FUSED (2-PASS OR 1-PASS):
       - Load entire row to registers
       - Compute max, exp, sum, divide
       - Single write
       Memory: O(2N) or O(N) passes
    
    3. ONLINE (STREAMING):
       - Process in blocks
       - Maintain running max and sum
       - Correct for max changes
       Memory: O(2N) but works for arbitrary length
    
    TRITON SOFTMAX TEMPLATE:
    
    @triton.jit
    def softmax_kernel(input_ptr, output_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(input_ptr + row * stride + cols, mask=mask, other=-inf)
        
        x_max = tl.max(x, axis=0)
        x_exp = tl.exp(x - x_max)
        x_exp = tl.where(mask, x_exp, 0.0)
        x_sum = tl.sum(x_exp, axis=0)
        out = x_exp / x_sum
        
        tl.store(output_ptr + row * stride + cols, out, mask=mask)
    
    MULTIMODAL IMPLICATIONS:
    - Attention softmax dominates long-context models
    - Fused softmax critical for efficiency
    - Online algorithm enables Flash Attention
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " TRITON SOFTMAX IMPLEMENTATION ".center(68) + "║")
    print("║" + " From naive to Flash Attention's online algorithm ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n WARNING: CUDA not available")
    
    experiment_softmax_correctness()
    experiment_softmax_performance()
    experiment_softmax_memory()
    experiment_online_softmax()
    print_softmax_summary()
    
    print("\n" + "="*70)
    print(" NEXT: Matrix multiplication and Flash Attention")
    print("="*70)
