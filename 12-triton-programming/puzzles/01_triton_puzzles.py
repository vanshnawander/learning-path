"""
01_triton_puzzles.py - Triton Programming Exercises

Practice problems to master Triton kernel development.
Inspired by Sasha Rush's Triton Puzzles.

Each puzzle has:
1. Problem description
2. Specification (what to implement)
3. Reference solution (hidden until you try!)
4. Test cases

Run: python 01_triton_puzzles.py
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ============================================================================
# PUZZLE 1: VECTOR ADD (Warmup)
# ============================================================================
"""
PUZZLE 1: Vector Addition

Implement a Triton kernel that adds two vectors element-wise.
    output[i] = a[i] + b[i]

This is the "Hello World" of Triton!

Concepts tested:
- Basic kernel structure
- Program IDs
- Memory loading/storing
- Masking
"""

@triton.jit
def puzzle1_vector_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement vector addition
    
    Hints:
    1. Get program ID with tl.program_id(0)
    2. Calculate offsets: pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    3. Create mask for boundary: offsets < n_elements
    4. Load a and b with tl.load(..., mask=mask)
    5. Add and store with tl.store(..., mask=mask)
    """
    # YOUR CODE HERE
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    output = a + b
    
    tl.store(output_ptr + offsets, output, mask=mask)


def test_puzzle1():
    """Test vector addition."""
    print("\n" + "="*60)
    print("PUZZLE 1: Vector Addition")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    # Test
    n = 1000
    a = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    puzzle1_vector_add_kernel[grid](a, b, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = a + b
    passed = torch.allclose(output, expected)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 2: FUSED ADD + RELU
# ============================================================================
"""
PUZZLE 2: Fused Add + ReLU

Implement a kernel that computes:
    output[i] = max(a[i] + b[i], 0)

This demonstrates the power of fusion - two operations in one kernel!

Concepts tested:
- Operation fusion
- Using tl.maximum
"""

@triton.jit
def puzzle2_fused_add_relu_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement fused add + ReLU
    
    Hint: Use tl.maximum(value, 0.0) for ReLU
    """
    # YOUR CODE HERE
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    result = tl.maximum(a + b, 0.0)
    
    tl.store(output_ptr + offsets, result, mask=mask)


def test_puzzle2():
    """Test fused add + ReLU."""
    print("\n" + "="*60)
    print("PUZZLE 2: Fused Add + ReLU")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n = 1000
    a = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')
    output = torch.empty_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    puzzle2_fused_add_relu_kernel[grid](a, b, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = F.relu(a + b)
    passed = torch.allclose(output, expected)
    
    print(f"Test passed: {passed}")
    return passed


# ============================================================================
# PUZZLE 3: ROW SUM (2D Reduction)
# ============================================================================
"""
PUZZLE 3: Row-wise Sum

Given a 2D tensor, compute the sum of each row.
    output[i] = sum(input[i, :])

Concepts tested:
- 2D tensor indexing
- Reductions with tl.sum
- Strides
"""

@triton.jit
def puzzle3_row_sum_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    input_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Sum each row of a 2D tensor
    
    Each program handles one row.
    
    Hints:
    1. row_idx = tl.program_id(0)
    2. row_ptr = input_ptr + row_idx * input_stride
    3. Loop over columns in blocks
    4. Use tl.sum for reduction
    """
    # YOUR CODE HERE
    row_idx = tl.program_id(0)
    row_ptr = input_ptr + row_idx * input_stride
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        values = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
        acc += values
    
    row_sum = tl.sum(acc, axis=0)
    tl.store(output_ptr + row_idx, row_sum)


def test_puzzle3():
    """Test row sum."""
    print("\n" + "="*60)
    print("PUZZLE 3: Row Sum")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n_rows, n_cols = 128, 512
    x = torch.randn(n_rows, n_cols, device='cuda')
    output = torch.empty(n_rows, device='cuda')
    
    grid = (n_rows,)
    puzzle3_row_sum_kernel[grid](
        x, output, n_rows, n_cols, x.stride(0),
        BLOCK_SIZE=256
    )
    
    expected = x.sum(dim=1)
    passed = torch.allclose(output, expected, rtol=1e-4)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 4: SOFTMAX (Single Row)
# ============================================================================
"""
PUZZLE 4: Softmax

Implement softmax for a single row:
    softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))

Concepts tested:
- Multiple passes (find max, compute exp sum, normalize)
- Numerical stability (subtract max)
- tl.max and tl.sum reductions
"""

@triton.jit
def puzzle4_softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement row-wise softmax
    
    Steps:
    1. Load entire row (assume fits in BLOCK_SIZE)
    2. Find max for numerical stability
    3. Compute exp(x - max)
    4. Compute sum of exp
    5. Divide by sum
    """
    # YOUR CODE HERE
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row_ptr = input_ptr + row_idx * input_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Max for stability
    x_max = tl.max(x, axis=0)
    
    # Exp
    x_exp = tl.exp(x - x_max)
    x_exp = tl.where(mask, x_exp, 0.0)
    
    # Sum
    x_sum = tl.sum(x_exp, axis=0)
    
    # Normalize
    softmax = x_exp / x_sum
    
    # Store
    out_ptr = output_ptr + row_idx * output_stride
    tl.store(out_ptr + col_offsets, softmax, mask=mask)


def test_puzzle4():
    """Test softmax."""
    print("\n" + "="*60)
    print("PUZZLE 4: Softmax")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n_rows, n_cols = 32, 256
    x = torch.randn(n_rows, n_cols, device='cuda')
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    puzzle4_softmax_kernel[grid](
        x, output, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=256
    )
    
    expected = F.softmax(x, dim=-1)
    passed = torch.allclose(output, expected, atol=1e-5)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 5: LAYER NORM
# ============================================================================
"""
PUZZLE 5: Layer Normalization

Implement layer norm for each row:
    mean = mean(x)
    var = mean((x - mean)²)
    output = (x - mean) / sqrt(var + eps) * gamma + beta

Concepts tested:
- Multiple statistics (mean, variance)
- Parameter loading (gamma, beta)
- Numerical precision (use float32 for stats)
"""

@triton.jit
def puzzle5_layernorm_kernel(
    input_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    n_cols,
    input_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement layer normalization
    
    Steps:
    1. Load row
    2. Compute mean
    3. Compute variance
    4. Normalize: (x - mean) / sqrt(var + eps)
    5. Scale and shift: output * gamma + beta
    """
    # YOUR CODE HERE
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row_ptr = input_ptr + row_idx * input_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    
    # Mean
    mean = tl.sum(x_fp32, axis=0) / n_cols
    
    # Variance
    x_centered = tl.where(mask, x_fp32 - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    
    # Normalize
    inv_std = tl.math.rsqrt(var + eps)
    x_normed = x_centered * inv_std
    
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    
    # Scale and shift
    output = x_normed * gamma + beta
    
    # Store
    out_ptr = output_ptr + row_idx * input_stride
    tl.store(out_ptr + col_offsets, output.to(x.dtype), mask=mask)


def test_puzzle5():
    """Test layer norm."""
    print("\n" + "="*60)
    print("PUZZLE 5: Layer Normalization")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n_rows, n_cols = 32, 256
    x = torch.randn(n_rows, n_cols, device='cuda')
    gamma = torch.ones(n_cols, device='cuda')
    beta = torch.zeros(n_cols, device='cuda')
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    puzzle5_layernorm_kernel[grid](
        x, output, gamma, beta, n_cols,
        x.stride(0), 1e-5,
        BLOCK_SIZE=256
    )
    
    layer_norm = torch.nn.LayerNorm(n_cols, device='cuda')
    expected = layer_norm(x)
    
    passed = torch.allclose(output, expected, atol=1e-4)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 6: MATMUL (Tiled)
# ============================================================================
"""
PUZZLE 6: Tiled Matrix Multiplication

Implement C = A @ B using tiling.
    C[i, j] = sum_k(A[i, k] * B[k, j])

This is the most important operation in deep learning!

Concepts tested:
- 2D program grid
- Tiling/blocking
- Accumulation
- tl.dot for hardware-accelerated matmul
"""

@triton.jit
def puzzle6_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    TODO: Implement tiled matrix multiplication
    
    Each program computes a BLOCK_M x BLOCK_N tile of C.
    
    Steps:
    1. Get tile indices from program IDs
    2. Initialize accumulator to zeros
    3. Loop over K dimension in blocks
    4. Load A and B tiles, accumulate with tl.dot
    5. Store result
    """
    # YOUR CODE HERE
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        
        # Load A tile
        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def test_puzzle6():
    """Test matmul."""
    print("\n" + "="*60)
    print("PUZZLE 6: Tiled Matrix Multiplication")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    M, K, N = 256, 256, 256
    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')
    c = torch.empty(M, N, device='cuda')
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    puzzle6_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    expected = a @ b
    passed = torch.allclose(c, expected, rtol=1e-3, atol=1e-3)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(c - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 7: FUSED GELU
# ============================================================================
"""
PUZZLE 7: GELU Activation (Fused)

Implement GELU:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

This is the activation function used in GPT/BERT.

Concepts tested:
- Math operations
- Using tl.libdevice for tanh
- Numerical constants
"""

@triton.jit
def puzzle7_gelu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Implement GELU activation
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    sqrt(2/π) ≈ 0.7978845608028654
    """
    # YOUR CODE HERE
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    tanh_inner = tl.libdevice.tanh(inner)
    gelu = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(output_ptr + offsets, gelu, mask=mask)


def test_puzzle7():
    """Test GELU."""
    print("\n" + "="*60)
    print("PUZZLE 7: GELU Activation")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n = 10000
    x = torch.randn(n, device='cuda')
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    puzzle7_gelu_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = F.gelu(x)
    passed = torch.allclose(output, expected, atol=1e-4)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# PUZZLE 8: ONLINE MAX (Streaming)
# ============================================================================
"""
PUZZLE 8: Online Maximum

Implement an online algorithm to find the maximum of a very long vector
that doesn't fit in a single block.

Process blocks sequentially, maintaining a running maximum.

Concepts tested:
- Online/streaming algorithms
- State across iterations
- Handling large inputs
"""

@triton.jit
def puzzle8_online_max_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TODO: Find maximum using online algorithm
    
    Process one block at a time, update running max.
    Only one program (pid=0) should run.
    """
    # YOUR CODE HERE
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    running_max = -float('inf')
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        block = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(block, axis=0)
        running_max = tl.maximum(running_max, block_max)
    
    tl.store(output_ptr, running_max)


def test_puzzle8():
    """Test online max."""
    print("\n" + "="*60)
    print("PUZZLE 8: Online Maximum")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n = 100000
    x = torch.randn(n, device='cuda')
    output = torch.empty(1, device='cuda')
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single program
    
    puzzle8_online_max_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = x.max()
    passed = torch.allclose(output[0], expected)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Expected: {expected.item()}, Got: {output[0].item()}")
    
    return passed


# ============================================================================
# CHALLENGE: RMSNORM (Unsloth-style)
# ============================================================================
"""
CHALLENGE: RMSNorm

Implement RMSNorm used in Llama/Mistral:
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

This is simpler than LayerNorm (no mean subtraction, no beta).

Concepts tested:
- All previous concepts combined
- Production-quality kernel
"""

@triton.jit
def challenge_rmsnorm_kernel(
    input_ptr, output_ptr, weight_ptr,
    n_cols, input_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    CHALLENGE: Implement RMSNorm
    
    RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight
    """
    # YOUR CODE HERE
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row_ptr = input_ptr + row_idx * input_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute RMS in FP32
    x_fp32 = x.to(tl.float32)
    variance = tl.sum(x_fp32 * x_fp32, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(variance + eps)
    
    # Normalize
    x_normed = x_fp32 * inv_rms
    
    # Load weight and apply
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = x_normed * weight
    
    # Store
    out_ptr = output_ptr + row_idx * input_stride
    tl.store(out_ptr + col_offsets, output.to(x.dtype), mask=mask)


def test_challenge_rmsnorm():
    """Test RMSNorm challenge."""
    print("\n" + "="*60)
    print("CHALLENGE: RMSNorm (Unsloth-style)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False
    
    n_rows, n_cols = 32, 256
    x = torch.randn(n_rows, n_cols, device='cuda')
    weight = torch.ones(n_cols, device='cuda')
    output = torch.empty_like(x)
    eps = 1e-6
    
    grid = (n_rows,)
    challenge_rmsnorm_kernel[grid](
        x, output, weight, n_cols,
        x.stride(0), eps,
        BLOCK_SIZE=256
    )
    
    # Reference implementation
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    expected = (x.float() / rms * weight).to(x.dtype)
    
    passed = torch.allclose(output, expected, atol=1e-4)
    
    print(f"Test passed: {passed}")
    if not passed:
        print(f"Max error: {(output - expected).abs().max()}")
    
    return passed


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_puzzles():
    """Run all puzzle tests."""
    print("╔" + "═"*58 + "╗")
    print("║" + " TRITON PUZZLES - Practice Problems ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    
    results = []
    
    results.append(("Puzzle 1: Vector Add", test_puzzle1()))
    results.append(("Puzzle 2: Fused Add+ReLU", test_puzzle2()))
    results.append(("Puzzle 3: Row Sum", test_puzzle3()))
    results.append(("Puzzle 4: Softmax", test_puzzle4()))
    results.append(("Puzzle 5: Layer Norm", test_puzzle5()))
    results.append(("Puzzle 6: Matmul", test_puzzle6()))
    results.append(("Puzzle 7: GELU", test_puzzle7()))
    results.append(("Puzzle 8: Online Max", test_puzzle8()))
    results.append(("Challenge: RMSNorm", test_challenge_rmsnorm()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASSED" if p else "✗ FAILED"
        print(f" {name:<30} {status}")
    
    print(f"\n Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n 🎉 Congratulations! You've mastered the basics of Triton!")
        print(" Next: Try the advanced kernels in 02_unsloth_kernels.py")


if __name__ == "__main__":
    run_all_puzzles()
