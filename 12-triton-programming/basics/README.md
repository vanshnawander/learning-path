# Triton Programming Basics

This directory covers the fundamentals of Triton GPU programming with profiled examples.

## Modules

### 01_triton_fundamentals.py
**Complete introduction to Triton programming**

- First Triton kernel (vector addition)
- Program IDs and work distribution
- Memory access patterns
- Auto-tuning
- Kernel fusion benefits
- Debugging techniques

**Key Profiled Experiments:**
- Triton vs PyTorch performance comparison
- Coalesced vs strided access
- Auto-tuning demonstration
- Fused GELU performance

**Run:** `python 01_triton_fundamentals.py`

## Learning Objectives

- [ ] Write basic Triton kernels
- [ ] Understand block-based programming model
- [ ] Use auto-tuning effectively
- [ ] Implement fused operations

## Triton vs CUDA

| Aspect | CUDA | Triton |
|--------|------|--------|
| Level | Thread | Block |
| Memory | Manual | Automatic |
| Tuning | Manual | Auto-tune |
| Language | C++ | Python |

## Expected Time

- Reading + Running: 2-3 hours
- Deep understanding: 1 day with Python syntax.

## Why Triton?

| Aspect | CUDA | Triton |
|--------|------|--------|
| Language | C++ | Python |
| Abstraction | Threads | Blocks |
| Shared mem | Manual | Automatic |
| Performance | Expert-level | Near-expert |
| Learning curve | Steep | Moderate |

## First Kernel: Vector Add

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID (which block)
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for out-of-bounds
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Launching Kernels

```python
def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Grid: how many blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

## Key Concepts
- `tl.program_id`: Which block am I?
- `tl.arange`: Create range within block
- `tl.load/tl.store`: Memory access with masks
- `tl.constexpr`: Compile-time constants
