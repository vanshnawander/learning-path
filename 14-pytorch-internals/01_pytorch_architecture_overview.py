"""
01_pytorch_architecture_overview.py - PyTorch Architecture Deep Dive

This is the MOST IMPORTANT module for understanding PyTorch internals.
Without understanding this architecture, you cannot contribute effectively.

PyTorch Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER CODE (Python)                                │
│                     import torch; x = torch.randn(3,3)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                           torch/ (Python Frontend)                          │
│ • torch.Tensor - Python tensor class                                       │
│ • torch.nn - Neural network modules                                        │
│ • torch.autograd - Automatic differentiation                               │
│ • torch.optim - Optimizers                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                    torch/csrc/ (C++ Python Bindings)                        │
│ • THPVariable - Python Tensor wrapper                                      │
│ • pybind11 bindings                                                        │
│ • Autograd engine                                                          │
│ • JIT compiler                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ATen/ (A Tensor Library)                            │
│ • Tensor operations implementation                                         │
│ • CPU kernels (native/)                                                    │
│ • CUDA kernels (cuda/)                                                     │
│ • Dispatcher system                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                       c10/ (Core Library)                                   │
│ • TensorImpl - Core tensor data structure                                  │
│ • Storage - Memory management                                              │
│ • Device, Layout, Dtype                                                    │
│ • Allocators                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                          HARDWARE                                           │
│                    CPU (OpenMP/MKL) | CUDA | ROCm | MPS                    │
└─────────────────────────────────────────────────────────────────────────────┘

Key Files to Study:
- c10/core/TensorImpl.h - The heart of tensors
- aten/src/ATen/native/native_functions.yaml - Operator definitions
- torch/csrc/autograd/engine.cpp - Autograd engine
- aten/src/ATen/Dispatch.h - Dtype dispatch macros

Run: python 01_pytorch_architecture_overview.py
"""

import torch
import torch.nn as nn
import time
import sys
import gc
from typing import List, Tuple, Optional

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_cuda(func, warmup=10, iterations=100):
    """Profile GPU operation with proper synchronization."""
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

def profile_cpu(func, warmup=5, iterations=50):
    """Profile CPU operation."""
    for _ in range(warmup):
        func()
    
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) * 1000 / iterations

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

def explain_directory_structure():
    """Explain PyTorch source code directory structure."""
    print("\n" + "="*70)
    print(" PYTORCH SOURCE CODE STRUCTURE")
    print(" Understanding where code lives is ESSENTIAL for contribution")
    print("="*70)
    
    print("""
    pytorch/
    ├── torch/                    # PYTHON FRONTEND
    │   ├── __init__.py          # Main torch module
    │   ├── tensor.py            # Tensor Python class (mostly stubs)
    │   ├── nn/                  # Neural network modules
    │   │   ├── modules/         # nn.Linear, nn.Conv2d, etc.
    │   │   └── functional.py    # F.relu, F.softmax, etc.
    │   ├── autograd/            # Python autograd interface
    │   ├── optim/               # Optimizers (SGD, Adam, etc.)
    │   ├── cuda/                # CUDA utilities
    │   ├── distributed/         # Distributed training
    │   ├── fx/                  # FX graph transformation
    │   ├── _dynamo/             # TorchDynamo graph capture
    │   ├── _inductor/           # Inductor compiler backend
    │   └── _functorch/          # Functorch transforms
    │
    ├── torch/csrc/              # C++ PYTHON BINDINGS
    │   ├── Module.cpp           # Main _C module
    │   ├── autograd/            # Autograd engine (C++)
    │   │   ├── engine.cpp       # THE autograd engine
    │   │   ├── function.h       # autograd::Node base
    │   │   └── variable.h       # Variable (tensor with grad)
    │   ├── jit/                 # TorchScript JIT compiler
    │   ├── api/                 # C++ frontend (libtorch)
    │   └── cuda/                # CUDA Python bindings
    │
    ├── aten/                    # ATEN: A TENSOR LIBRARY
    │   └── src/ATen/
    │       ├── native/          # Modern C++ operators
    │       │   ├── cpu/         # CPU implementations
    │       │   ├── cuda/        # CUDA implementations
    │       │   ├── *.cpp        # Cross-platform ops
    │       │   └── native_functions.yaml  # OPERATOR DEFINITIONS!
    │       ├── core/            # Core tensor types
    │       ├── TensorIterator.h # Efficient element-wise ops
    │       └── Dispatch.h       # Dtype dispatch macros
    │
    ├── c10/                     # CORE LIBRARY (Caffe2 + ATen = c10)
    │   ├── core/                # Core abstractions
    │   │   ├── TensorImpl.h     # THE tensor implementation
    │   │   ├── Storage.h        # Memory storage
    │   │   ├── Device.h         # Device abstraction
    │   │   ├── ScalarType.h     # Data types (float, int, etc.)
    │   │   └── DispatchKey.h    # Dispatch key system
    │   ├── cuda/                # CUDA allocator, streams
    │   └── util/                # Utilities
    │
    ├── torchgen/                # CODE GENERATION
    │   ├── gen.py               # Main codegen script
    │   └── model.py             # YAML parsing
    │
    ├── tools/                   # BUILD & DEV TOOLS
    │   ├── autograd/            # Autograd codegen
    │   └── nightly.py           # Nightly builds
    │
    └── test/                    # TESTS
        ├── test_torch.py        # Core tests
        ├── test_autograd.py     # Autograd tests
        └── test_nn.py           # NN tests
    
    KEY INSIGHT: Most code is AUTO-GENERATED!
    ─────────────────────────────────────────────────────────────────
    When you build PyTorch, these files are generated:
    
    • torch/csrc/autograd/generated/  - Autograd functions
    • aten/src/ATen/generated/        - ATen operators
    • torch/_C/                       - Python bindings
    
    The source of truth is:
    • aten/src/ATen/native/native_functions.yaml
    • tools/autograd/derivatives.yaml
    """)

# ============================================================================
# TENSOR INTERNALS
# ============================================================================

def explain_tensor_internals():
    """Deep dive into tensor data structure."""
    print("\n" + "="*70)
    print(" TENSOR INTERNALS: TensorImpl")
    print(" The heart of every PyTorch tensor")
    print("="*70)
    
    print("""
    TENSOR DATA STRUCTURE (c10/core/TensorImpl.h):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          TensorImpl                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Storage storage_            │ Pointer to actual data                    │
    │ ├── data_ptr               │ Raw memory pointer                        │
    │ ├── byte_size              │ Total bytes allocated                     │
    │ └── allocator              │ Memory allocator (CPU/CUDA)               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ sizes_and_strides_          │ Shape and stride information              │
    │ ├── sizes_                 │ [dim0, dim1, dim2, ...]                   │
    │ └── strides_               │ [stride0, stride1, stride2, ...]          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ storage_offset_             │ Offset into storage (for views)           │
    │ numel_                      │ Total number of elements                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ data_type_                  │ ScalarType (float32, int64, etc.)         │
    │ device_opt_                 │ Device (CPU, CUDA:0, etc.)                │
    │ key_set_                    │ DispatchKeySet (for dispatch)             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    WHY STRIDES MATTER:
    ─────────────────────────────────────────────────────────────────
    
    Physical memory is 1D. Strides map N-D indices to 1D offsets.
    
    tensor[i, j, k] = storage[offset + i*stride[0] + j*stride[1] + k*stride[2]]
    
    Example: 2x3 tensor stored row-major
    
    Logical:        Physical memory:
    [[1, 2, 3],     [1, 2, 3, 4, 5, 6]
     [4, 5, 6]]      ^     ^
                     │     └── tensor[1,0] = storage[0 + 1*3 + 0*1] = storage[3]
                     └── tensor[0,0] = storage[0 + 0*3 + 0*1] = storage[0]
    
    sizes = [2, 3]
    strides = [3, 1]  # Jump 3 to go to next row, 1 to go to next column
    """)
    
    # Live demonstration
    print("\n LIVE DEMONSTRATION:")
    print("-" * 50)
    
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f" Tensor:\n{x}")
    print(f" Shape: {x.shape}")
    print(f" Strides: {x.stride()}")
    print(f" Storage offset: {x.storage_offset()}")
    print(f" Is contiguous: {x.is_contiguous()}")
    
    # View vs Clone
    print(f"\n Views share storage:")
    y = x[1, :]  # View of second row
    print(f" y = x[1, :] = {y}")
    print(f" y.stride() = {y.stride()}")
    print(f" y.storage_offset() = {y.storage_offset()}")
    print(f" y shares storage with x: {y.storage().data_ptr() == x.storage().data_ptr()}")
    
    # Transpose changes strides
    print(f"\n Transpose changes strides, not data:")
    z = x.t()
    print(f" z = x.t()")
    print(f" z.stride() = {z.stride()}")  # Now [1, 3] instead of [3, 1]
    print(f" z.is_contiguous() = {z.is_contiguous()}")

# ============================================================================
# DISPATCH SYSTEM
# ============================================================================

def explain_dispatch_system():
    """Explain PyTorch's dispatch system."""
    print("\n" + "="*70)
    print(" DISPATCH SYSTEM")
    print(" How PyTorch routes operations to the right implementation")
    print("="*70)
    
    print("""
    WHEN YOU CALL torch.add(x, y):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. PYTHON → C++ BINDING                                                 │
    │    Python: torch.add(x, y)                                             │
    │         ↓                                                               │
    │    C++: THPVariable_add (auto-generated)                               │
    │         ↓                                                               │
    │    Parse arguments, release GIL                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 2. DISPATCHER LOOKUP                                                    │
    │    Get DispatchKeySet from tensors                                     │
    │         ↓                                                               │
    │    Find highest priority key that has registered kernel                │
    │         ↓                                                               │
    │    Route to appropriate implementation                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 3. DISPATCH KEY PRIORITY (highest first)                               │
    │                                                                         │
    │    Python (for torch_function)                                         │
    │         ↓                                                               │
    │    Autograd (records grad_fn)                                          │
    │         ↓                                                               │
    │    AutocastCUDA (mixed precision)                                      │
    │         ↓                                                               │
    │    Functionalize (for torch.compile)                                   │
    │         ↓                                                               │
    │    CUDA / CPU / MPS (actual kernel)                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 4. KERNEL EXECUTION                                                     │
    │                                                                         │
    │    AT_DISPATCH_ALL_TYPES (dtype dispatch)                              │
    │         ↓                                                               │
    │    Actual computation (TensorIterator or custom)                       │
    │         ↓                                                               │
    │    Return result                                                       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    DISPATCH KEYS (c10/core/DispatchKey.h):
    ─────────────────────────────────────────────────────────────────
    
    // High-level functionality (processed first)
    Python,              // __torch_function__ override
    PythonTLSSnapshot,   // TLS state capture
    
    // Autograd
    AutogradCPU,
    AutogradCUDA,
    AutogradOther,
    
    // Transforms
    Functionalize,       // For torch.compile
    FuncTorchBatched,    // vmap
    FuncTorchGradWrapper,// grad transform
    
    // Backend (actual kernels)
    CPU,
    CUDA,
    MPS,                 // Apple Metal
    XLA,                 // TPU
    
    // Quantization
    QuantizedCPU,
    QuantizedCUDA,
    
    HOW DISPATCH KEYS WORK:
    ─────────────────────────────────────────────────────────────────
    
    Each tensor has a DispatchKeySet (bitmask of active keys).
    When you call an operation:
    
    1. Compute union of all input tensor key sets
    2. Find highest priority key with registered kernel
    3. Call that kernel
    4. Kernel may "redispatch" to next key
    
    Example: CUDA tensor with requires_grad=True
    
    KeySet = {CUDA, AutogradCUDA}
    
    add(x, y) dispatches to:
    1. AutogradCUDA::add (records grad_fn)
    2. AutogradCUDA calls redispatch to CUDA::add
    3. CUDA::add executes actual kernel
    """)

# ============================================================================
# OPERATOR REGISTRATION
# ============================================================================

def explain_operator_registration():
    """Explain how operators are defined and registered."""
    print("\n" + "="*70)
    print(" OPERATOR REGISTRATION")
    print(" How to add new operators to PyTorch")
    print("="*70)
    
    print("""
    NATIVE_FUNCTIONS.YAML (aten/src/ATen/native/native_functions.yaml):
    ─────────────────────────────────────────────────────────────────
    
    This YAML file is THE source of truth for PyTorch operators.
    Example entry:
    
    - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      device_check: NoCheck
      structured_delegate: add.out
      variants: function, method
      dispatch:
        SparseCPU: add_sparse
        SparseCUDA: add_sparse_cuda
        SparseCsrCPU: add_sparse_csr
        MkldnnCPU: mkldnn_add
      tags: pointwise
    
    BREAKDOWN:
    
    - func: add.Tensor(...)
      │     │    └── Function signature
      │     └── Overload name (distinguishes add.Tensor from add.Scalar)
      └── Keyword for function definition
    
    - structured_delegate: add.out
      └── Use add.out's structured kernel implementation
    
    - dispatch:
        SparseCPU: add_sparse
      └── Sparse CPU uses add_sparse function instead of default
    
    - variants: function, method
      └── Available as torch.add() AND tensor.add()
    
    STRUCTURED KERNELS:
    ─────────────────────────────────────────────────────────────────
    
    Modern PyTorch uses "structured kernels" pattern:
    
    1. Meta function: Compute output shape/dtype (no computation)
    2. Impl function: Actual computation
    
    // In native_functions.yaml
    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      structured: True
      dispatch:
        CPU: add_out_cpu
        CUDA: add_out_cuda
    
    // In C++ (aten/src/ATen/native/BinaryOps.cpp)
    TORCH_META_FUNC(add) (
      const Tensor& self, const Tensor& other, const Scalar& alpha
    ) {
      // Just compute output shape
      build_borrowing_binary_op(maybe_get_output(), self, other);
    }
    
    TORCH_IMPL_FUNC(add_out) (
      const Tensor& self, const Tensor& other, const Scalar& alpha,
      const Tensor& result
    ) {
      // Actual computation using TensorIterator
      add_stub(device_type(), *this, alpha);
    }
    
    CODE GENERATION PIPELINE:
    ─────────────────────────────────────────────────────────────────
    
    native_functions.yaml
           ↓
    torchgen/gen.py (codegen script)
           ↓
    ┌──────┴──────┐
    │             │
    ▼             ▼
    ATen ops    Python bindings
    (C++)       (C++ → Python)
    """)

# ============================================================================
# AUTOGRAD INTERNALS
# ============================================================================

def explain_autograd():
    """Deep dive into autograd engine."""
    print("\n" + "="*70)
    print(" AUTOGRAD ENGINE INTERNALS")
    print(" How PyTorch computes gradients")
    print("="*70)
    
    print("""
    AUTOGRAD CONCEPTS:
    ─────────────────────────────────────────────────────────────────
    
    1. VARIABLE (tensor with grad tracking)
       - Wraps tensor with AutogradMeta
       - Contains grad_fn (how to compute gradient)
       - Contains grad (accumulated gradient)
    
    2. NODE (autograd::Node / torch.autograd.Function)
       - Represents an operation in computation graph
       - Has apply() method to compute gradients
       - Connected by edges (inputs/outputs)
    
    3. EDGE
       - Connection between nodes
       - Points to (function, input_nr) pair
       - Represents flow of gradients
    
    COMPUTATION GRAPH STRUCTURE:
    ─────────────────────────────────────────────────────────────────
    
    Forward: z = x * y + 1
    
    x (leaf)      y (leaf)
         \\          /
          \\        /
           MulBackward
                |
                + -------- 1 (constant)
                |
           AddBackward
                |
                z
    
    AUTOGRAD ENGINE (torch/csrc/autograd/engine.cpp):
    ─────────────────────────────────────────────────────────────────
    
    When you call loss.backward():
    
    1. CREATE GRAPH TASK
       - Start from output node (loss.grad_fn)
       - Track which nodes need to be executed
    
    2. TOPOLOGICAL SORT (kind of)
       - Use reference counting
       - Node ready when all outputs processed
    
    3. EXECUTE NODES
       - Call node.apply(grad_outputs) → grad_inputs
       - Accumulate gradients for leaf tensors
       - May use multiple threads (per device)
    
    4. HOOKS
       - Pre-hooks: modify grad before apply
       - Post-hooks: modify grad after apply
       - Tensor hooks: on specific tensors
    
    KEY CODE LOCATIONS:
    ─────────────────────────────────────────────────────────────────
    
    torch/csrc/autograd/
    ├── engine.cpp         # Main backward engine
    ├── function.h         # autograd::Node base class
    ├── variable.h         # Variable (tensor + grad)
    ├── autograd_meta.cpp  # AutogradMeta storage
    ├── custom_function.cpp# torch.autograd.Function
    └── graph_task.h       # Backward pass state
    
    tools/autograd/
    ├── derivatives.yaml   # Derivative definitions!
    └── gen_autograd.py    # Generates backward functions
    
    DERIVATIVES.YAML EXAMPLE:
    ─────────────────────────────────────────────────────────────────
    
    - name: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      self: grad
      other: maybe_multiply(grad, alpha)
    
    - name: mul.Tensor(Tensor self, Tensor other) -> Tensor
      self: mul_tensor_backward(grad, other, self.scalar_type())
      other: mul_tensor_backward(grad, self, other.scalar_type())
    
    - name: mm(Tensor self, Tensor mat2) -> Tensor
      self: mm_mat1_backward(grad, mat2, self, 1)
      mat2: mm_mat2_backward(grad, self, mat2, 1)
    """)
    
    # Live demonstration
    print("\n LIVE DEMONSTRATION:")
    print("-" * 50)
    
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    z = x * y + 1
    
    print(f" x = {x.item()}, y = {y.item()}")
    print(f" z = x * y + 1 = {z.item()}")
    print(f"\n z.grad_fn = {z.grad_fn}")
    print(f" z.grad_fn.next_functions = {z.grad_fn.next_functions}")
    
    # Get the mul node
    mul_node = z.grad_fn.next_functions[0][0]
    print(f"\n Mul node: {mul_node}")
    print(f" Mul inputs: {mul_node.next_functions}")
    
    # Backward
    z.backward()
    print(f"\n After backward():")
    print(f" x.grad = {x.grad} (dz/dx = y = 3)")
    print(f" y.grad = {y.grad} (dz/dy = x = 2)")

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def explain_memory_management():
    """Explain PyTorch memory management."""
    print("\n" + "="*70)
    print(" MEMORY MANAGEMENT")
    print(" How PyTorch allocates and manages memory")
    print("="*70)
    
    print("""
    CUDA CACHING ALLOCATOR (c10/cuda/CUDACachingAllocator.cpp):
    ─────────────────────────────────────────────────────────────────
    
    PyTorch uses a caching allocator to avoid expensive cudaMalloc calls.
    
    ALLOCATION STRATEGY:
    
    1. REQUEST: Need N bytes
    2. CHECK CACHE: Is there a free block >= N bytes?
       YES → Use cached block (possibly split)
       NO  → Call cudaMalloc
    3. ON FREE: Don't call cudaFree, add to cache
    
    BLOCK POOLS:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Small Pool (< 1MB blocks)                                       │
    │ └── For small allocations, reduces fragmentation               │
    ├─────────────────────────────────────────────────────────────────┤
    │ Large Pool (>= 1MB blocks)                                      │
    │ └── For large allocations                                      │
    └─────────────────────────────────────────────────────────────────┘
    
    MEMORY STATS:
    """)
    
    if torch.cuda.is_available():
        # Reset stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        print(f"\n Current CUDA memory state:")
        print(f" Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f" Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Allocate some tensors
        tensors = []
        for i in range(5):
            t = torch.randn(1000, 1000, device='cuda')
            tensors.append(t)
        
        print(f"\n After allocating 5 x (1000x1000) tensors:")
        print(f" Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f" Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Free tensors
        tensors.clear()
        gc.collect()
        torch.cuda.synchronize()
        
        print(f"\n After deleting tensors (memory cached, not freed):")
        print(f" Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f" Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Empty cache
        torch.cuda.empty_cache()
        
        print(f"\n After empty_cache() (actually freed):")
        print(f" Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f" Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    else:
        print("\n CUDA not available")
    
    print("""
    
    REFERENCE COUNTING:
    ─────────────────────────────────────────────────────────────────
    
    PyTorch uses intrusive_ptr (similar to shared_ptr):
    
    • TensorImpl has refcount
    • Storage has separate refcount
    • Views share Storage (increment Storage refcount)
    • When refcount → 0, memory freed (or returned to cache)
    
    MEMORY FRAGMENTATION:
    ─────────────────────────────────────────────────────────────────
    
    Problem: Many small allocations/frees create holes
    
    [████|    |████|    |████]  ← Fragmented
         └────────────────────── Can't allocate large block!
    
    Solutions:
    • Caching allocator coalesces adjacent free blocks
    • empty_cache() returns memory to CUDA
    • max_split_size_mb config controls splitting behavior
    
    DEBUGGING MEMORY:
    ─────────────────────────────────────────────────────────────────
    
    # Memory snapshot
    torch.cuda.memory._dump_snapshot("snapshot.pkl")
    
    # Memory summary
    print(torch.cuda.memory_summary())
    
    # Record memory history
    torch.cuda.memory._record_memory_history()
    # ... do stuff ...
    torch.cuda.memory._dump_snapshot("history.pkl")
    """)

# ============================================================================
# PROFILING PYTORCH INTERNALS
# ============================================================================

def profile_pytorch_internals():
    """Profile and examine PyTorch operations."""
    print("\n" + "="*70)
    print(" PROFILING PYTORCH INTERNALS")
    print(" Understanding what happens under the hood")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n CUDA not available for profiling demo")
        return
    
    # Profile a simple operation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    print("\n Profiling torch.add(x, y):")
    print("-" * 50)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            z = torch.add(x, y)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Profile with stack traces
    print("\n Profiling with dispatch keys visible:")
    print("-" * 50)
    
    # Show dispatch
    print(f" x.requires_grad = {x.requires_grad}")
    print(f" Dispatch keys would be: {{CUDA}}")
    
    x.requires_grad_(True)
    print(f"\n After requires_grad_(True):")
    print(f" Dispatch keys would be: {{CUDA, AutogradCUDA}}")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        z = x + y
        z.sum().backward()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

# ============================================================================
# SUMMARY
# ============================================================================

def print_architecture_summary():
    """Print architecture summary."""
    print("\n" + "="*70)
    print(" PYTORCH ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("""
    KEY CONCEPTS TO REMEMBER:
    
    1. TENSOR = TensorImpl + Storage
       • TensorImpl: metadata (sizes, strides, dtype, device)
       • Storage: actual memory
       • Views share Storage
    
    2. DISPATCH = Multi-level routing
       • DispatchKey: functionality to invoke
       • DispatchKeySet: bitmask of active keys
       • Dispatcher: routes to correct kernel
    
    3. AUTOGRAD = Dynamic computation graph
       • grad_fn: points to backward function
       • AutogradMeta: gradient storage
       • Engine: executes backward pass
    
    4. CODE GENERATION
       • native_functions.yaml → ATen operators
       • derivatives.yaml → Autograd functions
       • torchgen/ → Code generators
    
    5. MEMORY
       • CUDACachingAllocator: caches allocations
       • Reference counting: intrusive_ptr
       • empty_cache(): return to system
    
    WHERE TO START CONTRIBUTING:
    ─────────────────────────────────────────────────────────────────
    
    EASY:
    • Add tests (test/)
    • Fix documentation
    • Python utilities (torch/nn/, torch/optim/)
    
    MEDIUM:
    • Add new operator (native_functions.yaml + implementation)
    • Fix bugs in existing operators
    • Improve error messages
    
    HARD:
    • Dispatcher changes (c10/core/)
    • Autograd engine (torch/csrc/autograd/)
    • New backend support
    
    NEXT: Study 02_tensor_operations.py for operator deep dive
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH ARCHITECTURE DEEP DIVE ".center(68) + "║")
    print("║" + " Understanding the internals for contribution ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    
    explain_directory_structure()
    explain_tensor_internals()
    explain_dispatch_system()
    explain_operator_registration()
    explain_autograd()
    explain_memory_management()
    profile_pytorch_internals()
    print_architecture_summary()
    
    print("\n" + "="*70)
    print(" This is the foundation - study the source code next!")
    print("="*70)
