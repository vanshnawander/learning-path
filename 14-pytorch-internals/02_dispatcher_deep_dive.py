"""
02_dispatcher_deep_dive.py - PyTorch Dispatcher System

The Dispatcher is the ROUTING SYSTEM of PyTorch.
Every operation goes through it to find the right implementation.

Understanding the Dispatcher is CRITICAL for:
- Adding new backends (XLA, MPS, custom hardware)
- Implementing custom operators
- Understanding torch.compile, vmap, etc.
- Debugging performance issues

Key Source Files:
- c10/core/DispatchKey.h - All dispatch keys
- c10/core/DispatchKeySet.h - Key set operations
- aten/src/ATen/core/dispatch/Dispatcher.h - Main dispatcher
- aten/src/ATen/core/op_registration/ - Operator registration

Run: python 02_dispatcher_deep_dive.py
"""

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any, Dict, List
import functools

# ============================================================================
# DISPATCH KEY SYSTEM
# ============================================================================

def explain_dispatch_keys():
    """Explain the dispatch key system in detail."""
    print("\n" + "="*70)
    print(" DISPATCH KEY SYSTEM")
    print(" The foundation of PyTorch's extensibility")
    print("="*70)
    
    print("""
    WHAT IS A DISPATCH KEY?
    ─────────────────────────────────────────────────────────────────
    
    A DispatchKey is an enum that identifies a "functionality" or "backend".
    When you call an operation, PyTorch looks up which key should handle it.
    
    DISPATCH KEY CATEGORIES (c10/core/DispatchKey.h):
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ FUNCTIONALITY KEYS (Transform behavior)                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Python          │ __torch_function__ / __torch_dispatch__              │
    │ Autograd*       │ Record operations for backward                       │
    │ Autocast*       │ Automatic mixed precision                            │
    │ Batched         │ vmap batching                                        │
    │ Functionalize   │ Convert mutations to functional ops                  │
    │ FuncTorchGrad*  │ grad() transform                                     │
    │ Tracing         │ fx.symbolic_trace                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ BACKEND KEYS (Actual implementations)                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ CPU             │ CPU tensor operations                                │
    │ CUDA            │ NVIDIA GPU operations                                │
    │ MPS             │ Apple Metal Performance Shaders                      │
    │ XLA             │ TPU / XLA compiler                                   │
    │ Meta            │ Shape-only computation (no data)                     │
    │ SparseCPU/CUDA  │ Sparse tensor operations                            │
    │ QuantizedCPU    │ Quantized operations                                │
    │ MkldnnCPU       │ Intel MKL-DNN optimized ops                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    DISPATCH KEY PRIORITY (Higher = Processed First):
    ─────────────────────────────────────────────────────────────────
    
    Python (highest)
       ↓
    FuncTorchDynamicLayerFrontMode
       ↓
    Functionalize
       ↓
    FuncTorchBatched (vmap)
       ↓
    FuncTorchGradWrapper
       ↓
    AutogradFunctionality
       ↓
    Autocast
       ↓
    [Backend Keys: CUDA, CPU, etc.]
       ↓
    CompositeExplicitAutograd (lowest)
    
    WHY PRIORITY MATTERS:
    ─────────────────────────────────────────────────────────────────
    
    Consider: torch.add(x, y) where x is CUDA tensor with requires_grad=True
    
    DispatchKeySet = {CUDA, AutogradCUDA}
    
    Order of dispatch:
    1. AutogradCUDA::add (records grad_fn)
       - Saves inputs for backward
       - Calls redispatch to next key
    2. CUDA::add (actual computation)
       - Launches CUDA kernel
       - Returns result
    
    If vmap is active:
    DispatchKeySet = {CUDA, AutogradCUDA, FuncTorchBatched}
    
    1. FuncTorchBatched::add (handles batching)
    2. AutogradCUDA::add (records grad_fn)
    3. CUDA::add (actual computation)
    """)

# ============================================================================
# DISPATCH KEY SET
# ============================================================================

def explain_dispatch_key_set():
    """Explain DispatchKeySet operations."""
    print("\n" + "="*70)
    print(" DISPATCH KEY SET")
    print(" Bitmask of active dispatch keys")
    print("="*70)
    
    print("""
    DISPATCHKEYSET STRUCTURE:
    ─────────────────────────────────────────────────────────────────
    
    A DispatchKeySet is a 64-bit bitmask where each bit represents a key.
    
    Example:
    CPU tensor: key_set = 0b...00000001 (just CPU bit set)
    CUDA tensor: key_set = 0b...00000010 (just CUDA bit set)
    CUDA + grad: key_set = 0b...10000010 (CUDA + AutogradCUDA bits)
    
    OPERATIONS:
    ─────────────────────────────────────────────────────────────────
    
    // Union: combine key sets
    DispatchKeySet result = a | b;
    
    // Intersection
    DispatchKeySet result = a & b;
    
    // Check if key present
    bool has_key = key_set.has(DispatchKey::CUDA);
    
    // Get highest priority key
    DispatchKey highest = key_set.highestPriorityTypeId();
    
    // Remove a key (for redispatch)
    DispatchKeySet remaining = key_set.remove(DispatchKey::Autograd);
    
    HOW TENSOR GETS ITS KEY SET:
    ─────────────────────────────────────────────────────────────────
    
    When tensor is created:
    
    torch.randn(3, 3)  
    → key_set = {CPU}
    
    torch.randn(3, 3, device='cuda')
    → key_set = {CUDA}
    
    torch.randn(3, 3, device='cuda', requires_grad=True)
    → key_set = {CUDA, AutogradCUDA}
    
    torch.randn(3, 3).to_sparse()
    → key_set = {SparseCPU}
    """)
    
    # Live demonstration
    print("\n LIVE DEMONSTRATION:")
    print("-" * 50)
    
    # Check dispatch keys through internal APIs
    x_cpu = torch.randn(3, 3)
    print(f" CPU tensor:")
    print(f"   device = {x_cpu.device}")
    print(f"   requires_grad = {x_cpu.requires_grad}")
    
    if torch.cuda.is_available():
        x_cuda = torch.randn(3, 3, device='cuda')
        print(f"\n CUDA tensor:")
        print(f"   device = {x_cuda.device}")
        
        x_cuda_grad = torch.randn(3, 3, device='cuda', requires_grad=True)
        print(f"\n CUDA tensor with grad:")
        print(f"   device = {x_cuda_grad.device}")
        print(f"   requires_grad = {x_cuda_grad.requires_grad}")

# ============================================================================
# DISPATCHER OPERATION
# ============================================================================

def explain_dispatcher_operation():
    """Explain how dispatcher routes operations."""
    print("\n" + "="*70)
    print(" DISPATCHER OPERATION")
    print(" How operations get routed to implementations")
    print("="*70)
    
    print("""
    THE DISPATCH PROCESS:
    ─────────────────────────────────────────────────────────────────
    
    When you call torch.add(a, b):
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. COMPUTE DISPATCH KEY SET                                             │
    │                                                                         │
    │    key_set = a.key_set() | b.key_set() | thread_local_keys             │
    │                                                                         │
    │    Thread local keys include:                                           │
    │    - Autocast (if torch.autocast enabled)                              │
    │    - Grad mode (if torch.no_grad disabled)                             │
    │    - Inference mode                                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 2. FIND HIGHEST PRIORITY KEY WITH REGISTERED KERNEL                    │
    │                                                                         │
    │    for key in key_set (highest to lowest priority):                    │
    │        if op.hasKernelForKey(key):                                     │
    │            return op.kernelForKey(key)                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 3. CALL KERNEL                                                          │
    │                                                                         │
    │    kernel(a, b, remaining_key_set)                                     │
    │                                                                         │
    │    Kernel may:                                                         │
    │    - Execute directly (backend kernel)                                 │
    │    - Redispatch to next key (transform kernel)                        │
    └─────────────────────────────────────────────────────────────────────────┘
    
    KERNEL TYPES:
    ─────────────────────────────────────────────────────────────────
    
    1. BACKEND KERNEL (terminal)
       - Actually computes the result
       - CPU, CUDA, etc.
       
    2. WRAPPER KERNEL (transforms, redispatches)
       - Autograd: saves for backward, redispatches
       - Autocast: casts to lower precision, redispatches
       - vmap: adds batch dim, redispatches
       
    3. FALLBACK KERNEL
       - Used when no specific kernel registered
       - CompositeExplicitAutograd: decompose into other ops
       - CompositeImplicitAutograd: let autograd handle
    
    EXAMPLE: torch.mm(a, b) with autograd
    ─────────────────────────────────────────────────────────────────
    
    a = CUDA tensor, requires_grad=True
    b = CUDA tensor, requires_grad=True
    
    key_set = {CUDA, AutogradCUDA}
    
    Step 1: Dispatch to AutogradCUDA
    ┌─────────────────────────────────────────────────────────────────┐
    │ AutogradCUDA::mm kernel:                                        │
    │   1. Save a, b for backward                                    │
    │   2. Create MmBackward node                                    │
    │   3. Redispatch with key_set.remove(AutogradCUDA)              │
    │      → now just {CUDA}                                         │
    └─────────────────────────────────────────────────────────────────┘
    
    Step 2: Dispatch to CUDA
    ┌─────────────────────────────────────────────────────────────────┐
    │ CUDA::mm kernel:                                                │
    │   1. Call cuBLAS gemm                                          │
    │   2. Return result tensor                                      │
    └─────────────────────────────────────────────────────────────────┘
    
    Result has grad_fn = MmBackward
    """)

# ============================================================================
# TORCH DISPATCH MODE
# ============================================================================

def demonstrate_torch_dispatch():
    """Demonstrate __torch_dispatch__ for custom dispatch behavior."""
    print("\n" + "="*70)
    print(" __torch_dispatch__: CUSTOM DISPATCH BEHAVIOR")
    print(" Python-level hook into the dispatcher")
    print("="*70)
    
    print("""
    __torch_dispatch__ allows you to intercept ALL operations
    at the dispatch level. This is how torch.compile works!
    
    Use cases:
    - Tracing (FX, Dynamo)
    - Custom tensor types
    - Debugging/logging
    - Performance analysis
    """)
    
    # Simple logging mode
    class LoggingMode(TorchDispatchMode):
        def __init__(self):
            self.ops_called = []
        
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.ops_called.append(func.__name__)
            return func(*args, **kwargs)
    
    print("\n DEMO: Logging all operations")
    print("-" * 50)
    
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    
    with LoggingMode() as mode:
        z = x + y
        w = z.relu()
        result = w.sum()
    
    print(f" Operations called:")
    for op in mode.ops_called:
        print(f"   - {op}")
    
    # Counting mode
    class OpCounterMode(TorchDispatchMode):
        def __init__(self):
            self.op_counts: Dict[str, int] = {}
        
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.__name__
            self.op_counts[name] = self.op_counts.get(name, 0) + 1
            return func(*args, **kwargs)
    
    print("\n DEMO: Counting operations in a forward pass")
    print("-" * 50)
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    x = torch.randn(4, 10)
    
    with OpCounterMode() as counter:
        y = model(x)
    
    print(" Operation counts:")
    for op, count in sorted(counter.op_counts.items()):
        print(f"   {op}: {count}")

# ============================================================================
# OPERATOR REGISTRATION
# ============================================================================

def explain_operator_registration():
    """Explain how to register custom operators."""
    print("\n" + "="*70)
    print(" OPERATOR REGISTRATION")
    print(" How to add custom operations to PyTorch")
    print("="*70)
    
    print("""
    TWO WAYS TO REGISTER OPERATORS:
    ─────────────────────────────────────────────────────────────────
    
    1. PYTHON (torch.library)
       - Quick prototyping
       - Custom autograd
       - Limited performance
    
    2. C++ (TORCH_LIBRARY)
       - Production quality
       - Maximum performance
       - More complex
    
    PYTHON REGISTRATION (torch.library):
    ─────────────────────────────────────────────────────────────────
    """)
    
    # Python operator registration demo
    from torch.library import Library, impl
    
    # Create a library namespace
    my_lib = Library("myops", "DEF")
    
    # Define the operator schema
    my_lib.define("my_add(Tensor x, Tensor y) -> Tensor")
    
    # Implement for different backends
    @impl(my_lib, "my_add", "CPU")
    def my_add_cpu(x, y):
        print("  [CPU implementation called]")
        return x + y
    
    if torch.cuda.is_available():
        @impl(my_lib, "my_add", "CUDA")
        def my_add_cuda(x, y):
            print("  [CUDA implementation called]")
            return x + y
    
    # Implement autograd
    @impl(my_lib, "my_add", "Autograd")
    def my_add_autograd(x, y):
        print("  [Autograd wrapper called]")
        # Custom backward not shown for brevity
        return x + y
    
    print("\n DEMO: Custom operator dispatch")
    print("-" * 50)
    
    x_cpu = torch.randn(3, 3)
    y_cpu = torch.randn(3, 3)
    
    print(" Calling myops.my_add on CPU tensors:")
    result = torch.ops.myops.my_add(x_cpu, y_cpu)
    print(f" Result shape: {result.shape}")
    
    if torch.cuda.is_available():
        x_cuda = x_cpu.cuda()
        y_cuda = y_cpu.cuda()
        
        print("\n Calling myops.my_add on CUDA tensors:")
        result = torch.ops.myops.my_add(x_cuda, y_cuda)
        print(f" Result shape: {result.shape}")
    
    print("""
    
    C++ REGISTRATION (TORCH_LIBRARY):
    ─────────────────────────────────────────────────────────────────
    
    // In your_ops.cpp
    
    #include <torch/library.h>
    
    Tensor my_add(const Tensor& x, const Tensor& y) {
        return x + y;
    }
    
    // Register the library and operators
    TORCH_LIBRARY(myops, m) {
        m.def("my_add(Tensor x, Tensor y) -> Tensor");
    }
    
    // Register implementations per backend
    TORCH_LIBRARY_IMPL(myops, CPU, m) {
        m.impl("my_add", my_add_cpu);
    }
    
    TORCH_LIBRARY_IMPL(myops, CUDA, m) {
        m.impl("my_add", my_add_cuda);
    }
    
    TORCH_LIBRARY_IMPL(myops, Autograd, m) {
        m.impl("my_add", my_add_autograd);
    }
    
    SCHEMA LANGUAGE:
    ─────────────────────────────────────────────────────────────────
    
    "op_name(Tensor x, Tensor y, *, Scalar alpha=1) -> Tensor"
           │         │       │      │           │      └── Return type
           │         │       │      │           └── Default value
           │         │       │      └── Keyword-only args after *
           │         └───────┴── Positional args
           └── Operator name
    
    Special annotations:
    - Tensor(a!) out     # Mutates tensor, aliased to 'a'
    - Tensor?            # Optional tensor
    - Tensor[]           # List of tensors
    - Scalar             # Python scalar (int/float)
    - int[]              # List of ints
    - Device?            # Optional device
    """)

# ============================================================================
# FALLBACK AND DECOMPOSITIONS
# ============================================================================

def explain_fallbacks():
    """Explain fallback kernels and decompositions."""
    print("\n" + "="*70)
    print(" FALLBACKS AND DECOMPOSITIONS")
    print(" What happens when there's no specific kernel")
    print("="*70)
    
    print("""
    FALLBACK TYPES:
    ─────────────────────────────────────────────────────────────────
    
    1. CompositeExplicitAutograd
       - Decompose into simpler ops
       - Autograd sees the decomposition
       - Example: layer_norm → mean, var, normalize
    
    2. CompositeImplicitAutograd  
       - Decompose into simpler ops
       - Autograd uses formula for original op
       - Better gradients, more memory
    
    3. Backend Fallback
       - Fall back to different backend
       - Example: MPS falls back to CPU for unsupported ops
    
    4. Autogenerated
       - Generated from native_functions.yaml
       - Handles structured kernels
    
    DECOMPOSITION EXAMPLE:
    ─────────────────────────────────────────────────────────────────
    
    softmax(x, dim) decomposes to:
    
    def softmax_decomp(x, dim):
        x_max = x.max(dim, keepdim=True).values
        x_shifted = x - x_max  # Numerical stability
        exp_x = x_shifted.exp()
        return exp_x / exp_x.sum(dim, keepdim=True)
    
    WHY DECOMPOSITIONS MATTER FOR TORCH.COMPILE:
    ─────────────────────────────────────────────────────────────────
    
    torch.compile decomposes high-level ops into primitives:
    
    Original: layer_norm(x, normalized_shape, weight, bias)
    
    Decomposed:
    1. mean = x.mean(dim=-1, keepdim=True)
    2. var = x.var(dim=-1, keepdim=True)  
    3. x_norm = (x - mean) / sqrt(var + eps)
    4. out = x_norm * weight + bias
    
    Benefits:
    - Can fuse steps 1-4 into one kernel
    - Don't need to implement every op for every backend
    - Triton can generate efficient fused kernel
    
    PRIMS (Primitive Operations):
    ─────────────────────────────────────────────────────────────────
    
    PyTorch has a set of ~250 primitive operations (torch._prims).
    All higher-level ops can be expressed using prims.
    
    Used by:
    - torch.compile/Inductor
    - nvFuser
    - Custom backends
    
    Example prims:
    - prim.add
    - prim.mul
    - prim.exp
    - prim.reduce_sum
    - prim.broadcast_in_dim
    """)

# ============================================================================
# DISPATCHER FOR TORCH.COMPILE
# ============================================================================

def explain_compile_dispatch():
    """Explain how torch.compile uses the dispatcher."""
    print("\n" + "="*70)
    print(" DISPATCHER AND TORCH.COMPILE")
    print(" How compilation integrates with dispatch")
    print("="*70)
    
    print("""
    TORCH.COMPILE DISPATCH FLOW:
    ─────────────────────────────────────────────────────────────────
    
    When you call compiled_fn(x):
    
    1. TorchDynamo intercepts Python bytecode
    2. Traces operations through FX
    3. For each op, uses:
       - Functionalize: remove mutations
       - Decompose: break into prims
       - Meta dispatch: compute shapes
    
    FUNCTIONALIZE KEY:
    ─────────────────────────────────────────────────────────────────
    
    Problem: x.add_(y) mutates x
    
    torch.compile needs functional ops for optimization.
    
    Functionalize transforms:
    - x.add_(y) → x = x.add(y)
    
    This is done via the Functionalize dispatch key!
    
    META DISPATCH:
    ─────────────────────────────────────────────────────────────────
    
    Meta tensors have no data, just shape/dtype.
    
    Used for:
    - Shape inference at compile time
    - Memory planning
    - Graph optimization
    
    Example:
    x = torch.randn(3, 4, device='meta')  # No memory allocated!
    y = x.mm(torch.randn(4, 5, device='meta'))
    print(y.shape)  # (3, 5) - computed without actual data
    
    DISPATCH KEYS FOR COMPILATION:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ FuncTorchDynamicLayerFront  │ Dynamic dispatch layer          │
    │ Functionalize               │ Remove mutations                 │
    │ ProxyTorchDispatchMode      │ Tracing for torch.compile       │
    │ FakeTensor                  │ Meta tensors with device info   │
    │ Meta                        │ Shape-only computation          │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Demo meta tensors
    print("\n DEMO: Meta tensors for shape inference")
    print("-" * 50)
    
    # Create meta tensors
    x_meta = torch.randn(3, 4, device='meta')
    y_meta = torch.randn(4, 5, device='meta')
    
    print(f" x_meta.shape = {x_meta.shape}")
    print(f" x_meta.device = {x_meta.device}")
    print(f" x_meta.numel() = {x_meta.numel()}")
    
    # Compute shapes without data
    z_meta = x_meta.mm(y_meta)
    print(f"\n z = x.mm(y)")
    print(f" z_meta.shape = {z_meta.shape}")
    print(f" (Computed without actual matrix multiply!)")

# ============================================================================
# SUMMARY
# ============================================================================

def print_dispatcher_summary():
    """Print dispatcher summary."""
    print("\n" + "="*70)
    print(" DISPATCHER SUMMARY")
    print("="*70)
    
    print("""
    KEY CONCEPTS:
    
    1. DISPATCH KEY = Functionality identifier
       - Backends: CPU, CUDA, MPS, XLA
       - Transforms: Autograd, Autocast, vmap
       - Utilities: Meta, Python, Tracing
    
    2. DISPATCH KEY SET = Bitmask of active keys
       - Computed from tensor + thread-local state
       - Union of all input tensor key sets
    
    3. DISPATCHER = Routes ops to kernels
       - Finds highest priority key with kernel
       - Calls kernel, which may redispatch
    
    4. KERNEL REGISTRATION
       - Define schema: "op(Tensor x) -> Tensor"
       - Implement per backend
       - TORCH_LIBRARY in C++, torch.library in Python
    
    5. FALLBACKS & DECOMPOSITIONS
       - Not every op needs every backend kernel
       - Decompose into simpler ops
       - Prims = minimal primitive set
    
    HOW THIS HELPS CONTRIBUTION:
    ─────────────────────────────────────────────────────────────────
    
    Adding new operator:
    1. Add to native_functions.yaml
    2. Implement kernel in aten/src/ATen/native/
    3. Add derivative in derivatives.yaml
    
    Adding new backend:
    1. Create dispatch keys in DispatchKey.h
    2. Register fallback kernel
    3. Implement key operations
    
    Understanding errors:
    "No kernel found for dispatch key XXX"
    → Need to implement kernel for that key
    
    NEXT: Study 03_autograd_engine.py for autograd deep dive
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH DISPATCHER DEEP DIVE ".center(68) + "║")
    print("║" + " Understanding the routing system ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    
    explain_dispatch_keys()
    explain_dispatch_key_set()
    explain_dispatcher_operation()
    demonstrate_torch_dispatch()
    explain_operator_registration()
    explain_fallbacks()
    explain_compile_dispatch()
    print_dispatcher_summary()
    
    print("\n" + "="*70)
    print(" The dispatcher is the heart of PyTorch's extensibility!")
    print("="*70)
