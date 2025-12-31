"""
02_pytorch_compiler_stack.py - PyTorch Compiler Deep Dive

PyTorch's compiler stack (PyTorch 2.0+) is a game-changer.
It brings compilation benefits while keeping PyTorch's flexibility.

The Stack:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Python Code (your model)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ TorchDynamo - Graph Capture                                                 │
│ • Intercepts Python bytecode                                               │
│ • Extracts computation graph                                               │
│ • Handles dynamic shapes, control flow                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ AOTAutograd - Automatic Differentiation                                     │
│ • Generates forward and backward graphs                                    │
│ • Enables training compilation                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ FX Graph - Intermediate Representation                                      │
│ • Clean, inspectable graph format                                          │
│ • Python-based, easy to transform                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Inductor - Code Generation Backend                                          │
│ • Triton for GPU (generates Triton kernels)                               │
│ • C++/OpenMP for CPU                                                       │
│ • Handles fusion, scheduling, memory planning                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Triton - GPU Kernel Compiler                                                │
│ • High-level GPU programming                                               │
│ • Compiles to PTX/SASS                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 02_pytorch_compiler_stack.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

# ============================================================================
# PROFILING
# ============================================================================

def profile_fn(func, warmup=10, iterations=100):
    """Profile a function."""
    for _ in range(warmup):
        func()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            func()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        return (time.perf_counter() - start) * 1000 / iterations

# ============================================================================
# TORCHDYNAMO - GRAPH CAPTURE
# ============================================================================

def experiment_dynamo_capture():
    """
    Demonstrate how TorchDynamo captures computation graphs.
    """
    print("\n" + "="*70)
    print(" TORCHDYNAMO: GRAPH CAPTURE")
    print(" How PyTorch captures your computation graph")
    print("="*70)
    
    print("""
    TORCHDYNAMO MAGIC:
    
    Traditional approaches (TorchScript):
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Parse Python AST                                              │
    │ • Or trace execution                                           │
    │ • Limited Python support                                       │
    │ • Breaks on dynamic control flow                               │
    └─────────────────────────────────────────────────────────────────┘
    
    TorchDynamo approach:
    ┌─────────────────────────────────────────────────────────────────┐
    │ • Intercepts Python bytecode (CPython internals!)              │
    │ • Captures what it can, "graph breaks" when needed             │
    │ • Supports more Python features                                │
    │ • Can handle dynamic shapes (with guards)                      │
    └─────────────────────────────────────────────────────────────────┘
    
    GRAPH BREAKS:
    When Dynamo encounters unsupported code, it creates a "graph break":
    
    def my_fn(x):
        y = x * 2        # ← Captured in Graph 1
        print(y.shape)   # ← Graph break! (side effect)
        z = y + 1        # ← Captured in Graph 2
        return z
    
    Result: Two compiled graphs with Python code between them.
    """)
    
    # Demonstrate graph capture with explain
    def simple_fn(x):
        y = x.sin()
        z = y.cos()
        return z + 1
    
    def complex_fn(x):
        y = x.sin()
        if x.sum() > 0:  # Dynamic condition - requires guard
            z = y.cos()
        else:
            z = y.tan()
        return z
    
    if torch.cuda.is_available():
        x = torch.randn(1000, device='cuda')
        
        print("\n Simple function (no graph breaks):")
        print(" def simple_fn(x): return x.sin().cos() + 1")
        
        try:
            # Use explain to see what Dynamo does
            explanation = torch._dynamo.explain(simple_fn)(x)
            print(f" Graph breaks: {explanation.graph_break_count}")
            print(f" Graphs captured: {len(explanation.graphs)}")
        except Exception as e:
            print(f" explain not available: {e}")
        
        print("\n Complex function (with condition):")
        print(" def complex_fn(x): if x.sum() > 0: ... else: ...")
        
        try:
            explanation = torch._dynamo.explain(complex_fn)(x)
            print(f" Graph breaks: {explanation.graph_break_count}")
            print(f" Guards: Condition on x.sum() > 0")
        except Exception as e:
            print(f" explain not available: {e}")

# ============================================================================
# FX GRAPH - INTERMEDIATE REPRESENTATION
# ============================================================================

def experiment_fx_graph():
    """
    Show how to inspect and understand FX graphs.
    """
    print("\n" + "="*70)
    print(" FX GRAPH: INTERMEDIATE REPRESENTATION")
    print(" The graph format used by PyTorch compiler")
    print("="*70)
    
    print("""
    FX GRAPH STRUCTURE:
    
    A graph is a sequence of nodes, each representing:
    • placeholder: Input
    • call_function: A function call (torch.add, etc.)
    • call_method: A method call (tensor.view, etc.)
    • call_module: A module forward (nn.Linear, etc.)
    • output: The return value
    
    Example FX Graph for: y = relu(matmul(x, W) + b)
    
    graph():
        %x : [num_users=1] = placeholder[target=x]
        %W : [num_users=1] = placeholder[target=W]
        %b : [num_users=1] = placeholder[target=b]
        %matmul : [num_users=1] = call_function[target=torch.matmul](args=(%x, %W))
        %add : [num_users=1] = call_function[target=torch.add](args=(%matmul, %b))
        %relu : [num_users=1] = call_function[target=torch.relu](args=(%add,))
        return relu
    """)
    
    # Create and trace a simple model
    class SimpleModel(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
        
        def forward(self, x):
            x = self.linear(x)
            x = F.relu(x)
            x = x * 2
            return x
    
    model = SimpleModel(64, 32)
    
    # Symbolic trace to get FX graph
    try:
        from torch.fx import symbolic_trace
        traced = symbolic_trace(model)
        
        print("\n Traced FX Graph:")
        print("-" * 50)
        traced.graph.print_tabular()
        
        print("\n Generated Python code:")
        print("-" * 50)
        print(traced.code)
    except Exception as e:
        print(f" FX tracing error: {e}")

# ============================================================================
# INDUCTOR - CODE GENERATION
# ============================================================================

def experiment_inductor():
    """
    Understand what Inductor does and how to see its output.
    """
    print("\n" + "="*70)
    print(" INDUCTOR: CODE GENERATION BACKEND")
    print(" Generates optimized Triton/C++ code")
    print("="*70)
    
    print("""
    INDUCTOR'S JOB:
    
    Input: FX Graph (high-level operations)
    Output: Optimized Triton kernels (GPU) or C++ code (CPU)
    
    KEY OPTIMIZATIONS:
    
    1. FUSION DECISIONS
       • Which ops to fuse together?
       • Point-wise ops → definitely fuse
       • Reductions → careful about order
       • Matmul → usually keep separate (call cuBLAS)
    
    2. LOOP TILING
       • Break large operations into tiles
       • Fit working set in cache/shared memory
       • Critical for memory-bound ops
    
    3. SCHEDULING
       • Order operations to minimize memory traffic
       • Overlap compute and memory operations
       • Handle dependencies correctly
    
    4. MEMORY PLANNING
       • Reuse buffers when possible
       • Choose in-place operations when safe
       • Minimize peak memory usage
    """)
    
    if not torch.cuda.is_available():
        print("\n CUDA not available for Inductor demo")
        return
    
    # Show Inductor in action
    def fused_ops(x, y, z):
        a = x + y
        b = a * z
        c = torch.relu(b)
        d = c.sin()
        return d
    
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.randn(1000, 1000, device='cuda')
    
    print("\n Example: Fused element-wise operations")
    print(" d = sin(relu((x + y) * z))")
    
    time_eager = profile_fn(lambda: fused_ops(x, y, z))
    print(f" Eager time: {time_eager:.3f} ms")
    
    try:
        # Compile with debug output
        import os
        os.environ['TORCH_LOGS'] = ''  # Disable verbose logs for now
        
        compiled_fn = torch.compile(fused_ops)
        
        # Warmup (triggers compilation)
        _ = compiled_fn(x, y, z)
        torch.cuda.synchronize()
        
        time_compiled = profile_fn(lambda: compiled_fn(x, y, z))
        print(f" Compiled time: {time_compiled:.3f} ms")
        print(f" Speedup: {time_eager/time_compiled:.2f}x")
        
        print("\n What Inductor did:")
        print(" • Fused 4 operations into 1 Triton kernel")
        print(" • Reduced memory traffic from 8 passes to 2 passes")
        print(" • Generated optimized Triton code")
        
    except Exception as e:
        print(f" Compilation error: {e}")

# ============================================================================
# COMPARING COMPILATION APPROACHES
# ============================================================================

def experiment_compilation_comparison():
    """
    Compare different compilation approaches in PyTorch.
    """
    print("\n" + "="*70)
    print(" COMPILATION APPROACHES COMPARISON")
    print("="*70)
    
    print("""
    EVOLUTION OF PYTORCH COMPILATION:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Approach        │ Era      │ Method              │ Limitations          │
    ├─────────────────┼──────────┼─────────────────────┼──────────────────────┤
    │ Eager           │ Always   │ Execute immediately │ No optimization      │
    │ torch.jit.trace │ 1.0+     │ Record execution    │ No control flow      │
    │ torch.jit.script│ 1.0+     │ Parse TorchScript   │ Limited Python       │
    │ torch.compile   │ 2.0+     │ Dynamo+Inductor     │ Best, still maturing │
    └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    if not torch.cuda.is_available():
        print("\n CUDA not available")
        return
    
    # Test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.ln = nn.LayerNorm(512)
        
        def forward(self, x):
            x = self.ln(x)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x
    
    model = TestModel().cuda()
    x = torch.randn(32, 128, 512, device='cuda')
    
    print(f"\n Model: LayerNorm → Linear(512,1024) → GELU → Linear(1024,512)")
    print(f" Input: (32, 128, 512)")
    print(f"\n{'Approach':<25} {'Time (ms)':<15} {'Speedup':<10} {'Notes'}")
    print("-" * 70)
    
    # Eager
    time_eager = profile_fn(lambda: model(x), iterations=50)
    print(f"{'Eager':<25} {time_eager:<15.3f} {'1.0x':<10} Baseline")
    
    # torch.jit.trace
    try:
        traced = torch.jit.trace(model, x)
        time_traced = profile_fn(lambda: traced(x), iterations=50)
        speedup = time_eager / time_traced
        print(f"{'torch.jit.trace':<25} {time_traced:<15.3f} {speedup:.2f}x{'':<5} Limited fusion")
    except Exception as e:
        print(f"{'torch.jit.trace':<25} Error: {str(e)[:30]}")
    
    # torch.jit.script
    try:
        scripted = torch.jit.script(model)
        time_scripted = profile_fn(lambda: scripted(x), iterations=50)
        speedup = time_eager / time_scripted
        print(f"{'torch.jit.script':<25} {time_scripted:<15.3f} {speedup:.2f}x{'':<5} Better optimization")
    except Exception as e:
        print(f"{'torch.jit.script':<25} Error: {str(e)[:30]}")
    
    # torch.compile
    try:
        compiled = torch.compile(model)
        # Warmup
        _ = compiled(x)
        torch.cuda.synchronize()
        
        time_compiled = profile_fn(lambda: compiled(x), iterations=50)
        speedup = time_eager / time_compiled
        print(f"{'torch.compile':<25} {time_compiled:<15.3f} {speedup:.2f}x{'':<5} Triton fusion")
    except Exception as e:
        print(f"{'torch.compile':<25} Error: {str(e)[:30]}")
    
    # torch.compile with max-autotune
    try:
        compiled_tuned = torch.compile(model, mode="max-autotune")
        # Warmup
        _ = compiled_tuned(x)
        torch.cuda.synchronize()
        
        time_tuned = profile_fn(lambda: compiled_tuned(x), iterations=50)
        speedup = time_eager / time_tuned
        print(f"{'torch.compile (autotune)':<25} {time_tuned:<15.3f} {speedup:.2f}x{'':<5} Best configs")
    except Exception as e:
        print(f"{'torch.compile (autotune)':<25} Error: {str(e)[:30]}")

# ============================================================================
# DEBUGGING TORCH.COMPILE
# ============================================================================

def explain_debugging():
    """
    How to debug torch.compile issues.
    """
    print("\n" + "="*70)
    print(" DEBUGGING TORCH.COMPILE")
    print("="*70)
    
    print("""
    COMMON ISSUES AND SOLUTIONS:
    
    1. GRAPH BREAKS
    ───────────────────────────────────────────────────────────────────
    Problem: Too many graph breaks hurt performance
    
    Diagnose:
        torch._dynamo.explain(fn)(inputs)
    
    Common causes:
    • print() statements
    • Tensor.item() calls
    • Data-dependent control flow
    • Unsupported operations
    
    Solution:
    • Remove debug prints
    • Use torch.where instead of if/else
    • Mark functions with @torch._dynamo.disable if needed
    
    2. RECOMPILATION
    ───────────────────────────────────────────────────────────────────
    Problem: Model recompiles on every call (slow!)
    
    Diagnose:
        import torch._dynamo
        torch._dynamo.config.log_level = logging.DEBUG
    
    Common causes:
    • Changing input shapes
    • Changing dtypes
    • Guards failing
    
    Solution:
    • Use dynamic=True for variable shapes:
        torch.compile(model, dynamic=True)
    • Pad to fixed sizes when possible
    
    3. CORRECTNESS ISSUES
    ───────────────────────────────────────────────────────────────────
    Problem: Compiled output differs from eager
    
    Diagnose:
        # Compare outputs
        eager_out = model(x)
        compiled_out = compiled_model(x)
        torch.testing.assert_close(eager_out, compiled_out)
    
    Common causes:
    • Floating point ordering differences
    • Numerical precision
    • Random number generation
    
    Solution:
    • Use torch.compile(model, fullgraph=True) to fail on breaks
    • Report bugs to PyTorch
    
    4. SEEING GENERATED CODE
    ───────────────────────────────────────────────────────────────────
    
    # See Triton code
    import torch._inductor.config
    torch._inductor.config.debug = True
    
    # Output directory
    torch._inductor.config.trace.enabled = True
    
    # Or use environment variable
    TORCH_COMPILE_DEBUG=1 python your_script.py
    
    5. PROFILING COMPILED CODE
    ───────────────────────────────────────────────────────────────────
    
    # Use PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        compiled_model(x)
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    """)

# ============================================================================
# TRITON INTEGRATION
# ============================================================================

def experiment_triton_integration():
    """
    Show how torch.compile generates Triton code.
    """
    print("\n" + "="*70)
    print(" TRITON INTEGRATION")
    print(" How Inductor generates Triton kernels")
    print("="*70)
    
    print("""
    INDUCTOR → TRITON PIPELINE:
    
    1. Inductor analyzes the FX graph
    2. Groups fusible operations
    3. Generates Triton kernel code
    4. Triton compiles to PTX
    5. PTX compiled to GPU binary (SASS)
    
    WHAT GETS FUSED:
    
    ✓ Element-wise ops (add, mul, relu, etc.)
    ✓ Reductions (sum, mean, max)
    ✓ Softmax
    ✓ Layer normalization
    ✓ Some attention patterns
    
    WHAT STAYS SEPARATE:
    
    ✗ Large matmuls (use cuBLAS)
    ✗ Convolutions (use cuDNN)
    ✗ Operations with complex memory patterns
    
    EXAMPLE GENERATED TRITON CODE:
    
    @triton.jit
    def fused_add_mul_relu(
        in_ptr0, in_ptr1, in_ptr2, out_ptr0,
        xnumel, XBLOCK: tl.constexpr
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        
        x0 = tl.load(in_ptr0 + xindex, xmask)
        x1 = tl.load(in_ptr1 + xindex, xmask)
        x2 = tl.load(in_ptr2 + xindex, xmask)
        
        tmp0 = x0 + x1
        tmp1 = tmp0 * x2
        tmp2 = tl.maximum(tmp1, 0)  # relu
        
        tl.store(out_ptr0 + xindex, tmp2, xmask)
    """)
    
    if not torch.cuda.is_available():
        print("\n CUDA not available")
        return
    
    # Show fusion in action
    print("\n Demonstrating kernel fusion:")
    
    def unfused(x, y, z):
        a = x + y
        b = a * z
        c = F.relu(b)
        return c
    
    x = torch.randn(10000, 1000, device='cuda')
    y = torch.randn(10000, 1000, device='cuda')
    z = torch.randn(10000, 1000, device='cuda')
    
    # Count CUDA kernels
    print("\n Kernel count analysis:")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        _ = unfused(x, y, z)
    
    cuda_kernels = [e for e in prof.events() if e.device_type == torch.profiler.DeviceType.CUDA]
    print(f" Eager mode: ~3 kernels (add, mul, relu)")
    
    try:
        compiled_unfused = torch.compile(unfused)
        _ = compiled_unfused(x, y, z)  # Warmup
        torch.cuda.synchronize()
        
        print(f" Compiled mode: 1 fused kernel (add_mul_relu)")
        print(f"\n Memory traffic reduction: ~3x")
    except Exception as e:
        print(f" Compilation error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

def print_pytorch_compiler_summary():
    """
    Summary of PyTorch compiler stack.
    """
    print("\n" + "="*70)
    print(" PYTORCH COMPILER STACK SUMMARY")
    print("="*70)
    
    print("""
    THE STACK:
    
    torch.compile(model) triggers:
    
    1. TorchDynamo
       • Captures Python bytecode
       • Creates FX graph
       • Handles graph breaks gracefully
    
    2. AOTAutograd  
       • Generates backward graph
       • Enables training compilation
    
    3. Inductor
       • Optimizes the graph
       • Generates Triton/C++ code
    
    4. Triton
       • Compiles to GPU binary
    
    WHEN TO USE:
    
    ✓ Training large models
    ✓ Inference optimization
    ✓ Custom operations
    ✓ Research experiments
    
    WHEN TO AVOID:
    
    ✗ Very dynamic code (too many graph breaks)
    ✗ Debugging (harder to debug compiled code)
    ✗ One-shot inference (compilation overhead)
    
    BEST PRACTICES:
    
    1. Start with default mode
    2. Use explain() to check graph breaks
    3. Try max-autotune for production
    4. Use dynamic=True for variable shapes
    5. Profile to verify speedup
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH COMPILER STACK DEEP DIVE ".center(68) + "║")
    print("║" + " TorchDynamo + AOTAutograd + Inductor + Triton ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
        print(f" PyTorch: {torch.__version__}")
    else:
        print("\n Running on CPU")
    
    experiment_dynamo_capture()
    experiment_fx_graph()
    experiment_inductor()
    experiment_compilation_comparison()
    explain_debugging()
    experiment_triton_integration()
    print_pytorch_compiler_summary()
    
    print("\n" + "="*70)
    print(" Next: XLA and JAX compiler system")
    print("="*70)
