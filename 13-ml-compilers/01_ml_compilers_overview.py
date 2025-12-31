"""
01_ml_compilers_overview.py - Understanding ML Compilers: The Complete Picture

ML Compilers bridge the gap between high-level ML frameworks and hardware.
They are ESSENTIAL for performance - without them, your models would be 10-100x slower!

What ML Compilers Do:
1. Graph Optimization - Fuse operations, eliminate redundancy
2. Memory Planning - Minimize allocations, reuse buffers
3. Code Generation - Produce optimized kernels for target hardware
4. Hardware Abstraction - Same code runs on GPU, TPU, CPU

The ML Compiler Landscape:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HIGH-LEVEL FRAMEWORKS                                 │
│   PyTorch    │    JAX    │   TensorFlow   │   Mojo   │   ONNX Runtime      │
├─────────────────────────────────────────────────────────────────────────────┤
│                        INTERMEDIATE REPRESENTATIONS                          │
│   TorchScript │   HLO    │   TF Graph     │   MLIR   │   ONNX              │
├─────────────────────────────────────────────────────────────────────────────┤
│                        COMPILER BACKENDS                                     │
│   Triton     │   XLA    │   TensorRT     │   TVM    │   IREE              │
├─────────────────────────────────────────────────────────────────────────────┤
│                        LOW-LEVEL CODE GENERATION                             │
│   LLVM/PTX   │   CUDA   │   ROCm         │   Metal  │   CPU (AVX/NEON)    │
├─────────────────────────────────────────────────────────────────────────────┤
│                        HARDWARE                                              │
│   NVIDIA GPU │   AMD GPU│   Google TPU   │   Apple  │   Intel/ARM CPU     │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 01_ml_compilers_overview.py
"""

import torch
import time
import sys
from typing import Callable, List, Tuple

# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def profile_function(func: Callable, warmup: int = 10, iterations: int = 100) -> float:
    """Profile a function with proper warmup."""
    # Warmup
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
# WHY ML COMPILERS MATTER
# ============================================================================

def experiment_why_compilers_matter():
    """
    Demonstrate why ML compilers are essential for performance.
    """
    print("\n" + "="*70)
    print(" WHY ML COMPILERS MATTER")
    print(" The difference between naive and optimized execution")
    print("="*70)
    
    print("""
    WITHOUT COMPILATION (Eager Mode):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Python Code                                                     │
    │     ↓                                                           │
    │ Framework (PyTorch/JAX)                                        │
    │     ↓                                                           │
    │ INDIVIDUAL kernel launches for EACH operation                  │
    │     ↓                                                           │
    │ x = a + b    → Launch kernel 1                                 │
    │ y = x * c    → Launch kernel 2 (read x back from memory!)      │
    │ z = relu(y)  → Launch kernel 3 (read y back from memory!)      │
    │                                                                │
    │ PROBLEM: Each kernel reads/writes to slow global memory!       │
    └─────────────────────────────────────────────────────────────────┘
    
    WITH COMPILATION (Compiled Mode):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Python Code                                                     │
    │     ↓                                                           │
    │ COMPILER analyzes entire computation graph                     │
    │     ↓                                                           │
    │ FUSES operations: z = relu((a + b) * c)                        │
    │     ↓                                                           │
    │ SINGLE optimized kernel                                        │
    │     ↓                                                           │
    │ Read a, b, c ONCE, compute all, write z ONCE                   │
    │                                                                │
    │ BENEFIT: Minimal memory traffic, maximum compute!              │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    if not torch.cuda.is_available():
        print(" CUDA not available for demonstration")
        return
    
    # Demonstrate with actual code
    size = 10_000_000
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    c = torch.randn(size, device='cuda')
    
    # Eager mode - separate operations
    def eager_compute():
        x = a + b
        y = x * c
        z = torch.relu(y)
        return z
    
    time_eager = profile_function(eager_compute)
    
    # Compiled mode
    try:
        compiled_compute = torch.compile(eager_compute)
        # Warmup compilation
        _ = compiled_compute()
        torch.cuda.synchronize()
        
        time_compiled = profile_function(compiled_compute)
        
        print(f"\n Performance comparison ({size/1e6:.0f}M elements):")
        print(f"{'Mode':<25} {'Time (ms)':<15} {'Speedup'}")
        print("-" * 55)
        print(f"{'Eager (no compilation)':<25} {time_eager:<15.3f} 1.0x")
        print(f"{'Compiled (torch.compile)':<25} {time_compiled:<15.3f} {time_eager/time_compiled:.2f}x")
    except Exception as e:
        print(f" torch.compile not available: {e}")
        print(f" Eager time: {time_eager:.3f} ms")
    
    print(f"\n KEY INSIGHT:")
    print(f" - Compiler fuses 3 operations into 1 kernel")
    print(f" - Memory traffic reduced from 6 passes to 2 passes")
    print(f" - This is why compilation matters for ML!")

# ============================================================================
# THE COMPILATION PIPELINE
# ============================================================================

def explain_compilation_pipeline():
    """
    Explain the stages of ML compilation.
    """
    print("\n" + "="*70)
    print(" THE ML COMPILATION PIPELINE")
    print(" From Python to optimized machine code")
    print("="*70)
    
    print("""
    STAGE 1: GRAPH CAPTURE
    ─────────────────────────────────────────────────────────────────
    Convert dynamic Python code into a static computation graph.
    
    PyTorch approaches:
    • torch.jit.trace() - Run code, record operations
    • torch.jit.script() - Parse Python AST
    • torch.compile() - Use Dynamo to capture graph dynamically
    
    Challenges:
    • Control flow (if/else, loops)
    • Dynamic shapes
    • Python side effects
    
    STAGE 2: GRAPH OPTIMIZATION (High-Level)
    ─────────────────────────────────────────────────────────────────
    Transform the graph for efficiency without changing semantics.
    
    Common optimizations:
    • Constant folding: Precompute constant expressions
    • Dead code elimination: Remove unused computations
    • Common subexpression elimination: Reuse repeated computations
    • Operator fusion: Combine multiple ops into one
    
    Example fusion:
    Before: matmul → bias_add → relu → dropout
    After:  fused_linear_relu_dropout (single kernel!)
    
    STAGE 3: LOWERING
    ─────────────────────────────────────────────────────────────────
    Convert high-level ops to lower-level primitives.
    
    • LayerNorm → mean, variance, normalize, scale, shift
    • Attention → matmul, softmax, matmul
    • But keep fused ops when beneficial!
    
    STAGE 4: MEMORY PLANNING
    ─────────────────────────────────────────────────────────────────
    Decide where tensors live and when to allocate/free.
    
    • Buffer reuse: Tensors with non-overlapping lifetimes share memory
    • In-place operations: Modify inputs when safe
    • Memory layout: Choose optimal tensor layout (NCHW vs NHWC)
    
    STAGE 5: CODE GENERATION
    ─────────────────────────────────────────────────────────────────
    Generate actual code for target hardware.
    
    Options:
    • Call pre-written kernels (cuDNN, cuBLAS)
    • Generate custom kernels (Triton, TVM)
    • Use templates (CUTLASS)
    
    STAGE 6: RUNTIME
    ─────────────────────────────────────────────────────────────────
    Execute the compiled code efficiently.
    
    • Memory allocation
    • Kernel scheduling
    • Multi-stream execution
    • Profiling and debugging
    """)

# ============================================================================
# COMPILER COMPARISON TABLE
# ============================================================================

def print_compiler_comparison():
    """
    Compare different ML compiler systems.
    """
    print("\n" + "="*70)
    print(" ML COMPILER COMPARISON")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    ML COMPILER SYSTEMS COMPARISON                            │
    ├───────────────┬──────────────┬──────────────┬──────────────┬───────────────┤
    │ System        │ Framework    │ IR           │ Backend      │ Best For      │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ torch.compile │ PyTorch      │ FX Graph     │ Triton/C++   │ Training      │
    │ + Inductor    │              │              │              │ Flexibility   │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ XLA           │ JAX, TF      │ HLO          │ LLVM, TPU    │ TPU, Research │
    │               │              │              │              │ Functional    │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ TensorRT      │ Any (ONNX)   │ TRT IR       │ CUDA         │ Inference     │
    │               │              │              │              │ NVIDIA HW     │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ TVM           │ Any (Relay)  │ Relay/TIR    │ LLVM, CUDA   │ Edge, Mobile  │
    │               │              │              │              │ Multi-HW      │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ MLIR          │ Framework    │ Multi-level  │ LLVM         │ Compiler Dev  │
    │               │              │ dialects     │              │ Research      │
    ├───────────────┼──────────────┼──────────────┼──────────────┼───────────────┤
    │ Mojo/MAX      │ Mojo         │ MLIR         │ LLVM         │ Unified       │
    │               │              │              │              │ High-perf     │
    └───────────────┴──────────────┴──────────────┴──────────────┴───────────────┘
    
    DETAILED PROS AND CONS:
    
    ═══════════════════════════════════════════════════════════════════════════════
    PYTORCH COMPILER STACK (torch.compile + Triton)
    ═══════════════════════════════════════════════════════════════════════════════
    
    PROS:
    ✓ Seamless integration with PyTorch ecosystem
    ✓ Supports dynamic shapes (with some overhead)
    ✓ Good for training (backward pass compilation)
    ✓ Active development, rapid improvements
    ✓ Triton backend is accessible for customization
    ✓ Works with existing PyTorch code (mostly)
    
    CONS:
    ✗ Compilation overhead on first run
    ✗ Not all Python features supported
    ✗ Debugging compiled code is harder
    ✗ NVIDIA-focused (AMD support improving)
    ✗ Still maturing (occasional bugs)
    
    BEST FOR: Training, research, rapid prototyping
    
    ═══════════════════════════════════════════════════════════════════════════════
    XLA (JAX / TensorFlow)
    ═══════════════════════════════════════════════════════════════════════════════
    
    PROS:
    ✓ Excellent optimization quality
    ✓ First-class TPU support
    ✓ Mature, battle-tested (Google production)
    ✓ Great for functional programming style
    ✓ Automatic differentiation is seamless
    ✓ Good for large-scale distributed training
    
    CONS:
    ✗ Requires functional programming style
    ✗ Less flexible than PyTorch
    ✗ Steeper learning curve
    ✗ Debugging can be challenging
    ✗ Shape must be known at compile time (jit)
    ✗ Smaller ecosystem than PyTorch
    
    BEST FOR: TPU training, large-scale production, research labs
    
    ═══════════════════════════════════════════════════════════════════════════════
    TENSORRT (NVIDIA)
    ═══════════════════════════════════════════════════════════════════════════════
    
    PROS:
    ✓ Best inference performance on NVIDIA GPUs
    ✓ Extensive kernel library (highly optimized)
    ✓ INT8/FP16 quantization support
    ✓ Production-ready, well-supported
    ✓ Integration with Triton Inference Server
    
    CONS:
    ✗ Inference only (no training)
    ✗ NVIDIA GPUs only
    ✗ Model conversion can be tricky
    ✗ Less flexible (fixed graph)
    ✗ Limited dynamic shape support
    ✗ Proprietary
    
    BEST FOR: Production inference on NVIDIA GPUs
    
    ═══════════════════════════════════════════════════════════════════════════════
    TVM (Apache)
    ═══════════════════════════════════════════════════════════════════════════════
    
    PROS:
    ✓ Targets many hardware backends
    ✓ AutoTVM/Ansor for automatic tuning
    ✓ Good for edge deployment
    ✓ Open source, community-driven
    ✓ Research-friendly
    
    CONS:
    ✗ Smaller community than major frameworks
    ✗ Integration can be complex
    ✗ May not match hand-tuned kernels
    ✗ Tuning can be time-consuming
    ✗ Less mature than TensorRT for NVIDIA
    
    BEST FOR: Edge devices, multi-hardware deployment
    
    ═══════════════════════════════════════════════════════════════════════════════
    MOJO / MAX (Modular)
    ═══════════════════════════════════════════════════════════════════════════════
    
    PROS:
    ✓ Python-like syntax with C performance
    ✓ No two-language problem
    ✓ MLIR-based (cutting edge)
    ✓ Designed for AI from ground up
    ✓ Promising performance claims
    ✓ Memory safety without GC
    
    CONS:
    ✗ Very new (ecosystem still developing)
    ✗ Limited library support
    ✗ Not fully open source
    ✗ Learning new language
    ✗ Unproven at scale
    ✗ Small community (so far)
    
    BEST FOR: Future-looking projects, performance-critical code
    """)

# ============================================================================
# KEY OPTIMIZATIONS EXPLAINED
# ============================================================================

def explain_key_optimizations():
    """
    Explain the key optimizations that compilers perform.
    """
    print("\n" + "="*70)
    print(" KEY COMPILER OPTIMIZATIONS")
    print("="*70)
    
    print("""
    ═══════════════════════════════════════════════════════════════════════════════
    1. OPERATOR FUSION
    ═══════════════════════════════════════════════════════════════════════════════
    
    Combine multiple operations into a single kernel.
    
    BEFORE (3 kernels, 3 memory round-trips):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Kernel 1: y = matmul(x, W)    │ Read x, W → Write y            │
    │ Kernel 2: z = y + bias        │ Read y, bias → Write z         │
    │ Kernel 3: out = relu(z)       │ Read z → Write out             │
    └─────────────────────────────────────────────────────────────────┘
    Memory traffic: Read x, W; Write y; Read y, bias; Write z; Read z; Write out
    
    AFTER (1 fused kernel):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Fused Kernel: out = relu(matmul(x, W) + bias)                  │
    │ Read x, W, bias → Compute all → Write out                      │
    └─────────────────────────────────────────────────────────────────┘
    Memory traffic: Read x, W, bias; Write out (3x less!)
    
    ═══════════════════════════════════════════════════════════════════════════════
    2. MEMORY LAYOUT OPTIMIZATION
    ═══════════════════════════════════════════════════════════════════════════════
    
    Choose optimal tensor memory layout for each operation.
    
    Image tensors:
    • NCHW (PyTorch default): N×C×H×W - good for some ops
    • NHWC (cuDNN preferred): N×H×W×C - better for Tensor Cores!
    
    Compiler automatically:
    • Inserts layout transforms where needed
    • Minimizes total transform cost
    • Keeps data in optimal layout as long as possible
    
    ═══════════════════════════════════════════════════════════════════════════════
    3. KERNEL SELECTION
    ═══════════════════════════════════════════════════════════════════════════════
    
    Choose the best implementation for each operation.
    
    For matmul, options include:
    • cuBLAS (general, reliable)
    • cuDNN (for conv-like patterns)
    • CUTLASS (template-based, customizable)
    • Triton (JIT compiled)
    • Custom kernel (for specific shapes)
    
    Selection criteria:
    • Input shapes and dtypes
    • Hardware capabilities
    • Surrounding operations (fusion potential)
    
    ═══════════════════════════════════════════════════════════════════════════════
    4. CONSTANT FOLDING
    ═══════════════════════════════════════════════════════════════════════════════
    
    Precompute expressions involving constants at compile time.
    
    Before:
    scale = 1.0 / sqrt(64)  # Computed at runtime
    scores = matmul(Q, K) * scale
    
    After:
    scores = matmul(Q, K) * 0.125  # Constant precomputed
    
    ═══════════════════════════════════════════════════════════════════════════════
    5. BUFFER REUSE
    ═══════════════════════════════════════════════════════════════════════════════
    
    Reuse memory for tensors with non-overlapping lifetimes.
    
    y = f(x)      # x needed
    z = g(y)      # x no longer needed, y needed
    w = h(z)      # y no longer needed
    
    Memory allocation:
    • Buffer A: x, then reused for z
    • Buffer B: y, then reused for w
    
    Result: 2 buffers instead of 4!
    """)

# ============================================================================
# DEMO: TORCH.COMPILE IN ACTION
# ============================================================================

def demo_torch_compile():
    """
    Demonstrate torch.compile with different modes and backends.
    """
    print("\n" + "="*70)
    print(" DEMO: TORCH.COMPILE IN ACTION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return
    
    # A more complex function to compile
    def transformer_block_fn(x, qkv_weight, proj_weight, ff1_weight, ff2_weight):
        B, S, D = x.shape
        
        # Self-attention
        qkv = torch.matmul(x, qkv_weight)  # (B, S, 3*D)
        qkv = qkv.reshape(B, S, 3, 8, D // 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scale = (D // 8) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # Attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = torch.matmul(out, proj_weight)
        
        x = x + out
        
        # FFN
        ff = torch.matmul(x, ff1_weight)
        ff = torch.gelu(ff)
        ff = torch.matmul(ff, ff2_weight)
        
        return x + ff
    
    # Setup
    B, S, D = 4, 512, 512
    x = torch.randn(B, S, D, device='cuda')
    qkv_weight = torch.randn(D, 3*D, device='cuda')
    proj_weight = torch.randn(D, D, device='cuda')
    ff1_weight = torch.randn(D, 4*D, device='cuda')
    ff2_weight = torch.randn(4*D, D, device='cuda')
    
    def run_block():
        return transformer_block_fn(x, qkv_weight, proj_weight, ff1_weight, ff2_weight)
    
    # Eager baseline
    time_eager = profile_function(run_block, iterations=50)
    
    print(f"\n Transformer block: ({B}, {S}, {D})")
    print(f"{'Mode':<30} {'Time (ms)':<15} {'Speedup'}")
    print("-" * 60)
    print(f"{'Eager (no compile)':<30} {time_eager:<15.3f} 1.0x")
    
    # Try different compile modes
    modes = [
        ("default", {}),
        ("reduce-overhead", {"mode": "reduce-overhead"}),
        ("max-autotune", {"mode": "max-autotune"}),
    ]
    
    for mode_name, kwargs in modes:
        try:
            compiled_fn = torch.compile(run_block, **kwargs)
            
            # Warmup
            for _ in range(5):
                _ = compiled_fn()
            torch.cuda.synchronize()
            
            time_compiled = profile_function(compiled_fn, iterations=50)
            speedup = time_eager / time_compiled
            print(f"{'torch.compile (' + mode_name + ')':<30} {time_compiled:<15.3f} {speedup:.2f}x")
        except Exception as e:
            print(f"{'torch.compile (' + mode_name + ')':<30} Error: {str(e)[:30]}")
    
    print(f"\n COMPILE MODES EXPLAINED:")
    print(f" • default: Good balance of compile time and runtime")
    print(f" • reduce-overhead: Minimize kernel launch overhead")
    print(f" • max-autotune: Try more configs, longer compile, better runtime")

# ============================================================================
# SUMMARY
# ============================================================================

def print_summary():
    """
    Print summary of ML compilers.
    """
    print("\n" + "="*70)
    print(" ML COMPILERS SUMMARY")
    print("="*70)
    
    print("""
    KEY TAKEAWAYS:
    
    1. ML COMPILERS ARE ESSENTIAL
       • 2-10x speedup is common
       • Enable hardware-specific optimizations
       • Make ML workloads practical
    
    2. COMPILATION STAGES
       • Graph capture → Optimization → Lowering → Code generation
       • Each stage has specific optimizations
    
    3. CHOOSE THE RIGHT COMPILER
       • PyTorch training: torch.compile
       • JAX/TPU: XLA
       • NVIDIA inference: TensorRT
       • Multi-hardware: TVM
       • Cutting edge: Mojo/MAX
    
    4. UNDERSTAND THE TRADEOFFS
       • Compilation time vs runtime performance
       • Flexibility vs optimization
       • Ecosystem vs raw performance
    
    RECOMMENDATION FOR LEARNING:
    
    1. Start with torch.compile (easiest)
    2. Understand what fusion means
    3. Learn to read compiler output
    4. Explore XLA if using JAX
    5. Try TensorRT for production inference
    6. Watch Mojo for the future
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " ML COMPILERS: THE COMPLETE PICTURE ".center(68) + "║")
    print("║" + " From high-level code to optimized execution ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if torch.cuda.is_available():
        print(f"\n GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n Running on CPU (limited demos)")
    
    experiment_why_compilers_matter()
    explain_compilation_pipeline()
    print_compiler_comparison()
    explain_key_optimizations()
    demo_torch_compile()
    print_summary()
    
    print("\n" + "="*70)
    print(" Next: Deep dive into each compiler system")
    print("="*70)
