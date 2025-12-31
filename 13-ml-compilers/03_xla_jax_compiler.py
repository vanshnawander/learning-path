"""
03_xla_jax_compiler.py - XLA and JAX Compiler System

XLA (Accelerated Linear Algebra) is Google's ML compiler.
JAX is built on top of XLA and provides a NumPy-like interface.

XLA Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│ JAX / TensorFlow Code                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ HLO (High Level Operations) - XLA's IR                                      │
│ • Platform-independent representation                                       │
│ • ~100 operations (matmul, conv, reduce, etc.)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ XLA Optimizations                                                           │
│ • Fusion, layout optimization, algebraic simplification                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Backend Code Generation                                                     │
│ • GPU: LLVM → PTX → SASS                                                   │
│ • TPU: Custom backend                                                      │
│ • CPU: LLVM → x86/ARM                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

JAX Philosophy:
• Functional programming (pure functions, no side effects)
• Transformations: jit, grad, vmap, pmap
• NumPy API with automatic differentiation

Run: python 03_xla_jax_compiler.py
Note: Requires JAX installation: pip install jax jaxlib
"""

import time
import sys

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap, pmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed. Install with: pip install jax jaxlib")
    print("For GPU: pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

import numpy as np

# Also import PyTorch for comparison
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================================
# PROFILING
# ============================================================================

def profile_fn(func, warmup=10, iterations=100):
    """Profile a function."""
    for _ in range(warmup):
        result = func()
        if JAX_AVAILABLE and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = func()
        if JAX_AVAILABLE and hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    
    return (time.perf_counter() - start) * 1000 / iterations

# ============================================================================
# XLA OVERVIEW
# ============================================================================

def explain_xla():
    """
    Explain XLA's architecture and optimizations.
    """
    print("\n" + "="*70)
    print(" XLA: ACCELERATED LINEAR ALGEBRA")
    print(" Google's ML compiler used by JAX and TensorFlow")
    print("="*70)
    
    print("""
    WHAT IS XLA?
    
    XLA is a domain-specific compiler for linear algebra that can:
    • Fuse operations to reduce memory bandwidth
    • Optimize memory layout for target hardware
    • Generate efficient code for GPUs, TPUs, and CPUs
    
    HLO (High Level Operations):
    ─────────────────────────────────────────────────────────────────
    XLA's intermediate representation (IR).
    
    Key HLO operations:
    • Elementwise: add, multiply, exp, tanh, etc.
    • Reductions: reduce_sum, reduce_max, etc.
    • Matrix ops: dot (matmul), conv
    • Data movement: transpose, reshape, slice
    • Control flow: conditional, while_loop
    
    Example HLO for y = relu(matmul(x, W) + b):
    
    HloModule relu_matmul
    ENTRY main {
      x = f32[32,64] parameter(0)
      W = f32[64,128] parameter(1)
      b = f32[128] parameter(2)
      
      dot = f32[32,128] dot(x, W)
      broadcast = f32[32,128] broadcast(b)
      add = f32[32,128] add(dot, broadcast)
      zero = f32[] constant(0)
      zero_broadcast = f32[32,128] broadcast(zero)
      ROOT relu = f32[32,128] maximum(add, zero_broadcast)
    }
    
    XLA OPTIMIZATIONS:
    ─────────────────────────────────────────────────────────────────
    
    1. OPERATION FUSION
       • Fuse element-wise operations
       • Fuse reductions with producers
       • Fuse transpose with consumers
    
    2. LAYOUT OPTIMIZATION
       • Choose optimal memory layout
       • Minimize transpose operations
       • Hardware-specific layouts (NHWC vs NCHW)
    
    3. ALGEBRAIC SIMPLIFICATION
       • x * 1 → x
       • x + 0 → x
       • Constant folding
    
    4. MEMORY OPTIMIZATION
       • Buffer reuse
       • In-place operations
       • Memory coalescing
    """)

# ============================================================================
# JAX FUNDAMENTALS
# ============================================================================

def experiment_jax_fundamentals():
    """
    Demonstrate JAX's core concepts and transformations.
    """
    print("\n" + "="*70)
    print(" JAX FUNDAMENTALS")
    print(" Transformations: jit, grad, vmap, pmap")
    print("="*70)
    
    if not JAX_AVAILABLE:
        print("\n JAX not available. Showing conceptual examples only.")
        print("""
    JAX TRANSFORMATIONS:
    
    1. jit - Just-In-Time Compilation
    ─────────────────────────────────────────────────────────────────
    @jax.jit
    def my_function(x):
        return jnp.sin(x) * jnp.cos(x)
    
    # First call: traces and compiles
    # Subsequent calls: uses compiled version
    
    2. grad - Automatic Differentiation
    ─────────────────────────────────────────────────────────────────
    def loss_fn(params, x, y):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    gradients = grad_fn(params, x, y)
    
    3. vmap - Automatic Vectorization
    ─────────────────────────────────────────────────────────────────
    def single_example(x):
        return jnp.dot(x, W)
    
    batched_fn = jax.vmap(single_example)
    # Automatically handles batch dimension!
    
    4. pmap - Parallel Mapping (multi-device)
    ─────────────────────────────────────────────────────────────────
    @jax.pmap
    def parallel_fn(x):
        return x ** 2
    
    # Runs on multiple GPUs/TPUs automatically
        """)
        return
    
    print("\n JAX is available! Running live examples...")
    
    # 1. JIT Compilation
    print("\n 1. JIT COMPILATION:")
    print("-" * 50)
    
    def slow_fn(x):
        for _ in range(10):
            x = jnp.sin(x) * jnp.cos(x)
        return x
    
    fast_fn = jit(slow_fn)
    
    x = jnp.ones((1000, 1000))
    
    # Warmup (triggers compilation)
    _ = fast_fn(x).block_until_ready()
    
    time_eager = profile_fn(lambda: slow_fn(x), iterations=20)
    time_jit = profile_fn(lambda: fast_fn(x), iterations=20)
    
    print(f" Eager: {time_eager:.3f} ms")
    print(f" JIT:   {time_jit:.3f} ms")
    print(f" Speedup: {time_eager/time_jit:.2f}x")
    
    # 2. Automatic Differentiation
    print("\n 2. AUTOMATIC DIFFERENTIATION:")
    print("-" * 50)
    
    def simple_loss(x):
        return jnp.sum(x ** 2)
    
    grad_fn = grad(simple_loss)
    
    x = jnp.array([1.0, 2.0, 3.0])
    gradient = grad_fn(x)
    
    print(f" x = {x}")
    print(f" loss = sum(x^2) = {simple_loss(x)}")
    print(f" gradient = 2x = {gradient}")
    
    # 3. VMAP (Vectorization)
    print("\n 3. AUTOMATIC VECTORIZATION (vmap):")
    print("-" * 50)
    
    def single_fn(x):
        return jnp.dot(x, x)
    
    # Without vmap - need explicit loop
    def manual_batch(xs):
        return jnp.array([single_fn(x) for x in xs])
    
    # With vmap - automatic batching
    batched_fn = vmap(single_fn)
    
    xs = jnp.ones((100, 64))
    
    time_manual = profile_fn(lambda: manual_batch(xs), iterations=100)
    time_vmap = profile_fn(lambda: batched_fn(xs), iterations=100)
    
    print(f" Manual loop: {time_manual:.3f} ms")
    print(f" vmap:        {time_vmap:.3f} ms")
    print(f" Speedup: {time_manual/time_vmap:.2f}x")

# ============================================================================
# JAX VS PYTORCH COMPARISON
# ============================================================================

def experiment_jax_vs_pytorch():
    """
    Compare JAX and PyTorch approaches.
    """
    print("\n" + "="*70)
    print(" JAX VS PYTORCH: PHILOSOPHY AND PERFORMANCE")
    print("="*70)
    
    print("""
    FUNDAMENTAL DIFFERENCES:
    
    ┌────────────────────┬─────────────────────┬─────────────────────┐
    │ Aspect             │ PyTorch             │ JAX                 │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ Paradigm           │ Object-oriented     │ Functional          │
    │ State              │ Mutable tensors     │ Immutable arrays    │
    │ Compilation        │ Optional (2.0+)     │ Core (jit)          │
    │ Autodiff           │ Tape-based          │ Transform-based     │
    │ Random numbers     │ Global state        │ Explicit keys       │
    │ Control flow       │ Native Python       │ Traced or primitives│
    │ Ecosystem          │ Huge                │ Growing             │
    │ TPU support        │ Limited             │ First-class         │
    └────────────────────┴─────────────────────┴─────────────────────┘
    
    CODE STYLE COMPARISON:
    
    PyTorch (Imperative):
    ─────────────────────────────────────────────────────────────────
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)
        
        def forward(self, x):
            return F.relu(self.linear(x))
    
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    for batch in dataloader:
        loss = loss_fn(model(batch))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    JAX (Functional):
    ─────────────────────────────────────────────────────────────────
    def model(params, x):
        return jax.nn.relu(jnp.dot(x, params['w']) + params['b'])
    
    @jax.jit
    def train_step(params, batch):
        def loss_fn(params):
            return jnp.mean((model(params, batch) - target) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, loss
    
    for batch in dataloader:
        params, loss = train_step(params, batch)
    """)
    
    if not (JAX_AVAILABLE and TORCH_AVAILABLE):
        print("\n Need both JAX and PyTorch for comparison")
        return
    
    # Performance comparison
    print("\n PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    size = 2048
    
    # PyTorch
    x_torch = torch.randn(size, size)
    if torch.cuda.is_available():
        x_torch = x_torch.cuda()
    
    def pytorch_ops():
        y = x_torch @ x_torch.T
        y = torch.softmax(y, dim=-1)
        return y
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        _ = pytorch_ops()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(50):
        _ = pytorch_ops()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_pytorch = (time.perf_counter() - start) * 1000 / 50
    
    # JAX
    x_jax = jnp.array(np.random.randn(size, size).astype(np.float32))
    
    @jit
    def jax_ops(x):
        y = x @ x.T
        y = jax.nn.softmax(y, axis=-1)
        return y
    
    # Warmup
    _ = jax_ops(x_jax).block_until_ready()
    
    start = time.perf_counter()
    for _ in range(50):
        _ = jax_ops(x_jax).block_until_ready()
    time_jax = (time.perf_counter() - start) * 1000 / 50
    
    print(f" Operation: matmul + softmax ({size}x{size})")
    print(f" PyTorch: {time_pytorch:.3f} ms")
    print(f" JAX:     {time_jax:.3f} ms")
    print(f" Ratio:   {time_pytorch/time_jax:.2f}x")

# ============================================================================
# JAX FOR TRAINING
# ============================================================================

def explain_jax_training():
    """
    Explain how training works in JAX.
    """
    print("\n" + "="*70)
    print(" JAX TRAINING PATTERNS")
    print("="*70)
    
    print("""
    FUNCTIONAL TRAINING LOOP:
    
    In JAX, everything is a pure function:
    • Model is a function: output = model(params, input)
    • Training step is a function: new_params = train_step(params, batch)
    • No hidden state mutation!
    
    EXAMPLE TRAINING SETUP:
    ─────────────────────────────────────────────────────────────────
    
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
    
    # 1. Define model as pure function
    def mlp(params, x):
        for w, b in params[:-1]:
            x = jax.nn.relu(jnp.dot(x, w) + b)
        w, b = params[-1]
        return jnp.dot(x, w) + b
    
    # 2. Define loss function
    def loss_fn(params, x, y):
        pred = mlp(params, x)
        return jnp.mean((pred - y) ** 2)
    
    # 3. Compute gradients
    @jit
    def train_step(params, x, y, lr):
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        
        # Update params (functional style)
        new_params = jax.tree_map(
            lambda p, g: p - lr * g,
            params, grads
        )
        return new_params, loss
    
    # 4. Training loop
    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            params, loss = train_step(params, x_batch, y_batch, lr=0.001)
    
    RANDOM NUMBERS IN JAX:
    ─────────────────────────────────────────────────────────────────
    
    JAX requires explicit random keys (no global state):
    
    key = jax.random.PRNGKey(42)
    
    # Split key for each use
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (100, 64))
    
    key, subkey = jax.random.split(key)
    dropout_mask = jax.random.bernoulli(subkey, 0.9, x.shape)
    
    WHY FUNCTIONAL?
    ─────────────────────────────────────────────────────────────────
    
    Benefits:
    ✓ Easier to reason about (no hidden state)
    ✓ Better for compilation (pure functions)
    ✓ Natural parallelization (no shared state)
    ✓ Reproducibility (explicit randomness)
    
    Challenges:
    ✗ Different mental model from PyTorch
    ✗ Need to pass state explicitly
    ✗ Can be verbose for complex models
    
    LIBRARIES TO HELP:
    • Flax: Neural network library (Google)
    • Haiku: Neural network library (DeepMind)
    • Optax: Optimizers
    • Equinox: PyTorch-like syntax
    """)

# ============================================================================
# XLA OPTIMIZATIONS IN DETAIL
# ============================================================================

def explain_xla_optimizations():
    """
    Detail the optimizations XLA performs.
    """
    print("\n" + "="*70)
    print(" XLA OPTIMIZATIONS IN DETAIL")
    print("="*70)
    
    print("""
    1. FUSION STRATEGIES
    ═══════════════════════════════════════════════════════════════════
    
    XLA has sophisticated fusion heuristics:
    
    PRODUCER-CONSUMER FUSION:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Before: y = exp(x); z = y + 1                                  │
    │ After:  z = exp(x) + 1  (single kernel)                        │
    └─────────────────────────────────────────────────────────────────┘
    
    MULTI-OUTPUT FUSION:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Before: y = x * 2; z = x + 1  (x read twice)                   │
    │ After:  y, z = fused(x)  (x read once)                         │
    └─────────────────────────────────────────────────────────────────┘
    
    REDUCTION FUSION:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Before: y = x * 2; z = sum(y)                                  │
    │ After:  z = sum(x * 2)  (multiply during reduction)            │
    └─────────────────────────────────────────────────────────────────┘
    
    2. LAYOUT OPTIMIZATION
    ═══════════════════════════════════════════════════════════════════
    
    XLA automatically chooses optimal memory layouts:
    
    • Row-major vs column-major
    • NCHW vs NHWC for images
    • Padding for alignment
    
    Example:
    ┌─────────────────────────────────────────────────────────────────┐
    │ matmul(A, B) where A is row-major, B is column-major           │
    │                                                                │
    │ Options:                                                       │
    │ 1. Transpose A, then matmul (costly)                          │
    │ 2. Transpose B, then matmul (costly)                          │
    │ 3. Use different matmul algorithm (maybe best!)               │
    │                                                                │
    │ XLA analyzes the full graph to minimize total cost            │
    └─────────────────────────────────────────────────────────────────┘
    
    3. BUFFER ASSIGNMENT
    ═══════════════════════════════════════════════════════════════════
    
    XLA performs whole-program buffer planning:
    
    • Liveness analysis: When is each tensor alive?
    • Buffer reuse: Share memory for non-overlapping tensors
    • In-place updates: Modify input when safe
    
    4. TPU-SPECIFIC OPTIMIZATIONS
    ═══════════════════════════════════════════════════════════════════
    
    XLA has first-class TPU support:
    
    • MXU utilization: Maximize matrix unit usage
    • HBM bandwidth: Optimize memory access patterns
    • Infeed/outfeed: Overlap compute with data transfer
    • Cross-replica communication: Efficient all-reduce
    """)

# ============================================================================
# PROS AND CONS SUMMARY
# ============================================================================

def print_xla_jax_summary():
    """
    Summary of XLA and JAX.
    """
    print("\n" + "="*70)
    print(" XLA / JAX SUMMARY")
    print("="*70)
    
    print("""
    XLA PROS:
    ─────────────────────────────────────────────────────────────────
    ✓ Excellent optimization quality
    ✓ First-class TPU support
    ✓ Whole-program optimization
    ✓ Mature and battle-tested
    ✓ Good for large-scale training
    ✓ Efficient memory management
    
    XLA CONS:
    ─────────────────────────────────────────────────────────────────
    ✗ Requires static shapes (mostly)
    ✗ Longer compilation times
    ✗ Less flexible than eager execution
    ✗ Debugging compiled code is hard
    ✗ Some ops not well-supported
    
    JAX PROS:
    ─────────────────────────────────────────────────────────────────
    ✓ NumPy-compatible API
    ✓ Elegant transformation system (jit, grad, vmap)
    ✓ Great for research and experimentation
    ✓ Excellent automatic differentiation
    ✓ Good for functional programming
    ✓ Growing ecosystem (Flax, Haiku, Optax)
    
    JAX CONS:
    ─────────────────────────────────────────────────────────────────
    ✗ Smaller ecosystem than PyTorch
    ✗ Requires functional programming mindset
    ✗ Explicit random state management
    ✗ Less community support/tutorials
    ✗ Some PyTorch features missing
    
    WHEN TO USE JAX/XLA:
    ─────────────────────────────────────────────────────────────────
    ✓ Training on TPUs
    ✓ Large-scale distributed training
    ✓ Research requiring transformations (vmap, etc.)
    ✓ Functional programming preference
    ✓ Google Cloud infrastructure
    
    WHEN TO USE PYTORCH:
    ─────────────────────────────────────────────────────────────────
    ✓ Rapid prototyping
    ✓ Existing PyTorch codebase
    ✓ Need maximum flexibility
    ✓ Large library ecosystem (HuggingFace, etc.)
    ✓ NVIDIA GPU focus
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " XLA AND JAX COMPILER SYSTEM ".center(68) + "║")
    print("║" + " Google's ML compiler stack ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    if JAX_AVAILABLE:
        print(f"\n JAX version: {jax.__version__}")
        print(f" Devices: {jax.devices()}")
    else:
        print("\n JAX not installed")
    
    explain_xla()
    experiment_jax_fundamentals()
    experiment_jax_vs_pytorch()
    explain_jax_training()
    explain_xla_optimizations()
    print_xla_jax_summary()
    
    print("\n" + "="*70)
    print(" Next: NVIDIA compiler ecosystem (TensorRT, cuDNN, CUTLASS)")
    print("="*70)
