"""
03_autograd_engine.py - PyTorch Autograd Engine Deep Dive

The Autograd Engine is what makes PyTorch "PyTorch".
It enables automatic differentiation for gradient-based optimization.

Key Source Files:
- torch/csrc/autograd/engine.cpp - THE engine
- torch/csrc/autograd/function.h - Node (grad_fn) base class
- torch/csrc/autograd/variable.h - Variable (tensor with grad)
- tools/autograd/derivatives.yaml - Derivative definitions

Understanding autograd is ESSENTIAL for:
- Debugging gradient issues
- Implementing custom autograd functions
- Understanding memory usage in training
- Contributing to PyTorch core

Run: python 03_autograd_engine.py
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Function
import time
import gc
from typing import Tuple, Optional, List

# ============================================================================
# PROFILING
# ============================================================================

def profile_cuda(func, warmup=5, iterations=20):
    """Profile with proper CUDA sync."""
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
# COMPUTATION GRAPH STRUCTURE
# ============================================================================

def explain_computation_graph():
    """Explain how autograd builds computation graphs."""
    print("\n" + "="*70)
    print(" COMPUTATION GRAPH STRUCTURE")
    print(" The foundation of automatic differentiation")
    print("="*70)
    
    print("""
    AUTOGRAD BUILDS A DYNAMIC COMPUTATION GRAPH:
    ─────────────────────────────────────────────────────────────────
    
    Forward pass: z = x * y + 1
    
            x (leaf)      y (leaf)
            (grad_fn=None) (grad_fn=None)
                 \\          /
                  \\        /
                   MulBackward0
                        |
                        + -------- 1 (constant, no grad)
                        |
                   AddBackward0
                        |
                        z
                   (grad_fn=AddBackward0)
    
    KEY CONCEPTS:
    ─────────────────────────────────────────────────────────────────
    
    1. LEAF TENSOR
       - Created directly by user (not from operation)
       - grad_fn = None
       - .is_leaf = True
       - Gradients accumulated here
    
    2. NON-LEAF TENSOR (intermediate)
       - Result of an operation
       - grad_fn points to backward function
       - .is_leaf = False
       - Gradient NOT retained by default
    
    3. grad_fn (Node)
       - Stores how to compute gradient
       - Links to input tensors' grad_fns
       - next_functions = [(parent_grad_fn, output_nr), ...]
    
    4. requires_grad
       - Tensor participates in gradient computation
       - Propagates through operations
    """)
    
    # Live demo
    print("\n LIVE DEMONSTRATION:")
    print("-" * 50)
    
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    print(f" x = {x.item()}, requires_grad={x.requires_grad}, is_leaf={x.is_leaf}")
    print(f" y = {y.item()}, requires_grad={y.requires_grad}, is_leaf={y.is_leaf}")
    
    # Build graph
    mul = x * y
    z = mul + 1
    
    print(f"\n mul = x * y = {mul.item()}")
    print(f" mul.grad_fn = {mul.grad_fn}")
    print(f" mul.is_leaf = {mul.is_leaf}")
    
    print(f"\n z = mul + 1 = {z.item()}")
    print(f" z.grad_fn = {z.grad_fn}")
    
    # Traverse the graph
    print(f"\n Graph structure:")
    print(f" z.grad_fn = {z.grad_fn}")
    print(f" └── next_functions[0] = {z.grad_fn.next_functions[0]}")
    
    mul_backward = z.grad_fn.next_functions[0][0]
    print(f"     └── MulBackward0.next_functions:")
    for i, (fn, idx) in enumerate(mul_backward.next_functions):
        print(f"         [{i}] = ({fn}, {idx})")

# ============================================================================
# BACKWARD PASS
# ============================================================================

def explain_backward_pass():
    """Explain how backward() executes."""
    print("\n" + "="*70)
    print(" BACKWARD PASS EXECUTION")
    print(" How gradients flow through the graph")
    print("="*70)
    
    print("""
    WHEN YOU CALL loss.backward():
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. CREATE GRAPH TASK                                                    │
    │    - Root = loss.grad_fn                                               │
    │    - Initial gradient = 1.0 (for scalar loss)                          │
    │    - Track which nodes to execute                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 2. TOPOLOGICAL EXECUTION                                                │
    │    - Process nodes in reverse topological order                        │
    │    - Use reference counting (dependencies)                             │
    │    - Node ready when all dependents processed                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 3. FOR EACH NODE:                                                       │
    │    a. Collect input gradients (from dependents)                        │
    │    b. Call node.apply(grad_outputs) → grad_inputs                      │
    │    c. Distribute grad_inputs to parent nodes                           │
    │    d. For leaf tensors: accumulate into .grad                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 4. CLEANUP                                                              │
    │    - Release saved tensors                                             │
    │    - Clear graph (unless retain_graph=True)                           │
    └─────────────────────────────────────────────────────────────────────────┘
    
    GRADIENT ACCUMULATION:
    ─────────────────────────────────────────────────────────────────
    
    For leaf tensors, gradients ACCUMULATE:
    
    x.grad = x.grad + new_gradient
    
    This is why you need optimizer.zero_grad()!
    
    Without zeroing:
    - Epoch 1: x.grad = grad1
    - Epoch 2: x.grad = grad1 + grad2  # WRONG!
    
    With zeroing:
    - Epoch 1: x.grad = 0 + grad1 = grad1
    - Epoch 2: x.grad = 0 + grad2 = grad2  # Correct
    """)
    
    # Demo gradient accumulation
    print("\n DEMO: Gradient accumulation")
    print("-" * 50)
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # First backward
    y1 = x * 2
    y1.backward()
    print(f" After first backward: x.grad = {x.grad.item()}")
    
    # Second backward WITHOUT zeroing
    y2 = x * 3
    y2.backward()
    print(f" After second backward (no zero): x.grad = {x.grad.item()} (accumulated!)")
    
    # Reset and do it right
    x.grad.zero_()
    y3 = x * 3
    y3.backward()
    print(f" After zeroing and backward: x.grad = {x.grad.item()} (correct)")

# ============================================================================
# AUTOGRAD ENGINE INTERNALS
# ============================================================================

def explain_engine_internals():
    """Deep dive into autograd engine implementation."""
    print("\n" + "="*70)
    print(" AUTOGRAD ENGINE INTERNALS")
    print(" torch/csrc/autograd/engine.cpp")
    print("="*70)
    
    print("""
    ENGINE ARCHITECTURE:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Engine (singleton)                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ thread_pool_           │ Worker threads (one per device)               │
    │ ready_queues_          │ Per-device queues of ready nodes              │
    │ local_ready_queue_     │ Thread-local ready queue                      │
    └─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         GraphTask                                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ owner_                 │ Thread that owns this task                    │
    │ outstanding_tasks_     │ Number of pending nodes                       │
    │ dependencies_          │ Map: Node → number of unprocessed outputs     │
    │ not_ready_             │ Nodes waiting for inputs                      │
    │ captured_vars_         │ Gradients for outputs (if grad_tensors)       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    NODE EXECUTION (engine.cpp::evaluate_function):
    ─────────────────────────────────────────────────────────────────
    
    1. Pop node from ready_queue
    
    2. Prepare inputs:
       - Gather gradients from InputBuffer
       - Apply hooks (if any)
    
    3. Execute node:
       outputs = node->apply(inputs)
    
    4. Process outputs:
       for each (next_node, output_nr) in node.next_functions:
           if next_node is AccumulateGrad:
               # Leaf tensor - accumulate gradient
               next_node.variable.grad += output
           else:
               # Non-leaf - add to InputBuffer
               input_buffer[next_node][output_nr] = output
               if next_node.all_inputs_ready():
                   ready_queue.push(next_node)
    
    5. Decrement outstanding_tasks
    
    MULTI-DEVICE EXECUTION:
    ─────────────────────────────────────────────────────────────────
    
    Each CUDA device has its own:
    - Worker thread
    - Ready queue
    - CUDA stream for backward
    
    CPU operations run on calling thread.
    
    Synchronization:
    - GraphTask coordinates across devices
    - Events ensure correct ordering
    """)

# ============================================================================
# CUSTOM AUTOGRAD FUNCTIONS
# ============================================================================

def explain_custom_autograd():
    """Explain how to create custom autograd functions."""
    print("\n" + "="*70)
    print(" CUSTOM AUTOGRAD FUNCTIONS")
    print(" Extending autograd with custom derivatives")
    print("="*70)
    
    print("""
    torch.autograd.Function STRUCTURE:
    ─────────────────────────────────────────────────────────────────
    
    class MyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, ...):
            # Save tensors for backward
            ctx.save_for_backward(input, ...)
            # Compute and return output
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve saved tensors
            input, ... = ctx.saved_tensors
            # Compute gradient w.r.t. each input
            grad_input = ...
            return grad_input, ...
    
    RULES:
    ─────────────────────────────────────────────────────────────────
    
    1. backward() must return same number of gradients as forward() inputs
    2. Return None for inputs that don't need gradients
    3. Use ctx.save_for_backward() for tensors (memory efficient)
    4. Use ctx.attr = value for non-tensors
    5. ctx.needs_input_grad[i] tells if input i needs gradient
    """)
    
    # Custom ReLU implementation
    class MyReLU(Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(min=0)
        
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input
    
    print("\n DEMO: Custom ReLU function")
    print("-" * 50)
    
    x = torch.randn(5, requires_grad=True)
    print(f" x = {x.data}")
    
    # Forward
    y = MyReLU.apply(x)
    print(f" y = MyReLU(x) = {y.data}")
    
    # Backward
    y.sum().backward()
    print(f" x.grad = {x.grad}")
    print(f" (Gradient is 1 where x > 0, 0 where x <= 0)")
    
    # Verify against PyTorch relu
    x2 = x.detach().clone().requires_grad_(True)
    torch.relu(x2).sum().backward()
    print(f"\n Matches PyTorch: {torch.allclose(x.grad, x2.grad)}")
    
    # Custom function with multiple outputs
    print("\n DEMO: Custom function with multiple outputs")
    print("-" * 50)
    
    class SplitFunction(Function):
        @staticmethod
        def forward(ctx, input, split_size):
            ctx.split_size = split_size
            return input[:split_size], input[split_size:]
        
        @staticmethod
        def backward(ctx, grad_first, grad_second):
            # Concatenate gradients back
            return torch.cat([grad_first, grad_second]), None
    
    x = torch.randn(10, requires_grad=True)
    first, second = SplitFunction.apply(x, 4)
    loss = first.sum() + second.sum() * 2
    loss.backward()
    print(f" Split at index 4:")
    print(f" Gradient for first part (weight 1): {x.grad[:4]}")
    print(f" Gradient for second part (weight 2): {x.grad[4:]}")

# ============================================================================
# GRADIENT CHECKPOINTING
# ============================================================================

def explain_gradient_checkpointing():
    """Explain gradient checkpointing for memory efficiency."""
    print("\n" + "="*70)
    print(" GRADIENT CHECKPOINTING")
    print(" Trading compute for memory")
    print("="*70)
    
    print("""
    THE MEMORY PROBLEM:
    ─────────────────────────────────────────────────────────────────
    
    Forward pass saves activations for backward:
    
    Layer 1: Save activation (e.g., 1GB)
    Layer 2: Save activation (e.g., 1GB)
    ...
    Layer N: Save activation (e.g., 1GB)
    
    Total memory: N × activation_size
    
    For deep networks (e.g., GPT with 96 layers), this is HUGE!
    
    CHECKPOINTING SOLUTION:
    ─────────────────────────────────────────────────────────────────
    
    Instead of saving all activations:
    1. Save only checkpoint activations
    2. During backward, recompute from checkpoints
    
    Example with checkpoint every 4 layers:
    
    Normal:     [1][2][3][4][5][6][7][8] - 8 activations saved
    Checkpoint: [1]      [4]      [8]    - 3 activations saved
    
    During backward for layer 3:
    - Recompute from checkpoint 1: layers 1→2→3
    - Then compute gradient
    
    TRADEOFF:
    ─────────────────────────────────────────────────────────────────
    
    Memory: O(sqrt(N)) instead of O(N)
    Compute: ~1.3x more (33% overhead)
    
    Worth it for large models!
    """)
    
    from torch.utils.checkpoint import checkpoint
    
    # Demo memory savings
    class HeavyModule(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.linear = nn.Linear(size, size)
        
        def forward(self, x):
            return torch.relu(self.linear(x))
    
    def run_with_checkpointing(use_checkpoint: bool, num_layers: int = 8):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        size = 1024
        layers = nn.ModuleList([HeavyModule(size) for _ in range(num_layers)])
        if torch.cuda.is_available():
            layers = layers.cuda()
        
        x = torch.randn(64, size, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forward
        for i, layer in enumerate(layers):
            if use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        
        loss = x.sum()
        loss.backward()
        
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
        return 0
    
    if torch.cuda.is_available():
        print("\n DEMO: Memory savings from checkpointing")
        print("-" * 50)
        
        mem_normal = run_with_checkpointing(False)
        mem_checkpoint = run_with_checkpointing(True)
        
        print(f" Normal: {mem_normal:.1f} MB peak memory")
        print(f" Checkpoint: {mem_checkpoint:.1f} MB peak memory")
        print(f" Savings: {(1 - mem_checkpoint/mem_normal)*100:.1f}%")
    else:
        print("\n CUDA not available for memory demo")

# ============================================================================
# AUTOGRAD HOOKS
# ============================================================================

def explain_autograd_hooks():
    """Explain autograd hooks for debugging and customization."""
    print("\n" + "="*70)
    print(" AUTOGRAD HOOKS")
    print(" Intercepting gradients for debugging and modification")
    print("="*70)
    
    print("""
    HOOK TYPES:
    ─────────────────────────────────────────────────────────────────
    
    1. TENSOR HOOKS (register_hook)
       - Called when gradient computed for this tensor
       - Can modify gradient before accumulation
       - tensor.register_hook(lambda grad: ...)
    
    2. MODULE HOOKS (for nn.Module)
       - register_forward_hook: after forward
       - register_backward_hook: during backward (deprecated)
       - register_full_backward_hook: after backward
    
    3. GLOBAL HOOKS
       - register_multi_grad_hook: fire when all grads ready
    """)
    
    # Tensor hook demo
    print("\n DEMO: Tensor hook for gradient clipping")
    print("-" * 50)
    
    x = torch.randn(5, requires_grad=True)
    
    def clip_grad(grad):
        print(f"  Hook called! Original grad norm: {grad.norm().item():.4f}")
        return grad.clamp(-0.5, 0.5)
    
    hook = x.register_hook(clip_grad)
    
    y = (x * 10).sum()  # Large gradients
    y.backward()
    
    print(f" After clipping: x.grad = {x.grad}")
    print(f" Gradient is clipped to [-0.5, 0.5]")
    
    hook.remove()  # Clean up
    
    # Module hook demo
    print("\n DEMO: Module hook for activation statistics")
    print("-" * 50)
    
    class HookedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.activation_stats = {}
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = HookedModel()
    
    def forward_hook(module, input, output):
        name = type(module).__name__
        model.activation_stats[name] = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'max': output.max().item(),
        }
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(forward_hook))
    
    # Forward pass
    x = torch.randn(4, 10)
    y = model(x)
    
    print(" Activation statistics:")
    for name, stats in model.activation_stats.items():
        print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    # Clean up hooks
    for handle in handles:
        handle.remove()

# ============================================================================
# PROFILING AUTOGRAD
# ============================================================================

def profile_autograd():
    """Profile autograd operations."""
    print("\n" + "="*70)
    print(" PROFILING AUTOGRAD")
    print(" Understanding backward pass performance")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n CUDA not available for profiling")
        return
    
    # Create a model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).cuda()
    
    x = torch.randn(32, 512, device='cuda')
    
    # Profile forward + backward
    print("\n Forward + Backward profiling:")
    print("-" * 50)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            y = model(x)
            loss = y.sum()
            loss.backward()
            model.zero_grad()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# ============================================================================
# SUMMARY
# ============================================================================

def print_autograd_summary():
    """Print autograd summary."""
    print("\n" + "="*70)
    print(" AUTOGRAD ENGINE SUMMARY")
    print("="*70)
    
    print("""
    KEY CONCEPTS:
    
    1. COMPUTATION GRAPH
       - Built dynamically during forward
       - Nodes = grad_fn (backward functions)
       - Edges = data flow
       - Cleared after backward (unless retain_graph)
    
    2. BACKWARD EXECUTION
       - Reverse topological order
       - Reference counting for dependencies
       - Multi-threaded (per device)
       - Gradients accumulate in leaf tensors
    
    3. CUSTOM FUNCTIONS
       - Extend torch.autograd.Function
       - Implement forward() and backward()
       - Use ctx.save_for_backward() for tensors
    
    4. CHECKPOINTING
       - Trade compute for memory
       - Essential for large models
       - torch.utils.checkpoint.checkpoint()
    
    5. HOOKS
       - Intercept gradients
       - Debug and modify
       - Tensor, module, and global hooks
    
    COMMON ISSUES:
    ─────────────────────────────────────────────────────────────────
    
    "Trying to backward through graph a second time"
    → Use retain_graph=True or don't reuse outputs
    
    "One of the differentiated Tensors does not require grad"
    → Ensure inputs have requires_grad=True
    
    "Gradients are None"
    → Check is_leaf, or use retain_grad()
    
    "Gradients accumulate unexpectedly"
    → Call optimizer.zero_grad() before backward
    
    SOURCE CODE TO STUDY:
    ─────────────────────────────────────────────────────────────────
    
    torch/csrc/autograd/
    ├── engine.cpp           # Main engine
    ├── function.h           # Node base class
    ├── variable.h           # Variable = Tensor + grad
    └── custom_function.cpp  # Python Function binding
    
    tools/autograd/
    ├── derivatives.yaml     # Derivative formulas
    └── gen_autograd.py      # Code generator
    
    NEXT: Study 04_pytorch_ecosystem.py for ecosystem overview
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH AUTOGRAD ENGINE DEEP DIVE ".center(68) + "║")
    print("║" + " Understanding automatic differentiation ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    
    explain_computation_graph()
    explain_backward_pass()
    explain_engine_internals()
    explain_custom_autograd()
    explain_gradient_checkpointing()
    explain_autograd_hooks()
    profile_autograd()
    print_autograd_summary()
    
    print("\n" + "="*70)
    print(" Autograd is the magic behind gradient-based learning!")
    print("="*70)
