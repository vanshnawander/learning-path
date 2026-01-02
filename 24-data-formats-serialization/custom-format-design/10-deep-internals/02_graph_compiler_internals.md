# The FFCV Compiler: AST Generation and JIT Linking

## The Core Problem: Python Overhead

Even with optimized C++ or Numba functions, Python itself becomes the bottleneck:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    PYTHON OVERHEAD BREAKDOWN                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Naive Pipeline:                                                               │
│  ─────────────────                                                             │
│                                                                                │
│  for i in range(batch_size):          # Python loop overhead: ~100ns/iter    │
│      sample = read_sample(indices[i]) # Python call: ~50ns                    │
│      image = decode_jpeg(sample)      # Fast C++: 3ms                         │
│      image = resize(image)            # Python call: ~50ns                    │
│      image = resize_call(image)       # Fast C++: 0.5ms                       │
│      image = normalize(image)         # Python call: ~50ns                    │
│      image = normalize_call(image)    # Fast Numba: 0.1ms                     │
│      images[i] = image                # Python assignment: ~50ns              │
│                                                                                │
│  For batch_size=256:                                                          │
│  - C++ work: 256 × 3.6ms = 921ms                                             │
│  - Python overhead: 256 × ~300ns × 4 calls ≈ 0.3ms                           │
│                                                                                │
│  Seems small? Not when you scale up:                                          │
│  - More transforms: each adds ~50ns per call                                  │
│  - More samples: linear scaling                                               │
│  - Parallel workers: GIL contention magnifies overhead                        │
│                                                                                │
│  In real workloads, Python overhead can be 10-30% of total time!             │
│                                                                                │
│  THE SOLUTION: Generate a single function that does EVERYTHING,              │
│  then JIT-compile that entire function to machine code.                       │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## How FFCV Solves This: Code Generation

FFCV takes a radical approach:
1. At pipeline setup, analyze all operations and their memory needs.
2. Generate Python source code for a single "mega-function" that does the entire pipeline.
3. JIT-compile that function with Numba.
4. At runtime, only the compiled machine code runs—no Python interpreter in the loop.

```python
# BEFORE: What the user writes (clean, modular)
pipeline = [
    decode_jpeg(),
    random_resized_crop(224),
    random_horizontal_flip(),
    normalize(mean, std),
    to_tensor()
]

# AFTER: What FFCV generates and compiles (single fused function)
@numba.njit(parallel=True, nogil=True)
def stage_func_0(batch_indices, metadata, storage_state, mem_0, mem_1, mem_2):
    for i in prange(len(batch_indices)):
        sample_id = batch_indices[i]
        
        # Decode (C++ call via ctypes pointer)
        decode_impl(
            metadata[sample_id]['data_ptr'],
            metadata[sample_id]['data_size'],
            mem_0[i]  # Pre-allocated output buffer
        )
        
        # Random resized crop (JIT-compiled Numba)
        crop_params = get_crop_params(...)  # All inlined
        crop_impl(mem_0[i], crop_params, mem_1[i])
        
        # Horizontal flip (JIT-compiled, conditionally executed)
        if random() < 0.5:
            flip_impl(mem_1[i], mem_1[i])  # In-place
        
        # Normalize (JIT-compiled, vectorized)
        normalize_impl(mem_1[i], mem_2[i], mean, std)
    
    return mem_2

# The result: one function call processes the entire batch
# No Python interpreter involvement during the loop
```

## The AST Generation Pipeline

### Step 1: Build the Computation Graph

FFCV represents the pipeline as a directed graph of `Node` objects:

```python
from dataclasses import dataclass
from typing import List, Optional, Any
import ast

@dataclass
class Node:
    """Base class for all nodes in the computation graph."""
    node_id: str
    operation: Any  # The Operation instance
    parent: Optional['Node']
    children: List['Node']
    
    # Computed during analysis
    output_state: Optional['State'] = None
    memory_allocation: Optional['AllocationQuery'] = None
    
    def generate_code_ast(self, context: dict) -> List[ast.stmt]:
        """
        Generate AST statements for this node's operation.
        
        Returns a list of AST statements to be inserted into the
        generated function body.
        """
        raise NotImplementedError


class DecoderNode(Node):
    """
    Node that reads and decodes from storage.
    
    This is always a root node (no parent).
    """
    
    def generate_code_ast(self, context: dict) -> List[ast.stmt]:
        # Get the implementation function name
        impl_name = f"decoder_{self.node_id}"
        
        # Generate: result_X = decoder_X(sample_id, metadata, storage, mem_X)
        code = f"""
{self.node_id}_result = {impl_name}(
    batch_indices[i],
    metadata,
    storage_state,
    memory_{self.node_id}[i]
)
"""
        return ast.parse(code).body


class TransformNode(Node):
    """
    Node that transforms data from a parent node.
    """
    
    def generate_code_ast(self, context: dict) -> List[ast.stmt]:
        impl_name = f"transform_{self.node_id}"
        parent_result = f"{self.parent.node_id}_result"
        
        # Generate: result_X = transform_X(parent_result, mem_X)
        code = f"""
{self.node_id}_result = {impl_name}(
    {parent_result},
    memory_{self.node_id}[i]
)
"""
        return ast.parse(code).body


class Graph:
    """
    The complete computation graph for a data loading pipeline.
    """
    
    def __init__(self, pipelines: dict):
        """
        Args:
            pipelines: Dict mapping field name to list of operations.
                       E.g., {'image': [Decode(), Resize(), Flip()], 'label': [Decode()]}
        """
        self.pipelines = pipelines
        self.nodes: List[Node] = []
        self.root_nodes: List[Node] = []
        
        self._build_graph()
    
    def _build_graph(self):
        """Convert operation lists to node graph."""
        for field_name, operations in self.pipelines.items():
            parent = None
            
            for i, op in enumerate(operations):
                node_id = f"{field_name}_{i}"
                
                if i == 0:
                    # First operation is a decoder
                    node = DecoderNode(
                        node_id=node_id,
                        operation=op,
                        parent=None,
                        children=[]
                    )
                    self.root_nodes.append(node)
                else:
                    node = TransformNode(
                        node_id=node_id,
                        operation=op,
                        parent=parent,
                        children=[]
                    )
                    parent.children.append(node)
                
                self.nodes.append(node)
                parent = node
```

### Step 2: Collect Memory Requirements

Before generating code, we traverse the graph to determine output shapes:

```python
def collect_requirements(self):
    """
    Traverse graph to determine memory allocations needed.
    
    Each operation declares:
    1. Its output State (shape, dtype)
    2. Memory allocation query (how much buffer space)
    
    This is done ONCE at pipeline setup, not per-batch.
    """
    for root in self.root_nodes:
        self._collect_recursive(root, initial_state=None)

def _collect_recursive(self, node: Node, initial_state):
    """
    Depth-first traversal collecting requirements.
    """
    # Ask operation what it needs
    output_state, allocation = node.operation.declare_state_and_memory(initial_state)
    
    node.output_state = output_state
    node.memory_allocation = allocation
    
    # Recurse to children
    for child in node.children:
        self._collect_recursive(child, output_state)
```

### Step 3: Generate the AST

Now we build the actual function AST:

```python
def generate_stage_function(self, stage_nodes: List[Node]) -> ast.FunctionDef:
    """
    Generate a complete function AST for a pipeline stage.
    
    The generated function will:
    1. Take batch indices, metadata, and pre-allocated memory buffers
    2. Loop over batch indices
    3. Call each operation in topological order
    4. Return the final output buffers
    """
    
    # ==========================================
    # Step A: Create function signature
    # ==========================================
    
    # Base arguments: (batch_indices, metadata, storage_state)
    args = [
        ast.arg(arg='batch_indices', annotation=None),
        ast.arg(arg='metadata', annotation=None),
        ast.arg(arg='storage_state', annotation=None),
    ]
    
    # Add memory buffer arguments for each node
    for node in stage_nodes:
        args.append(ast.arg(arg=f'memory_{node.node_id}', annotation=None))
    
    # ==========================================
    # Step B: Build loop body
    # ==========================================
    
    # We use numba.prange for parallel iteration
    # The generated code looks like:
    #
    # for i in prange(len(batch_indices)):
    #     sample_id = batch_indices[i]
    #     result_0 = decoder_0(...)
    #     result_1 = transform_1(result_0, ...)
    #     ...
    
    loop_body = []
    
    # Add: sample_id = batch_indices[i]
    loop_body.append(ast.parse("sample_id = batch_indices[i]").body[0])
    
    # Add operation calls in topological order
    for node in self._topological_sort(stage_nodes):
        loop_body.extend(node.generate_code_ast({}))
    
    # ==========================================
    # Step C: Create the for loop
    # ==========================================
    
    # for i in prange(len(batch_indices)):
    loop = ast.For(
        target=ast.Name(id='i', ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id='prange', ctx=ast.Load()),
            args=[ast.Call(
                func=ast.Name(id='len', ctx=ast.Load()),
                args=[ast.Name(id='batch_indices', ctx=ast.Load())],
                keywords=[]
            )],
            keywords=[]
        ),
        body=loop_body,
        orelse=[]
    )
    
    # ==========================================
    # Step D: Create return statement
    # ==========================================
    
    # Find the leaf nodes (final outputs)
    leaf_nodes = [n for n in stage_nodes if len(n.children) == 0]
    
    # return (memory_leaf1, memory_leaf2, ...)
    return_value = ast.Tuple(
        elts=[ast.Name(id=f'memory_{n.node_id}', ctx=ast.Load()) for n in leaf_nodes],
        ctx=ast.Load()
    )
    return_stmt = ast.Return(value=return_value)
    
    # ==========================================
    # Step E: Assemble function
    # ==========================================
    
    func_def = ast.FunctionDef(
        name='stage_func_0',
        args=ast.arguments(
            args=args,
            posonlyargs=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None
        ),
        body=[loop, return_stmt],
        decorator_list=[],
        returns=None
    )
    
    return func_def

def _topological_sort(self, nodes: List[Node]) -> List[Node]:
    """
    Sort nodes so that each node comes after its parent.
    
    This ensures operations are called in the correct order.
    """
    sorted_nodes = []
    visited = set()
    
    def visit(node):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        
        if node.parent and node.parent.node_id not in visited:
            visit(node.parent)
        
        sorted_nodes.append(node)
    
    for node in nodes:
        visit(node)
    
    return sorted_nodes
```

### Step 4: Compile with `exec` and Numba

```python
def compile_stage(self, stage_nodes: List[Node]):
    """
    Compile a stage into optimized machine code.
    
    1. Generate AST
    2. Create namespace with operation implementations
    3. Execute to create Python function
    4. JIT-compile with Numba
    """
    
    # Generate AST
    func_ast = self.generate_stage_function(stage_nodes)
    
    # Wrap in module
    module = ast.Module(body=[func_ast], type_ignores=[])
    ast.fix_missing_locations(module)
    
    # Create namespace with all required implementations
    namespace = {
        'prange': numba.prange,
        'len': len,
    }
    
    # Add operation implementations
    for node in stage_nodes:
        impl = node.operation.generate_code()  # Returns JIT-ready function
        namespace[f"decoder_{node.node_id}" if isinstance(node, DecoderNode) 
                  else f"transform_{node.node_id}"] = impl
    
    # Execute the AST to create the function
    code = compile(module, '<generated>', 'exec')
    exec(code, namespace)
    
    # Get the function
    stage_func = namespace['stage_func_0']
    
    # Apply Numba JIT
    compiled = numba.njit(parallel=True, nogil=True)(stage_func)
    
    # Warm up (trigger compilation)
    # Would call with dummy data here in production
    
    return compiled
```

## Complete Example: From Pipeline to Machine Code

```python
# User code
from ffcv import DatasetReader, Pipeline
from ffcv.transforms import RandomResizedCrop, Normalize, ToTensor

pipeline = {
    'image': [
        SimpleJPEGDecoder(),
        RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.75, 1.33), size=224),
        RandomHorizontalFlip(prob=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    'label': [
        IntDecoder(),
    ]
}

# FFCV internally does this:

# 1. Build graph
graph = Graph(pipeline)

# 2. Collect memory requirements
graph.collect_requirements()
# Result: knows image pipeline needs 224×224×3 uint8 buffer per sample
#         knows label pipeline needs 1 int64 per sample

# 3. Pre-allocate memory (once, before training)
batch_size = 256
image_buffer = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
label_buffer = np.zeros((batch_size,), dtype=np.int64)

# 4. Generate and compile stage function
stage_func = graph.compile_stage(graph.nodes)

# The generated code (conceptually) looks like:
"""
@numba.njit(parallel=True, nogil=True)
def stage_func_0(batch_indices, metadata, storage, mem_image, mem_label):
    for i in prange(len(batch_indices)):
        sample_id = batch_indices[i]
        
        # Image pipeline
        image_0_result = decoder_image_0(sample_id, metadata, storage, mem_image[i])
        image_1_result = transform_image_1(image_0_result, mem_image[i])  # Crop
        image_2_result = transform_image_2(image_1_result, mem_image[i])  # Flip
        image_3_result = transform_image_3(image_2_result, mem_image[i])  # Normalize
        
        # Label pipeline
        label_0_result = decoder_label_0(sample_id, metadata, storage, mem_label[i])
    
    return (mem_image, mem_label)
"""

# 5. At runtime, just call the compiled function
batch_indices = np.array([0, 1, 2, ...])
images, labels = stage_func(batch_indices, metadata, storage, image_buffer, label_buffer)

# No Python interpreter in the hot path!
```

## Why This Matters

The generated code has several key properties:

1.  **Zero Python overhead in loop**: The `for i in prange(...)` loop runs entirely in machine code.

2.  **Inlining**: Numba inlines the operation implementations, eliminating function call overhead.

3.  **Loop fusion**: Multiple operations that iterate over the same data can be fused by the compiler.

4.  **Vectorization**: SIMD instructions are used where possible.

5.  **Parallelization**: `prange` spawns threads without GIL contention (all code is `nogil`).

## Exercises

1.  **Print Generated AST**: Use `ast.dump()` to inspect the generated AST for a simple pipeline.

2.  **Add Conditional Node**: Implement a node that generates an `ast.If` for conditional execution.

3.  **Benchmark Fusion**: Compare the performance of separate Numba functions vs. a fused generated function.

4.  **Debug Mode**: Add a flag that generates readable Python source code (not AST) for debugging.
