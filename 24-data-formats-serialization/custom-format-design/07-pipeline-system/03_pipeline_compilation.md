# Pipeline Compilation: AST-Based Code Generation

## The Problem: Python Overhead in Pipelines

Even if each individual transform is JIT-compiled, calling them sequentially from Python incurs overhead:

```python
# Each arrow is a Python function call with overhead
result = decode(data)        # ~1µs overhead
result = resize(result)      # ~1µs overhead
result = normalize(result)   # ~1µs overhead
result = to_tensor(result)   # ~1µs overhead
# Total overhead: ~4µs per sample × 1000 samples/batch = 4ms per batch!
```

This "glue code" overhead can be 10-30% of total training time.

## FFCV's Solution: AST Compilation

FFCV solves this by **generating a single Python function at runtime** that contains all operations, then JIT-compiling that entire function.

```python
# FFCV generates something like this:
@numba.njit
def compiled_pipeline(indices, metadata, storage, mem_0, mem_1, mem_2):
    for i in range(len(indices)):
        # Operation 1: Decode
        decode_impl(metadata[indices[i]], storage, mem_0[i])
        
        # Operation 2: Resize
        resize_impl(mem_0[i], mem_1[i])
        
        # Operation 3: Normalize
        normalize_impl(mem_1[i], mem_2[i])
    
    return mem_2
```

This is ONE function call with ZERO Python overhead inside the loop.

## How AST Generation Works

### Step 1: Build the Pipeline Graph

```python
from collections import defaultdict
from typing import List, Dict

class PipelineGraph:
    """
    Represents the pipeline as a directed graph of operations.
    """
    
    def __init__(self):
        self.nodes: List[OperationNode] = []
        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        self.root_nodes: List[int] = []
        self.leaf_nodes: List[int] = []
    
    def add_operation(self, op: Operation, parent_id: int = None):
        """Add an operation to the graph."""
        node_id = len(self.nodes)
        node = OperationNode(node_id, op)
        self.nodes.append(node)
        
        if parent_id is not None:
            self.adjacency[parent_id].append(node_id)
        else:
            self.root_nodes.append(node_id)
        
        return node_id
    
    def build_from_spec(self, spec: dict):
        """
        Build graph from a pipeline specification.
        
        spec = {
            'image': [
                RGBImageDecoder(),
                RandomResizedCrop((224, 224)),
                RandomHorizontalFlip(0.5),
                Normalize(),
                ToTensor(),
            ],
            'label': [
                IntDecoder(),
            ]
        }
        """
        for field_name, ops in spec.items():
            parent = None
            for op in ops:
                parent = self.add_operation(op, parent)
            self.leaf_nodes.append(parent)
```

### Step 2: Collect Requirements

Before generating code, we need to know each operation's state and memory needs.

```python
class PipelineCompiler:
    """
    Compiles a pipeline graph into optimized code.
    """
    
    def __init__(self, graph: PipelineGraph, metadata, memory_read):
        self.graph = graph
        self.metadata = metadata
        self.memory_read = memory_read
        
        self.states: Dict[int, State] = {}
        self.allocations: Dict[int, AllocationQuery] = {}
        self.funcs: Dict[int, Callable] = {}
    
    def collect_requirements(self, initial_state: State):
        """
        Traverse the graph and collect state/memory requirements.
        """
        # Process nodes in topological order
        for node_id in self._topo_sort():
            node = self.graph.nodes[node_id]
            op = node.operation
            
            # Get parent state (or initial state for roots)
            if node_id in self.graph.root_nodes:
                prev_state = initial_state
            else:
                parent_id = self._get_parent(node_id)
                prev_state = self.states[parent_id]
            
            # Pass globals to operation
            op.accept_globals(self.metadata, self.memory_read)
            
            # Declare state and memory
            new_state, allocation = op.declare_state_and_memory(prev_state)
            
            self.states[node_id] = new_state
            self.allocations[node_id] = allocation
    
    def _topo_sort(self) -> List[int]:
        """Topological sort of the graph."""
        visited = set()
        result = []
        
        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for child in self.graph.adjacency[node_id]:
                dfs(child)
            result.append(node_id)
        
        for root in self.graph.root_nodes:
            dfs(root)
        
        return result[::-1]  # Reverse for correct order
```

### Step 3: Generate Code with AST

This is the key innovation. FFCV uses Python's `ast` module to programmatically construct Python source code.

```python
import ast

class ASTCodeGenerator:
    """
    Generates Python AST for the compiled pipeline.
    """
    
    def __init__(self, graph: PipelineGraph, funcs: Dict[int, Callable]):
        self.graph = graph
        self.funcs = funcs
    
    def generate_stage_function(self, node_ids: List[int], stage_index: int) -> str:
        """
        Generate a function that executes a stage of the pipeline.
        """
        func_name = f"stage_{stage_index}"
        
        # Build argument list
        args = [
            'batch_indices',
            'metadata',
            'storage_state',
        ]
        for node_id in node_ids:
            args.append(f'memory_{node_id}')
        
        # Build function body as AST
        body = []
        
        for node_id in node_ids:
            # Generate: result_{node_id} = code_{node_id}(input, memory_{node_id})
            
            if node_id in self.graph.root_nodes:
                # Decoder: input is batch_indices + metadata + storage
                input_arg = 'batch_indices'
                extra_args = ['metadata', 'storage_state']
            else:
                # Transform: input is result from parent
                parent_id = self._get_parent(node_id)
                input_arg = f'result_{parent_id}'
                extra_args = []
            
            call = ast.Call(
                func=ast.Name(id=f'code_{node_id}', ctx=ast.Load()),
                args=[
                    ast.Name(id=input_arg, ctx=ast.Load()),
                    ast.Name(id=f'memory_{node_id}', ctx=ast.Load()),
                ] + [ast.Name(id=a, ctx=ast.Load()) for a in extra_args],
                keywords=[],
            )
            
            assign = ast.Assign(
                targets=[ast.Name(id=f'result_{node_id}', ctx=ast.Store())],
                value=call,
            )
            
            body.append(assign)
        
        # Add return statement
        return_nodes = [n for n in node_ids if n in self.graph.leaf_nodes]
        if return_nodes:
            return_tuple = ast.Tuple(
                elts=[ast.Name(id=f'result_{n}', ctx=ast.Load()) for n in return_nodes],
                ctx=ast.Load(),
            )
            body.append(ast.Return(value=return_tuple))
        
        # Build function definition
        func_def = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=a) for a in args],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
        )
        
        return func_def
    
    def compile_function(self, func_def: ast.FunctionDef, namespace: dict) -> Callable:
        """
        Compile an AST function definition and execute it.
        """
        # Wrap in a module
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)
        
        # Compile to code object
        code = compile(module, '<generated>', 'exec')
        
        # Execute to get the function
        exec(code, namespace)
        
        return namespace[func_def.name]
```

### Step 4: JIT Compile the Generated Function

The final step is to JIT-compile the generated Python function:

```python
from numba import njit

class PipelineCompiler:
    # ... (continued)
    
    def compile(self, batch_size: int):
        """
        Compile the entire pipeline.
        """
        # Step 1: Collect requirements
        self.collect_requirements(INITIAL_STATE)
        
        # Step 2: Generate code for each operation
        for node_id, node in enumerate(self.graph.nodes):
            self.funcs[node_id] = node.operation.generate_code()
        
        # Step 3: Group operations into JIT stages
        stages = self._group_into_stages()
        
        # Step 4: Generate and compile each stage
        compiled_stages = []
        ast_gen = ASTCodeGenerator(self.graph, self.funcs)
        
        for stage_idx, node_ids in enumerate(stages):
            # Build namespace with compiled operation functions
            namespace = {}
            for node_id in node_ids:
                func = self.funcs[node_id]
                # JIT compile individual functions
                if self.states[node_id].jit_mode:
                    func = njit(nogil=True, parallel=True)(func)
                namespace[f'code_{node_id}'] = func
            
            # Generate stage function AST
            stage_ast = ast_gen.generate_stage_function(node_ids, stage_idx)
            
            # Compile the stage function
            stage_func = ast_gen.compile_function(stage_ast, namespace)
            
            # JIT compile the stage function if all ops are JIT-able
            if all(self.states[n].jit_mode for n in node_ids):
                stage_func = njit(nogil=True)(stage_func)
            
            compiled_stages.append(stage_func)
        
        return CompiledPipeline(compiled_stages, self.allocations, batch_size)
    
    def _group_into_stages(self) -> List[List[int]]:
        """
        Group operations into stages based on JIT compatibility.
        
        Operations that can be JIT compiled are grouped together.
        Non-JIT operations break the grouping.
        """
        stages = []
        current_stage = []
        current_jit_mode = None
        
        for node_id in self._topo_sort():
            node_jit_mode = self.states[node_id].jit_mode
            
            if current_jit_mode is None or node_jit_mode == current_jit_mode:
                current_stage.append(node_id)
                current_jit_mode = node_jit_mode
            else:
                # JIT mode changed, start new stage
                if current_stage:
                    stages.append(current_stage)
                current_stage = [node_id]
                current_jit_mode = node_jit_mode
        
        if current_stage:
            stages.append(current_stage)
        
        return stages


class CompiledPipeline:
    """
    A compiled, ready-to-run pipeline.
    """
    
    def __init__(self, stages: List[Callable], allocations: Dict[int, AllocationQuery], batch_size: int):
        self.stages = stages
        self.allocations = allocations
        self.batch_size = batch_size
        
        # Pre-allocate memory buffers
        self.memory_pools = self._allocate_memory()
    
    def _allocate_memory(self) -> Dict[int, np.ndarray]:
        pools = {}
        for node_id, alloc in self.allocations.items():
            if alloc is not None:
                shape = (self.batch_size,) + alloc.shape
                pools[node_id] = np.empty(shape, dtype=alloc.dtype)
        return pools
    
    def __call__(self, batch_indices: np.ndarray, metadata: np.ndarray, storage_state: tuple):
        """
        Execute the compiled pipeline.
        """
        # Build memory arguments for each stage
        # (Implementation depends on stage structure)
        
        result = None
        for stage in self.stages:
            # Call each stage, passing appropriate memory buffers
            result = stage(
                batch_indices,
                metadata,
                storage_state,
                # Memory buffers for this stage's operations...
            )
        
        return result
```

## The Generated Code (Visualization)

For a simple image pipeline:
```
Decode → Resize → Normalize → ToTensor
```

FFCV generates something like:

```python
# Auto-generated by PipelineCompiler

def stage_0(batch_indices, metadata, storage_state, memory_0, memory_1, memory_2, memory_3):
    # Operation 0: Decode
    result_0 = code_0(batch_indices, memory_0, metadata, storage_state)
    
    # Operation 1: Resize
    result_1 = code_1(result_0, memory_1)
    
    # Operation 2: Normalize
    result_2 = code_2(result_1, memory_2)
    
    # Operation 3: ToTensor
    result_3 = code_3(result_2, memory_3)
    
    return result_3
```

Which is then JIT compiled to:

```python
@numba.njit(nogil=True, parallel=True)
def stage_0(batch_indices, metadata, storage_state, memory_0, memory_1, memory_2, memory_3):
    # All operations inlined, all loops fused where possible
    # No Python overhead whatsoever
    ...
```

## Benefits of AST Compilation

1.  **Zero Python overhead**: The compiled function has no Python interpreter involvement.
2.  **Single function call**: One call per batch instead of N calls per sample × M operations.
3.  **Loop fusion**: The compiler can fuse loops across operations.
4.  **Memory optimization**: Intermediate buffers are known at compile time.
5.  **Parallelization**: The entire batch can be parallelized with `prange`.

## Advanced: Handling Non-JIT Operations

Some operations can't be JIT compiled (e.g., those using PIL, TurboJPEG, or SciPy). FFCV handles this by creating "hybrid" pipelines:

```
Stage 0 (Non-JIT): Decode (calls C library)
Stage 1 (JIT):     Resize → Normalize → ToTensor
```

Between stages, data is passed through Python, but within a stage, everything is JIT-compiled.

## Exercises

1.  **Print Generated Code**: Use Python's `astor` library to pretty-print the generated AST.

2.  **Measure Compilation Time**: Profile how long it takes to compile a pipeline vs. run it.

3.  **Add Conditional Execution**: Extend the AST generator to support conditional operations (e.g., apply augmentation only 50% of the time).

4.  **Benchmark**: Compare the speed of the AST-compiled pipeline vs. a naive loop calling each operation.
