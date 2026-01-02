# The FFCV Compiler: AST Generation and JIT Linking

The "Compiler" in FFCV is not a byte-code compiler like `gcc`. Instead, it is a **Python Source Code Generator** that uses `ast` (Abstract Syntax Tree) manipulation to build a highly optimized function at runtime, which is then compiled by Numba.

## The Problem: Python Overhead in Loops

A naive pipeline looks like this:

```python
# Naive Pipeline
for batch in dataset:                 # Python Loop
    for sample in batch:              # Python Loop
        img = decode(sample)          # Python Call -> C++
        img = resize(img)             # Python Call -> C++
        img = augment(img)            # Python Call -> C++
```

Even if `decode`, `resize`, and `augment` are fast C++ functions, the *Python interpreter* has to run between each call to manage variables, reference counts, and the loop itself. This "glue code" overhead is substantial (10-30% of total time).

## The Solution: The Generated "Stage" Function

FFCV generates a single function that does **everything** for a whole batch, and then compiles that **entire function** to machine code.

```python
# Generated "Fused" Function (simplified)
@numba.njit(nogil=True)
def stage_code_0(batch_indices, metadata, storage_state, mem_0, mem_1):
    for i in range(len(batch_indices)):
        # Raw C-speed pointer arithmetic and function calls
        # NO Python interpreter involvement here!
        sample_ix = batch_indices[i]
        
        # Operation 1: Decode
        # Direct call to jitted function
        res_decode = op_decode_impl(metadata[sample_ix], storage_state, mem_0[i])
        
        # Operation 2: Resize
        res_resize = op_resize_impl(res_decode, mem_1[i])
        
        # ...
    return mem_1
```

## How `Graph.py` Builds This

The `Graph` class orchestrates this process.

### 1. The Dependency Graph (`Node` classes)

The pipeline is represented as a graph of `Node` objects:
*   `DecoderNode`: The start of a pipeline branch (reads from file).
*   `TransformNode`: Modifies data (parent -> child).
*   `RefNode`: References data from another pipeline branch (e.g., getting image size for label processing).

### 2. Memory Analysis (`collect_requirements`)

Before generating code, the graph is traversed to calculate strict memory requirements.

```python
# Graph.collect_requirements (Recursive)
state, allocation = operation.declare_state_and_memory(previous_state)
```

Every operation *must* declare exactly what efficient NumPy representation it outputs (shape, dtype) given its input. This allows FFCV to **pre-allocate all memory buffers** before training starts. **Zero dynamic allocation** during the loop!

### 3. Code Generation (`codegen_stage`)

This is the most complex part. FFCV uses the python `ast` module to construct the `stage_code_X` function programmatically.

#### A. The Template
It starts with an empty template AST:

```python
def stage_code_X(batch_indices, metadata, storage_state):
    pass
```

#### B. Argument Injection
It injects arguments for every memory buffer needed by the operations in this stage:

```python
# In codegen_stage:
base_code.args.args.extend([
    ast.arg(arg=f'memory_{node_id}') for node_id in stage
])
```

#### C. Body Construction
It iterates through the sorted nodes in the stage and appends their function calls to the body.

Each `Node` knows how to generate its own AST call:
```python
# Node.func_call_ast
# Generates: result_id = code_id(arg_id, memory_id)
tree = ast.parse(f"{self.result_id} = {pipeline_identifier}({self.arg_id}, {memory_identifier})")
```

#### D. Dynamic Linking (`exec`)
Once the AST is complete, it is compiled into a Python code object and executed in a local namespace to create the actual function object.

```python
module = ast.fix_missing_locations(ast.Module(body=[base_code]))
namespace = {
    'op_impl_1': op1.generate_code(), # The actual Numba implementation
    'op_impl_2': op2.generate_code(),
}
exec(compile(module, '', 'exec'), namespace)
final_func = namespace['stage_code_0']
```

### 4. Final Compilation

The `final_func` is now a pure Python function that calls other JIT-compiled functions. FFCV then runs `numba.njit(final_func)`. Numba inlines the inner calls, optimizes the loops, vectorizes where possible, and produces a single block of optimized machine code.

## Why This is "Cracked" Level Engineering

1.  **Metaprogramming**: It writes code that writes code.
2.  **Zero-Cost Abstraction**: The complex graph of Python objects evaporates completely at runtime, leaving only raw assembly.
3.  **Static Memory**: By resolving shapes at "compile time" (pipeline setup), it eliminates `malloc` from the hot path entirely.

## Creating Your Own Compiler Hooks

If you want to add a new "Type" of node or a custom execution flow (e.g., conditional execution based on data content):

1.  Subclass `Node`.
2.  Implement `func_call_ast` to return the specific AST for your control flow (e.g., an `ast.If` block).
3.  Register it in `Graph`.

This allows for incredibly complex pipelines (like branching logic) that still compile down to linear, fast code.
