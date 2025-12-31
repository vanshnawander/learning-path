# PyTorch Autograd

Automatic differentiation engine.

## How It Works

### Forward Pass
- Operations build a DAG (Directed Acyclic Graph)
- Each operation has a backward function
- Tensors track their gradient function

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y + 1
# Graph: x -> mul -> y -> add -> z
```

### Backward Pass
- Traverse graph in reverse topological order
- Chain rule: multiply gradients
- Accumulate gradients at leaves

```python
z.backward()
# dz/dz = 1
# dz/dy = 1
# dz/dx = dz/dy * dy/dx = 1 * 2 = 2
print(x.grad)  # tensor([2.])
```

## Key Components

### grad_fn
Each tensor with gradients has a `grad_fn`:
```python
y = x * 2
print(y.grad_fn)  # <MulBackward0>
```

### AccumulateGrad
Special function at leaves to accumulate gradients.

### Hooks
Customize gradient computation:
```python
x.register_hook(lambda grad: grad * 2)
```

## Source Code Locations
- `torch/autograd/` - Python API
- `torch/csrc/autograd/` - C++ engine
- `torch/csrc/autograd/engine.cpp` - Core engine

## Gradient Checkpointing
Trade compute for memory:
```python
from torch.utils.checkpoint import checkpoint
y = checkpoint(expensive_fn, x)
```

Doesn't save activations, recomputes in backward.
