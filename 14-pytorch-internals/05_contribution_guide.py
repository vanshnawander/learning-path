"""
05_contribution_guide.py - PyTorch Contribution Guide

This module prepares you to contribute to PyTorch.
It covers the practical aspects of making your first PR.

Key Resources:
- CONTRIBUTING.md in PyTorch repo
- PyTorch Dev Discuss: https://dev-discuss.pytorch.org/
- PyTorch Wiki: https://github.com/pytorch/pytorch/wiki

Run: python 05_contribution_guide.py
"""

import torch
import os
import sys
from pathlib import Path

# ============================================================================
# DEVELOPMENT ENVIRONMENT SETUP
# ============================================================================

def explain_dev_setup():
    """Explain how to set up PyTorch development environment."""
    print("\n" + "="*70)
    print(" DEVELOPMENT ENVIRONMENT SETUP")
    print(" Getting ready to build PyTorch from source")
    print("="*70)
    
    print("""
    PREREQUISITES:
    ─────────────────────────────────────────────────────────────────
    
    Linux (Recommended):
    • GCC 9+ or Clang 9+
    • CMake 3.18+
    • Python 3.8+
    • CUDA 11.8+ (for GPU support)
    • Ninja (recommended for faster builds)
    
    macOS:
    • Xcode Command Line Tools
    • CMake 3.18+
    • Python 3.8+
    
    Windows:
    • Visual Studio 2019+ with C++ workload
    • CMake 3.18+
    • Python 3.8+
    • CUDA (optional)
    
    CLONE REPOSITORY:
    ─────────────────────────────────────────────────────────────────
    
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    
    # Update submodules
    git submodule sync
    git submodule update --init --recursive
    
    INSTALL DEPENDENCIES:
    ─────────────────────────────────────────────────────────────────
    
    # Create conda environment (recommended)
    conda create -n pytorch-dev python=3.10
    conda activate pytorch-dev
    
    # Install dependencies
    conda install cmake ninja
    pip install -r requirements.txt
    
    # For development
    pip install -e . -v --no-build-isolation
    # OR for faster iteration (Python only)
    python setup.py develop
    
    BUILD OPTIONS:
    ─────────────────────────────────────────────────────────────────
    
    # CPU only (faster build)
    USE_CUDA=0 python setup.py develop
    
    # With CUDA
    USE_CUDA=1 python setup.py develop
    
    # Debug build
    DEBUG=1 python setup.py develop
    
    # Specify CUDA architecture
    TORCH_CUDA_ARCH_LIST="8.0" python setup.py develop
    
    # Faster rebuilds with ninja
    CMAKE_GENERATOR=Ninja python setup.py develop
    
    BUILD TIME EXPECTATIONS:
    ─────────────────────────────────────────────────────────────────
    
    First build:
    • CPU only: 30-60 minutes
    • With CUDA: 1-2 hours
    
    Incremental rebuild (after small change):
    • 1-5 minutes (with ninja + ccache)
    
    TIPS FOR FASTER BUILDS:
    ─────────────────────────────────────────────────────────────────
    
    1. Use ccache
       export USE_CCACHE=1
       
    2. Use ninja
       pip install ninja
       export CMAKE_GENERATOR=Ninja
    
    3. Build only what you need
       python setup.py develop --cmake-only
       cd build && ninja torch_cpu
    
    4. Use a fast linker (lld or mold)
       export CMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld"
    """)

# ============================================================================
# CODE NAVIGATION
# ============================================================================

def explain_code_navigation():
    """Explain how to navigate PyTorch codebase."""
    print("\n" + "="*70)
    print(" NAVIGATING THE CODEBASE")
    print(" Finding your way around 10+ million lines of code")
    print("="*70)
    
    print("""
    FINDING WHERE CODE LIVES:
    ─────────────────────────────────────────────────────────────────
    
    Q: Where is torch.add implemented?
    
    1. Start in Python (torch/__init__.py)
       → Dispatches to C++ via torch._C
    
    2. C++ binding (torch/csrc/...)
       → Generated code in build/
    
    3. ATen operator (aten/src/ATen/native/)
       → native_functions.yaml defines it
       → Implementation in BinaryOps.cpp
    
    4. Actual kernel
       → CPU: aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
       → CUDA: aten/src/ATen/native/cuda/BinaryOpsKernel.cu
    
    USEFUL SEARCH PATTERNS:
    ─────────────────────────────────────────────────────────────────
    
    # Find operator definition
    grep -r "func: add" aten/src/ATen/native/native_functions.yaml
    
    # Find implementation
    grep -r "TORCH_IMPL_FUNC(add_out)" aten/
    
    # Find Python binding
    grep -r "def add" torch/
    
    # Find test
    grep -r "def test_add" test/
    
    TOOLS FOR NAVIGATION:
    ─────────────────────────────────────────────────────────────────
    
    1. VS Code with C++ extension
       • Go to definition (F12)
       • Find references (Shift+F12)
       • Search symbols (Ctrl+T)
    
    2. clangd for C++ LSP
       • Better than Microsoft's C++ extension
       • Requires compile_commands.json
    
    3. GitHub code search
       • Use repo:pytorch/pytorch
       • Syntax highlighting
    
    READING GENERATED CODE:
    ─────────────────────────────────────────────────────────────────
    
    After building, check these directories:
    
    build/
    ├── aten/src/ATen/
    │   ├── Operators.h          # Operator declarations
    │   ├── RegisterCPU.cpp      # CPU kernel registration
    │   └── RegisterCUDA.cpp     # CUDA kernel registration
    └── torch/csrc/autograd/
        └── generated/
            ├── Functions.h       # Autograd functions
            └── python_torch_functions.cpp  # Python bindings
    
    torch/
    └── _C/
        └── _VariableFunctions.pyi  # Python type stubs
    """)

# ============================================================================
# ADDING A NEW OPERATOR
# ============================================================================

def explain_adding_operator():
    """Step-by-step guide to adding a new operator."""
    print("\n" + "="*70)
    print(" ADDING A NEW OPERATOR")
    print(" Step-by-step guide")
    print("="*70)
    
    print("""
    STEP 1: DEFINE IN NATIVE_FUNCTIONS.YAML
    ─────────────────────────────────────────────────────────────────
    
    File: aten/src/ATen/native/native_functions.yaml
    
    - func: my_op(Tensor self, Tensor other, float alpha=1.0) -> Tensor
      variants: function, method  # torch.my_op and tensor.my_op
      dispatch:
        CPU: my_op_cpu
        CUDA: my_op_cuda
    
    Schema syntax:
    - Tensor       → Required tensor
    - Tensor?      → Optional tensor  
    - Tensor(a!)   → Mutated tensor (returns aliased)
    - Scalar       → Python scalar (int/float)
    - int[]        → List of ints
    - float=1.0    → Default value
    
    STEP 2: IMPLEMENT CPU KERNEL
    ─────────────────────────────────────────────────────────────────
    
    File: aten/src/ATen/native/MyOp.cpp
    
    #include <ATen/ATen.h>
    #include <ATen/Dispatch.h>
    
    namespace at::native {
    
    Tensor my_op_cpu(const Tensor& self, const Tensor& other, double alpha) {
        // 1. Input validation
        TORCH_CHECK(self.device().is_cpu(), "Expected CPU tensor");
        TORCH_CHECK(self.sizes() == other.sizes(), "Size mismatch");
        
        // 2. Allocate output
        Tensor result = at::empty_like(self);
        
        // 3. Dispatch on dtype
        AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "my_op_cpu", [&] {
            // scalar_t is now the actual type (float/double)
            auto self_data = self.data_ptr<scalar_t>();
            auto other_data = other.data_ptr<scalar_t>();
            auto result_data = result.data_ptr<scalar_t>();
            
            int64_t n = self.numel();
            for (int64_t i = 0; i < n; i++) {
                result_data[i] = self_data[i] + alpha * other_data[i];
            }
        });
        
        return result;
    }
    
    } // namespace at::native
    
    STEP 3: IMPLEMENT CUDA KERNEL (if needed)
    ─────────────────────────────────────────────────────────────────
    
    File: aten/src/ATen/native/cuda/MyOp.cu
    
    #include <ATen/ATen.h>
    #include <ATen/cuda/CUDAContext.h>
    
    namespace at::native {
    
    template <typename scalar_t>
    __global__ void my_op_kernel(
        const scalar_t* self,
        const scalar_t* other,
        scalar_t* result,
        scalar_t alpha,
        int64_t n
    ) {
        int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = self[idx] + alpha * other[idx];
        }
    }
    
    Tensor my_op_cuda(const Tensor& self, const Tensor& other, double alpha) {
        Tensor result = at::empty_like(self);
        int64_t n = self.numel();
        
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "my_op_cuda", [&] {
            my_op_kernel<<<blocks, threads>>>(
                self.data_ptr<scalar_t>(),
                other.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                static_cast<scalar_t>(alpha),
                n
            );
        });
        
        return result;
    }
    
    } // namespace at::native
    
    STEP 4: ADD DERIVATIVE (for autograd)
    ─────────────────────────────────────────────────────────────────
    
    File: tools/autograd/derivatives.yaml
    
    - name: my_op(Tensor self, Tensor other, float alpha=1.0) -> Tensor
      self: grad
      other: grad * alpha
    
    STEP 5: ADD TESTS
    ─────────────────────────────────────────────────────────────────
    
    File: test/test_ops.py (or appropriate test file)
    
    def test_my_op(self):
        x = torch.randn(10)
        y = torch.randn(10)
        
        # Test forward
        result = torch.my_op(x, y, alpha=0.5)
        expected = x + 0.5 * y
        self.assertEqual(result, expected)
        
        # Test backward
        x.requires_grad_(True)
        y.requires_grad_(True)
        result = torch.my_op(x, y, alpha=0.5)
        result.sum().backward()
        
        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, torch.full_like(y, 0.5))
    
    STEP 6: BUILD AND TEST
    ─────────────────────────────────────────────────────────────────
    
    # Rebuild
    python setup.py develop
    
    # Run your test
    python -m pytest test/test_ops.py::TestOps::test_my_op -v
    
    # Run linter
    lintrunner
    """)

# ============================================================================
# TESTING
# ============================================================================

def explain_testing():
    """Explain PyTorch testing practices."""
    print("\n" + "="*70)
    print(" TESTING YOUR CHANGES")
    print(" How to write and run tests")
    print("="*70)
    
    print("""
    TEST ORGANIZATION:
    ─────────────────────────────────────────────────────────────────
    
    test/
    ├── test_torch.py        # Core tensor operations
    ├── test_nn.py           # Neural network modules
    ├── test_autograd.py     # Autograd system
    ├── test_ops.py          # Operator tests
    ├── test_cuda.py         # CUDA-specific tests
    └── ...
    
    RUNNING TESTS:
    ─────────────────────────────────────────────────────────────────
    
    # Run specific test file
    python -m pytest test/test_torch.py -v
    
    # Run specific test
    python -m pytest test/test_torch.py::TestTorchDeviceType::test_add -v
    
    # Run with CUDA
    python -m pytest test/test_cuda.py -v
    
    # Run in parallel
    python -m pytest test/test_torch.py -n auto
    
    # Show slow tests
    python -m pytest --durations=10
    
    WRITING TESTS:
    ─────────────────────────────────────────────────────────────────
    
    import torch
    from torch.testing._internal.common_utils import TestCase, run_tests
    from torch.testing._internal.common_device_type import instantiate_device_type_tests
    
    class TestMyOp(TestCase):
        def test_my_op_basic(self):
            x = torch.randn(10)
            y = torch.randn(10)
            result = torch.my_op(x, y)
            expected = x + y
            self.assertEqual(result, expected)
        
        def test_my_op_edge_cases(self):
            # Empty tensor
            x = torch.tensor([])
            y = torch.tensor([])
            result = torch.my_op(x, y)
            self.assertEqual(result.numel(), 0)
            
            # Single element
            x = torch.tensor([1.0])
            y = torch.tensor([2.0])
            result = torch.my_op(x, y)
            self.assertEqual(result, torch.tensor([3.0]))
    
    # Test on multiple devices
    class TestMyOpDevice(TestCase):
        def test_my_op(self, device):
            x = torch.randn(10, device=device)
            y = torch.randn(10, device=device)
            result = torch.my_op(x, y)
            self.assertEqual(result.device, x.device)
    
    instantiate_device_type_tests(TestMyOpDevice, globals())
    
    if __name__ == '__main__':
        run_tests()
    
    IMPORTANT TEST UTILITIES:
    ─────────────────────────────────────────────────────────────────
    
    # Assert tensor equality
    self.assertEqual(a, b)  # Uses tolerance for floats
    
    # Assert tensors close
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    
    # Skip test conditionally
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_op(self):
        ...
    
    # Test multiple dtypes
    @dtypes(torch.float32, torch.float64)
    def test_my_op_dtypes(self, dtype):
        x = torch.randn(10, dtype=dtype)
        ...
    """)

# ============================================================================
# PR WORKFLOW
# ============================================================================

def explain_pr_workflow():
    """Explain the pull request workflow."""
    print("\n" + "="*70)
    print(" PULL REQUEST WORKFLOW")
    print(" From code to merge")
    print("="*70)
    
    print("""
    BEFORE SUBMITTING:
    ─────────────────────────────────────────────────────────────────
    
    1. Run linter
       lintrunner
       # Or fix automatically
       lintrunner -a
    
    2. Run relevant tests
       python -m pytest test/test_ops.py -v
    
    3. Check type annotations (if changed Python)
       mypy torch/
    
    4. Write/update documentation
    
    CREATING THE PR:
    ─────────────────────────────────────────────────────────────────
    
    1. Fork pytorch/pytorch on GitHub
    
    2. Create branch
       git checkout -b my-feature
    
    3. Make changes and commit
       git add .
       git commit -m "[module] Description of change"
       
       Commit message format:
       [area] Short description
       
       Longer description if needed.
       
       Fixes #issue_number (if applicable)
    
    4. Push and create PR
       git push origin my-feature
       # Go to GitHub and create PR
    
    PR DESCRIPTION TEMPLATE:
    ─────────────────────────────────────────────────────────────────
    
    ## Summary
    Brief description of what this PR does.
    
    ## Motivation
    Why is this change needed?
    
    ## Changes
    - List of specific changes
    - Another change
    
    ## Test Plan
    How was this tested?
    
    ## Related Issues
    Fixes #1234
    
    CI CHECKS:
    ─────────────────────────────────────────────────────────────────
    
    CI will run automatically:
    
    • Linux build + tests
    • Windows build + tests  
    • macOS build + tests
    • CUDA tests
    • Linting
    • Type checking
    • Documentation build
    
    CI takes 2-4 hours. Common failures:
    
    • Linting: Run lintrunner -a locally
    • Type errors: Fix mypy issues
    • Test failures: Debug and fix
    • Build failures: Check C++ errors
    
    REVIEW PROCESS:
    ─────────────────────────────────────────────────────────────────
    
    1. Automated checks must pass
    2. At least one maintainer approval required
    3. Address review feedback
    4. May need multiple rounds
    5. Maintainer merges when ready
    
    TIPS FOR SMOOTH REVIEW:
    ─────────────────────────────────────────────────────────────────
    
    • Keep PRs focused (one logical change)
    • Write clear description
    • Add tests
    • Respond to feedback promptly
    • Be patient - maintainers are busy
    • Ask questions if unclear
    """)

# ============================================================================
# DEBUGGING
# ============================================================================

def explain_debugging():
    """Explain debugging techniques for PyTorch development."""
    print("\n" + "="*70)
    print(" DEBUGGING PYTORCH")
    print(" Finding and fixing bugs")
    print("="*70)
    
    print("""
    PYTHON DEBUGGING:
    ─────────────────────────────────────────────────────────────────
    
    # Use pdb
    import pdb; pdb.set_trace()
    
    # Or breakpoint() in Python 3.7+
    breakpoint()
    
    # Print tensor info
    print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    
    # Check for NaN/Inf
    torch.isnan(tensor).any()
    torch.isinf(tensor).any()
    
    C++ DEBUGGING:
    ─────────────────────────────────────────────────────────────────
    
    # Build with debug symbols
    DEBUG=1 python setup.py develop
    
    # Use gdb
    gdb python
    (gdb) run script.py
    (gdb) bt  # backtrace on crash
    
    # Use lldb (macOS)
    lldb python
    (lldb) run script.py
    
    # Print from C++
    std::cout << tensor << std::endl;
    TORCH_WARN("Debug: ", value);
    
    CUDA DEBUGGING:
    ─────────────────────────────────────────────────────────────────
    
    # Synchronous CUDA (catch errors immediately)
    CUDA_LAUNCH_BLOCKING=1 python script.py
    
    # Check for CUDA errors
    torch.cuda.synchronize()
    
    # Use compute-sanitizer
    compute-sanitizer --tool memcheck python script.py
    
    # Print CUDA info
    print(torch.cuda.memory_summary())
    
    COMMON ISSUES:
    ─────────────────────────────────────────────────────────────────
    
    1. "CUDA error: device-side assert triggered"
       → Index out of bounds in kernel
       → Use CUDA_LAUNCH_BLOCKING=1 to find location
    
    2. "RuntimeError: expected scalar type Float but found Double"
       → Dtype mismatch
       → Use .float() or .double() to convert
    
    3. "RuntimeError: Expected all tensors to be on the same device"
       → Move tensors to same device
       → tensor.to(device)
    
    4. "Gradients are None"
       → Check requires_grad=True
       → Use retain_grad() for non-leaf tensors
    
    5. Slow performance
       → Profile with torch.profiler
       → Check for CPU-GPU sync points
       → Verify CUDA kernels are being used
    
    USEFUL ENVIRONMENT VARIABLES:
    ─────────────────────────────────────────────────────────────────
    
    # Debug autograd
    TORCH_SHOW_CPP_STACKTRACES=1
    
    # Debug dispatcher
    TORCH_SHOW_DISPATCH_TRACE=1
    
    # Debug CUDA
    CUDA_LAUNCH_BLOCKING=1
    
    # Verbose compilation
    TORCH_COMPILE_DEBUG=1
    """)

# ============================================================================
# SUMMARY
# ============================================================================

def print_contribution_summary():
    """Print contribution summary."""
    print("\n" + "="*70)
    print(" CONTRIBUTION GUIDE SUMMARY")
    print("="*70)
    
    print("""
    CONTRIBUTION CHECKLIST:
    
    □ Set up development environment
    □ Find an issue to work on
    □ Create a branch
    □ Make changes
    □ Run linter (lintrunner)
    □ Add/update tests
    □ Run tests locally
    □ Write clear commit message
    □ Create PR with description
    □ Address CI failures
    □ Respond to review feedback
    □ Get approval and merge!
    
    RESOURCES:
    ─────────────────────────────────────────────────────────────────
    
    • CONTRIBUTING.md: Setup and guidelines
    • PyTorch Wiki: Detailed documentation
    • Dev Discuss Forum: Ask questions
    • Office Hours: Talk to maintainers
    • Good First Issues: Starter tasks
    
    WHERE TO GET HELP:
    ─────────────────────────────────────────────────────────────────
    
    • GitHub Issues: Bug reports, feature requests
    • Dev Discuss: Development questions
    • Stack Overflow: Usage questions
    • Discord: Real-time chat
    
    REMEMBER:
    ─────────────────────────────────────────────────────────────────
    
    • Start small - documentation, tests, simple fixes
    • Read existing code to learn patterns
    • Ask questions when stuck
    • Be patient with reviews
    • Every contribution matters!
    
    You now have the foundation to contribute to PyTorch!
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH CONTRIBUTION GUIDE ".center(68) + "║")
    print("║" + " Your path to contributing ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    
    explain_dev_setup()
    explain_code_navigation()
    explain_adding_operator()
    explain_testing()
    explain_pr_workflow()
    explain_debugging()
    print_contribution_summary()
    
    print("\n" + "="*70)
    print(" Now go make your first contribution!")
    print("="*70)
