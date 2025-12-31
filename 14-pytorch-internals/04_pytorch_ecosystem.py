"""
04_pytorch_ecosystem.py - PyTorch Ecosystem Overview

PyTorch is more than just the core library.
Understanding the ecosystem is essential for production deployment.

PyTorch Ecosystem:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYTORCH CORE                                      │
│                 torch, torch.nn, torch.autograd                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ TRAINING & OPTIMIZATION    │ DEPLOYMENT & INFERENCE                        │
│ ─────────────────────────  │ ──────────────────────────                    │
│ torch.distributed          │ TorchServe (serving)                          │
│ torch.compile              │ ExecuTorch (mobile/edge)                      │
│ FSDP, DDP                  │ TorchScript (optimization)                    │
│ AMP (mixed precision)      │ torch.export (export)                         │
│ torch.profiler             │ ONNX export                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ DOMAIN LIBRARIES           │ ECOSYSTEM TOOLS                               │
│ ─────────────────────────  │ ──────────────────────────                    │
│ torchvision               │ TensorBoard                                   │
│ torchaudio                │ Captum (interpretability)                     │
│ torchtext                 │ TorchData (data loading)                      │
│ PyTorch Geometric         │ functorch (transforms)                        │
└─────────────────────────────────────────────────────────────────────────────┘

Run: python 04_pytorch_ecosystem.py
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional

# ============================================================================
# EXECUTORCH - MOBILE/EDGE DEPLOYMENT
# ============================================================================

def explain_executorch():
    """Explain ExecuTorch for mobile and edge deployment."""
    print("\n" + "="*70)
    print(" EXECUTORCH: MOBILE AND EDGE DEPLOYMENT")
    print(" PyTorch's next-gen on-device inference")
    print("="*70)
    
    print("""
    WHAT IS EXECUTORCH?
    ─────────────────────────────────────────────────────────────────
    
    ExecuTorch is PyTorch's solution for deploying models to:
    - Mobile devices (iOS, Android)
    - Edge devices (IoT, embedded)
    - Wearables
    - AR/VR devices
    
    Key goals:
    - Minimal binary size
    - Low memory footprint
    - High performance
    - Easy integration
    
    EXECUTORCH ARCHITECTURE:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        PyTorch Model                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                      torch.export()                                     │
    │                    (Export to ExportedProgram)                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                    ExecuTorch Compiler                                  │
    │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
    │  │ Edge Dialect │ │ Quantization │ │ Optimization │                    │
    │  │ Lowering     │ │ Pass         │ │ Passes       │                    │
    │  └──────────────┘ └──────────────┘ └──────────────┘                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                      .pte File (Portable)                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                    ExecuTorch Runtime                                   │
    │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
    │  │ CPU Backend  │ │ XNNPACK      │ │ Custom       │                    │
    │  │              │ │ (optimized)  │ │ Delegate     │                    │
    │  └──────────────┘ └──────────────┘ └──────────────┘                    │
    └─────────────────────────────────────────────────────────────────────────┘
    
    EXPORT WORKFLOW:
    ─────────────────────────────────────────────────────────────────
    
    # 1. Define your model
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x).relu()
    
    model = MyModel()
    
    # 2. Export using torch.export
    example_input = (torch.randn(1, 10),)
    exported = torch.export.export(model, example_input)
    
    # 3. Lower to ExecuTorch (requires executorch package)
    # from executorch.exir import to_edge
    # edge_program = to_edge(exported)
    # et_program = edge_program.to_executorch()
    # 
    # # 4. Save for deployment
    # with open("model.pte", "wb") as f:
    #     f.write(et_program.buffer)
    
    EXECUTORCH BACKENDS:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────┬───────────────────────────────────────────────────────┐
    │ Backend         │ Description                                           │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ Portable        │ Pure C++ reference implementation                     │
    │ XNNPACK         │ Optimized CPU kernels (ARM, x86)                     │
    │ Vulkan          │ GPU via Vulkan API                                   │
    │ CoreML          │ Apple Neural Engine (iOS/macOS)                      │
    │ MPS             │ Metal Performance Shaders                            │
    │ Qualcomm QNN    │ Qualcomm Hexagon DSP/NPU                            │
    │ Custom          │ Your own delegate!                                   │
    └─────────────────┴───────────────────────────────────────────────────────┘
    
    WHY NOT JUST USE TORCHSCRIPT?
    ─────────────────────────────────────────────────────────────────
    
    TorchScript:
    - Larger runtime (~30MB+)
    - Full Python semantics
    - Flexible but heavy
    
    ExecuTorch:
    - Tiny runtime (~100KB-1MB)
    - Ahead-of-time compiled
    - Optimized for inference
    - Better for resource-constrained devices
    """)

# ============================================================================
# TORCH.EXPORT - THE NEW EXPORT SYSTEM
# ============================================================================

def explain_torch_export():
    """Explain torch.export for model export."""
    print("\n" + "="*70)
    print(" TORCH.EXPORT: SOUNDLY CAPTURING PYTORCH PROGRAMS")
    print(" The foundation for ExecuTorch and AOT compilation")
    print("="*70)
    
    print("""
    WHAT IS TORCH.EXPORT?
    ─────────────────────────────────────────────────────────────────
    
    torch.export captures a PyTorch model as an ExportedProgram:
    - Sound: Captures EXACTLY what the model does
    - Complete: Handles dynamic shapes
    - Standardized: Single export format
    
    KEY DIFFERENCE FROM TORCHSCRIPT:
    ─────────────────────────────────────────────────────────────────
    
    TorchScript (torch.jit.trace/script):
    - May silently produce wrong results
    - Tracing loses control flow
    - Scripting requires TorchScript-compatible code
    
    torch.export:
    - Fails loudly if can't capture exactly
    - Preserves control flow (with constraints)
    - Works with more Python code
    
    EXPORT EXAMPLE:
    ─────────────────────────────────────────────────────────────────
    """)
    
    # Demo torch.export
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
        
        def forward(self, x):
            return torch.relu(self.fc(x))
    
    model = SimpleModel()
    example_input = (torch.randn(2, 10),)
    
    print(" Exporting simple model:")
    print("-" * 50)
    
    try:
        exported = torch.export.export(model, example_input)
        print(f" ✓ Export successful!")
        print(f" ExportedProgram type: {type(exported)}")
        
        # Show the graph
        print(f"\n Graph module:")
        print(f" {exported.graph_module}")
        
    except Exception as e:
        print(f" Export failed: {e}")
    
    print("""
    
    DYNAMIC SHAPES:
    ─────────────────────────────────────────────────────────────────
    
    Export with dynamic batch size:
    
    from torch.export import Dim
    
    batch = Dim("batch", min=1, max=32)
    dynamic_shapes = {"x": {0: batch}}
    
    exported = torch.export.export(
        model, 
        example_input,
        dynamic_shapes=dynamic_shapes
    )
    
    # Now works with any batch size 1-32
    
    CONSTRAINTS:
    ─────────────────────────────────────────────────────────────────
    
    torch.export captures constraints (guards):
    - Shape relationships: batch_size > 0
    - Data-dependent control flow captured symbolically
    
    If constraints violated at runtime → error
    (Unlike tracing which silently produces wrong output)
    """)

# ============================================================================
# TORCHSERVE - MODEL SERVING
# ============================================================================

def explain_torchserve():
    """Explain TorchServe for model serving."""
    print("\n" + "="*70)
    print(" TORCHSERVE: PRODUCTION MODEL SERVING")
    print(" Scalable inference serving for PyTorch models")
    print("="*70)
    
    print("""
    WHAT IS TORCHSERVE?
    ─────────────────────────────────────────────────────────────────
    
    TorchServe is a performant, flexible inference server:
    - Multi-model serving
    - Batch inference
    - Model versioning
    - Metrics and logging
    - REST & gRPC APIs
    
    TORCHSERVE ARCHITECTURE:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TorchServe                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Frontend (Java)          │ Backend (Python)                            │
    │ ─────────────────────    │ ──────────────────────────                  │
    │ • REST API               │ • Model loading                             │
    │ • gRPC API               │ • Inference workers                         │
    │ • Request batching       │ • Custom handlers                           │
    │ • Load balancing         │ • Pre/post processing                       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    DEPLOYMENT WORKFLOW:
    ─────────────────────────────────────────────────────────────────
    
    1. PACKAGE MODEL (.mar file)
       torch-model-archiver --model-name mymodel \\
           --version 1.0 \\
           --model-file model.py \\
           --serialized-file model.pt \\
           --handler handler.py
    
    2. START SERVER
       torchserve --start --model-store model_store \\
           --models mymodel=mymodel.mar
    
    3. INFERENCE
       curl http://localhost:8080/predictions/mymodel \\
           -T input.json
    
    CUSTOM HANDLER:
    ─────────────────────────────────────────────────────────────────
    
    class MyHandler(BaseHandler):
        def initialize(self, context):
            # Load model
            self.model = torch.jit.load('model.pt')
            self.model.eval()
        
        def preprocess(self, data):
            # Transform input
            return transform(data)
        
        def inference(self, data):
            # Run model
            with torch.no_grad():
                return self.model(data)
        
        def postprocess(self, data):
            # Format output
            return data.tolist()
    
    FEATURES:
    ─────────────────────────────────────────────────────────────────
    
    • Dynamic batching: Combine requests for efficiency
    • Model versioning: A/B testing, rollback
    • Metrics: Prometheus-compatible
    • Management API: Register/unregister models
    • Multi-GPU: Distribute across GPUs
    • Kubernetes: Production deployment
    """)

# ============================================================================
# DISTRIBUTED TRAINING
# ============================================================================

def explain_distributed_training():
    """Explain PyTorch distributed training."""
    print("\n" + "="*70)
    print(" DISTRIBUTED TRAINING")
    print(" Scaling PyTorch across multiple GPUs/nodes")
    print("="*70)
    
    print("""
    DISTRIBUTED STRATEGIES:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────┬───────────────────────────────────────────────────────┐
    │ Strategy        │ Description                                           │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ DataParallel    │ Single process, splits batch across GPUs              │
    │ (DP)            │ Simple but inefficient (GIL bottleneck)               │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ DistributedData │ Multi-process, one per GPU                           │
    │ Parallel (DDP)  │ Efficient gradient sync via AllReduce                │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ Fully Sharded   │ Shard model params across GPUs                       │
    │ DDP (FSDP)      │ Memory efficient for huge models                     │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ Pipeline        │ Split model layers across GPUs                       │
    │ Parallelism     │ For very deep models                                 │
    ├─────────────────┼───────────────────────────────────────────────────────┤
    │ Tensor          │ Split individual tensors across GPUs                 │
    │ Parallelism     │ For wide layers (e.g., large embeddings)            │
    └─────────────────┴───────────────────────────────────────────────────────┘
    
    DDP WORKFLOW:
    ─────────────────────────────────────────────────────────────────
    
    # Each process runs this code
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # 1. Initialize process group
    dist.init_process_group(backend='nccl')  # NCCL for GPUs
    
    # 2. Set device for this process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    # 3. Wrap model with DDP
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    
    # 4. Use DistributedSampler for data
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler)
    
    # 5. Training loop (gradients synced automatically!)
    for batch in loader:
        loss = model(batch)
        loss.backward()  # Gradients synchronized here
        optimizer.step()
    
    # Launch with:
    # torchrun --nproc_per_node=4 train.py
    
    FSDP FOR LARGE MODELS:
    ─────────────────────────────────────────────────────────────────
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=transformer_auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
    )
    
    # Each GPU only holds a shard of parameters!
    # Parameters gathered on-demand for forward/backward
    
    FSDP benefits:
    - Train models larger than single GPU memory
    - Efficient communication overlap
    - Activation checkpointing integration
    """)

# ============================================================================
# TORCH.COMPILE ECOSYSTEM
# ============================================================================

def explain_compile_ecosystem():
    """Explain torch.compile and related tools."""
    print("\n" + "="*70)
    print(" TORCH.COMPILE ECOSYSTEM")
    print(" PyTorch 2.0+ compilation stack")
    print("="*70)
    
    print("""
    TORCH.COMPILE STACK:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     torch.compile(model)                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ TorchDynamo (torch._dynamo)                                            │
    │ • Captures Python bytecode                                             │
    │ • Creates FX graph                                                     │
    │ • Handles graph breaks                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ AOTAutograd (torch._functorch.aot_autograd)                            │
    │ • Traces forward graph                                                 │
    │ • Generates backward graph                                             │
    │ • Enables training compilation                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Inductor (torch._inductor)                                             │
    │ • Generates Triton kernels (GPU)                                       │
    │ • Generates C++/OpenMP (CPU)                                           │
    │ • Fusion, scheduling, optimization                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    COMPILE MODES:
    ─────────────────────────────────────────────────────────────────
    
    # Default mode - good balance
    compiled = torch.compile(model)
    
    # Reduce overhead - minimize graph breaks
    compiled = torch.compile(model, mode="reduce-overhead")
    
    # Max autotune - try more configs
    compiled = torch.compile(model, mode="max-autotune")
    
    # Fullgraph - error on graph breaks
    compiled = torch.compile(model, fullgraph=True)
    
    # Dynamic shapes
    compiled = torch.compile(model, dynamic=True)
    
    BACKENDS:
    ─────────────────────────────────────────────────────────────────
    
    # Inductor (default) - Triton/C++ codegen
    torch.compile(model, backend="inductor")
    
    # Eager (debugging) - no optimization
    torch.compile(model, backend="eager")
    
    # AOT eager (debugging) - traces but no codegen
    torch.compile(model, backend="aot_eager")
    
    # Custom backend
    @torch._dynamo.register_backend
    def my_backend(gm, example_inputs):
        return gm  # or optimized version
    
    torch.compile(model, backend="my_backend")
    """)
    
    # Demo torch.compile
    if torch.cuda.is_available():
        print("\n DEMO: torch.compile speedup")
        print("-" * 50)
        
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).cuda()
        
        x = torch.randn(64, 512, device='cuda')
        
        # Eager
        def eager_forward():
            return model(x)
        
        # Warmup
        for _ in range(10):
            _ = eager_forward()
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = eager_forward()
        end.record()
        torch.cuda.synchronize()
        eager_time = start.elapsed_time(end) / 100
        
        # Compiled
        try:
            compiled_model = torch.compile(model)
            
            def compiled_forward():
                return compiled_model(x)
            
            # Warmup (includes compilation)
            for _ in range(10):
                _ = compiled_forward()
            torch.cuda.synchronize()
            
            start.record()
            for _ in range(100):
                _ = compiled_forward()
            end.record()
            torch.cuda.synchronize()
            compiled_time = start.elapsed_time(end) / 100
            
            print(f" Eager: {eager_time:.3f} ms")
            print(f" Compiled: {compiled_time:.3f} ms")
            print(f" Speedup: {eager_time/compiled_time:.2f}x")
        except Exception as e:
            print(f" Compilation failed: {e}")
    else:
        print("\n CUDA not available for demo")

# ============================================================================
# DOMAIN LIBRARIES
# ============================================================================

def explain_domain_libraries():
    """Explain PyTorch domain libraries."""
    print("\n" + "="*70)
    print(" PYTORCH DOMAIN LIBRARIES")
    print(" Specialized tools for different domains")
    print("="*70)
    
    print("""
    TORCHVISION (Computer Vision)
    ─────────────────────────────────────────────────────────────────
    
    • Datasets: ImageNet, COCO, CIFAR, etc.
    • Models: ResNet, ViT, EfficientNet, YOLO, etc.
    • Transforms: Augmentation, preprocessing
    • Ops: NMS, RoI pooling, etc.
    
    import torchvision
    model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    transform = torchvision.transforms.Compose([...])
    
    TORCHAUDIO (Audio)
    ─────────────────────────────────────────────────────────────────
    
    • Datasets: LibriSpeech, CommonVoice, etc.
    • Models: Wav2Vec2, HuBERT, etc.
    • Transforms: Spectrogram, MFCC, etc.
    • I/O: Audio file loading
    
    import torchaudio
    waveform, sample_rate = torchaudio.load('audio.wav')
    spectrogram = torchaudio.transforms.Spectrogram()(waveform)
    
    TORCHTEXT (NLP) - Note: Being deprecated
    ─────────────────────────────────────────────────────────────────
    
    • Datasets: IMDB, WikiText, etc.
    • Vocab: Tokenization, vocabulary building
    • Being replaced by HuggingFace ecosystem
    
    PYTORCH GEOMETRIC (Graphs)
    ─────────────────────────────────────────────────────────────────
    
    • Graph neural networks
    • Message passing
    • Graph datasets
    • Spatial-temporal models
    
    from torch_geometric.nn import GCNConv
    conv = GCNConv(in_channels, out_channels)
    
    TORCHDATA (Data Loading)
    ─────────────────────────────────────────────────────────────────
    
    • DataPipes: Composable data loading primitives
    • Works with torch.utils.data
    • Efficient for large datasets
    
    from torchdata.datapipes.iter import IterableWrapper
    dp = IterableWrapper(range(10)).map(lambda x: x * 2)
    """)

# ============================================================================
# CONTRIBUTION CONSIDERATIONS
# ============================================================================

def explain_contribution_considerations():
    """Important considerations for PyTorch contribution."""
    print("\n" + "="*70)
    print(" CONTRIBUTION CONSIDERATIONS")
    print(" What to know before contributing")
    print("="*70)
    
    print("""
    PYTORCH GOVERNANCE:
    ─────────────────────────────────────────────────────────────────
    
    • Meta (Facebook) is primary maintainer
    • Open source under BSD license
    • PyTorch Foundation (Linux Foundation)
    • Community contributions welcome!
    
    ROADMAP AREAS (2024):
    ─────────────────────────────────────────────────────────────────
    
    1. COMPILER (torch.compile)
       • Better coverage
       • Faster compilation
       • More backends
    
    2. DISTRIBUTED
       • Better FSDP
       • More parallelism strategies
       • Easier debugging
    
    3. EXPORT (torch.export)
       • Better coverage
       • Dynamic shapes
       • Edge deployment
    
    4. EAGER EXECUTION
       • Performance
       • Memory efficiency
       • New operations
    
    WHERE TO CONTRIBUTE:
    ─────────────────────────────────────────────────────────────────
    
    EASY (Good First Issues):
    • Documentation fixes
    • Test improvements
    • Error message improvements
    • Type annotations
    
    MEDIUM:
    • Bug fixes in operators
    • New operator implementations
    • Performance improvements
    • Ecosystem tools
    
    HARD:
    • Compiler work (Dynamo, Inductor)
    • Distributed systems
    • Core infrastructure
    • New backends
    
    HOW TO START:
    ─────────────────────────────────────────────────────────────────
    
    1. Read CONTRIBUTING.md
    2. Set up development environment
    3. Find "good first issue" on GitHub
    4. Join PyTorch Dev Discuss forum
    5. Attend office hours (if available)
    
    CODING STANDARDS:
    ─────────────────────────────────────────────────────────────────
    
    Python:
    • Follow PEP 8
    • Type annotations preferred
    • Use lintrunner for checks
    
    C++:
    • Follow Google C++ style (mostly)
    • Use clang-format
    • Document public APIs
    
    Tests:
    • Required for new features
    • Use pytest
    • Test edge cases
    """)

# ============================================================================
# SUMMARY
# ============================================================================

def print_ecosystem_summary():
    """Print ecosystem summary."""
    print("\n" + "="*70)
    print(" PYTORCH ECOSYSTEM SUMMARY")
    print("="*70)
    
    print("""
    KEY COMPONENTS:
    
    1. DEPLOYMENT
       • ExecuTorch: Mobile/edge (new, recommended)
       • TorchScript: Legacy but stable
       • torch.export: New export system
       • TorchServe: Serving
    
    2. TRAINING
       • DDP: Multi-GPU training
       • FSDP: Large model training
       • torch.compile: Training speedup
       • AMP: Mixed precision
    
    3. ECOSYSTEM
       • torchvision, torchaudio: Domains
       • Captum: Interpretability
       • TensorBoard: Visualization
    
    CHOOSING THE RIGHT TOOL:
    ─────────────────────────────────────────────────────────────────
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Use Case                      │ Tool                                   │
    ├───────────────────────────────┼────────────────────────────────────────┤
    │ Mobile deployment             │ ExecuTorch                             │
    │ Server deployment             │ TorchServe or TorchScript              │
    │ Multi-GPU training            │ DDP                                    │
    │ Huge model training           │ FSDP                                   │
    │ Speed up training             │ torch.compile                          │
    │ Export for other frameworks   │ ONNX or torch.export                   │
    │ Edge/IoT                      │ ExecuTorch with XNNPACK               │
    └───────────────────────────────┴────────────────────────────────────────┘
    
    NEXT: Study 05_contribution_guide.py for detailed contribution guide
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║" + " PYTORCH ECOSYSTEM OVERVIEW ".center(68) + "║")
    print("║" + " Beyond the core library ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\n PyTorch version: {torch.__version__}")
    
    explain_executorch()
    explain_torch_export()
    explain_torchserve()
    explain_distributed_training()
    explain_compile_ecosystem()
    explain_domain_libraries()
    explain_contribution_considerations()
    print_ecosystem_summary()
    
    print("\n" + "="*70)
    print(" The ecosystem makes PyTorch production-ready!")
    print("="*70)
