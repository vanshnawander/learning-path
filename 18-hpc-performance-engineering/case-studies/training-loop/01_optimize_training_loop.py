"""
Training Loop Optimization Case Study

This module demonstrates common optimizations for PyTorch training loops,
progressing from naive to highly optimized implementations.

Run with: python 01_optimize_training_loop.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from contextlib import contextmanager

# Synthetic dataset
def create_dataset(num_samples=10000, input_dim=1024, num_classes=100):
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, num_classes=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

@contextmanager
def timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f} ms")

# ============================================================
# Version 1: Naive Implementation
# ============================================================

def train_v1_naive(model, loader, optimizer, criterion, device):
    """
    Naive training loop with common mistakes.
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        # Problem 1: Synchronous data transfer
        data = data.to(device)
        target = target.to(device)
        
        # Problem 2: Not using zero_grad(set_to_none=True)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        # Problem 3: Synchronous backward
        loss.backward()
        optimizer.step()
        
        # Problem 4: Moving tensor to CPU for logging
        total_loss += loss.item()  # Synchronization point!
    
    return total_loss / len(loader)

# ============================================================
# Version 2: Basic Optimizations
# ============================================================

def train_v2_basic_opt(model, loader, optimizer, criterion, device):
    """
    Basic optimizations:
    - set_to_none=True
    - Reduced synchronization
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Faster than zero_grad()
        optimizer.zero_grad(set_to_none=True)
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Only sync every N batches for logging
        if batch_idx % 10 == 0:
            total_loss += loss.item()
    
    return total_loss / (len(loader) // 10)

# ============================================================
# Version 3: Mixed Precision Training
# ============================================================

def train_v3_amp(model, loader, optimizer, criterion, device):
    """
    Automatic Mixed Precision (AMP) training.
    Uses FP16 for forward/backward, FP32 for optimizer.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Automatic mixed precision
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 10 == 0:
            total_loss += loss.item()
    
    return total_loss / (len(loader) // 10)

# ============================================================
# Version 4: torch.compile (PyTorch 2.0+)
# ============================================================

def train_v4_compiled(model, loader, optimizer, criterion, device):
    """
    Using torch.compile for kernel fusion and optimization.
    """
    # Compile the model (do this once, outside training loop)
    compiled_model = torch.compile(model)
    
    compiled_model.train()
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            output = compiled_model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 10 == 0:
            total_loss += loss.item()
    
    return total_loss / (len(loader) // 10)

# ============================================================
# Version 5: Gradient Accumulation
# ============================================================

def train_v5_grad_accum(model, loader, optimizer, criterion, device, 
                        accum_steps=4):
    """
    Gradient accumulation for effective larger batch size.
    Useful when GPU memory is limited.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Use no_sync context for distributed (no-op for single GPU)
        with torch.cuda.amp.autocast():
            output = model(data)
            # Scale loss by accumulation steps
            loss = criterion(output, target) / accum_steps
        
        scaler.scale(loss).backward()
        
        # Only step every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        if batch_idx % 10 == 0:
            total_loss += loss.item() * accum_steps
    
    return total_loss / (len(loader) // 10)

# ============================================================
# Version 6: CUDA Graphs (for static shapes)
# ============================================================

def train_v6_cuda_graphs(model, example_input, example_target, 
                         optimizer, criterion, device):
    """
    CUDA Graphs capture and replay for minimal CPU overhead.
    Only works with static tensor shapes!
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output = model(example_input)
                loss = criterion(output, example_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture graph
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    
    with torch.cuda.graph(g):
        with torch.cuda.amp.autocast():
            static_output = model(example_input)
            static_loss = criterion(static_output, example_target)
        scaler.scale(static_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # Replay function
    def replay_step(input_data, target_data):
        example_input.copy_(input_data)
        example_target.copy_(target_data)
        g.replay()
        return static_loss
    
    return replay_step

# ============================================================
# Benchmark
# ============================================================

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Setup
    dataset = create_dataset(num_samples=10000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, 
                        num_workers=4, pin_memory=True)
    
    results = {}
    
    # V1: Naive
    print("\n=== V1: Naive ===")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    with timer("V1 Naive"):
        for epoch in range(3):
            loss = train_v1_naive(model, loader, optimizer, criterion, device)
    
    # V2: Basic optimizations
    print("\n=== V2: Basic Optimizations ===")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters())
    
    with timer("V2 Basic"):
        for epoch in range(3):
            loss = train_v2_basic_opt(model, loader, optimizer, criterion, device)
    
    # V3: AMP
    print("\n=== V3: Mixed Precision (AMP) ===")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters())
    
    with timer("V3 AMP"):
        for epoch in range(3):
            loss = train_v3_amp(model, loader, optimizer, criterion, device)
    
    # V4: Compiled (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("\n=== V4: torch.compile ===")
        model = SimpleModel().to(device)
        optimizer = optim.Adam(model.parameters())
        
        with timer("V4 Compiled"):
            for epoch in range(3):
                loss = train_v4_compiled(model, loader, optimizer, criterion, device)
    
    # V5: Gradient Accumulation
    print("\n=== V5: Gradient Accumulation (4x) ===")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters())
    
    with timer("V5 Grad Accum"):
        for epoch in range(3):
            loss = train_v5_grad_accum(model, loader, optimizer, criterion, device)
    
    print("\n=== Summary ===")
    print("Key optimizations applied:")
    print("1. non_blocking=True for async data transfer")
    print("2. set_to_none=True for faster gradient zeroing")
    print("3. Automatic Mixed Precision (AMP)")
    print("4. torch.compile for kernel fusion")
    print("5. Gradient accumulation for memory efficiency")
    print("6. CUDA Graphs for minimal CPU overhead (static shapes)")

if __name__ == "__main__":
    benchmark()
