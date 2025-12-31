"""
07_end_to_end_pipeline_profile.py - Complete Training Pipeline Profiling

This is the MASTER profiling script. Run this to understand
where time goes in your entire training pipeline.

Usage: python 07_end_to_end_pipeline_profile.py
"""

import time
import sys
from dataclasses import dataclass, field
from typing import List, Dict
from contextlib import contextmanager

@dataclass
class TimingResult:
    name: str
    times: List[float] = field(default_factory=list)
    
    @property
    def avg_ms(self) -> float:
        return sum(self.times) / len(self.times) * 1000 if self.times else 0
    
    @property
    def total_ms(self) -> float:
        return sum(self.times) * 1000

class PipelineProfiler:
    """
    Complete training pipeline profiler.
    
    Measures:
    - Data loading time
    - CPU→GPU transfer time
    - Forward pass time
    - Loss computation time
    - Backward pass time
    - Optimizer step time
    - Total iteration time
    """
    
    def __init__(self):
        self.stages = {
            'data_load': TimingResult('Data Loading'),
            'to_gpu': TimingResult('CPU→GPU Transfer'),
            'forward': TimingResult('Forward Pass'),
            'loss': TimingResult('Loss Computation'),
            'backward': TimingResult('Backward Pass'),
            'optimizer': TimingResult('Optimizer Step'),
            'total': TimingResult('Total Iteration'),
        }
        self._current_stage = None
        self._stage_start = None
        self._iter_start = None
    
    def start_iteration(self):
        self._iter_start = time.perf_counter()
    
    def end_iteration(self):
        if self._iter_start:
            self.stages['total'].times.append(time.perf_counter() - self._iter_start)
    
    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        yield
        if name in self.stages:
            self.stages[name].times.append(time.perf_counter() - start)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE PROFILING SUMMARY")
        print("=" * 70)
        
        total_avg = self.stages['total'].avg_ms
        
        print(f"\n{'Stage':<25} {'Avg (ms)':<12} {'%':<8} {'Visualization':<20}")
        print("-" * 70)
        
        for key in ['data_load', 'to_gpu', 'forward', 'loss', 'backward', 'optimizer']:
            result = self.stages[key]
            if result.times:
                pct = result.avg_ms / total_avg * 100 if total_avg > 0 else 0
                bar = "█" * int(pct / 3)
                print(f"{result.name:<25} {result.avg_ms:<12.2f} {pct:<8.1f} {bar}")
        
        print("-" * 70)
        print(f"{'TOTAL ITERATION':<25} {total_avg:<12.2f} {'100.0':<8}")
        
        # Identify bottleneck
        bottleneck = max(
            [(k, v.avg_ms) for k, v in self.stages.items() if k != 'total' and v.times],
            key=lambda x: x[1]
        )
        
        print(f"\n🔥 BOTTLENECK: {self.stages[bottleneck[0]].name} ({bottleneck[1]:.1f} ms)")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 70)
        
        if bottleneck[0] == 'data_load':
            print("""
    DATA LOADING is the bottleneck!
    
    Solutions:
    ✓ Increase num_workers in DataLoader
    ✓ Use pin_memory=True
    ✓ Use persistent_workers=True
    ✓ Pre-process and cache data (FFCV, memory-mapped)
    ✓ Use faster storage (NVMe SSD)
    ✓ Pre-resize images to training size
    ✓ Use GPU decoding (NVIDIA DALI)
            """)
        elif bottleneck[0] == 'to_gpu':
            print("""
    CPU→GPU TRANSFER is the bottleneck!
    
    Solutions:
    ✓ Use pin_memory=True in DataLoader
    ✓ Use non_blocking=True in .cuda()
    ✓ Use CUDA streams for overlap
    ✓ Reduce batch size to fit transfer budget
    ✓ Keep more data on GPU (gradient checkpointing for memory)
            """)
        elif bottleneck[0] in ['forward', 'backward']:
            print(f"""
    {bottleneck[0].upper()} PASS is the bottleneck!
    
    This is GOOD - GPU is being utilized!
    
    Further optimization:
    ✓ Use mixed precision (torch.cuda.amp)
    ✓ Use torch.compile() for kernel fusion
    ✓ Use Flash Attention for transformers
    ✓ Consider smaller model or gradient checkpointing
    ✓ Profile GPU kernels with torch.profiler
            """)
        elif bottleneck[0] == 'optimizer':
            print("""
    OPTIMIZER STEP is the bottleneck!
    
    Solutions:
    ✓ Use fused optimizers (apex.optimizers.FusedAdam)
    ✓ Use gradient accumulation
    ✓ Consider simpler optimizer (SGD vs Adam)
            """)

def run_simulated_training():
    """Simulate training loop with realistic timings."""
    print("\n" + "█" * 70)
    print("█  SIMULATED TRAINING PIPELINE PROFILING")
    print("█" * 70)
    
    try:
        import torch
        import torch.nn as nn
        has_torch = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {device}")
    except ImportError:
        has_torch = False
        device = 'cpu'
        print("\nPyTorch not installed, using simulated timings")
    
    profiler = PipelineProfiler()
    num_iterations = 20
    batch_size = 32
    
    if has_torch and device == 'cuda':
        # Real PyTorch profiling
        model = nn.Sequential(
            nn.Linear(224*224*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        print(f"\nRunning {num_iterations} iterations...")
        
        for i in range(num_iterations):
            profiler.start_iteration()
            
            # Data loading (simulated)
            with profiler.stage('data_load'):
                time.sleep(0.005)  # Simulate 5ms data loading
                data = torch.randn(batch_size, 224*224*3)
                labels = torch.randint(0, 1000, (batch_size,))
            
            # CPU to GPU transfer
            with profiler.stage('to_gpu'):
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                torch.cuda.synchronize()
            
            # Forward pass
            with profiler.stage('forward'):
                output = model(data)
                torch.cuda.synchronize()
            
            # Loss computation
            with profiler.stage('loss'):
                loss = criterion(output, labels)
                torch.cuda.synchronize()
            
            # Backward pass
            with profiler.stage('backward'):
                optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
            
            # Optimizer step
            with profiler.stage('optimizer'):
                optimizer.step()
                torch.cuda.synchronize()
            
            profiler.end_iteration()
            
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}/{num_iterations}")
    
    else:
        # Simulated timings
        print(f"\nSimulating {num_iterations} iterations...")
        
        for i in range(num_iterations):
            profiler.start_iteration()
            
            with profiler.stage('data_load'):
                time.sleep(0.015)  # 15ms
            
            with profiler.stage('to_gpu'):
                time.sleep(0.003)  # 3ms
            
            with profiler.stage('forward'):
                time.sleep(0.008)  # 8ms
            
            with profiler.stage('loss'):
                time.sleep(0.001)  # 1ms
            
            with profiler.stage('backward'):
                time.sleep(0.012)  # 12ms
            
            with profiler.stage('optimizer'):
                time.sleep(0.002)  # 2ms
            
            profiler.end_iteration()
    
    profiler.print_summary()
    
    # Additional metrics
    total_time = sum(profiler.stages['total'].times)
    samples_per_sec = num_iterations * batch_size / total_time
    
    print("\n" + "=" * 70)
    print("THROUGHPUT METRICS")
    print("=" * 70)
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"  Time per sample: {1000/samples_per_sec:.2f} ms")

if __name__ == "__main__":
    run_simulated_training()
    
    print("\n" + "=" * 70)
    print("HOW TO PROFILE YOUR OWN TRAINING")
    print("=" * 70)
    print("""
    Copy this pattern into your training loop:
    
    ```python
    profiler = PipelineProfiler()
    
    for batch in dataloader:
        profiler.start_iteration()
        
        with profiler.stage('data_load'):
            # DataLoader already loaded 'batch'
            pass
        
        with profiler.stage('to_gpu'):
            batch = batch.cuda(non_blocking=True)
            torch.cuda.synchronize()
        
        with profiler.stage('forward'):
            output = model(batch)
            torch.cuda.synchronize()
        
        # ... rest of training loop
        
        profiler.end_iteration()
    
    profiler.print_summary()
    ```
    
    For more detailed GPU profiling, use:
    
    ```python
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    ) as prof:
        # Your training loop
        prof.step()
    ```
    """)
