"""
Roofline Model Calculator and Visualizer

Compute arithmetic intensity and plot roofline for various operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class HardwareSpec:
    """Hardware specifications for roofline analysis."""
    name: str
    peak_flops: float      # GFLOPS
    peak_bandwidth: float  # GB/s
    
    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at ridge point (FLOP/Byte)."""
        return self.peak_flops / self.peak_bandwidth
    
    def attainable_performance(self, ai: float) -> float:
        """Calculate attainable performance for given arithmetic intensity."""
        return min(self.peak_flops, self.peak_bandwidth * ai)


@dataclass  
class KernelProfile:
    """Profile of a computational kernel."""
    name: str
    flops: float           # Total floating point operations
    bytes_transferred: float  # Total bytes moved from/to memory
    achieved_gflops: Optional[float] = None  # Measured performance
    
    @property
    def arithmetic_intensity(self) -> float:
        """FLOP/Byte ratio."""
        return self.flops / self.bytes_transferred if self.bytes_transferred > 0 else float('inf')


# Common hardware configurations
HARDWARE_SPECS = {
    'A100_FP32': HardwareSpec('NVIDIA A100 (FP32)', 19500, 2039),
    'A100_FP16': HardwareSpec('NVIDIA A100 (FP16 TC)', 312000, 2039),
    'A100_FP64': HardwareSpec('NVIDIA A100 (FP64)', 9700, 2039),
    'H100_FP32': HardwareSpec('NVIDIA H100 (FP32)', 51200, 3350),
    'H100_FP16': HardwareSpec('NVIDIA H100 (FP16 TC)', 989000, 3350),
    'V100_FP32': HardwareSpec('NVIDIA V100 (FP32)', 15700, 900),
    'RTX4090_FP32': HardwareSpec('NVIDIA RTX 4090 (FP32)', 82600, 1008),
    'Intel_Xeon': HardwareSpec('Intel Xeon 8380 (AVX-512)', 4800, 410),
    'AMD_EPYC': HardwareSpec('AMD EPYC 9654 (AVX-512)', 5100, 460),
    'Apple_M2': HardwareSpec('Apple M2 Ultra', 27200, 800),
}


def calculate_gemm_ai(M: int, N: int, K: int, dtype_bytes: int = 4) -> KernelProfile:
    """Calculate arithmetic intensity for matrix multiplication C = A @ B."""
    flops = 2.0 * M * N * K  # multiply-add
    
    # Naive: read A, B once; write C once
    bytes_read = (M * K + K * N) * dtype_bytes
    bytes_write = M * N * dtype_bytes
    total_bytes = bytes_read + bytes_write
    
    return KernelProfile(
        name=f'GEMM [{M}x{K}] @ [{K}x{N}]',
        flops=flops,
        bytes_transferred=total_bytes
    )


def calculate_conv2d_ai(N: int, C: int, H: int, W: int, 
                        K: int, R: int, S: int, dtype_bytes: int = 4) -> KernelProfile:
    """Calculate AI for 2D convolution."""
    H_out = H - R + 1
    W_out = W - S + 1
    
    # FLOPs: each output element requires R*S*C multiply-adds
    flops = 2.0 * N * K * H_out * W_out * C * R * S
    
    # Bytes: input, weights, output
    bytes_input = N * C * H * W * dtype_bytes
    bytes_weights = K * C * R * S * dtype_bytes
    bytes_output = N * K * H_out * W_out * dtype_bytes
    total_bytes = bytes_input + bytes_weights + bytes_output
    
    return KernelProfile(
        name=f'Conv2D [{N},{C},{H},{W}] kernel={R}x{S} filters={K}',
        flops=flops,
        bytes_transferred=total_bytes
    )


def calculate_elementwise_ai(N: int, num_ops: int = 1, 
                             inputs: int = 2, dtype_bytes: int = 4) -> KernelProfile:
    """Calculate AI for element-wise operations."""
    flops = num_ops * N
    total_bytes = (inputs + 1) * N * dtype_bytes  # inputs + output
    
    return KernelProfile(
        name=f'Elementwise (ops={num_ops}, inputs={inputs})',
        flops=flops,
        bytes_transferred=total_bytes
    )


def calculate_reduction_ai(N: int, dtype_bytes: int = 4) -> KernelProfile:
    """Calculate AI for reduction (sum, mean, etc.)."""
    flops = N - 1  # N-1 additions
    total_bytes = N * dtype_bytes + dtype_bytes  # read N, write 1
    
    return KernelProfile(
        name=f'Reduction (N={N})',
        flops=flops,
        bytes_transferred=total_bytes
    )


def calculate_attention_ai(seq_len: int, head_dim: int, 
                          num_heads: int = 1, dtype_bytes: int = 4) -> KernelProfile:
    """Calculate AI for attention (Q @ K^T @ V)."""
    # QK^T: [S, D] @ [D, S] -> [S, S]
    flops_qk = 2.0 * seq_len * seq_len * head_dim
    # Softmax: ~5 ops per element
    flops_softmax = 5.0 * seq_len * seq_len
    # Attention @ V: [S, S] @ [S, D] -> [S, D]
    flops_av = 2.0 * seq_len * seq_len * head_dim
    
    total_flops = num_heads * (flops_qk + flops_softmax + flops_av)
    
    # Memory: Q, K, V input; output
    bytes_input = 3 * seq_len * head_dim * num_heads * dtype_bytes
    bytes_output = seq_len * head_dim * num_heads * dtype_bytes
    # Intermediate (if not fused): attention matrix
    bytes_intermediate = seq_len * seq_len * num_heads * dtype_bytes
    
    total_bytes = bytes_input + bytes_output + bytes_intermediate
    
    return KernelProfile(
        name=f'Attention (seq={seq_len}, dim={head_dim}, heads={num_heads})',
        flops=total_flops,
        bytes_transferred=total_bytes
    )


def plot_roofline(hw: HardwareSpec, kernels: List[KernelProfile], 
                  title: Optional[str] = None):
    """Plot roofline model with kernel positions."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # AI range for plotting
    ai_min, ai_max = 0.01, 10000
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 1000)
    
    # Compute roofline
    performance = np.minimum(hw.peak_flops, hw.peak_bandwidth * ai_range)
    
    # Plot roofline
    ax.loglog(ai_range, performance, 'b-', linewidth=2, label='Roofline')
    
    # Mark ridge point
    ridge_ai = hw.ridge_point
    ridge_perf = hw.attainable_performance(ridge_ai)
    ax.axvline(x=ridge_ai, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Ridge Point\n({ridge_ai:.1f} FLOP/B)', 
                xy=(ridge_ai, ridge_perf/2), fontsize=9, ha='center')
    
    # Plot kernels
    colors = plt.cm.tab10(np.linspace(0, 1, len(kernels)))
    for kernel, color in zip(kernels, colors):
        ai = kernel.arithmetic_intensity
        attainable = hw.attainable_performance(ai)
        
        # Plot attainable (on roofline)
        ax.plot(ai, attainable, 'o', color=color, markersize=10, 
                label=f'{kernel.name} (AI={ai:.2f})')
        
        # Plot achieved if available
        if kernel.achieved_gflops:
            ax.plot(ai, kernel.achieved_gflops, '^', color=color, markersize=10)
            # Draw line from achieved to attainable
            ax.plot([ai, ai], [kernel.achieved_gflops, attainable], 
                    '--', color=color, alpha=0.5)
            efficiency = kernel.achieved_gflops / attainable * 100
            ax.annotate(f'{efficiency:.0f}%', xy=(ai, kernel.achieved_gflops),
                       xytext=(5, 0), textcoords='offset points', fontsize=8)
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(title or f'Roofline Model: {hw.name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(1, hw.peak_flops * 2)
    
    # Add regions annotation
    ax.text(ai_min * 2, hw.peak_flops * 0.5, 'Memory\nBound', fontsize=10, alpha=0.7)
    ax.text(ai_max / 10, hw.peak_flops * 0.8, 'Compute\nBound', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def analyze_kernel(hw: HardwareSpec, kernel: KernelProfile):
    """Print detailed analysis of a kernel."""
    ai = kernel.arithmetic_intensity
    attainable = hw.attainable_performance(ai)
    
    print(f"\n{'='*60}")
    print(f"Kernel: {kernel.name}")
    print(f"{'='*60}")
    print(f"FLOPs:           {kernel.flops:.2e}")
    print(f"Bytes:           {kernel.bytes_transferred:.2e}")
    print(f"Arithmetic Int.: {ai:.2f} FLOP/Byte")
    print(f"Ridge Point:     {hw.ridge_point:.2f} FLOP/Byte")
    print(f"Bound:           {'Compute' if ai > hw.ridge_point else 'Memory'}")
    print(f"Attainable:      {attainable:.1f} GFLOPS")
    
    if kernel.achieved_gflops:
        efficiency = kernel.achieved_gflops / attainable * 100
        print(f"Achieved:        {kernel.achieved_gflops:.1f} GFLOPS ({efficiency:.1f}%)")


def main():
    # Select hardware
    hw = HARDWARE_SPECS['A100_FP32']
    print(f"Hardware: {hw.name}")
    print(f"Peak Compute: {hw.peak_flops} GFLOPS")
    print(f"Peak Bandwidth: {hw.peak_bandwidth} GB/s")
    print(f"Ridge Point: {hw.ridge_point:.2f} FLOP/Byte")
    
    # Create kernel profiles
    kernels = [
        calculate_gemm_ai(4096, 4096, 4096),
        calculate_gemm_ai(256, 256, 256),
        calculate_conv2d_ai(32, 64, 224, 224, 128, 3, 3),
        calculate_elementwise_ai(10000000),
        calculate_reduction_ai(10000000),
        calculate_attention_ai(2048, 64, 8),
    ]
    
    # Analyze each kernel
    for kernel in kernels:
        analyze_kernel(hw, kernel)
    
    # Plot
    fig, ax = plot_roofline(hw, kernels)
    plt.savefig('roofline.png', dpi=150)
    print("\nRoofline plot saved to roofline.png")
    plt.show()


if __name__ == '__main__':
    main()
