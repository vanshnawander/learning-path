# GPU Memory Hierarchy

This directory provides a deep dive into GPU memory hierarchy - the most critical factor for GPU performance.

## 📚 Modules

### 01_memory_deep_dive.py
**Complete exploration of GPU memory levels**

- Memory hierarchy overview (Registers → Shared → L1 → L2 → HBM → Host)
- Cache effects and working set size
- Memory access patterns (coalesced vs strided)
- Shared memory benefits
- Bandwidth vs compute analysis

**Key Profiled Experiments:**
- Working set size vs bandwidth
- Coalesced vs strided access patterns
- Tiling benefit calculation
- Arithmetic intensity analysis

**Run:** `python 01_memory_deep_dive.py`

## 🎯 Learning Objectives

- [ ] Understand GPU memory hierarchy levels
- [ ] Know latency and bandwidth at each level
- [ ] Identify memory vs compute bottlenecks
- [ ] Apply optimization strategies

## 🔗 Memory Quick Reference

| Level | Latency | Size | Scope |
|-------|---------|------|-------|
| Registers | ~0 cycles | 256KB/SM | Thread |
| Shared/L1 | ~20-30 cy | 48-228KB/SM | Block |
| L2 Cache | ~200 cy | 40-50MB | Device |
| HBM | ~400-600 cy | 40-80GB | Device |
| CPU RAM | ~10000+ cy | 64-256GB | Host |

## ⏱️ Expected Time

- Reading + Running: 2-3 hours
- Deep understanding: 1 day
