# CPU Pipeline and Instruction-Level Parallelism

Understanding how modern CPUs execute instructions.

## Files in This Directory

| File | Description |
|------|-------------|
| `01_pipeline_basics.c` | Pipeline effects, branch prediction |

## CPU Pipeline Stages

```
Fetch → Decode → Execute → Memory → Writeback
```

Modern CPUs pipeline: while one instruction executes, next is decoded, etc.

## Superscalar Execution

CPUs can execute multiple instructions per cycle:
- Multiple execution units (ALU, FPU, Load, Store)
- Out-of-order execution
- 4-6 instructions per cycle possible

## Hazards

### Data Hazards
```c
a = b + c;
d = a + e;  // Must wait for 'a'
```
Dependency chain limits parallelism.

### Branch Hazards
```c
if (condition) {
    // CPU speculates which branch
}
```
Misprediction: ~15 cycle penalty!

## Branch Prediction

Modern CPUs predict branches with >95% accuracy.
But unpredictable branches hurt performance.

```c
// Predictable (always true)
if (x >= 0) { ... }

// Unpredictable (50/50)
if (x >= 50) { ... }  // where x is random 0-99
```

## ML Implications

1. **Avoid branches in hot loops** - use masking
2. **Reduce dependency chains** - unroll loops
3. **GPU kernels** - branches cause warp divergence
4. **Triton** - block-based to avoid branching

## Exercises

1. Measure branch misprediction cost
2. Compare dependency chain vs independent ops
3. Profile with `perf stat -e branch-misses`
