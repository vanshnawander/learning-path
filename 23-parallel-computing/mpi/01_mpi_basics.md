# MPI: Message Passing Interface

## What is MPI?

MPI (Message Passing Interface) is a standardized API for distributed-memory parallel programming. Unlike shared-memory models (OpenMP), MPI processes have separate address spaces and communicate via explicit message passing.

## Core Concepts

### Process Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        MPI Application                          │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ Rank 0  │    │ Rank 1  │    │ Rank 2  │    │ Rank 3  │     │
│  │ (Node A)│    │ (Node A)│    │ (Node B)│    │ (Node B)│     │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                    Network (InfiniBand, Ethernet)              │
└─────────────────────────────────────────────────────────────────┘

Key terms:
- Rank: Unique process ID (0 to N-1)
- Communicator: Group of processes (MPI_COMM_WORLD is default)
- Size: Total number of processes
```

## Basic MPI Program Structure

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get process info
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("Hello from rank %d of %d\n", rank, size);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
```

Compile and run:
```bash
mpicc -o hello hello.c
mpirun -n 4 ./hello
```

## Point-to-Point Communication

### Blocking Send/Receive

```c
// Rank 0 sends to Rank 1
if (rank == 0) {
    int data = 42;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    //       buffer, count, type, dest, tag, comm
} 
else if (rank == 1) {
    int data;
    MPI_Status status;
    MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    //       buffer, count, type, source, tag, comm, status
    printf("Received: %d\n", data);
}
```

### Non-Blocking Communication

```c
MPI_Request request;
int data = 42;

// Non-blocking send
MPI_Isend(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &request);

// Do other work while message is in flight...

// Wait for completion
MPI_Wait(&request, MPI_STATUS_IGNORE);
```

## Collective Operations

### Broadcast

```c
int data;
if (rank == 0) {
    data = 100;  // Only root has data initially
}

// Broadcast from rank 0 to all
MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Now all ranks have data = 100
```

### Reduce

```c
int local_sum = rank + 1;  // Each rank has different value
int global_sum;

// Sum all values to rank 0
MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
    printf("Global sum: %d\n", global_sum);  // 1+2+3+4 = 10 for 4 ranks
}
```

### Allreduce (Most Common in ML)

```c
float local_gradient[1000];
float global_gradient[1000];

// Compute local gradients...

// Sum gradients across all ranks, result to ALL ranks
MPI_Allreduce(local_gradient, global_gradient, 1000, 
              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

### Scatter/Gather

```c
// Scatter: Distribute array from root to all ranks
int sendbuf[4] = {1, 2, 3, 4};  // On rank 0
int recvbuf;

MPI_Scatter(sendbuf, 1, MPI_INT, &recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Rank 0 gets 1, Rank 1 gets 2, etc.

// Gather: Collect values from all ranks to root
int gathered[4];
MPI_Gather(&recvbuf, 1, MPI_INT, gathered, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### Allgather

```c
int local_data = rank;
int all_data[4];

// Gather from all to all
MPI_Allgather(&local_data, 1, MPI_INT, all_data, 1, MPI_INT, MPI_COMM_WORLD);
// All ranks now have [0, 1, 2, 3]
```

## Allreduce Algorithms

### Ring Allreduce (Bandwidth Optimal)

```
For N ranks, each with data of size S:

Step 1: Reduce-scatter (N-1 steps)
  Each rank sends/receives S/N data per step
  Total data moved: (N-1) * S/N per rank

Step 2: Allgather (N-1 steps)
  Each rank sends/receives S/N data per step
  Total data moved: (N-1) * S/N per rank

Total: 2 * (N-1) * S/N per rank
     = 2S * (N-1)/N ≈ 2S for large N

This is bandwidth-optimal!
```

```
Ring Allreduce Visualization (4 ranks):

Reduce-Scatter Phase:
Step 1: 0→1, 1→2, 2→3, 3→0 (chunk 0,1,2,3)
Step 2: 0→1, 1→2, 2→3, 3→0 (different chunks)
Step 3: 0→1, 1→2, 2→3, 3→0 (different chunks)

After: Rank i has reduced chunk i

Allgather Phase:
Step 1: 0→1, 1→2, 2→3, 3→0 (gathered chunks)
Step 2: 0→1, 1→2, 2→3, 3→0
Step 3: 0→1, 1→2, 2→3, 3→0

After: All ranks have all reduced chunks
```

### Tree-Based (Latency Optimal)

```
For small messages where latency dominates:

Binary tree reduce + broadcast
Latency: O(log N) vs O(N) for ring

But bandwidth: O(S * log N) vs O(S) for ring
```

## MPI with Python (mpi4py)

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# NumPy array allreduce
local_data = np.ones(1000, dtype=np.float32) * rank
global_data = np.zeros_like(local_data)

comm.Allreduce(local_data, global_data, op=MPI.SUM)

if rank == 0:
    print(f"Sum: {global_data[0]}")  # 0+1+2+3 = 6 for 4 ranks
```

## MPI + CUDA (GPU-Aware MPI)

```c
// With GPU-aware MPI, can pass device pointers directly
float* d_data;
cudaMalloc(&d_data, size);

// MPI directly reads/writes GPU memory
MPI_Allreduce(MPI_IN_PLACE, d_data, count, MPI_FLOAT, 
              MPI_SUM, MPI_COMM_WORLD);
// No cudaMemcpy needed!

// Requires: CUDA-aware MPI (OpenMPI with CUDA, MVAPICH2-GDR)
```

## Best Practices

1. **Overlap communication and computation**
```c
MPI_Request req;
MPI_Iallreduce(..., &req);  // Start async allreduce
compute_independent_work();  // Do other work
MPI_Wait(&req, ...);         // Wait for completion
```

2. **Use collective operations over point-to-point**
   - Collectives are highly optimized
   - Automatic algorithm selection

3. **Match message sizes to network**
   - Small messages: latency-bound
   - Large messages: bandwidth-bound
   - Batch small messages when possible

4. **Consider process placement**
```bash
# Bind ranks to nodes intelligently
mpirun --map-by ppr:4:node -n 16 ./myapp
```

## References

- MPI Standard: https://www.mpi-forum.org/
- "Using MPI" - Gropp, Lusk, Skjellum
- NCCL for GPU collective operations
