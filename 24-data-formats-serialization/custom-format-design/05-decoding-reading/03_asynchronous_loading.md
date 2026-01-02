# Asynchronous Loading: The Producer-Consumer Pattern

## The Problem: Sequential Execution

In a naive training loop, operations happen one after another:

```python
# Naive training loop
for epoch in range(num_epochs):
    for batch in dataloader:       # 1. Load data (CPU, Disk)
        outputs = model(batch)      # 2. Forward pass (GPU)
        loss = criterion(outputs)
        loss.backward()             # 3. Backward pass (GPU)
        optimizer.step()            # 4. Update weights (GPU)
```

**Timeline of a single batch:**

```
Time:   0ms            50ms           100ms         150ms         200ms
        |───────────────|───────────────|───────────────|───────────────|
        
Disk:   [ Read Batch ] [     IDLE     ] [     IDLE     ] [     IDLE     ]
CPU:    [    IDLE    ] [ Decode/Aug   ] [     IDLE     ] [     IDLE     ]
GPU:    [    IDLE    ] [    IDLE      ] [  Forward     ] [  Backward    ]
```

**Total iteration time**: 200ms
**GPU utilization**: 50% (GPU is idle for 100ms waiting for data)

## The Solution: Pipelining

By preparing the next batch **while** the current batch is training, we overlap I/O with computation:

```
Time:   0ms            50ms           100ms         150ms         200ms
        |───────────────|───────────────|───────────────|───────────────|
        
Disk:   [ Read B0 ] [ Read B1 ] [ Read B2 ] [ Read B3 ]
CPU:    [ Decode B0] [ Decode B1] [ Decode B2] [ Decode B3]
GPU:                   [ Train B0 ] [ Train B1 ] [ Train B2 ]

        ↑              ↑
     Preparing     Training
     Batch 1       Batch 0
```

**GPU utilization**: 100% (GPU never waits for data)

## The Producer-Consumer Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      ASYNC LOADER ARCHITECTURE                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────┐                                                          │
│  │  Index Schedule │ ← Which samples to load in what order (shuffled)        │
│  └────────┬────────┘                                                          │
│           │                                                                    │
│           ▼                                                                    │
│  ┌─────────────────┐         ┌─────────────────┐         ┌──────────────────┐│
│  │  PRODUCER       │  batch  │  BUFFER POOL    │  batch  │  CONSUMER        ││
│  │  THREADS        │ ──────► │  (Ready Queue)  │ ──────► │  (Training Loop) ││
│  │                 │         │                 │         │                  ││
│  │  - Read mmap    │         │  [Buffer 0]     │         │  - model(batch)  ││
│  │  - Decode       │         │  [Buffer 1]     │         │  - backward()    ││
│  │  - Transform    │         │  [Buffer 2]     │         │  - step()        ││
│  │  - Enqueue      │         │  [Buffer 3]     │         │                  ││
│  └─────────────────┘         └─────────────────┘         └──────────────────┘│
│                                                                                │
│  The Producer runs in background threads.                                     │
│  The Consumer runs in the main thread (or GPU thread).                        │
│  The Buffer Pool acts as a synchronized queue with pre-allocated memory.      │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Why Threads, Not Processes?

Python's Global Interpreter Lock (GIL) normally prevents true thread parallelism. But FFCV uses:
1.  **Memory-mapped I/O**: `mmap` access doesn't hold the GIL.
2.  **Numba `@njit(nogil=True)`**: JIT-compiled code releases the GIL.

This means producer threads can run truly in parallel with the training thread, without expensive inter-process communication (IPC) that multiprocessing would require.

## Double Buffering (N-Buffering)

To avoid data races, we use multiple pre-allocated buffers:

```
Producer writes to:   Buffer[i % N]
Consumer reads from:  Buffer[(i - lag) % N]

If N >= 2 (double buffering), they never touch the same buffer.
```

**Triple buffering (N=3)** is common because it allows:
- Buffer 0: Being consumed by GPU
- Buffer 1: Ready in queue
- Buffer 2: Being filled by CPU

```python
import numpy as np
import threading
from queue import Queue
from typing import Iterator, Tuple

class BufferPool:
    """
    A pool of pre-allocated buffers for zero-copy data transfer.
    """
    
    def __init__(self, num_buffers: int, buffer_shape: Tuple, dtype: np.dtype):
        self.num_buffers = num_buffers
        self.buffers = [np.empty(buffer_shape, dtype=dtype) for _ in range(num_buffers)]
        self.available = Queue(maxsize=num_buffers)
        
        # Initially, all buffers are available
        for buf in self.buffers:
            self.available.put(buf)
    
    def acquire(self) -> np.ndarray:
        """Get a buffer. Blocks if none available."""
        return self.available.get()
    
    def release(self, buffer: np.ndarray):
        """Return a buffer to the pool."""
        self.available.put(buffer)
```

## Complete Implementation

```python
import numpy as np
import threading
from queue import Queue
from typing import Callable, Iterator, Tuple, Any
from dataclasses import dataclass

@dataclass
class BatchResult:
    """Container for a completed batch."""
    data: dict           # Field name -> numpy array
    indices: np.ndarray  # Sample indices in this batch
    buffer_id: int       # Which buffer was used (for recycling)

class AsyncDataLoader:
    """
    Asynchronous data loader with producer-consumer pattern.
    
    This loader:
    1. Pre-allocates multiple output buffers.
    2. Runs producer threads that fill buffers.
    3. Yields completed batches to the training loop.
    4. Recycles buffers after the training loop consumes them.
    """
    
    def __init__(
        self,
        dataset_reader,         # Object that provides mmap, metadata, etc.
        pipeline: Callable,     # Compiled pipeline function
        batch_size: int,
        num_workers: int = 4,   # Producer threads
        buffers_ahead: int = 3, # How many batches to prepare ahead
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.reader = dataset_reader
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.buffers_ahead = buffers_ahead
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate sizes
        self.num_samples = len(dataset_reader)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        # Pre-allocate output buffers
        self.buffer_pools = self._allocate_buffers()
        
        # Communication
        self.task_queue = Queue()           # Batches of indices to process
        self.result_queue = Queue()         # Completed BatchResults
        self.release_queue = Queue()        # Buffers to recycle
        self.stop_event = threading.Event()
    
    def _allocate_buffers(self) -> dict:
        """
        Pre-allocate buffers for each output field.
        
        Returns:
            Dict[field_name, BufferPool]
        """
        pools = {}
        
        for field_name, (shape, dtype) in self.pipeline.output_specs.items():
            buffer_shape = (self.batch_size,) + shape
            pools[field_name] = BufferPool(
                num_buffers=self.buffers_ahead,
                buffer_shape=buffer_shape,
                dtype=dtype,
            )
        
        return pools
    
    def _producer_loop(self, worker_id: int):
        """
        Producer thread main loop.
        
        Each producer:
        1. Gets a task (batch indices) from task_queue.
        2. Acquires buffers from buffer pools.
        3. Runs the JIT pipeline (releases GIL).
        4. Puts the result in result_queue.
        """
        while not self.stop_event.is_set():
            try:
                # Get next task (batch indices)
                task = self.task_queue.get(timeout=0.1)
            except:
                continue  # Check stop_event
            
            if task is None:  # Poison pill
                break
            
            batch_indices, batch_id = task
            
            # Acquire buffers for each field
            buffers = {
                field: pool.acquire()
                for field, pool in self.buffer_pools.items()
            }
            
            try:
                # Run the JIT pipeline
                # This is the critical section that releases the GIL
                self.pipeline(
                    batch_indices,
                    self.reader.metadata,
                    self.reader.storage_state,
                    buffers,
                )
            except Exception as e:
                # On error, release buffers and re-raise
                for field, buf in buffers.items():
                    self.buffer_pools[field].release(buf)
                raise
            
            # Put result in queue
            result = BatchResult(
                data=buffers,
                indices=batch_indices,
                buffer_id=batch_id,
            )
            self.result_queue.put(result)
    
    def _buffer_recycler_loop(self):
        """
        Background thread that recycles buffers after consumption.
        
        The training loop puts consumed BatchResults in release_queue.
        This thread returns the buffers to the pools.
        """
        while not self.stop_event.is_set():
            try:
                result = self.release_queue.get(timeout=0.1)
            except:
                continue
            
            if result is None:  # Poison pill
                break
            
            for field, buf in result.data.items():
                self.buffer_pools[field].release(buf)
    
    def _generate_schedule(self) -> np.ndarray:
        """Generate sample order for the epoch (with shuffling)."""
        indices = np.arange(self.num_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
            self.seed += 1  # Different shuffle each epoch
        
        return indices
    
    def __iter__(self) -> Iterator[BatchResult]:
        """
        Iterate over batches for one epoch.
        """
        # Generate schedule for this epoch
        schedule = self._generate_schedule()
        
        # Start producer threads
        producers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._producer_loop, args=(i,))
            t.daemon = True
            t.start()
            producers.append(t)
        
        # Start buffer recycler
        recycler = threading.Thread(target=self._buffer_recycler_loop)
        recycler.daemon = True
        recycler.start()
        
        try:
            # Enqueue all batches
            for batch_id in range(self.num_batches):
                start = batch_id * self.batch_size
                end = min(start + self.batch_size, self.num_samples)
                batch_indices = schedule[start:end]
                self.task_queue.put((batch_indices, batch_id))
            
            # Yield results as they complete
            for _ in range(self.num_batches):
                result = self.result_queue.get()
                yield result
                
                # After the caller is done with the result, recycle it
                # (Caller should call loader.release(result) or we do it automatically)
                self.release_queue.put(result)
        
        finally:
            # Shutdown
            self.stop_event.set()
            
            # Send poison pills
            for _ in range(self.num_workers):
                self.task_queue.put(None)
            self.release_queue.put(None)
            
            # Wait for threads
            for t in producers:
                t.join(timeout=1.0)
            recycler.join(timeout=1.0)
            
            self.stop_event.clear()
    
    def __len__(self) -> int:
        return self.num_batches
```

## Integration with madvise Prefetching

To further optimize, the producer can prefetch pages for upcoming batches:

```python
def _producer_loop_with_prefetch(self, worker_id: int):
    """
    Producer with prefetching of next batch's pages.
    """
    pending_prefetch = None
    
    while not self.stop_event.is_set():
        # Get current task
        task = self.task_queue.get(timeout=0.1)
        if task is None:
            break
        
        batch_indices, batch_id = task
        
        # Start prefetching NEXT batch while processing current
        if pending_prefetch is None:
            next_task = self.task_queue.peek_next()  # Hypothetical
            if next_task:
                pending_prefetch = self._start_prefetch(next_task[0])
        
        # Acquire buffers
        buffers = {field: pool.acquire() for field, pool in self.buffer_pools.items()}
        
        # Process current batch
        self.pipeline(batch_indices, ...)
        
        # Complete prefetch for next iteration
        pending_prefetch = None
        
        # ... (rest of loop)


def _start_prefetch(self, batch_indices: np.ndarray) -> None:
    """
    Issue madvise WILLNEED for pages this batch will access.
    """
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    MADV_WILLNEED = 3
    
    for sample_id in batch_indices:
        ptr = self.reader.metadata[sample_id]['data_ptr']
        size = self.reader.metadata[sample_id]['data_size']
        
        # Round to page boundaries
        page_start = (ptr // 4096) * 4096
        page_end = ((ptr + size + 4095) // 4096) * 4096
        
        addr = self.reader.mmap.ctypes.data + page_start
        libc.madvise(addr, page_end - page_start, MADV_WILLNEED)
```

## Performance Tuning

### How Many Buffers?

| `buffers_ahead` | Behavior |
|-----------------|----------|
| 1 | Ping-pong: producer waits for consumer. Low memory, potential stalls. |
| 2 | Double buffering: basic overlap. Good for balanced I/O and compute. |
| 3-4 | Triple/quad buffering: absorbs variance. Recommended default. |
| 8+ | Diminishing returns. Only useful for very long I/O latencies. |

### How Many Workers?

| `num_workers` | Guideline |
|---------------|-----------|
| 1 | Minimal overhead. Use if decode is very fast (e.g., pre-decoded). |
| 2-4 | Good for most cases. Matches I/O parallelism of SSDs. |
| 8+ | Use if decode is CPU-heavy (e.g., video decompression). |

### Memory Considerations

Each buffer consumes memory proportional to `batch_size × output_size`. For ImageNet with batch 256 and float32 224×224×3:

```
256 × 224 × 224 × 3 × 4 bytes = 154 MB per buffer
Triple buffering: 462 MB
```

For video or audio, this can be much larger. Tune accordingly!

## Exercises

1.  **Implement Prefetching**: Add `madvise(MADV_WILLNEED)` calls to prefetch the next batch's pages while the current batch is processing.

2.  **Measure Latency**: Log the time each batch spends in the queue waiting to be consumed. Adjust `buffers_ahead` to minimize waiting.

3.  **Handle Errors Gracefully**: If the pipeline raises an exception, propagate it to the main thread via the result queue (wrap in a special "error result" object).

4.  **Dynamic Worker Scaling**: Implement a mechanism to spin up more workers if the result queue is empty (consumer is starved) or spin down if the queue is always full (workers are outpacing consumer).
