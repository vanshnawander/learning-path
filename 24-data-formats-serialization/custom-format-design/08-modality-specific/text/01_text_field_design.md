# Text Field Design: From Strings to Token Streams

## Understanding Text Data

Text is fundamentally different from images or audio. It's **symbolic** (discrete tokens) rather than signal-based (continuous values), and it has highly variable length.

### The Scale of Text

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        TEXT DATA CHARACTERISTICS                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  LENGTH VARIABILITY:                                                           │
│  ───────────────────                                                           │
│                                                                                │
│  Content Type          Characters    Tokens (GPT-4)    Storage                │
│  ────────────          ──────────    ──────────────    ───────                │
│  Tweet                 280           ~50-100           ~300 B                  │
│  Email                 1,000         ~200-500          ~1 KB                   │
│  News article          5,000         ~1,000-2,000      ~5 KB                   │
│  Research paper        40,000        ~8,000-15,000     ~40 KB                  │
│  Novel                 500,000       ~100,000-200,000  ~500 KB                 │
│  Wikipedia dump        22 GB         ~5B tokens        ~22 GB                  │
│                                                                                │
│  TOKENIZATION METHODS:                                                         │
│  ─────────────────────                                                         │
│                                                                                │
│  Word-level:      "Hello world" → ["Hello", "world"]                          │
│                   Vocab size: ~50,000-200,000                                  │
│                   Problem: OOV (out-of-vocabulary) words                       │
│                                                                                │
│  Character-level: "Hello" → ["H", "e", "l", "l", "o"]                         │
│                   Vocab size: ~100-500                                         │
│                   Problem: Very long sequences                                 │
│                                                                                │
│  Subword (BPE):   "unbelievable" → ["un", "believ", "able"]                   │
│                   Vocab size: ~30,000-100,000                                  │
│                   Best balance of vocab size and sequence length               │
│                                                                                │
│  TOKENIZATION OVERHEAD:                                                        │
│  ──────────────────────                                                        │
│                                                                                │
│  Tokenizer           Throughput (tokens/sec)    Latency per doc               │
│  ─────────           ─────────────────────────  ──────────────                │
│  HuggingFace (slow)  100,000-500,000            1-10 ms                       │
│  HuggingFace (fast)  1,000,000-5,000,000        0.1-1 ms                      │
│  tiktoken            5,000,000-20,000,000       0.05-0.2 ms                   │
│                                                                                │
│  For large-scale training, PRE-TOKENIZE AND STORE TOKEN IDs!                 │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### The Padding Problem

Transformers require fixed-length inputs within a batch. Naive padding wastes compute and memory:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         THE PADDING PROBLEM                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Batch of 4 sequences (padded to max_length=512):                             │
│                                                                                │
│  Sequence 1: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  len=50/512  (90% pad)│
│  Sequence 2: [█████████████████████████████░░░░░░░░░░░░░]  len=230/512 (55% pad)│
│  Sequence 3: [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]  len=128/512 (75% pad)│
│  Sequence 4: [██████████████████████████████████████████]  len=512/512 (0% pad) │
│                                                                                │
│  Total tokens: 50 + 230 + 128 + 512 = 920 real tokens                         │
│  Total compute: 4 × 512 = 2048 positions                                      │
│  Wasted compute: 55%!                                                          │
│                                                                                │
│  SOLUTIONS:                                                                    │
│  ──────────                                                                    │
│                                                                                │
│  1. Dynamic Padding: Pad to longest in batch, not global max                  │
│  2. Bucketing: Group similar-length sequences together                        │
│  3. Packing: Concatenate sequences, no padding at all                         │
│  4. Flash Attention: Variable-length attention (no padding needed)            │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Storage Strategy 1: Raw Text (UTF-8)

The simplest approach: store raw text, tokenize at load time.

```python
import numpy as np
from typing import Tuple, Type
import struct

class RawTextField:
    """
    Store text as raw UTF-8 bytes.
    
    This provides maximum flexibility: you can change tokenizers
    without re-creating the dataset.
    
    Use when:
    - Experimenting with different tokenizers
    - Storage is not a concern
    - Dataset is small enough that tokenization overhead is acceptable
    """
    
    TYPE_ID = 40
    
    def __init__(
        self,
        max_chars: int = 100000,
        encoding: str = 'utf-8',
    ):
        """
        Args:
            max_chars: Maximum characters to store (truncate longer texts).
            encoding: Text encoding (utf-8 recommended).
        """
        self.max_chars = max_chars
        self.encoding = encoding
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),       # Pointer to text bytes
            ('byte_length', '<u4'),    # Length in bytes
            ('char_length', '<u4'),    # Length in characters (may differ from bytes for UTF-8)
        ], align=True)
    
    def encode(self, text: str) -> Tuple[np.ndarray, bytes]:
        """
        Encode text to bytes.
        
        Args:
            text: Input string.
        
        Returns:
            (metadata, text_bytes)
        """
        # Truncate if needed
        if len(text) > self.max_chars:
            text = text[:self.max_chars]
        
        # Encode to bytes
        text_bytes = text.encode(self.encoding)
        
        # Create metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0  # Filled by writer
        metadata['byte_length'] = len(text_bytes)
        metadata['char_length'] = len(text)
        
        return metadata, text_bytes
    
    def to_binary(self) -> bytes:
        return struct.pack('<I', self.max_chars)
    
    @classmethod
    def from_binary(cls, data: bytes) -> 'RawTextField':
        max_chars, = struct.unpack('<I', data[:4])
        return cls(max_chars=max_chars)
    
    def get_decoder_class(self) -> Type:
        return RawTextDecoder


class RawTextDecoder:
    """
    Decoder for raw text with optional tokenization.
    """
    
    def __init__(self, field: RawTextField, metadata: np.ndarray, memory_read, tokenizer=None):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        self.tokenizer = tokenizer
    
    def decode_text(self, sample_id: int) -> str:
        """Decode to raw string."""
        ptr = self.metadata[sample_id]['data_ptr']
        byte_length = self.metadata[sample_id]['byte_length']
        
        raw_bytes = self.memory_read(ptr, None)[:byte_length]
        return bytes(raw_bytes).decode('utf-8')
    
    def decode_tokenized(self, sample_id: int, max_length: int = 512) -> dict:
        """Decode and tokenize."""
        text = self.decode_text(sample_id)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for decode_tokenized")
        
        return self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,  # Don't pad yet
            return_tensors='np',
        )
    
    def generate_code(self):
        """Generate batch decode function."""
        metadata = self.metadata
        mem_read = self.memory_read
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            texts = []
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                
                ptr = metadata[sample_id]['data_ptr']
                byte_length = metadata[sample_id]['byte_length']
                
                raw_bytes = mem_read(ptr, storage_state)[:byte_length]
                text = bytes(raw_bytes).decode('utf-8')
                texts.append(text)
            
            return texts  # Returns list of strings
        
        return decode
```

## Storage Strategy 2: Pre-Tokenized Sequences

Store token IDs directly for maximum loading speed.

```python
class TokenizedTextField:
    """
    Store pre-tokenized text as token IDs.
    
    This is the most common approach for large-scale training.
    Tokenization is done once during dataset creation.
    
    Token ID dtype selection:
    - uint8:  vocab < 256 (rare, only character-level)
    - uint16: vocab < 65536 (most BPE tokenizers)
    - int32:  vocab < 2B (very large vocabularies)
    """
    
    TYPE_ID = 41
    
    def __init__(
        self,
        tokenizer,
        max_tokens: int = 2048,
        add_special_tokens: bool = True,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer (or compatible).
            max_tokens: Maximum tokens to store per sample.
            add_special_tokens: Whether to include [CLS], [SEP], etc.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.add_special_tokens = add_special_tokens
        
        # Determine optimal dtype
        vocab_size = tokenizer.vocab_size
        if vocab_size <= 256:
            self.token_dtype = np.uint8
        elif vocab_size <= 65536:
            self.token_dtype = np.uint16
        else:
            self.token_dtype = np.int32
        
        self.bytes_per_token = np.dtype(self.token_dtype).itemsize
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_tokens', '<u4'),
        ], align=True)
    
    def encode(self, text: str) -> Tuple[np.ndarray, bytes]:
        """
        Tokenize and encode text.
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        
        token_ids = encoding['input_ids']
        
        # Convert to numpy
        tokens = np.array(token_ids, dtype=self.token_dtype)
        
        # Create metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['data_ptr'] = 0
        metadata['num_tokens'] = len(tokens)
        
        return metadata, tokens.tobytes()
    
    def to_binary(self) -> bytes:
        # Store tokenizer name and config for verification
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
        name_bytes = tokenizer_name.encode('utf-8')[:128].ljust(128, b'\x00')
        
        return struct.pack('<I', self.max_tokens) + \
               struct.pack('<I', self.tokenizer.vocab_size) + \
               struct.pack('<B', self.bytes_per_token) + \
               name_bytes
    
    @classmethod
    def from_binary(cls, data: bytes, tokenizer) -> 'TokenizedTextField':
        max_tokens, vocab_size, bytes_per_token = struct.unpack('<IIB', data[:9])
        name = data[9:137].rstrip(b'\x00').decode('utf-8')
        
        # Verify tokenizer compatibility
        if tokenizer.vocab_size != vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: "
                f"file has {vocab_size}, provided tokenizer has {tokenizer.vocab_size}"
            )
        
        return cls(tokenizer=tokenizer, max_tokens=max_tokens)
    
    def get_decoder_class(self) -> Type:
        return TokenizedTextDecoder


class TokenizedTextDecoder:
    """
    Decoder for pre-tokenized text.
    """
    
    def __init__(self, field: TokenizedTextField, metadata: np.ndarray, memory_read):
        self.field = field
        self.metadata = metadata
        self.memory_read = memory_read
        
        self.max_tokens = int(metadata['num_tokens'].max())
        self.token_dtype = field.token_dtype
        self.bytes_per_token = field.bytes_per_token
    
    def declare_state_and_memory(self, previous_state):
        from dataclasses import dataclass
        
        @dataclass
        class State:
            shape: tuple
            dtype: np.dtype
            jit_mode: bool
        
        @dataclass
        class AllocationQuery:
            shape: tuple
            dtype: np.dtype
        
        # Output: fixed-length padded sequence
        new_state = State(
            shape=(self.max_tokens,),
            dtype=np.int64,  # Standard for PyTorch
            jit_mode=True,
        )
        allocation = AllocationQuery(
            shape=(self.max_tokens,),
            dtype=np.int64,
        )
        return new_state, allocation
    
    def generate_code(self):
        metadata = self.metadata
        mem_read = self.memory_read
        token_dtype = self.token_dtype
        bytes_per_token = self.bytes_per_token
        pad_token_id = self.field.tokenizer.pad_token_id or 0
        
        import numba as nb
        
        @nb.njit(parallel=True, nogil=True)
        def _pad_tokens(tokens, output, pad_id):
            """Pad/copy tokens to output."""
            n = min(len(tokens), len(output))
            for i in nb.prange(n):
                output[i] = tokens[i]
            for i in nb.prange(n, len(output)):
                output[i] = pad_id
        
        def decode(batch_indices, destination, metadata_arg, storage_state):
            """
            Decode batch of tokenized sequences.
            
            Returns:
                destination: (batch, max_tokens) int64 array
            """
            for batch_idx in range(len(batch_indices)):
                sample_id = batch_indices[batch_idx]
                
                ptr = metadata[sample_id]['data_ptr']
                num_tokens = metadata[sample_id]['num_tokens']
                
                # Read token bytes
                byte_size = num_tokens * bytes_per_token
                raw_bytes = mem_read(ptr, storage_state)[:byte_size]
                
                # View as token array
                tokens = np.frombuffer(raw_bytes, dtype=token_dtype)
                
                # Pad to max length
                _pad_tokens(tokens, destination[batch_idx], pad_token_id)
            
            return destination[:len(batch_indices)]
        
        decode.is_parallel = True
        return decode
```

## Storage Strategy 3: Packed Sequences

Eliminate padding waste entirely by concatenating sequences.

```python
class PackedTokenField:
    """
    Store tokens in packed format (no padding between sequences).
    
    Format:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Sequence Index Table                                            │
    │ ├─ cumulative_offset[0]: uint64                                 │
    │ ├─ cumulative_offset[1]: uint64                                 │
    │ └─ ...                                                          │
    ├─────────────────────────────────────────────────────────────────┤
    │ Packed Tokens                                                   │
    │ [seq0_tokens][seq1_tokens][seq2_tokens]...                      │
    └─────────────────────────────────────────────────────────────────┘
    
    This is ideal for language model pre-training where sequences
    can be arbitrarily concatenated.
    
    Benefits:
    - Zero padding waste
    - Efficient storage
    - Natural for causal LM training
    
    Works with:
    - Flash Attention (native variable-length support)
    - Sequence packing in transformers
    """
    
    TYPE_ID = 42
    
    def __init__(
        self,
        tokenizer,
        chunk_tokens: int = 4096,
        add_eos_between_docs: bool = True,
    ):
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.add_eos_between_docs = add_eos_between_docs
        
        self.eos_token_id = tokenizer.eos_token_id
        
        if tokenizer.vocab_size <= 65536:
            self.token_dtype = np.uint16
        else:
            self.token_dtype = np.int32
    
    @property
    def metadata_type(self) -> np.dtype:
        """Per-document metadata."""
        return np.dtype([
            ('doc_id', '<u4'),
            ('start_offset', '<u8'),   # Position in global token stream
            ('num_tokens', '<u4'),
        ], align=True)
    
    def create_packed_dataset(self, documents, output_path: str):
        """
        Pack all documents into a single token stream.
        
        Args:
            documents: Iterator of (doc_id, text) tuples.
            output_path: Path to output file.
        """
        import mmap
        
        all_tokens = []
        metadata_list = []
        current_offset = 0
        
        for doc_id, text in documents:
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            if self.add_eos_between_docs and self.eos_token_id is not None:
                tokens = tokens + [self.eos_token_id]
            
            # Record metadata
            meta = np.zeros(1, dtype=self.metadata_type)[0]
            meta['doc_id'] = doc_id
            meta['start_offset'] = current_offset
            meta['num_tokens'] = len(tokens)
            metadata_list.append(meta)
            
            # Accumulate tokens
            all_tokens.extend(tokens)
            current_offset += len(tokens)
        
        # Convert to numpy
        token_array = np.array(all_tokens, dtype=self.token_dtype)
        metadata_array = np.array(metadata_list)
        
        # Write to file
        # (In practice, use a proper writer class)
        with open(output_path, 'wb') as f:
            # Header: total tokens, num docs
            f.write(struct.pack('<QQ', len(token_array), len(metadata_array)))
            
            # Metadata table
            f.write(metadata_array.tobytes())
            
            # Token data
            f.write(token_array.tobytes())
        
        return len(token_array), len(metadata_array)


class PackedTokenLoader:
    """
    Loader that creates fixed-size chunks from packed tokens.
    
    For each batch, we:
    1. Extract a chunk of chunk_tokens consecutive tokens
    2. Create position IDs and document boundaries
    3. Return the batch
    """
    
    def __init__(
        self,
        packed_file: str,
        chunk_tokens: int = 4096,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        self.chunk_tokens = chunk_tokens
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Memory-map the token file
        import mmap as mmap_module
        
        self.file = open(packed_file, 'rb')
        self.mm = mmap_module.mmap(self.file.fileno(), 0, access=mmap_module.ACCESS_READ)
        
        # Read header
        self.total_tokens, self.num_docs = struct.unpack('<QQ', self.mm[:16])
        
        # Read metadata
        metadata_size = self.num_docs * np.dtype([
            ('doc_id', '<u4'),
            ('start_offset', '<u8'),
            ('num_tokens', '<u4'),
        ]).itemsize
        
        self.metadata = np.frombuffer(
            self.mm[16:16 + metadata_size],
            dtype=np.dtype([
                ('doc_id', '<u4'),
                ('start_offset', '<u8'),
                ('num_tokens', '<u4'),
            ])
        )
        
        self.tokens_offset = 16 + metadata_size
        
        # Total chunks
        self.num_chunks = self.total_tokens // chunk_tokens
    
    def __len__(self):
        return self.num_chunks // self.batch_size
    
    def __iter__(self):
        # Generate chunk indices
        chunk_indices = np.arange(self.num_chunks)
        
        if self.shuffle:
            np.random.shuffle(chunk_indices)
        
        for batch_start in range(0, len(chunk_indices) - self.batch_size + 1, self.batch_size):
            batch_chunk_ids = chunk_indices[batch_start:batch_start + self.batch_size]
            
            # Build batch
            input_ids = np.zeros((self.batch_size, self.chunk_tokens), dtype=np.int64)
            
            for i, chunk_id in enumerate(batch_chunk_ids):
                # Calculate byte offset
                token_start = chunk_id * self.chunk_tokens
                byte_start = self.tokens_offset + token_start * 2  # uint16
                byte_end = byte_start + self.chunk_tokens * 2
                
                # Read tokens
                tokens = np.frombuffer(
                    self.mm[byte_start:byte_end],
                    dtype=np.uint16
                )
                input_ids[i] = tokens
            
            yield {
                'input_ids': input_ids,
                'labels': input_ids.copy(),  # For causal LM
            }
    
    def close(self):
        self.mm.close()
        self.file.close()
```

## Storage Strategy 4: Bucketed Sequences

Group sequences by length to minimize padding within each batch.

```python
class BucketedTokenField:
    """
    Store sequences organized by length buckets.
    
    This enables mining batches with similar-length sequences,
    minimizing padding waste while still allowing dynamic batching.
    
    Bucket structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Bucket 0 (len 1-64)                                             │
    │ ├─ Sequence offsets table                                       │
    │ └─ Sequence data                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ Bucket 1 (len 65-128)                                           │
    │ ├─ Sequence offsets table                                       │
    │ └─ Sequence data                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ ...                                                             │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    TYPE_ID = 43
    
    def __init__(
        self,
        tokenizer,
        bucket_boundaries: list = [64, 128, 256, 512, 1024, 2048],
        max_tokens: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.bucket_boundaries = sorted(bucket_boundaries)
        self.max_tokens = max_tokens
        
        # Add final bucket for sequences up to max_tokens
        if self.bucket_boundaries[-1] < max_tokens:
            self.bucket_boundaries.append(max_tokens)
    
    def get_bucket_id(self, length: int) -> int:
        """Determine which bucket a sequence belongs to."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries) - 1
    
    # ... (implementation similar to PackedTokenField but organized by bucket)


class BucketedLoader:
    """
    Loader that samples batches from length buckets.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group sample indices by bucket
        self.buckets = self._build_buckets()
    
    def _build_buckets(self):
        """Group samples by length bucket."""
        buckets = {i: [] for i in range(len(self.dataset.bucket_boundaries))}
        
        for idx in range(len(self.dataset.metadata)):
            length = self.dataset.metadata[idx]['num_tokens']
            bucket_id = self.dataset.get_bucket_id(length)
            buckets[bucket_id].append(idx)
        
        return buckets
    
    def __iter__(self):
        # Shuffle within each bucket
        shuffled_buckets = {}
        for bucket_id, indices in self.buckets.items():
            shuffled = np.array(indices)
            np.random.shuffle(shuffled)
            shuffled_buckets[bucket_id] = shuffled
        
        # Generate batches from each bucket
        for bucket_id in shuffled_buckets:
            indices = shuffled_buckets[bucket_id]
            
            for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                
                # Load and pad to bucket max length
                # (padding is minimal since all sequences are similar length)
                yield self._load_batch(batch_indices, bucket_id)
```

## JIT-Compiled Text Operations

```python
import numba as nb
import numpy as np

@nb.njit
def pad_sequence(tokens: np.ndarray, max_length: int, pad_value: int) -> np.ndarray:
    """Pad a single sequence to max_length."""
    result = np.full(max_length, pad_value, dtype=np.int64)
    length = min(len(tokens), max_length)
    result[:length] = tokens[:length]
    return result


@nb.njit(parallel=True, nogil=True)
def batch_pad_sequences(
    token_ptrs: np.ndarray,    # (batch,) pointers to token arrays
    token_lengths: np.ndarray, # (batch,) lengths
    output: np.ndarray,        # (batch, max_len) output buffer
    pad_value: int,
):
    """Pad a batch of sequences in parallel."""
    batch_size, max_len = output.shape
    
    # Initialize with pad value
    for b in nb.prange(batch_size):
        for i in range(max_len):
            output[b, i] = pad_value
    
    # Copy tokens
    for b in nb.prange(batch_size):
        length = min(token_lengths[b], max_len)
        for i in range(length):
            output[b, i] = token_ptrs[b, i]  # Simplified; actual impl needs mem access
    
    return output


@nb.njit
def create_attention_mask(length: int, max_length: int) -> np.ndarray:
    """Create a simple attention mask (1 for tokens, 0 for padding)."""
    mask = np.zeros(max_length, dtype=np.int64)
    mask[:min(length, max_length)] = 1
    return mask


@nb.njit
def create_causal_mask(seq_length: int) -> np.ndarray:
    """
    Create a causal (lower-triangular) attention mask.
    
    Used for autoregressive models like GPT.
    """
    # Upper triangle is -inf (masked), lower triangle is 0 (attend)
    # For additive masking: softmax(QK^T + mask)
    mask = np.zeros((seq_length, seq_length), dtype=np.float32)
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            mask[i, j] = -np.inf
    return mask


@nb.njit
def find_document_boundaries(input_ids: np.ndarray, eos_token: int) -> np.ndarray:
    """
    Find document boundaries in a packed sequence.
    
    Returns array of (start, end) positions for each document.
    """
    boundaries = []
    start = 0
    
    for i in range(len(input_ids)):
        if input_ids[i] == eos_token:
            boundaries.append((start, i + 1))
            start = i + 1
    
    # Last document (if not ending with EOS)
    if start < len(input_ids):
        boundaries.append((start, len(input_ids)))
    
    return np.array(boundaries, dtype=np.int64)
```

## Performance Comparison

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      TEXT FIELD PERFORMANCE COMPARISON                         │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Strategy              Storage        Load Time       Best Use Case            │
│  ────────              ───────        ─────────       ─────────────            │
│                                                                                │
│  Raw UTF-8             1.0x           10-50 ms        Experimentation          │
│  (tokenize at load)    (baseline)     (tokenization)  Multiple tokenizers      │
│                                                                                │
│  Pre-tokenized         0.5-0.8x       0.1-1 ms        Large-scale training     │
│  (uint16 tokens)       (smaller)      (memcpy)        Fixed tokenizer          │
│                                                                                │
│  Packed                0.4-0.6x       0.1-1 ms        Language modeling        │
│  (no padding)          (smallest)     (memcpy)        Flash Attention          │
│                                                                                │
│  Bucketed              0.5-0.8x       0.2-2 ms        Variable-length tasks    │
│                        (depends)      (+ binning)     Minimal padding          │
│                                                                                │
│                                                                                │
│  THROUGHPUT (samples/sec, single thread):                                      │
│                                                                                │
│  Raw UTF-8:            1,000-10,000   (bottleneck: tokenization)               │
│  Pre-tokenized:        100,000+       (bottleneck: I/O)                        │
│  Packed:               500,000+       (bottleneck: I/O, no per-sample overhead)│
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Exercises

1.  **Implement SentencePiece Integration**: Create a `SentencePieceTextField` that uses SentencePiece for tokenization during encoding.

2.  **Token Caching**: Build a caching layer that stores recently-used sequences in RAM to avoid re-reading from disk.

3.  **Dynamic Batching**: Implement a dataloader that dynamically creates batches to maximize tokens-per-batch while staying under a GPU memory limit.

4.  **Benchmark Tokenizers**: Compare the throughput of HuggingFace (slow/fast), tiktoken, and SentencePiece on your hardware.
