# Text Field Design: From Strings to Token Streams

## Text Data Characteristics

Text is unique among modalities:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TEXT DATA CHARACTERISTICS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Property 1: HIGHLY VARIABLE LENGTH                                     │
│  ──────────────────────────────────                                      │
│                                                                          │
│  • Tweet: ~50 chars                                                      │
│  • Document: 10,000+ chars                                              │
│  • Book: 500,000+ chars                                                 │
│                                                                          │
│  Property 2: MULTIPLE REPRESENTATIONS                                   │
│  ────────────────────────────────────                                   │
│                                                                          │
│  Raw String  →  "Hello, world!"                                         │
│  UTF-8 Bytes →  [72, 101, 108, 108, 111, 44, 32, ...]                  │
│  Token IDs   →  [15496, 11, 995, 0]  (GPT-2 tokenizer)                 │
│  Embeddings  →  [[0.1, -0.3, ...], [...], ...]  float32                │
│                                                                          │
│  Property 3: PREPROCESSING IS EXPENSIVE                                 │
│  ──────────────────────────────────────                                 │
│                                                                          │
│  Tokenization can be 10-100x slower than reading from disk!            │
│  → Pre-tokenize and store token IDs                                    │
│                                                                          │
│  Property 4: PADDING OVERHEAD                                           │
│  ───────────────────────────────                                         │
│                                                                          │
│  Batch of sequences with max_length=512:                                │
│  • Sequence lengths: [23, 156, 89, 512, 45]                            │
│  • Average padding: 70%+ wasted memory!                                 │
│  → Use packing or dynamic batching                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Text Storage Strategies

### Strategy 1: Raw UTF-8 Bytes

Simple but requires tokenization at load time:

```python
import numpy as np
from typing import Tuple

class RawTextField:
    """
    Store text as raw UTF-8 bytes.
    
    Use when:
    - Tokenizer may change during training
    - Multiple tokenizers needed
    - Storage space is abundant
    """
    
    type_id = 20
    
    def __init__(
        self,
        max_length: int = 10000,  # Max chars
        encoding: str = 'utf-8'
    ):
        self.max_length = max_length
        self.encoding = encoding
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('byte_length', '<u4'),
            ('char_length', '<u4'),  # Unicode chars (may differ from bytes)
        ], align=True)
    
    def encode(self, text: str) -> Tuple[np.ndarray, bytes]:
        """Encode text to bytes."""
        # Truncate if needed
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        # Encode to bytes
        text_bytes = text.encode(self.encoding)
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['byte_length'] = len(text_bytes)
        metadata['char_length'] = len(text)
        
        return metadata, text_bytes
    
    def decode(self, metadata, read_fn) -> str:
        """Decode bytes to text."""
        ptr = metadata['data_ptr']
        size = metadata['byte_length']
        
        raw_bytes = read_fn(ptr, size)
        return bytes(raw_bytes).decode(self.encoding)


class RawTextDecoder:
    """Decoder with on-the-fly tokenization."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def decode(self, metadata, read_fn):
        ptr = metadata['data_ptr']
        size = metadata['byte_length']
        
        raw_bytes = read_fn(ptr, size)
        text = bytes(raw_bytes).decode('utf-8')
        
        # Tokenize
        return self.tokenizer(text, return_tensors='np')
```

### Strategy 2: Pre-Tokenized Sequences

Store token IDs directly for maximum speed:

```python
class TokenizedTextField:
    """
    Store pre-tokenized text as token IDs.
    
    Use when:
    - Tokenizer is fixed
    - Maximum loading speed needed
    - Token IDs fit in int32 (vocab < 2B)
    """
    
    type_id = 21
    
    def __init__(
        self,
        max_tokens: int = 512,
        vocab_size: int = 50000,
        tokenizer = None,
        store_special_tokens: bool = True
    ):
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.store_special_tokens = store_special_tokens
        
        # Determine dtype based on vocab size
        if vocab_size <= 256:
            self.token_dtype = np.uint8
        elif vocab_size <= 65536:
            self.token_dtype = np.uint16
        else:
            self.token_dtype = np.int32
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('data_ptr', '<u8'),
            ('num_tokens', '<u2'),
            ('pad_token_id', '<u2'),
        ], align=True)
    
    def encode(self, text: str) -> Tuple[np.ndarray, bytes]:
        """Tokenize and encode."""
        # Tokenize
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=self.store_special_tokens
        )['input_ids']
        
        # Convert to numpy
        token_array = np.array(tokens, dtype=self.token_dtype)
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['num_tokens'] = len(tokens)
        metadata['pad_token_id'] = self.tokenizer.pad_token_id or 0
        
        return metadata, token_array.tobytes()
    
    def decode(self, metadata, read_fn) -> np.ndarray:
        """Load token IDs."""
        ptr = metadata['data_ptr']
        num_tokens = metadata['num_tokens']
        
        raw_bytes = read_fn(ptr, num_tokens * np.dtype(self.token_dtype).itemsize)
        tokens = np.frombuffer(raw_bytes, dtype=self.token_dtype)
        
        return tokens


class TokenizedTextDecoder:
    """Decoder with optional padding."""
    
    def __init__(
        self,
        max_length: int,
        token_dtype: np.dtype,
        pad_token_id: int = 0
    ):
        self.max_length = max_length
        self.token_dtype = token_dtype
        self.pad_token_id = pad_token_id
    
    def decode(self, metadata, read_fn) -> dict:
        """Decode to dict with input_ids and attention_mask."""
        ptr = metadata['data_ptr']
        num_tokens = metadata['num_tokens']
        
        # Read tokens
        raw_bytes = read_fn(ptr, num_tokens * np.dtype(self.token_dtype).itemsize)
        tokens = np.frombuffer(raw_bytes, dtype=self.token_dtype)
        
        # Pad to max length
        input_ids = np.full(self.max_length, self.pad_token_id, dtype=np.int64)
        input_ids[:num_tokens] = tokens
        
        # Create attention mask
        attention_mask = np.zeros(self.max_length, dtype=np.int64)
        attention_mask[:num_tokens] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': num_tokens
        }
```

### Strategy 3: Packed Sequences (No Padding)

Eliminate padding waste:

```python
class PackedTextField:
    """
    Store sequences packed together without padding.
    
    Format:
    ┌─────────────────────────────────────────────┐
    │ Sequence 1 tokens │ Sequence 2 tokens │ ... │
    └─────────────────────────────────────────────┘
    
    Metadata stores (start_idx, length) for each sequence.
    
    Benefits:
    - No padding waste
    - ~30% storage reduction typical
    - Perfect for language modeling
    """
    
    type_id = 22
    
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 4096,  # Tokens per chunk
        overlap: int = 0
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('chunk_ptr', '<u8'),      # Pointer to token chunk
            ('local_start', '<u4'),    # Start within chunk
            ('length', '<u4'),         # Sequence length
            ('doc_id', '<u4'),         # Document ID
        ], align=True)
    
    def encode_document(self, text: str, doc_id: int) -> Tuple[list, bytes]:
        """Encode a document, possibly split into chunks."""
        # Tokenize entire document
        tokens = self.tokenizer(text)['input_ids']
        token_array = np.array(tokens, dtype=np.int32)
        
        # Split into chunks
        chunks = []
        metadata_list = []
        
        for start in range(0, len(tokens), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = token_array[start:end]
            
            meta = np.zeros(1, dtype=self.metadata_type)[0]
            meta['local_start'] = 0  # Will be updated by writer
            meta['length'] = len(chunk_tokens)
            meta['doc_id'] = doc_id
            
            chunks.append(chunk_tokens.tobytes())
            metadata_list.append(meta)
        
        return metadata_list, chunks


class PackedSequenceLoader:
    """
    Loader that creates packed batches.
    
    Instead of padding, concatenates sequences
    and uses position IDs to track boundaries.
    """
    
    def __init__(
        self,
        reader,
        batch_tokens: int = 4096,  # Tokens per batch
        shuffle: bool = True
    ):
        self.reader = reader
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
    
    def __iter__(self):
        """Generate packed batches."""
        # Get all sequence lengths
        lengths = self.reader.metadata['length']
        indices = np.arange(len(lengths))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Pack sequences into batches
        batch_tokens = []
        batch_positions = []
        batch_doc_ids = []
        current_pos = 0
        
        for idx in indices:
            meta = self.reader.metadata[idx]
            length = meta['length']
            
            # Would this overflow the batch?
            if current_pos + length > self.batch_tokens:
                # Yield current batch
                yield self._finalize_batch(
                    batch_tokens, batch_positions, batch_doc_ids
                )
                # Start new batch
                batch_tokens = []
                batch_positions = []
                batch_doc_ids = []
                current_pos = 0
            
            # Load tokens
            tokens = self._load_tokens(idx)
            batch_tokens.append(tokens)
            batch_positions.append(np.arange(length))
            batch_doc_ids.append(np.full(length, meta['doc_id']))
            current_pos += length
        
        # Yield final batch
        if batch_tokens:
            yield self._finalize_batch(
                batch_tokens, batch_positions, batch_doc_ids
            )
    
    def _finalize_batch(self, tokens, positions, doc_ids):
        return {
            'input_ids': np.concatenate(tokens),
            'position_ids': np.concatenate(positions),
            'document_ids': np.concatenate(doc_ids),
            'num_sequences': len(tokens)
        }
    
    def _load_tokens(self, idx):
        meta = self.reader.metadata[idx]
        return self.reader.read_tokens(meta)
```

### Strategy 4: Hierarchical Text (Documents > Paragraphs > Sentences)

For document-level tasks:

```python
class HierarchicalTextField:
    """
    Store text with hierarchical structure.
    
    Format:
    ┌─────────────────────────────────────────────┐
    │ Document Header                             │
    │ ├─ num_paragraphs                          │
    │ ├─ paragraph_offsets[]                     │
    │ └─ paragraph_lengths[]                     │
    ├─────────────────────────────────────────────┤
    │ Paragraph 0 (tokenized)                    │
    ├─────────────────────────────────────────────┤
    │ Paragraph 1 (tokenized)                    │
    ├─────────────────────────────────────────────┤
    │ ...                                         │
    └─────────────────────────────────────────────┘
    
    Use for:
    - Document summarization
    - Question answering over documents
    - Hierarchical attention models
    """
    
    type_id = 23
    
    def __init__(
        self,
        tokenizer,
        max_paragraphs: int = 100,
        max_tokens_per_paragraph: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_paragraphs = max_paragraphs
        self.max_tokens_per_para = max_tokens_per_paragraph
    
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('doc_id', '<u4'),
            ('num_paragraphs', '<u2'),
            ('total_tokens', '<u4'),
            ('header_ptr', '<u8'),
            ('data_ptr', '<u8'),
            ('data_size', '<u4'),
        ], align=True)
    
    def encode(self, text: str, doc_id: int) -> Tuple[np.ndarray, bytes]:
        """Encode hierarchical document."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraphs = paragraphs[:self.max_paragraphs]
        
        # Tokenize each paragraph
        para_tokens = []
        for para in paragraphs:
            tokens = self.tokenizer(
                para,
                truncation=True,
                max_length=self.max_tokens_per_para
            )['input_ids']
            para_tokens.append(np.array(tokens, dtype=np.int32))
        
        # Build offset table
        offsets = np.zeros(len(paragraphs), dtype='<u4')
        lengths = np.zeros(len(paragraphs), dtype='<u2')
        
        offset = 0
        for i, tokens in enumerate(para_tokens):
            offsets[i] = offset
            lengths[i] = len(tokens)
            offset += len(tokens) * 4  # int32
        
        # Pack header: num_paragraphs, offsets, lengths
        header = np.array([len(paragraphs)], dtype='<u2')
        header_bytes = header.tobytes() + offsets.tobytes() + lengths.tobytes()
        
        # Pack token data
        token_bytes = b''.join(t.tobytes() for t in para_tokens)
        
        # Combine
        all_data = header_bytes + token_bytes
        
        # Metadata
        metadata = np.zeros(1, dtype=self.metadata_type)[0]
        metadata['doc_id'] = doc_id
        metadata['num_paragraphs'] = len(paragraphs)
        metadata['total_tokens'] = sum(len(t) for t in para_tokens)
        metadata['data_size'] = len(all_data)
        
        return metadata, all_data
    
    def decode(self, metadata, read_fn) -> dict:
        """Decode hierarchical document."""
        data = read_fn(metadata['data_ptr'], metadata['data_size'])
        
        # Parse header
        num_paras = np.frombuffer(data[:2], dtype='<u2')[0]
        header_size = 2 + num_paras * 4 + num_paras * 2
        
        offsets = np.frombuffer(data[2:2 + num_paras * 4], dtype='<u4')
        lengths = np.frombuffer(data[2 + num_paras * 4:header_size], dtype='<u2')
        
        # Parse paragraphs
        token_data = data[header_size:]
        paragraphs = []
        
        for i in range(num_paras):
            start = offsets[i]
            length = lengths[i]
            tokens = np.frombuffer(
                token_data[start:start + length * 4],
                dtype=np.int32
            )
            paragraphs.append(tokens)
        
        return {
            'paragraphs': paragraphs,
            'num_paragraphs': num_paras,
            'paragraph_lengths': lengths
        }
```

## JIT-Compiled Text Operations

```python
import numba as nb

@nb.njit
def pad_sequence(
    tokens: np.ndarray,
    max_length: int,
    pad_value: int
) -> np.ndarray:
    """Pad or truncate sequence to fixed length."""
    result = np.full(max_length, pad_value, dtype=tokens.dtype)
    length = min(len(tokens), max_length)
    result[:length] = tokens[:length]
    return result


@nb.njit
def create_attention_mask(
    length: int,
    max_length: int
) -> np.ndarray:
    """Create attention mask."""
    mask = np.zeros(max_length, dtype=np.int64)
    mask[:length] = 1
    return mask


@nb.njit
def create_causal_mask(length: int) -> np.ndarray:
    """Create causal (autoregressive) attention mask."""
    mask = np.tril(np.ones((length, length), dtype=np.float32))
    return mask


@nb.njit(parallel=True)
def batch_pad_sequences(
    sequences: list,
    max_length: int,
    pad_value: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad batch of sequences."""
    batch_size = len(sequences)
    
    input_ids = np.full((batch_size, max_length), pad_value, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.int64)
    
    for i in nb.prange(batch_size):
        seq = sequences[i]
        length = min(len(seq), max_length)
        input_ids[i, :length] = seq[:length]
        attention_mask[i, :length] = 1
    
    return input_ids, attention_mask
```

## Text Collation Strategies

```python
class TextCollator:
    """
    Collates text samples into batches.
    Supports multiple padding strategies.
    """
    
    def __init__(
        self,
        max_length: int = 512,
        padding: str = 'max_length',  # 'max_length', 'longest', 'none'
        pad_token_id: int = 0,
        return_tensors: str = 'np'  # 'np', 'pt'
    ):
        self.max_length = max_length
        self.padding = padding
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors
    
    def __call__(self, samples: list) -> dict:
        """Collate samples into batch."""
        
        if self.padding == 'none':
            # Return packed format
            return self._pack_sequences(samples)
        
        elif self.padding == 'longest':
            # Pad to longest in batch
            max_len = min(
                max(len(s['input_ids']) for s in samples),
                self.max_length
            )
            return self._pad_to_length(samples, max_len)
        
        else:  # max_length
            return self._pad_to_length(samples, self.max_length)
    
    def _pad_to_length(self, samples, length):
        batch_size = len(samples)
        
        input_ids = np.full(
            (batch_size, length),
            self.pad_token_id,
            dtype=np.int64
        )
        attention_mask = np.zeros((batch_size, length), dtype=np.int64)
        
        for i, sample in enumerate(samples):
            seq_len = min(len(sample['input_ids']), length)
            input_ids[i, :seq_len] = sample['input_ids'][:seq_len]
            attention_mask[i, :seq_len] = 1
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if self.return_tensors == 'pt':
            import torch
            result = {k: torch.from_numpy(v) for k, v in result.items()}
        
        return result
    
    def _pack_sequences(self, samples):
        # Concatenate all sequences
        all_ids = np.concatenate([s['input_ids'] for s in samples])
        
        # Create position IDs that reset per sequence
        positions = []
        for s in samples:
            positions.append(np.arange(len(s['input_ids'])))
        all_positions = np.concatenate(positions)
        
        return {
            'input_ids': all_ids,
            'position_ids': all_positions,
            'sequence_lengths': [len(s['input_ids']) for s in samples]
        }
```

## Performance Summary

| Strategy | Storage | Load Speed | Flexibility |
|----------|---------|------------|-------------|
| Raw UTF-8 | Large | Slow (tokenize) | High |
| Pre-tokenized | Small | Fast | Low (fixed tokenizer) |
| Packed | Smallest | Fast | Medium |
| Hierarchical | Medium | Medium | High (structure) |

## Next Steps

- See [02_text_augmentations.md](02_text_augmentations.md) for text augmentations
- See [../multimodal/02_text_image_pairs.md](../multimodal/02_text_image_pairs.md) for text+image
