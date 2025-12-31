# Text/NLP Glossary

## Tokenization Terms

| Term | Definition |
|------|------------|
| **BPE** | Byte-Pair Encoding - merges frequent character pairs |
| **WordPiece** | Google's tokenizer using likelihood-based merging |
| **Unigram** | Probabilistic tokenizer using language model |
| **SentencePiece** | Language-agnostic tokenizer library |
| **tiktoken** | OpenAI's fast BPE tokenizer |
| **Vocabulary** | Set of all tokens the model knows |
| **OOV** | Out-of-vocabulary token |
| **[UNK]** | Unknown token placeholder |
| **[CLS]** | Classification token (BERT) |
| **[SEP]** | Separator token |
| **[PAD]** | Padding token |
| **[MASK]** | Mask token for MLM |

## Embedding Terms

| Term | Definition |
|------|------------|
| **Word2Vec** | Neural word embeddings (CBOW, Skip-gram) |
| **GloVe** | Global Vectors from co-occurrence |
| **FastText** | Subword-aware embeddings |
| **Contextual Embedding** | Position-dependent representation |
| **Sentence Embedding** | Fixed-size sentence representation |
| **Embedding Dimension** | Size of embedding vector |

## Attention Terms

| Term | Definition |
|------|------------|
| **Self-Attention** | Attention within same sequence |
| **Cross-Attention** | Attention between different sequences |
| **Multi-Head Attention** | Parallel attention heads |
| **MHA** | Multi-Head Attention |
| **MQA** | Multi-Query Attention (shared KV) |
| **GQA** | Grouped-Query Attention |
| **KV Cache** | Cached key-value pairs for inference |
| **Flash Attention** | Memory-efficient attention algorithm |
| **Causal Mask** | Prevents attending to future tokens |

## Architecture Terms

| Term | Definition |
|------|------------|
| **Transformer** | Attention-based architecture |
| **Encoder** | Bidirectional context processor |
| **Decoder** | Autoregressive generator |
| **FFN** | Feed-Forward Network |
| **LayerNorm** | Layer Normalization |
| **RMSNorm** | Root Mean Square Normalization |
| **RoPE** | Rotary Position Embedding |
| **ALiBi** | Attention with Linear Biases |
| **GLU** | Gated Linear Unit |
| **SwiGLU** | Swish-Gated Linear Unit |
| **MoE** | Mixture of Experts |

## Language Model Terms

| Term | Definition |
|------|------------|
| **LLM** | Large Language Model |
| **CLM** | Causal Language Modeling |
| **MLM** | Masked Language Modeling |
| **Perplexity** | Exp of cross-entropy loss |
| **Autoregressive** | Predicting next token |
| **Context Length** | Maximum sequence length |
| **Temperature** | Sampling randomness control |
| **Top-k** | Sample from k most likely tokens |
| **Top-p (Nucleus)** | Sample from cumulative probability p |

## Training Terms

| Term | Definition |
|------|------------|
| **Pretraining** | Initial training on large corpus |
| **Fine-tuning** | Task-specific training |
| **SFT** | Supervised Fine-Tuning |
| **RLHF** | RL from Human Feedback |
| **DPO** | Direct Preference Optimization |
| **LoRA** | Low-Rank Adaptation |
| **QLoRA** | Quantized LoRA |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **Gradient Checkpointing** | Memory-saving recomputation |

## Inference Terms

| Term | Definition |
|------|------------|
| **Prefill** | Initial prompt processing |
| **Decode** | Token-by-token generation |
| **Speculative Decoding** | Draft model acceleration |
| **Continuous Batching** | Dynamic batch management |
| **PagedAttention** | vLLM's memory management |
| **Quantization** | Reduced precision weights |
| **GPTQ** | Post-training quantization |
| **AWQ** | Activation-aware quantization |

## Metrics

| Term | Definition |
|------|------------|
| **BLEU** | Machine translation metric |
| **ROUGE** | Summarization metric |
| **BERTScore** | Semantic similarity metric |
| **Perplexity** | Language model quality |
| **F1** | Harmonic mean of precision/recall |
| **Accuracy** | Correct predictions ratio |
