# Text Modality: Complete Learning Path

A comprehensive curriculum for understanding text processing, tokenization, language models, and LLM systems. Built from foundational NLP to cutting-edge research (GPT-4, Llama 3, DeepSeek V3).

## Curriculum Philosophy

- **Core Depth Over Abstractions**: Every concept includes profiled implementations
- **Research-Grounded**: Each module references seminal papers and latest research
- **Practical Focus**: Runnable code, benchmarks, and production considerations

---

## Module Overview (50+ Files Planned)

```
26-text-modality/
├── 01-foundations/                      # Core text concepts (8 files)
│   ├── 00_nlp_history_statistical_to_neural.md  # N-grams → RNN → Transformers
│   ├── 01_text_preprocessing_fundamentals.md    # Cleaning, normalization, regex
│   ├── 02_text_preprocessing_profiled.py        # Benchmarked implementations
│   ├── 03_unicode_encodings_deep_dive.md        # UTF-8, UTF-16, normalization forms
│   ├── 04_regular_expressions_nlp.py            # Regex patterns for NLP
│   ├── 05_text_fundamentals.c                   # Pure C string processing
│   ├── 06_linguistic_features.md                # POS, NER, dependency parsing
│   └── 07_spacy_nltk_comparison.py              # Library comparison with benchmarks
│
├── 02-tokenization/                     # Tokenization deep dive (10 files)
│   ├── 01_tokenization_fundamentals.md          # Why tokenization matters
│   ├── 02_bpe_algorithm_deep_dive.md            # Byte-Pair Encoding from scratch
│   ├── 03_bpe_implementation.py                 # Complete BPE implementation
│   ├── 04_wordpiece_algorithm.md                # WordPiece (BERT tokenizer)
│   ├── 05_unigram_sentencepiece.md              # Unigram LM tokenization
│   ├── 06_sentencepiece_implementation.py       # SentencePiece training/inference
│   ├── 07_tiktoken_analysis.md                  # OpenAI's tiktoken internals
│   ├── 08_tiktoken_custom_training.py           # Custom vocabulary training
│   ├── 09_tokenizer_comparison_benchmark.py     # Speed/quality comparison
│   └── 10_byte_level_bpe.md                     # GPT-2/3/4 style tokenization
│
├── 03-embeddings/                       # Text representations (8 files)
│   ├── 01_word_embeddings_history.md            # One-hot → Word2Vec → Transformers
│   ├── 02_word2vec_implementation.py            # CBOW and Skip-gram from scratch
│   ├── 03_glove_fasttext.md                     # GloVe, FastText architectures
│   ├── 04_contextual_embeddings.md              # ELMo → BERT contextualization
│   ├── 05_sentence_transformers.md              # Sentence-BERT, contrastive learning
│   ├── 06_sentence_embeddings_profiled.py       # Embedding generation benchmarks
│   ├── 07_embedding_similarity_search.py        # Cosine similarity, FAISS, ANN
│   └── 08_embedding_visualization.ipynb         # t-SNE, UMAP visualization
│
├── 04-attention-mechanisms/             # Attention deep dive (7 files)
│   ├── 01_attention_fundamentals.md             # Seq2seq attention origins
│   ├── 02_self_attention_math.md                # QKV, scaled dot-product
│   ├── 03_multi_head_attention.py               # MHA implementation from scratch
│   ├── 04_attention_variants.md                 # MQA, GQA, sliding window
│   ├── 05_flash_attention_explained.md          # Memory-efficient attention
│   ├── 06_flash_attention_cuda.cu               # CUDA FlashAttention kernel
│   └── 07_attention_visualization.ipynb         # Attention pattern analysis
│
├── 05-transformer-architecture/         # Transformer internals (8 files)
│   ├── 01_transformer_original_paper.md         # "Attention Is All You Need"
│   ├── 02_encoder_decoder_architecture.md       # Full architecture breakdown
│   ├── 03_transformer_from_scratch.py           # Complete implementation
│   ├── 04_positional_encodings.md               # Sinusoidal, learned, RoPE, ALiBi
│   ├── 05_rope_implementation.py                # Rotary Position Embedding
│   ├── 06_layer_normalization.md                # Pre-LN vs Post-LN, RMSNorm
│   ├── 07_feed_forward_networks.md              # FFN, GLU, SwiGLU variants
│   └── 08_transformer_profiled.py               # Performance analysis
│
├── 06-language-models/                  # LM architectures (10 files)
│   ├── 01_language_modeling_fundamentals.md     # Perplexity, autoregressive LM
│   ├── 02_bert_architecture.md                  # BERT, masked LM, NSP
│   ├── 03_gpt_architecture_evolution.md         # GPT-1 → GPT-2 → GPT-3 → GPT-4
│   ├── 04_llama_architecture.md                 # Llama 1/2/3, architectural choices
│   ├── 05_mistral_mixtral.md                    # Mistral, Mixtral MoE
│   ├── 06_deepseek_architecture.md              # DeepSeek V2/V3, MLA attention
│   ├── 07_moe_mixture_of_experts.md             # Sparse MoE, routing, load balancing
│   ├── 08_llm_comparison_table.md               # Architecture comparison matrix
│   ├── 09_small_language_models.md              # Phi, Gemma, efficient LLMs
│   └── 10_llm_from_scratch.py                   # Mini-LLM implementation
│
├── 07-training-methods/                 # Training techniques (8 files)
│   ├── 01_pretraining_objectives.md             # CLM, MLM, span corruption
│   ├── 02_instruction_tuning.md                 # SFT, instruction datasets
│   ├── 03_rlhf_explained.md                     # RLHF pipeline, reward modeling
│   ├── 04_dpo_direct_preference.md              # DPO, IPO, KTO alternatives
│   ├── 05_lora_qlora.md                         # Parameter-efficient fine-tuning
│   ├── 06_lora_implementation.py                # LoRA from scratch
│   ├── 07_full_finetuning_vs_peft.md            # When to use what
│   └── 08_training_recipes.py                   # Complete training scripts
│
├── 08-inference-optimization/           # LLM inference (8 files)
│   ├── 01_kv_cache_explained.md                 # KV caching mechanics
│   ├── 02_kv_cache_implementation.py            # KV cache from scratch
│   ├── 03_quantization_methods.md               # INT8, INT4, GPTQ, AWQ, GGUF
│   ├── 04_quantization_benchmark.py             # Quality vs speed tradeoffs
│   ├── 05_speculative_decoding.md               # Draft model acceleration
│   ├── 06_continuous_batching.md                # vLLM, TensorRT-LLM batching
│   ├── 07_vllm_tensorrt_comparison.py           # Inference engine benchmarks
│   └── 08_serving_optimization.md               # Production deployment
│
├── 09-text-generation/                  # Generation methods (6 files)
│   ├── 01_decoding_strategies.md                # Greedy, beam, sampling
│   ├── 02_sampling_methods.py                   # Top-k, top-p, temperature
│   ├── 03_constrained_generation.md             # Structured output, JSON mode
│   ├── 04_prompt_engineering.md                 # Few-shot, CoT, system prompts
│   ├── 05_rag_retrieval_augmented.md            # RAG architecture
│   └── 06_rag_implementation.py                 # Complete RAG pipeline
│
├── 10-nlp-tasks/                        # Classic NLP tasks (6 files)
│   ├── 01_text_classification.py                # Sentiment, topic classification
│   ├── 02_named_entity_recognition.py           # NER with transformers
│   ├── 03_question_answering.md                 # Extractive, generative QA
│   ├── 04_summarization.md                      # Abstractive, extractive
│   ├── 05_machine_translation.md                # Seq2seq, multilingual
│   └── 06_semantic_similarity.py                # STS benchmarks
│
├── 11-optimization-profiling/           # Performance engineering (4 files)
│   ├── 01_text_data_loading.md                  # Efficient text datasets
│   ├── 02_huggingface_datasets_profiled.py      # HF datasets optimization
│   ├── 03_tokenizer_parallelization.py          # Parallel tokenization
│   └── 04_memory_optimization.md                # Gradient checkpointing, offloading
│
├── 12-practical-notebooks/              # Hands-on experiments (5 files)
│   ├── 01_tokenizer_from_scratch.ipynb          # Build BPE tokenizer
│   ├── 02_transformer_from_scratch.ipynb        # Build transformer
│   ├── 03_finetune_llm_qlora.ipynb              # QLoRA fine-tuning
│   ├── 04_exercises_and_solutions.py            # Graded exercises
│   └── 05_llm_inference_optimization.ipynb      # Optimization techniques
│
├── 13-advanced-topics/                  # Cutting-edge research (5 files)
│   ├── 01_long_context_methods.md               # RoPE scaling, landmark attention
│   ├── 02_multimodal_text_integration.md        # Text in VLMs, audio LLMs
│   ├── 03_reasoning_models.md                   # Chain-of-thought, o1, R1
│   ├── 04_agents_tool_use.md                    # Function calling, agents
│   └── 05_latest_research_2025.md               # Most recent developments
│
├── papers/                              # Reference materials
│   └── paper_summaries.md                       # All papers summarized
│
├── resources/                           # Learning resources
│   ├── glossary.md                              # 100+ NLP terms defined
│   └── external_links.md                        # Datasets, tools, community
│
└── README.md                            # This file
```

---

## Learning Progression

### Phase 1: Foundations (Week 1-2)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 01 | Text Preprocessing | - |
| 01 | Unicode & Encodings | - |
| 02 | Tokenization Fundamentals | [BPE 2015](https://arxiv.org/abs/1508.07909) |
| 02 | BPE, WordPiece, Unigram | [SentencePiece 2018](https://arxiv.org/abs/1808.06226) |

### Phase 2: Embeddings & Attention (Week 3-4)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 03 | Word2Vec, GloVe | [Word2Vec 2013](https://arxiv.org/abs/1301.3781) |
| 03 | Sentence Transformers | [Sentence-BERT 2019](https://arxiv.org/abs/1908.10084) |
| 04 | Attention Mechanisms | [Attention 2014](https://arxiv.org/abs/1409.0473) |
| 04 | Flash Attention | [FlashAttention 2022](https://arxiv.org/abs/2205.14135) |

### Phase 3: Transformers (Week 5-6)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 05 | Transformer Architecture | [AIAYN 2017](https://arxiv.org/abs/1706.03762) |
| 05 | Positional Encodings | [RoPE 2021](https://arxiv.org/abs/2104.09864) |
| 06 | BERT Architecture | [BERT 2018](https://arxiv.org/abs/1810.04805) |
| 06 | GPT Architecture | [GPT-2 2019](https://openai.com/research/better-language-models) |

### Phase 4: Modern LLMs (Week 7-9)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 06 | Llama Architecture | [Llama 2 2023](https://arxiv.org/abs/2307.09288) |
| 06 | Mixture of Experts | [Mixtral 2024](https://arxiv.org/abs/2401.04088) |
| 06 | DeepSeek | [DeepSeek V3 2024](https://arxiv.org/abs/2412.19437) |
| 07 | RLHF & DPO | [DPO 2023](https://arxiv.org/abs/2305.18290) |

### Phase 5: Training & Inference (Week 10-12)
| Module | Topic | Key Papers |
|--------|-------|------------|
| 07 | LoRA, QLoRA | [LoRA 2021](https://arxiv.org/abs/2106.09685) |
| 08 | KV Cache | - |
| 08 | Quantization | [GPTQ 2022](https://arxiv.org/abs/2210.17323) |
| 08 | Speculative Decoding | [Spec Decoding 2023](https://arxiv.org/abs/2302.01318) |

### Phase 6: Production (Week 13-14)
| Module | Topic | Resources |
|--------|-------|-----------|
| 09 | RAG Systems | - |
| 11 | Data Loading | - |
| 12 | Practical Notebooks | - |

---

## Key Research Papers

### Foundational (2013-2018)
1. **Word2Vec** (2013) - Efficient word embeddings
2. **GloVe** (2014) - Global vectors for word representation
3. **Attention** (2014) - Neural machine translation attention
4. **Transformer** (2017) - "Attention Is All You Need"
5. **BERT** (2018) - Bidirectional pre-training
6. **GPT** (2018) - Generative pre-training

### Modern LLMs (2019-2023)
7. **GPT-2/3** (2019/2020) - Scaling language models
8. **T5** (2019) - Text-to-text framework
9. **RoPE** (2021) - Rotary positional embeddings
10. **InstructGPT** (2022) - RLHF for alignment
11. **Llama** (2023) - Open-weight LLMs
12. **Mistral/Mixtral** (2023/2024) - Efficient architectures

### Cutting-Edge (2024-2025)
13. **Llama 3** (2024) - Latest Meta LLM
14. **DeepSeek V3** (2024) - MLA attention, efficient training
15. **Qwen 2.5** (2024) - Alibaba's LLM series
16. **o1/R1** (2024/2025) - Reasoning models

---

## Profiling Focus Areas

### Memory & Bandwidth
- Tokenization: Vocabulary size impact on memory
- Embeddings: Float32 vs Float16 vs INT8
- KV Cache: Memory growth with sequence length
- Batch processing vs streaming patterns

### Computation
- Attention: O(n²) complexity, FlashAttention optimization
- FFN: Parameter count vs compute
- Inference: Prefill vs decode phases

### Model Comparisons
| Model | Parameters | Context | Architecture |
|-------|------------|---------|--------------|
| GPT-2 | 1.5B | 1K | Standard transformer |
| Llama 2 | 7-70B | 4K | RoPE, GQA |
| Mistral | 7B | 32K | Sliding window |
| Mixtral | 8x7B | 32K | Sparse MoE |
| DeepSeek V3 | 671B | 128K | MLA, MoE |

---

## Prerequisites

1. **Python**: Intermediate level
2. **PyTorch**: Basic tensor operations
3. **Linear Algebra**: Matrix operations, attention math
4. **Probability**: Language modeling basics

---

## Quick Start

```bash
# Setup environment
pip install torch transformers datasets tokenizers
pip install sentencepiece tiktoken
pip install accelerate bitsandbytes  # For efficient inference

# Clone reference implementations
git clone https://github.com/karpathy/nanoGPT
git clone https://github.com/huggingface/transformers
```

---

## Status Tracker

| Module | Status | Last Updated |
|--------|--------|--------------|
| 01-foundations | 🟡 Planned | Dec 2024 |
| 02-tokenization | 🟡 Planned | Dec 2024 |
| 03-embeddings | 🟡 Planned | Dec 2024 |
| 04-attention-mechanisms | 🟡 Planned | Dec 2024 |
| 05-transformer-architecture | 🟡 Planned | Dec 2024 |
| 06-language-models | 🟡 Planned | Dec 2024 |
| 07-training-methods | 🟡 Planned | Dec 2024 |
| 08-inference-optimization | 🟡 Planned | Dec 2024 |
| 09-text-generation | 🟡 Planned | Dec 2024 |
| 10-nlp-tasks | 🟡 Planned | Dec 2024 |
| 11-optimization-profiling | 🟡 Planned | Dec 2024 |
| 12-practical-notebooks | 🟡 Planned | Dec 2024 |
| 13-advanced-topics | 🟡 Planned | Dec 2024 |

---

## Estimated Time: 14-16 weeks
