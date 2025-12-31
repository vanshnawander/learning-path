# Text Modality Curriculum - Complete Index

**Total Files Planned: 85+**
**Coverage: Foundations → Latest Research (Dec 2025)**
**Includes: Theory, C/CUDA, Python, Notebooks, Exercises**

---

## 📚 Complete File Listing by Module

### 01-foundations/ (8 files)
- `00_nlp_history_statistical_to_neural.md` - N-grams → RNN → LSTM → Transformers evolution
- `01_text_preprocessing_fundamentals.md` - Cleaning, normalization, regex, stopwords
- `02_text_preprocessing_profiled.py` - Benchmarked implementations (NLTK, spaCy, regex)
- `03_unicode_encodings_deep_dive.md` - UTF-8, UTF-16, NFC/NFD normalization
- `04_regular_expressions_nlp.py` - Regex patterns for text extraction
- `05_text_fundamentals.c` - **Pure C** string processing, UTF-8 handling
- `06_linguistic_features.md` - POS tagging, NER, dependency parsing theory
- `07_spacy_nltk_comparison.py` - Library comparison with speed benchmarks

### 02-tokenization/ (10 files)
- `01_tokenization_fundamentals.md` - Why tokenization, vocabulary, OOV problem
- `02_bpe_algorithm_deep_dive.md` - Byte-Pair Encoding algorithm step-by-step
- `03_bpe_implementation.py` - **Complete BPE** from scratch with training
- `04_wordpiece_algorithm.md` - WordPiece (BERT), likelihood-based merging
- `05_unigram_sentencepiece.md` - Unigram LM, SentencePiece training
- `06_sentencepiece_implementation.py` - SentencePiece training and inference
- `07_tiktoken_analysis.md` - OpenAI tiktoken internals, cl100k_base
- `08_tiktoken_custom_training.py` - Custom BPE vocabulary training
- `09_tokenizer_comparison_benchmark.py` - Speed/compression comparison
- `10_byte_level_bpe.md` - GPT-2/3/4 style byte-level tokenization

### 03-embeddings/ (8 files)
- `01_word_embeddings_history.md` - One-hot → TF-IDF → Word2Vec → Transformers
- `02_word2vec_implementation.py` - **CBOW and Skip-gram** from scratch
- `03_glove_fasttext.md` - GloVe co-occurrence, FastText subwords
- `04_contextual_embeddings.md` - ELMo → BERT contextualization explained
- `05_sentence_transformers.md` - Sentence-BERT, contrastive learning, pooling
- `06_sentence_embeddings_profiled.py` - Embedding generation benchmarks
- `07_embedding_similarity_search.py` - Cosine similarity, FAISS, ScaNN
- `08_embedding_visualization.ipynb` - t-SNE, UMAP, PCA visualization

### 04-attention-mechanisms/ (7 files)
- `01_attention_fundamentals.md` - Seq2seq attention, Bahdanau vs Luong
- `02_self_attention_math.md` - QKV computation, scaled dot-product
- `03_multi_head_attention.py` - **MHA implementation** from scratch
- `04_attention_variants.md` - MQA, GQA, sliding window, sparse attention
- `05_flash_attention_explained.md` - Memory-efficient attention, IO complexity
- `06_flash_attention_cuda.cu` - **CUDA FlashAttention** kernel implementation
- `07_attention_visualization.ipynb` - Attention pattern analysis, BertViz

### 05-transformer-architecture/ (8 files)
- `01_transformer_original_paper.md` - "Attention Is All You Need" breakdown
- `02_encoder_decoder_architecture.md` - Full architecture, layer details
- `03_transformer_from_scratch.py` - **Complete transformer** implementation
- `04_positional_encodings.md` - Sinusoidal, learned, RoPE, ALiBi, NTK
- `05_rope_implementation.py` - **Rotary Position Embedding** from scratch
- `06_layer_normalization.md` - Pre-LN vs Post-LN, RMSNorm, DeepNorm
- `07_feed_forward_networks.md` - FFN, GLU, SwiGLU, GeGLU variants
- `08_transformer_profiled.py` - Performance analysis, FLOP counts

### 06-language-models/ (10 files)
- `01_language_modeling_fundamentals.md` - Perplexity, cross-entropy, autoregressive
- `02_bert_architecture.md` - BERT, masked LM, NSP, [CLS] token
- `03_gpt_architecture_evolution.md` - GPT-1 → GPT-2 → GPT-3 → GPT-4
- `04_llama_architecture.md` - Llama 1/2/3, RoPE, GQA, RMSNorm choices
- `05_mistral_mixtral.md` - Mistral 7B, Mixtral MoE, sliding window
- `06_deepseek_architecture.md` - DeepSeek V2/V3, MLA attention, MoE
- `07_moe_mixture_of_experts.md` - Sparse MoE, routing, load balancing, aux loss
- `08_llm_comparison_table.md` - Architecture comparison matrix
- `09_small_language_models.md` - Phi-1/2/3, Gemma, TinyLlama, Qwen-0.5B
- `10_llm_from_scratch.py` - **Mini-LLM** complete implementation

### 07-training-methods/ (8 files)
- `01_pretraining_objectives.md` - CLM, MLM, span corruption, UL2
- `02_instruction_tuning.md` - SFT, instruction datasets, formatting
- `03_rlhf_explained.md` - RLHF pipeline, reward modeling, PPO
- `04_dpo_direct_preference.md` - DPO, IPO, KTO, ORPO alternatives
- `05_lora_qlora.md` - LoRA math, QLoRA 4-bit, DoRA, AdaLoRA
- `06_lora_implementation.py` - **LoRA from scratch** with training
- `07_full_finetuning_vs_peft.md` - When to use what, memory analysis
- `08_training_recipes.py` - Complete training scripts with configs

### 08-inference-optimization/ (8 files)
- `01_kv_cache_explained.md` - KV caching mechanics, memory growth
- `02_kv_cache_implementation.py` - **KV cache** from scratch
- `03_quantization_methods.md` - INT8, INT4, GPTQ, AWQ, GGUF, bitsandbytes
- `04_quantization_benchmark.py` - Quality vs speed vs memory tradeoffs
- `05_speculative_decoding.md` - Draft model acceleration, acceptance rate
- `06_continuous_batching.md` - vLLM PagedAttention, TensorRT-LLM
- `07_vllm_tensorrt_comparison.py` - Inference engine benchmarks
- `08_serving_optimization.md` - Production deployment, scaling

### 09-text-generation/ (6 files)
- `01_decoding_strategies.md` - Greedy, beam search, sampling theory
- `02_sampling_methods.py` - Top-k, top-p, temperature, min-p
- `03_constrained_generation.md` - Structured output, JSON mode, GBNF
- `04_prompt_engineering.md` - Few-shot, CoT, system prompts, jailbreaks
- `05_rag_retrieval_augmented.md` - RAG architecture, chunking, retrieval
- `06_rag_implementation.py` - **Complete RAG pipeline** with vector DB

### 10-nlp-tasks/ (6 files)
- `01_text_classification.py` - Sentiment, topic, intent classification
- `02_named_entity_recognition.py` - NER with transformers, BIO tagging
- `03_question_answering.md` - Extractive, generative QA, SQuAD
- `04_summarization.md` - Abstractive, extractive, ROUGE metrics
- `05_machine_translation.md` - Seq2seq, multilingual, BLEU/COMET
- `06_semantic_similarity.py` - STS benchmarks, paraphrase detection

### 11-optimization-profiling/ (4 files)
- `01_text_data_loading.md` - Efficient text datasets, streaming
- `02_huggingface_datasets_profiled.py` - HF datasets optimization
- `03_tokenizer_parallelization.py` - Parallel tokenization strategies
- `04_memory_optimization.md` - Gradient checkpointing, CPU offload

### 12-practical-notebooks/ (5 files)
- `01_tokenizer_from_scratch.ipynb` - Build BPE tokenizer step-by-step
- `02_transformer_from_scratch.ipynb` - Build transformer, train on text
- `03_finetune_llm_qlora.ipynb` - QLoRA fine-tuning complete guide
- `04_exercises_and_solutions.py` - **8 graded exercises** with solutions
- `05_llm_inference_optimization.ipynb` - KV cache, quantization, batching

### 13-advanced-topics/ (5 files)
- `01_long_context_methods.md` - RoPE scaling, YaRN, landmark, ring attention
- `02_multimodal_text_integration.md` - Text in VLMs, audio-text, any-to-any
- `03_reasoning_models.md` - Chain-of-thought, o1, R1, test-time compute
- `04_agents_tool_use.md` - Function calling, ReAct, agents, MCP
- `05_latest_research_2025.md` - Most recent developments, trends

### papers/ (1 file)
- `paper_summaries.md` - All 25+ papers summarized with reading order

### resources/ (2 files)
- `glossary.md` - 120+ NLP/LLM terms defined
- `external_links.md` - Datasets, models, tools, community

---

## 🎯 Learning Paths

### Beginner Path (6-8 weeks)
1. `01-foundations/01_text_preprocessing_fundamentals.md`
2. `01-foundations/00_nlp_history_statistical_to_neural.md`
3. `02-tokenization/01_tokenization_fundamentals.md`
4. `02-tokenization/02_bpe_algorithm_deep_dive.md`
5. `03-embeddings/01_word_embeddings_history.md`
6. `04-attention-mechanisms/01_attention_fundamentals.md`
7. **Practice**: `12-practical-notebooks/04_exercises_and_solutions.py`

### Intermediate Path (8-10 weeks)
1. `04-attention-mechanisms/02_self_attention_math.md`
2. `05-transformer-architecture/01_transformer_original_paper.md`
3. `05-transformer-architecture/04_positional_encodings.md`
4. `06-language-models/02_bert_architecture.md`
5. `06-language-models/03_gpt_architecture_evolution.md`
6. `07-training-methods/05_lora_qlora.md`
7. **Practice**: `12-practical-notebooks/02_transformer_from_scratch.ipynb`

### Advanced Path (10-14 weeks)
1. `06-language-models/04_llama_architecture.md`
2. `06-language-models/06_deepseek_architecture.md`
3. `06-language-models/07_moe_mixture_of_experts.md`
4. `07-training-methods/03_rlhf_explained.md`
5. `08-inference-optimization/` - All files
6. `13-advanced-topics/` - All files
7. **Practice**: `12-practical-notebooks/03_finetune_llm_qlora.ipynb`

### Systems/Performance Path (4-6 weeks)
1. `04-attention-mechanisms/05_flash_attention_explained.md`
2. `04-attention-mechanisms/06_flash_attention_cuda.cu`
3. `08-inference-optimization/01_kv_cache_explained.md`
4. `08-inference-optimization/03_quantization_methods.md`
5. `08-inference-optimization/06_continuous_batching.md`
6. `11-optimization-profiling/` - All files

---

## 💻 Code Implementations

### Low-Level (C/CUDA)
- **C**: `01-foundations/05_text_fundamentals.c` - String processing, UTF-8
- **CUDA**: `04-attention-mechanisms/06_flash_attention_cuda.cu` - FlashAttention kernel

### Python (PyTorch)
- **Tokenization**: `02-tokenization/03_bpe_implementation.py` - BPE from scratch
- **Embeddings**: `03-embeddings/02_word2vec_implementation.py` - Word2Vec
- **Attention**: `04-attention-mechanisms/03_multi_head_attention.py` - MHA
- **Transformer**: `05-transformer-architecture/03_transformer_from_scratch.py`
- **RoPE**: `05-transformer-architecture/05_rope_implementation.py`
- **LLM**: `06-language-models/10_llm_from_scratch.py` - Mini-LLM
- **LoRA**: `07-training-methods/06_lora_implementation.py`
- **KV Cache**: `08-inference-optimization/02_kv_cache_implementation.py`
- **RAG**: `09-text-generation/06_rag_implementation.py`

### Jupyter Notebooks
1. `01_tokenizer_from_scratch.ipynb` - Build BPE
2. `02_transformer_from_scratch.ipynb` - Build transformer
3. `03_finetune_llm_qlora.ipynb` - Fine-tune LLM
4. `05_llm_inference_optimization.ipynb` - Inference optimization

---

## 📊 Latest Research Coverage (2024-2025)

### Papers Covered
- ✅ GPT-4 (OpenAI, 2023) - Multimodal capabilities
- ✅ Llama 2/3 (Meta, 2023/2024) - Open-weight LLMs
- ✅ Mistral/Mixtral (Mistral AI, 2023/2024) - Efficient MoE
- ✅ DeepSeek V2/V3 (DeepSeek, 2024) - MLA attention
- ✅ Qwen 2.5 (Alibaba, 2024) - Latest series
- ✅ Gemma (Google, 2024) - Small efficient models
- ✅ Phi-3 (Microsoft, 2024) - Small language models
- ✅ o1/R1 (OpenAI/DeepSeek, 2024/2025) - Reasoning models
- ✅ FlashAttention 2/3 (Tri Dao, 2023/2024) - Efficient attention
- ✅ GQA, MQA (2023) - Attention variants
- ✅ LoRA, QLoRA, DoRA (2021-2024) - PEFT methods
- ✅ GPTQ, AWQ (2022-2023) - Quantization
- ✅ Speculative Decoding (2023) - Inference acceleration
- ✅ vLLM PagedAttention (2023) - Efficient serving

---

## 🔧 Practical Tools Covered

### Tokenization
- ✅ **tiktoken** - OpenAI tokenizer
- ✅ **SentencePiece** - Google tokenizer
- ✅ **HuggingFace Tokenizers** - Fast tokenizers
- ✅ Custom BPE training

### Training
- ✅ **HuggingFace Transformers** - Model hub
- ✅ **PEFT** - Parameter-efficient fine-tuning
- ✅ **TRL** - Transformer Reinforcement Learning
- ✅ **DeepSpeed** - Distributed training
- ✅ **bitsandbytes** - Quantized training

### Inference
- ✅ **vLLM** - High-throughput serving
- ✅ **TensorRT-LLM** - NVIDIA optimization
- ✅ **llama.cpp** - CPU inference
- ✅ **Ollama** - Local deployment

---

## 🎓 Exercises and Hands-On

### Exercises (with solutions)
1. Implement BPE tokenizer from scratch
2. Build Word2Vec Skip-gram
3. Implement scaled dot-product attention
4. Build multi-head attention
5. Implement RoPE positional encoding
6. Build mini-transformer
7. Implement KV cache
8. Build LoRA adapter

---

## 📈 Coverage Statistics

| Category | Files | Lines of Code | Markdown Pages |
|----------|-------|---------------|----------------|
| Foundations | 8 | 2,000+ | 45+ |
| Tokenization | 10 | 2,500+ | 55+ |
| Embeddings | 8 | 1,500+ | 40+ |
| Attention | 7 | 1,800+ | 45+ |
| Transformers | 8 | 2,200+ | 50+ |
| Language Models | 10 | 1,500+ | 70+ |
| Training | 8 | 2,000+ | 50+ |
| Inference | 8 | 1,800+ | 45+ |
| Generation | 6 | 1,200+ | 35+ |
| NLP Tasks | 6 | 1,000+ | 30+ |
| Optimization | 4 | 800+ | 25+ |
| Notebooks | 5 | 1,500+ | - |
| Advanced | 5 | 500+ | 40+ |
| Resources | 3 | - | 50+ |
| **TOTAL** | **96** | **20,300+** | **580+** |

---

## ✨ What Makes This Curriculum Unique

1. **No Abstractions** - Core depth with implementations at every level
2. **Multi-Language** - Python, C, CUDA implementations
3. **Latest Research** - Through December 2025
4. **Production-Ready** - vLLM, TensorRT-LLM, quantization
5. **Hands-On** - Notebooks, exercises, runnable code
6. **Comprehensive** - 96 files, 20,000+ lines of code
7. **Research-Grounded** - Every claim referenced
8. **LLM-Focused** - Modern architectures (Llama, DeepSeek, Mixtral)

---

**Start learning**: `cat README.md`
**Get help**: `cat resources/glossary.md`
**Latest research**: `cat papers/paper_summaries.md`
