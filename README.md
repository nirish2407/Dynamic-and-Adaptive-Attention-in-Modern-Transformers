# Dynamic and Adaptive Attention in Modern Transformers

Seminar project for **Newest Trends in High-Performance Data Analytics**, exploring how attention mechanisms have evolved from the original Transformer formulation toward **dynamic, adaptive, sparse, memory-efficient, and interpretable computation**.

---

## Overview

Attention is a fundamental mechanism in modern neural network architectures. The original motivation for attention was to overcome limitations of recurrent sequence-to-sequence models, particularly their difficulty with long-range dependencies and fixed-size information bottlenecks.

With the introduction of the Transformer, attention became the central mechanism for modeling relationships between tokens. However, as Transformer models and context lengths have grown, full attention has introduced significant computational and memory challenges.

This project studies how modern Transformer architectures address these challenges by making attention and computation more **selective, dynamic and adaptive**.

The seminar covers:

- Self-attention and scaled dot-product attention
- Multi-head attention
- Encoder-decoder attention
- Rotary Positional Embeddings (RoPE)
- KV-cache and the memory wall
- Grouped-Query Attention (GQA)
- Multi-Query Attention (MQA)
- PagedAttention
- FlashAttention
- Model quantization
- Sparse and linear attention
- Learned and dynamic sparse attention
- Adaptive attention spans
- Dynamic attention masks
- Top-k attention
- Dynamic token selection and pruning
- Token merging
- Attention interpretability
- Attention rollout and attention flow

The overall perspective is that attention is increasingly becoming a mechanism for **dynamic allocation of computational resources**, rather than simply a mechanism for assigning importance to input tokens.

## Repository contents

```
.
├── Bart_Paraphrase_Type_Detection/
├── Presentation/
│   └── Attention_Presentation.pdf
├── Report/
│   └── Attention_Report.pdf
├── GPT_2_Style_Model.ipynb
├── Attention_Practical_Implementation_1.ipynb
├── Attention_Practical_Implementation_2.ipynb
├── Attention_Practical_Implementation_3.ipynb
├── Attention_Practical_Implementation_4.ipynb
├── Attention_Practical_Implementation_5.ipynb
└── README.md
```

## Practical implementation notebooks

| # | Notebook | What it covers |
|---|---|---|
| 1 | [GPT_2_Style_Model.ipynb](GPT_2_Style_Model.ipynb) | Builds a GPT-2 small (124M parameters) in plain PyTorch and trains it from scratch on Tiny Shakespeare using the GPT-2 BPE tokenizer (tiktoken). Covers causal self-attention, MLP and transformer blocks, weight tying, AdamW with gradient accumulation and clipping, a warmup and cosine LR schedule, bfloat16 autocast, top-k/temperature text generation, and checkpoint saving to Google Drive. Runs on a Colab T4 GPU. |
| 1 | [Attention_Practical_Implementation_1.ipynb](Attention_Practical_Implementation_1.ipynb) | Scaled dot-product attention, multi-head attention, attention visualization, RoPE relative-position property, KV cache speed test, GQA/MQA cache-size formula, sliding-window mask, BERT attention rollout, quantization memory |
| 2 | [Attention_Practical_Implementation_2.ipynb](Attention_Practical_Implementation_2.ipynb) | Tiled FlashAttention with online softmax, Longformer/BigBird masks, entmax/sparsemax, top-k/LSH/routing masks, learnable attention span, token pruning, ToMe merging, H2O cache eviction, rollout vs. flow |
| 3 | [Attention_Practical_Implementation_3.ipynb](Attention_Practical_Implementation_3.ipynb) | BERT query-dependence, KV cache on GPT-2 and Qwen (GQA), OPT-125m quantization, FlashAttention vs. PyTorch SDPA, sparse attention on GPT-2, per-head span calibration, ToMe in ViT-B/16, rollout and max-flow attribution |
| 4 | [Attention_Practical_Implementation_4.ipynb](Attention_Practical_Implementation_4.ipynb) | GPT-2 and DistilBERT attention inspection, post-hoc top-k, per-head effective span, attention-based token masking |
| 5 | [Attention_Practical_Implementation_5.ipynb](Attention_Practical_Implementation_5.ipynb) | RoPE offset invariance and frequency spectrum, attention entropy and k90, top-p and block-routing attention, head ablation and pruning, dynamic per-token head gating |
| 6 | [Bart_Paraphrase_Type_Detection](Bart%20Paraphrase%20Type%20Detection/) | Multi-label paraphrase type detection on the ETPC corpus (26 fine-grained types, severe class imbalance) by fine-tuning BART-large, with dev MCC as the main metric. Performs 13 controlled experiments: early stopping, LR warmup/decay, iterative multilabel stratified splitting, dropout and gradient clipping, BCEWithLogitsLoss with class weights, per-class MCC-optimized thresholds, pooling comparison (mean/CLS/max/EOS), a deeper classification head, and Optuna hyperparameter search. Contrastive pretraining, spaCy linguistic features, and Asymmetric Loss were tested and kept only as optional flags. |

# Key Takeaways

1. **Attention solved an important bottleneck in recurrent sequence-to-sequence models** by providing direct access to relevant representations.
2. **Transformers replaced recurrence with attention-based computation**, enabling highly parallel sequence processing.
3. **Full attention becomes expensive for long sequences**, primarily because of quadratic interaction patterns and increasing memory requirements.
4. **Modern Transformer optimization occurs at multiple levels**, including attention heads, memory management, kernels, numerical precision, and token interactions.
5. **GQA and MQA reduce KV-cache requirements** through key-value sharing.
6. **PagedAttention improves KV-cache memory management**, while **FlashAttention reduces memory traffic without approximating the attention result**.
7. **Sparse and linear attention reduce the cost of long-context processing**, but may introduce restrictions or approximations.
8. **Dynamic attention makes computation input-dependent**, allowing the model to determine which interactions are useful.
9. **Adaptive attention span allows different heads to learn different contextual ranges.**
10. **Token pruning and merging reduce the number of representations that need to be processed.**
11. **Interpretability techniques such as attention rollout and attention flow provide ways to study information propagation through attention layers.**

# References

The following references are among the key works discussed in the report:

- Vaswani et al. — *Attention Is All You Need* (2017)
- Bahdanau, Cho & Bengio — *Neural Machine Translation by Jointly Learning to Align and Translate* (2015)
- Ainslie et al. — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)
- Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)
- Beltagy, Peters & Cohan — *Longformer: The Long-Document Transformer* (2020)
- Kitaev, Kaiser & Levskaya — *Reformer: The Efficient Transformer* (2020)
- Dai et al. — *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* (2019)
- Choromanski et al. — *Rethinking Attention with Performers* (2021)
- Kwon et al. — *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)
- Goyal et al. — *PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination* (2020)
- Kim et al. — *Learned Token Pruning for Transformers* (2022)
- Bolya et al. — *Token Merging: Your ViT but Faster* (2023)
- Rao et al. — *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification* (2021)
- Abnar & Zuidema — *Quantifying Attention Flow in Transformers* (2020)

---