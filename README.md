# Dynamic and Adaptive Attention in Modern Transformers

Seminar project for **Newest Trends in High-Performance Data Analytics**, Georg-August-Universität Göttingen, Institute of Computer Science.

| | |
|---|---|
| **Author** | Nirish Samant |
| **Supervisor** | Lauritz Rasbach |

---

## Overview

This repository accompanies the seminar study on how attention evolved from a fix for the limitations of recurrent networks into a flexible framework in which computation is allocated dynamically, according to the input, the available resources, and the modeling requirements.

## Repository contents

```
.
├── Attention_Presentation.pdf                  
├── Attention_Report.pdf                        
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