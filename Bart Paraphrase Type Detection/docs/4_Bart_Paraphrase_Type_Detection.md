# ETPC Paraphrase Type Detection with BART

## Overview

This pipeline implements a **fine-grained multi-label paraphrase type detection system** using the **Extended Paraphrase Typology Corpus (ETPC)**.

Given a pair of sentences, the model predicts which of the **26 valid paraphrase types** occur between them. ETPC detection is a substantially more challenging **multi-label classification problem** because a single sentence pair can simultaneously exhibit several different linguistic transformations. For example, a sentence pair may contain multiple paraphrase phenomena such as lexical substitution, morphological changes, and syntactic transformations at the same time.

---

# 1. ETPC Detection

The objective is to identify the specific linguistic transformations that make one sentence a paraphrase of another.

The model receives:

```text
Sentence 1
Sentence 2
```

and predicts:

```text
[y1, y2, ..., y26]
```

where each yi is binary:

```text
yi = 1  → paraphrase type i is present
yi = 0  → paraphrase type i is absent
```

The task therefore uses **26 independent binary decisions** rather than a single multiclass prediction.

The ETPC task contains 26 valid paraphrase categories. Several original label IDs in the source data are unassigned and are therefore removed during preprocessing. The implementation removes labels:

```text
12, 19, 20, 23, 27
```

and maps the remaining 26 labels to consecutive indices:

```text
0 ... 25
```

This produces a fixed-size 26-dimensional target vector for every training example.

The dataset is highly imbalanced: some paraphrase types are common while others occur only rarely. This imbalance is one of the central challenges addressed by the methodology.

---

# 2. Label Preprocessing

The original ETPC data contains label IDs ranging beyond the 26 valid classes.

The implementation explicitly removes:

```text
12
19
20
23
27
```

The remaining labels are remapped from their original IDs to compact indices:

```text
original ETPC IDs
        ↓
remove invalid/unassigned IDs
        ↓
compact mapping
        ↓
0 ... 25
```

The labels are converted into binary vectors.

For example, if a sentence pair contains:

```text
[6, 25, 29]
```

the resulting target becomes a 26-dimensional vector:

```text
[0, 0, ..., 1, ..., 1, ..., 1, ...]
```

Duplicate labels are removed using a set before constructing the binary vector.

This preprocessing ensures that every example has exactly 26 output targets.

---

# 3. Baseline Architecture

The starting point is a straightforward pretrained Transformer classifier:

```text
Sentence 1 + Sentence 2
          │
          ▼
    BART tokenizer
          │
          ▼
     BART-large
          │
          ▼
   Sentence representation
          │
          ▼
      Linear layer
          │
          ▼
       26 logits
          │
          ▼
       Sigmoid
          │
          ▼
   26 binary predictions
```

The baseline uses:

| Component | Configuration |
|---|---|
| Backbone | facebook/bart-large |
| Task | Multi-label classification |
| Output dimension | 26 |
| Maximum sequence length | 512 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | `1.5e-5` |
| Loss | Binary Cross Entropy |
| Prediction threshold | 0.5 |
| Train/dev split | 80/20 |
| Random seed | 11711 |

The baseline is intentionally simple: the pretrained BART representation is passed to a single linear classification layer. This provides a meaningful reference point for evaluating subsequent modifications.

---

# 4. MCC as the Primary Selection Metric

Development performance is evaluated using:

```text
Accuracy
MCC
```

However, MCC is the primary metric used for model selection.

This is important because the ETPC task is highly imbalanced.

A model can achieve high accuracy by predicting the majority negative class for many rare labels while still performing poorly on the actual paraphrase-type detection problem.

MCC is therefore used to provide a more informative assessment of the quality of binary predictions across imbalanced classes.

The implementation computes MCC separately for each of the 26 labels and then averages the resulting coefficients:

```text
MCC =
mean(
    MCC_class_1,
    MCC_class_2,
    ...,
    MCC_class_26
)
```

The same averaging procedure is used for label-level accuracy.

---

# 5. Early Stopping and Checkpoint Selection

The model is not necessarily trained for the maximum number of epochs.

Instead, development MCC is monitored after every epoch.

The best checkpoint is stored whenever:

```python
dev_mcc > best_dev_mcc
```

If the metric does not improve for:

```text
patience = 2
```

consecutive epochs, training terminates.

The best checkpoint is then restored:

```python
model.load_state_dict(
    torch.load(best_model_path)
)
```

The resulting model is therefore the checkpoint with the highest observed development MCC rather than simply the final epoch.

This reduces the likelihood of selecting an overfitted late-training model.

---

# 6. Learning-Rate Warmup and Decay

The training schedule uses:

```python
get_linear_schedule_with_warmup(...)
```

The number of warmup steps is calculated as:

```python
warmup_steps = int(
    warmup_ratio * total_steps
)
```

with:

```text
warmup_ratio = 0.10 (Can be passed as an argument)
```

The learning rate therefore follows the general pattern:

```text
Learning rate

       /\
      /  \
     /    \
    /      \
   /        \
  /          \
 /            \
/              \____
──────────────────────► training
     warmup       decay
```

The initial warmup phase reduces the risk of unstable updates when fine-tuning the pretrained BART parameters.

After warmup, the learning rate linearly decays.

---

# 7. AdamW

The model uses AdamW:

```python
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
```

with the default configuration:

```text
Learning rate = 1.5 × 10⁻⁵
Weight decay  = 0.01
```

AdamW is particularly suitable for fine-tuning a pretrained Transformer because it separates weight decay from the adaptive gradient update.

The weight decay also provides additional regularization during fine-tuning.

---

# 8. Multi-Label Stratified Train/Development Split

A standard random train/dev split can produce undesirable differences in the distribution of rare labels.

This is particularly problematic for ETPC because the 26 labels have very different frequencies.

The implementation therefore uses:

```python
MultilabelStratifiedShuffleSplit
```

with:

```text
test_size = 0.2
random_state = 11711
```

The pipeline is:

```text
ETPC training dataset
        │
        ▼
26-dimensional label matrix
        │
        ▼
Multi-label stratification
        │
   ┌────┴────┐
   ▼         ▼
  80%       20%
 Train       Dev
```

The purpose is to preserve the distribution of all 26 labels as consistently as possible between the two splits.

This is especially important when evaluating MCC for rare paraphrase types.

The project documentation reports that the stratified split was used to improve the reliability of development evaluation. 

---

# 9. Dropout Regularization

The deeper classifier includes:

```python
nn.Dropout(0.2)
```

between the GELU layer and final output layer.

The classifier is therefore:

```text
Linear
  ↓
GELU
  ↓
Dropout
  ↓
Linear
```

Dropout is used to reduce over-reliance on individual hidden features and mitigate overfitting.

The project experiments found that adding dropout and gradient clipping improved the robustness of the training configuration, although dropout itself is considered a regularization mechanism rather than a fundamental backbone architecture change.

---

# 10. Gradient Clipping

Transformer fine-tuning can occasionally produce large gradients.

The training loop therefore applies:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

before the optimizer update.

The training sequence is:

```text
loss.backward()
      ↓
gradient clipping
      ↓
optimizer.step()
```

Gradient clipping constrains the total gradient norm and helps prevent unstable parameter updates.

This is a training-stability modification rather than a change to the model architecture.

---

# 11. BCEWithLogitsLoss

The model outputs raw logits.

The sigmoid is therefore **not** applied inside `forward()`.

Instead:

```text
BART
  ↓
classifier
  ↓
raw logits
  ↓
BCEWithLogitsLoss
```

`BCEWithLogitsLoss` combines the sigmoid operation with binary cross entropy in a numerically stable implementation.

During inference, sigmoid is applied explicitly:

```python
probs = torch.sigmoid(outputs)
```

because probabilities are required for thresholding.

---

# 12. Handling Severe Class Imbalance

ETPC contains highly imbalanced paraphrase categories.

Some labels are very frequent while others are extremely rare. For example, the project data contains categories with only a handful of positive examples compared with thousands of negative examples.

A naive BCE objective can therefore be dominated by negative labels.

To address this, the implementation calculates a separate positive-class weight for every label.

For each class:

```text
positive_count = number of positive examples

negative_count = total_examples - positive_count
```

The initial weight is:

```text
negative_count
----------------
positive_count
```

The implementation then applies square-root scaling:

```python
pos_weights = np.sqrt(pos_weights)
```

Therefore:

```text
raw imbalance ratio
        ↓
square-root scaling
        ↓
class-specific positive weight
```

This is a compromise between:

- treating all classes equally, and
- aggressively weighting extremely rare classes.

The resulting weights are supplied to:

```python
nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)
```

---

# 13. Class-Specific Decision Thresholds

A major part of the prediction methodology is replacing the conventional global threshold:

```text
threshold = 0.5
```

with **26 independently optimized thresholds**.

The reason is that the optimal operating point can differ substantially between common and rare paraphrase types.

For each class:

```text
class i
   │
   ├── predicted probabilities
   │
   ▼
candidate thresholds
   │
   ▼
calculate MCC for each threshold
   │
   ▼
select threshold with highest MCC
```

Mathematically:

```text
t_i = argmax_t MCC(y_i, I(p_i > t))
```

where:

- `p_i` is the predicted probability for class `i`;
- `y_i` is the true binary label;
- `t_i` is the selected threshold.

The implementation evaluates thresholds derived from the unique predicted probabilities and selects the threshold producing the highest training MCC for each class.

---

# 14. Why Per-Class Thresholding Is Important?

Using:

```text
p > 0.5
```

for every class assumes that every paraphrase category has the same probability calibration and the same optimal precision/recall trade-off.

That assumption is unlikely to hold in a strongly imbalanced multi-label problem.

Instead, the final system can learn something conceptually like:

```text
Type 1  → threshold 0.xx
Type 2  → threshold 0.xx
Type 3  → threshold 0.xx
...
Type 26 → threshold 0.xx
```

This gives each paraphrase category its own decision boundary.

Importantly, threshold optimization is performed separately from the neural network forward pass. The model continues to output probabilities; threshold optimization changes how those probabilities are converted into final binary decisions.

---

# 15. Thresholding and Model Evaluation

During development, thresholds are estimated from the training split and then applied to the development split.

The procedure is:

```text
TRAIN
  │
  ▼
train model
  │
  ▼
find 26 optimal thresholds
  │
  ▼
DEV
  │
  ▼
calculate MCC
```

This prevents development labels from being directly used to optimize the thresholds used for development evaluation.

After model selection, the labeled training and development datasets are combined:

```text
TRAIN + DEV
     │
     ▼
find final thresholds
     │
     ▼
TEST
```

The final test predictions therefore use thresholds estimated from all available labeled development data.

---

# 16. Sentence Pair Representation and Classification Capacity

The main model extends the baseline by improving both the **sentence-pair representation** and the **classification capacity**.

The two most important architectural decisions are:

1) **Mean pooling over BART's contextualized token representations**
2) **A deeper nonlinear classification head**

The implementation keeps BART-large as the pretrained backbone and modifies the task-specific layers around it.

---

# 17. BART-large Backbone

The model loads:

```python
BartModel.from_pretrained(
    "facebook/bart-large",
    local_files_only=True
)
```

The pretrained BART-large encoder generates a contextual representation for every input token.

For an input batch, the resulting hidden-state tensor has the conceptual shape:

```text
[batch_size, sequence_length, 1024]
```

where:

- `batch_size` = number of sentence pairs in the batch
- `sequence_length` = tokenized sequence length
- `1024` = BART-large hidden dimension

The important point is that BART does not directly produce a single representation for the complete sentence pair. A pooling operation is therefore required before classification.

---

# 18. Sentence-Pair Representation

A major part of the experimentation focuses on how to convert the token-level BART representation into a single fixed-dimensional representation.

Four pooling strategies were implemented:

```text
1. Token-0 / CLS-style pooling
2. Mean pooling
3. Max pooling
4. EOS pooling
```

These alternatives were evaluated under the same general training framework.

---

## 18.1 Token-0 Pooling

The token-0 representation is extracted using:

```python
pooled_output = last_hidden_state[:, 0, :]
```

This is analogous to using a single special representation as the sequence representation.

However, BART does not use a BERT-style `[CLS]` token. Therefore, in this implementation, "CLS pooling" refers specifically to using the representation at position zero.

The advantage is simplicity and computational efficiency.

The disadvantage is that all information used by the classifier must be encoded into or recoverable from one token representation.

---

## 18.2 Mean Pooling

Mean pooling computes the average of all non-padding contextualized token representations.

The implementation first expands the attention mask:

```python
mask = attention_mask.unsqueeze(-1).float()
```

and masks padding tokens:

```python
masked_hidden_state = last_hidden_state * mask
```

The representation is then calculated as:

```text
sum of valid token representations
-----------------------------------
      number of valid tokens
```

or:

```python
pooled_output = sum_hidden_states / token_count
```

This is a **masked mean**, meaning padding tokens do not influence the resulting representation.

The resulting representation has dimension:

```text
[batch_size, 1024]
```

### Motivation

Mean pooling allows information distributed throughout the complete sentence pair to contribute to the representation.

This is particularly useful for ETPC detection because paraphrase transformations may occur anywhere in either sentence rather than being concentrated around a single token.

---

## 18.3 Max Pooling

Max pooling selects the maximum activation independently for every hidden dimension.

Padding positions are masked with a very large negative value so they cannot become the maximum.

Conceptually:

```text
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼──► dimension-wise maximum
Token 4 ─┤
Token N ─┘
```

The motivation is to test whether the strongest individual contextual features provide a better signal than an average representation.

---

## 18.4 EOS Pooling

BART does not rely on the same `[CLS]` representation convention as BERT.

Therefore, an alternative representation is obtained from the final EOS token.

The implementation identifies EOS positions:

```python
eos_mask = input_ids.eq(
    self.bart.config.eos_token_id
)
```

and selects the final EOS representation.

This allows the experiment to test whether BART's EOS representation provides a stronger sentence-pair summary than either token-0 or aggregate pooling.

---

# 18.5 Pooling Comparison

All four strategies were compared:

| Pooling | Best Dev Accuracy | Best Dev MCC |
|---|---:|---:|
| **Mean** | **0.9093** | **0.2354** |
| CLS / token-0 | 0.9049 | 0.2157 |
| EOS | 0.9016 | 0.2083 |
| Max | 0.9003 | 0.1824 |

Mean pooling produced the strongest development MCC and was therefore selected for the subsequent architecture.

### Interpretation

The results suggest that aggregating information across all valid input tokens was more effective than relying on a single token representation or only the strongest activation.

This is particularly appropriate for fine-grained paraphrase detection because the evidence for a paraphrase type can be distributed across different parts of the sentence pair.

The selected representation is therefore:

```text
BART token states
       ↓
masked mean pooling
       ↓
1024-dimensional sentence-pair representation
```

---

# 19. Deeper Classification Head

The second major architectural improvement is the replacement of the baseline linear classification layer.

## Baseline

The baseline effectively performs:

```text
BART representation
        ↓
Linear
        ↓
26 logits
```

This provides only a linear transformation between the BART representation and the output labels.

## Proposed classifier

The improved architecture is:

```text
BART representation
        ↓
Linear(1024 → 1024)
        ↓
GELU
        ↓
Dropout(0.2)
        ↓
Linear(1024 → 26)
        ↓
26 logits
```

Implemented as:

```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_size, classifier_hidden_size),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(classifier_hidden_size, num_labels)
)
```

with:

```text
hidden_size = 1024
classifier_hidden_size = 1024
dropout = 0.2
num_labels = 26
```

---

## 19.1 Why add a nonlinear head?

The pretrained BART encoder provides a rich contextual representation, but the baseline linear classifier can only separate the paraphrase classes through linear decision boundaries in that representation space.

The deeper head allows the model to learn:

```text
h → Linear → GELU → Linear → predictions
```

rather than:

```text
h → Linear → predictions
```

The GELU activation introduces nonlinear modeling capacity, while the additional projection allows the classifier to transform the BART representation before making the final 26 independent predictions.

This is especially useful because ETPC labels correspond to linguistically different phenomena that may not be linearly separable in the original BART representation space.

---

## 19.2 Effect of the Deeper Classification Head

The deeper head produced the strongest development performance among the investigated architectural configurations. This experiment was therefore selected for the final architecture.

The progression can be summarized as:

```text
Baseline representation/classifier
              │
              ▼
       Better pooling
              │
              ▼
       Mean pooling
              │
              ▼
      Deeper nonlinear head
              │
              ▼
        Selected model
```

---

# 20. Optional Asymmetric Loss

Because ETPC is strongly multi-label and imbalanced, an alternative loss function was also implemented. When:

```bash
--loss ASL
```

is passed, the training process uses Asymmetric Loss.

The default ASL configuration is:

```text
gamma_pos = 0.0
gamma_neg = 4.0
clip      = 0.05
```

The motivation is to reduce the influence of easy negative examples.

The loss uses:

1. independent positive and negative probabilities;
2. asymmetric probability clipping;
3. asymmetric focusing;
4. stronger focusing on negative examples.

The negative focusing term is:

```text
gamma_neg = 4
```

while:

```text
gamma_pos = 0
```

This makes the loss substantially more focused on difficult negative labels.

The experiment produced a development MCC of approximately:

```text
0.220
```

and therefore did not outperform the selected BCE-based configuration. It was retained as an optional experimental loss rather than used as the default.

---

# 21. Optional Contrastive Pretraining

An additional experimental branch implements contrastive pretraining before supervised ETPC classification.

When:

```bash
--contrastive
```

is enabled, the training process becomes two-stage.

## Stage 1 — Contrastive Pretraining

```text
ETPC training pairs
        │
        ▼
Sentence encoder
        │
        ▼
BART representation
        │
        ▼
256-dimensional projection
        │
        ▼
Contrastive objective
```

The contrastive model uses a 256-dimensional projection space.

Only the training split is used for contrastive pretraining to avoid introducing development information into the learned representation.

## Stage 2 — Supervised ETPC Classification

The pretrained BART weights are transferred:

```text
Contrastively trained BART
          │
          ▼
transfer BART parameters
          │
          ▼
ETPC classification model
          │
          ▼
mean pooling
          │
          ▼
deeper classifier
          │
          ▼
26 labels
```

The projection head is not used as the final ETPC classifier.

Instead, its purpose is to improve the underlying BART representation before supervised fine-tuning.

Although this is a more unusual and creative training strategy, the downstream development performance did not exceed the selected supervised configuration, so it was not chosen as the default final approach.

---

# 22. Additional Linguistic Features

Another experiment tested whether handcrafted linguistic features could provide information complementary to BART.

Six features were extracted using spaCy:

```text
1. Lexical overlap
2. Token difference
3. Length difference
4. POS similarity
5. NER overlap
6. Dependency similarity
```

The architecture becomes conceptually:

```text
BART representation
       │
       ├───────────────┐
       │               │
       ▼               ▼
Neural representation  Linguistic features
       │               │
       └───────┬───────┘
               ▼
          Concatenation
               │
               ▼
          Classification
```

The motivation was to combine learned contextual features with explicit linguistic signals.

However, the experiment achieved approximately:

```text
Dev Accuracy = 0.906
Dev MCC      = 0.222
```

which was below the selected deeper-head configuration.

The added linguistic features therefore increased system complexity without providing sufficient generalization improvement and were not selected for the final model.

This is an important methodological result: **more features do not automatically produce a better model**.

---

# 23. Hyperparameter Optimization

Manual experimentation was complemented with automated hyperparameter search using **Optuna**.

The search considers:

```text
batch_size
classifier_hidden_size
dropout
learning_rate
warmup_ratio
weight_decay
```

The objective is based on development MCC.

This is especially appropriate for the ETPC task because:

- batch size influences gradient noise;
- classifier size controls task-specific capacity;
- dropout controls regularization;
- learning rate controls adaptation of the pretrained backbone;
- warmup controls early fine-tuning stability;
- weight decay controls regularization.

The search therefore explores both **optimization** and **model-capacity** dimensions rather than only tuning one parameter at a time.

---

# 24. Experimental Design

```text
Baseline
   │
   ├── Pooling
   │     ├── token-0
   │     ├── mean
   │     ├── max
   │     └── EOS
   │
   ├── Classification head
   │     ├── linear
   │     └── deeper nonlinear head
   │
   ├── Loss
   │     ├── BCE
   │     ├── weighted BCE
   │     └── ASL
   │
   ├── Regularization
   │     ├── dropout
   │     └── gradient clipping
   │
   ├── Optimization
   │     ├── AdamW
   │     ├── warmup
   │     ├── LR decay
   │     └── early stopping
   │
   ├── Data handling
   │     └── multi-label stratification
   │
   ├── Prediction
   │     └── per-class threshold optimization
   │
   ├── Representation learning
   │     └── contrastive pretraining
   │
   ├── External features
   │     └── linguistic feature fusion
   │
   └── Hyperparameter Search / Optimization
```

This allows each experiment to answer a specific question rather than simply increasing model complexity without justification.

---

# 25. Selected Final Architecture

The strongest architecture is:

```text
                 ┌─────────────────────┐
                 │ Sentence Pair       │
                 │ Sentence 1 + 2      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ BART-large          │
                 │ pretrained encoder  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌───────────────────────┐
                 │ Token representations │
                 │ [B, L, 1024]          │
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Masked Mean Pooling │
                 └──────────┬──────────┘
                            │
                            ▼
                       [B, 1024]
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Linear 1024 → 1024  │
                 └──────────┬──────────┘
                            │
                            ▼
                         GELU
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Dropout p = 0.2     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Linear 1024 → 26    │
                 └──────────┬──────────┘
                            │
                            ▼
                       26 logits
                            │
                            ▼
                       Sigmoid
                            │
                            ▼
              ┌─────────────────────────┐
              │ 26 class-specific       │
              │ optimized thresholds    │
              └────────────┬────────────┘
                           │
                           ▼
                     26 predictions
```

## Overall Pipeline

The system consists of four stages:

1) **Data preparation** – Multilabel stratified splitting and conversion of original type IDs into a compact 26-dimensional multi hot representation
2) **(Optional) Contrastive Pre-training** – A sentence encoder stage that learns semantically meaningful representations from the paraphrase pairs themselves.
3) **Supervised Multi-label Fine-tuning** – The core classification stage.
4) **Post-hoc Threshold Optimisation** – Independent MCC-maximising thresholds for each of the 26 classes.

Only the training split is ever used for contrastive pre-training or for computing class weights and initial thresholds, guaranteeing that no development information leaks into the model.

---

# 26. Training Pipeline

The complete supervised training procedure is:

```text
1. Load ETPC dataset
        ↓
2. Convert original ETPC IDs to 26-dimensional labels
        ↓
3. Perform multi-label stratified 80/20 split
        ↓
4. Calculate class-specific positive weights
        ↓
5. Tokenize sentence pairs using BART tokenizer
        ↓
6. Initialize BART-large
        ↓
7. Mean-pool contextualized token representations
        ↓
8. Apply deeper nonlinear classifier
        ↓
9. Produce 26 logits
        ↓
10. Calculate weighted BCE loss
        ↓
11. Backpropagate
        ↓
12. Clip gradients
        ↓
13. AdamW update
        ↓
14. Update learning-rate scheduler
        ↓
15. Evaluate training data
        ↓
16. Optimize per-class thresholds on training data
        ↓
17. Evaluate development MCC
        ↓
18. Save best checkpoint
        ↓
19. Early stopping
        ↓
20. Restore best checkpoint
```

---

# 27. Final Inference Pipeline

Once training is complete:

```text
TRAIN + DEV
     │
     ▼
final threshold estimation
     │
     ▼
trained BART model
     │
     ▼
TEST sentence pairs
     │
     ▼
BART token representations
     │
     ▼
mean pooling
     │
     ▼
deeper classification head
     │
     ▼
26 logits
     │
     ▼
sigmoid probabilities
     │
     ▼
26 class-specific thresholds
     │
     ▼
26-dimensional binary vector
```

The final predictions are stored in:

```text
predictions/bart/
```

with columns:

```text
id
Predicted_Paraphrase_Types
```

where `Predicted_Paraphrase_Types` contains a 26-dimensional binary prediction vector.

---

# 28. Reproducibility

The implementation uses a deterministic seed configuration:

```python
seed_everything(seed)
```

which sets seeds for:

```text
Python random
NumPy
PyTorch
CUDA
```

and configures deterministic CUDA/cuDNN behavior.

The default seed is:

```text
11711
```

The train/dev split also uses the same seed.

The DataLoader does not shuffle the data:

```python
shuffle = False
```

This is particularly important during inference because predictions must remain aligned with the original test IDs.

---