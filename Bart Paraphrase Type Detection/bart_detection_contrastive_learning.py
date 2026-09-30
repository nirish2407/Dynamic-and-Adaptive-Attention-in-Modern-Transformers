import argparse
import random
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW
import ast
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class BartSentenceEncoder(nn.Module):
    """
    BART encoder used for contrastive sentence representation learning.

    Each sentence is encoded independently into a fixed-size vector.
    """

    def __init__(self, projection_dim=256, pooling="mean"):
        super(BartSentenceEncoder, self).__init__()

        self.bart = BartModel.from_pretrained(
            "facebook/bart-large",
            local_files_only=True
        )

        # Pooling strategy:
        # "cls"  -> first/token-0 representation
        # "mean" -> masked mean over all real tokens
        # "max"  -> masked max over all real tokens
        # "eos"  -> last EOS-token representation
        self.pooling = pooling

        hidden_size = self.bart.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state

        # --------------------------------------------------
        # CLS / token-0 pooling
        # --------------------------------------------------

        if self.pooling == "cls":

            # Original baseline approach:
            # Take the representation of the first token
            pooled_output = last_hidden_state[:, 0, :]

        # --------------------------------------------------
        # Mean pooling
        # --------------------------------------------------

        elif self.pooling == "mean":

            # attention_mask:
            #   1 = real token
            #   0 = padding token
            mask = attention_mask.unsqueeze(-1).float()

            # Remove padding-token representations by multiplying
            # them by zero.
            masked_hidden_state = last_hidden_state * mask

            # Sum the hidden states of all real tokens.
            sum_hidden_states = masked_hidden_state.sum(dim=1)

            # Count how many real tokens each example contains.
            token_count = mask.sum(dim=1).clamp(min=1e-9)

            # Mean pooling: sum(hidden states) / number of real tokens
            pooled_output = sum_hidden_states / token_count

        # --------------------------------------------------
        # Max pooling
        # --------------------------------------------------

        elif self.pooling == "max":

            # Mask padding tokens with a very large negative number
            # so that they cannot become the maximum.
            mask = attention_mask.unsqueeze(-1).bool()

            masked_hidden_state = last_hidden_state.masked_fill(
                ~mask,
                torch.finfo(last_hidden_state.dtype).min
            )

            # Take the maximum value for every hidden dimension.
            pooled_output = masked_hidden_state.max(dim=1).values

        # --------------------------------------------------
        # EOS pooling
        # --------------------------------------------------

        elif self.pooling == "eos":

            # ------------------------------------------------------------
            # EOS-token pooling
            # Pool the EOS-token representation.
            # ------------------------------------------------------------
            # BART does not use a BERT-style [CLS] token.
            # Instead, use the last EOS-token representation as the
            # representation of the complete sentence pair.
            # ------------------------------------------------------------
            eos_mask = input_ids.eq(self.bart.config.eos_token_id).to(last_hidden_state.device)

            # Select hidden states at EOS positions, then take the LAST one
            # per sequence (in case of multiple EOS tokens, e.g. sentence pairs).
            pooled_output = last_hidden_state[eos_mask, :].view(
                last_hidden_state.size(0), -1, last_hidden_state.size(-1)
            )[:, -1, :]

        else:

            raise ValueError(
                f"Unknown pooling method: {self.pooling}. "
                f"Choose from 'cls', 'mean', 'max' or 'eos'."
            )

        # Contrastive projection
        projection = self.projection(pooled_output)

        # Normalize for cosine similarity
        projection = nn.functional.normalize(projection, p=2, dim=1)

        return pooled_output, projection

def contrastive_loss(sentence_embeddings_1, sentence_embeddings_2, temperature=0.05):
    """
    Symmetric InfoNCE contrastive loss.

    sentence_embeddings_1[i] and sentence_embeddings_2[i]
    form a positive pair.

    Other sentences in the batch act as negatives.
    """

    # Normalize embeddings so that dot product becomes
    # cosine similarity.
    z1 = nn.functional.normalize(
        sentence_embeddings_1,
        p=2,
        dim=1
    )

    z2 = nn.functional.normalize(
        sentence_embeddings_2,
        p=2,
        dim=1
    )

    # Similarity matrix.
    #
    # similarity[i, j] measures how similar sentence1[i]
    # is to sentence2[j].
    similarity = torch.matmul(
        z1,
        z2.T
    ) / temperature

    # Correct matching pairs are on the diagonal:
    #
    # sentence1[0] <-> sentence2[0]
    # sentence1[1] <-> sentence2[1]
    # sentence1[2] <-> sentence2[2]
    #
    # Therefore the correct class for each row is its index.
    labels = torch.arange(
        similarity.size(0),
        device=similarity.device
    )

    # Direction 1:
    # sentence1 -> sentence2
    loss_1 = nn.functional.cross_entropy(
        similarity,
        labels
    )

    # Direction 2:
    # sentence2 -> sentence1
    loss_2 = nn.functional.cross_entropy(
        similarity.T,
        labels
    )

    # Symmetric contrastive loss.
    loss = (loss_1 + loss_2) / 2

    return loss

def transform_contrastive_data(dataset, max_length=256):
    """
    Prepare ETPC sentence pairs for contrastive learning.

    Unlike Stage 2 (ETPC Classification), sentence1 and sentence2 are tokenized separately.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large",
        local_files_only=True
    )

    sentence1 = dataset["sentence1"].tolist()
    sentence2 = dataset["sentence2"].tolist()

    encoding1 = tokenizer(
        sentence1,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoding2 = tokenizer(
        sentence2,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    dataset_tensor = TensorDataset(
        encoding1["input_ids"],
        encoding1["attention_mask"],
        encoding2["input_ids"],
        encoding2["attention_mask"]
    )

    return DataLoader(
        dataset_tensor,
        batch_size=32,
        shuffle=True
    )

def train_contrastive(model, train_data, device, epochs=3, lr=1e-5, temperature=0.05, disable_tqdm=False):
    """
    Contrastively pretrain BART using ETPC sentence pairs.

    Each ETPC sentence pair is treated as a positive pair.
    Other sentences in the same batch act as negatives.
    """

    optimizer = AdamW(
        model.parameters(),
        lr=lr
    )

    total_steps = len(train_data) * epochs

    warmup_steps = int(
        0.1 * total_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        progress = tqdm(
            train_data,
            desc=f"Contrastive Epoch {epoch + 1}/{epochs}",
            disable=disable_tqdm
        )

        for batch in progress:

            input_ids_1 = batch[0].to(device)
            attention_mask_1 = batch[1].to(device)

            input_ids_2 = batch[2].to(device)
            attention_mask_2 = batch[3].to(device)

            optimizer.zero_grad()

            # --------------------------------------------------
            # Encode sentence 1
            # --------------------------------------------------

            embedding_1, z1 = model(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1
            )

            # --------------------------------------------------
            # Encode sentence 2
            # --------------------------------------------------

            embedding_2, z2 = model(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2
            )

            # --------------------------------------------------
            # Contrastive loss
            # --------------------------------------------------

            loss = contrastive_loss(
                z1,
                z2,
                temperature=temperature
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_data)

        print(
            f"Contrastive Epoch {epoch + 1}: "
            f"Loss={avg_loss:.4f}"
        )

    return model
