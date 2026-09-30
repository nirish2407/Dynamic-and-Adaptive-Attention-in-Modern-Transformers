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
from bart_detection_contrastive_learning import *

class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26, dropout=0.2, pooling="mean", classifier_hidden_size=1024):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=True)

        # Pooling strategy:
        # "cls"  -> first/token-0 representation
        # "mean" -> masked mean over all real tokens
        # "max"  -> masked max over all real tokens
        # "eos"  -> last EOS-token representation
        self.pooling = pooling

        # ------------------------------------------------------------
        # Deeper classification head
        #
        # First linear layer: 1024 -> classifier_hidden_size
        # GELU: introduces non-linearity
        # Dropout: regularization
        # Final linear layer: classifier_hidden_size -> 26
        # ------------------------------------------------------------

        hidden_size = self.bart.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):

        # Get the contextualized representation of every token.
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # ------------------------------------------------------------
        # Pool the token representations
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------

        # Pass the pooled BART representation through the deeper nonlinear classification head
        logits = self.classifier(pooled_output)

        return logits

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    gamma_pos: focusing parameter for positive labels
    gamma_neg: focusing parameter for negative labels
    clip: probability margin applied to negative samples
    """

    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True
    ):
        super().__init__()

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):

        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # --------------------------------------------------
        # Asymmetric clipping
        # --------------------------------------------------

        if self.clip is not None and self.clip > 0:

            xs_neg = (
                xs_neg + self.clip
            ).clamp(max=1.0)

        # --------------------------------------------------
        # Basic BCE
        # --------------------------------------------------

        loss_pos = targets * torch.log(
            xs_pos.clamp(min=self.eps)
        )

        loss_neg = (1.0 - targets) * torch.log(
            xs_neg.clamp(min=self.eps)
        )

        loss = loss_pos + loss_neg

        # --------------------------------------------------
        # Asymmetric focusing
        # --------------------------------------------------

        if self.gamma_neg > 0 or self.gamma_pos > 0:

            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():

                    pt = (
                        xs_pos * targets
                        + xs_neg * (1.0 - targets)
                    )

                    one_sided_gamma = (
                        self.gamma_pos * targets
                        + self.gamma_neg * (1.0 - targets)
                    )

                    asymmetric_weight = torch.pow(
                        1.0 - pt,
                        one_sided_gamma
                    )

            else:

                pt = (
                    xs_pos * targets
                    + xs_neg * (1.0 - targets)
                )

                one_sided_gamma = (
                    self.gamma_pos * targets
                    + self.gamma_neg * (1.0 - targets)
                )

                asymmetric_weight = torch.pow(
                    1.0 - pt,
                    one_sided_gamma
                )

            loss *= asymmetric_weight

        # Mean over batch and labels
        return -loss.mean()

def calculate_pos_weights(dataset):
    """
    Calculate one positive-class weight for each of the 26 paraphrase types.

    pos_weight[class] =
        number_of_negative_examples /
        number_of_positive_examples

    The weights are calculated from the given dataset only.
    """

    # Convert the original labels into a 26-dimensional binary matrix.
    labels = create_binary_labels(dataset)

    # Count positive examples for each class.
    # Shape: [26]
    positive_counts = labels.sum(axis=0)

    # Total number of examples.
    total_examples = labels.shape[0]

    # Count negative examples for each class.
    negative_counts = total_examples - positive_counts

    # Calculate the positive weight for each class.
    #
    # If a class occurs rarely, positive_counts will be small,
    # resulting in a larger weight.
    # np.maximum prevents division by zero if a class happens
    # to have no positive examples.
    pos_weights = negative_counts / np.maximum(positive_counts, 1)
    pos_weights = np.sqrt(pos_weights)

    # Convert to a PyTorch tensor because BCEWithLogitsLoss
    # expects pos_weight to be a tensor.
    pos_weights = torch.tensor(
        pos_weights,
        dtype=torch.float
    )

    return pos_weights

def find_optimal_thresholds(model, train_data, device, num_labels=26):
    """
    Find one prediction threshold for each class using the train set.

    The threshold for each class is chosen to maximize MCC on the train set.

    Returns:
        optimal_thresholds: numpy array of shape [26]
    """

    model.eval()

    all_probs = []
    all_labels = []

    # ------------------------------------------------------------
    # Collect model probabilities and true labels from train set
    # ------------------------------------------------------------

    with torch.no_grad():

        for batch in train_data:

            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Model outputs raw logits.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Convert logits to probabilities.
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # Combine all train batches.
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # ------------------------------------------------------------
    # Search for the best threshold independently for each class
    # ------------------------------------------------------------

    optimal_thresholds = []

    # Find the best threshold independently for each of the 26 classes.
    for class_idx in range(num_labels):

        best_threshold = 0.5
        best_mcc = -1.0

        y_true = all_labels[:, class_idx]
        y_prob = all_probs[:, class_idx]

        # Candidate thresholds are the unique predicted probabilities.
        unique_probs = np.unique(y_prob)

        # Evaluate every threshold at which the binary predictions
        # can potentially change.
        candidate_thresholds = np.concatenate([
            [0.0],
            (unique_probs[:-1] + unique_probs[1:]) / 2,
            [1.0]
        ])

        # Try thresholds from Candidate thresholds
        for threshold in candidate_thresholds:

            predictions = (y_prob > threshold).astype(int)

            mcc = matthews_corrcoef(y_true, predictions)

            if mcc > best_mcc:

                best_mcc = mcc
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)

    optimal_thresholds = np.array(
        optimal_thresholds,
        dtype=np.float32
    )

    return optimal_thresholds

def create_binary_labels(dataset):

    # These lables will be dropped
    # These label IDs do not correspond to any valid paraphrase type according to the project specification,
    # so they must be ignored.
    labels_to_drop = [12, 19, 20, 23, 27]

    # Mapping from (1-31) label indices to new indices (0-25), we drop not needed labels
    # Create a mapping from the original label IDs (1-31) to new consecutive indices (0-25).
    # Example:
    # Original labels:
    #   1,2,3,...,11,13,...,31
    # become
    #   0,1,2,...,10,11,...,25
    label_mapping = {}
    current_label_idx = 0

    # Iterate through every possible original label.
    for original_label in range(1, 32):  # Original labels 1 through 31
            
        # Ignore labels that should be removed.
        if original_label not in labels_to_drop:
                
            # Store the new compact index.
            label_mapping[original_label] = current_label_idx
                
            # Move to the next available binary position.
            current_label_idx += 1

    # Convert str lists to actual lists
    # Dataset labels may already be Python lists or stored as strings
    # such as "[6, 25, 29]".
    # Convert string representations safely into actual Python lists.
    def parse_label(x):
        if isinstance(x, str):
            x = ast.literal_eval(x)
        return x

    # Apply parsing to every row in the dataset.
    label_lists = dataset["paraphrase_type_ids"].apply(parse_label)

    # Store binary vectors for every training example.
    labels = []

    # Process each sentence pair independently.
    for label_list in label_lists:

        # Remove duplicate labels using set().
        # Ignore label 0 (not a valid class).
        # Remove labels specified by the project instructions.
        indices = [i for i in set(label_list) if i != 0 and i not in labels_to_drop]

        # Initialize a binary vector of length 26.
        # Every position corresponds to one paraphrase type.
        # Initially all labels are absent.
        bin_vec = [0] * 26  # 26 dim binary vector for each sample

        # Mark every existing paraphrase type.
        for idx in indices:

            # Convert original label ID into compact index.
            mapped_idx = label_mapping[idx]  # Use our mapping to get the index of the label

            # Set the corresponding class to 1,
            # indicating that this paraphrase type exists.
            bin_vec[mapped_idx] = 1

        # Store this sample's multi-label binary representation.
        labels.append(bin_vec)

    return np.array(labels, dtype=np.int64)

def transform_data(dataset, max_length=512, batch_size=4):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [6, 6, 6, 25, 25, 29]. This means that
    the sentence pair contains type 6, 25, and 29. Turn this into a binary form, where the
    label becomes [0, 0, 0, 0, 0, 1, ..., 1, 0, 0, 1, 0, 0].
    IMPORTANT: You will find that the dataset contains types up to 31, but some are not
    assigned. You need to drop 12, 19, 20, 23 and 27 when creating the binary labels.
    This way you should end up with a binary label of size 26.     
    Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    # raise NotImplementedError

    # Load the pretrained BART tokenizer.
    # local_files_only=True ensures that the tokenizer is loaded only from locally
    # cached files (useful on GPU clusters (GWDG) without internet access).
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    # Keep shuffle disabled to preserve the original order of samples.
    # This is especially important during prediction so that generated
    # outputs correspond to the correct dataset IDs.
    shuffle = False

    # Extract both sentences from the dataframe and convert them into Python lists.
    # Each element at the same index represents one sentence pair.
    sentence1 = dataset["sentence1"].tolist()
    sentence2 = dataset["sentence2"].tolist()

    # Tokenize as pairs. Tokenize sentence pairs together.
    # The tokenizer automatically inserts the required special tokens,
    # pads every sequence to max_length, truncates longer sequences,
    # and returns PyTorch tensors.
    encoding = tokenizer(
        sentence1,
        sentence2,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Tensor containing token ids that will be fed into BART.
    input_ids = encoding["input_ids"]

    # Attention mask indicates which tokens are real tokens (1) and which correspond to padding (0).
    attention_mask = encoding["attention_mask"]

    # Handle cases with and without labels
    # Handle both training/validation datasets (with labels) and test datasets (without labels).
    if "paraphrase_type_ids" in dataset.columns:

        labels = create_binary_labels(dataset)

        # Convert the complete list of binary vectors into a PyTorch tensor.
        # Float datatype is required for BCEWithLogitsLoss.
        labels = torch.tensor(labels, dtype=torch.float)

        # Combine inputs and labels into one dataset object.
        tensor_dataset = TensorDataset(input_ids, attention_mask, labels)

    else:

        # Test dataset has no ground-truth labels.
        # Store only model inputs.
        tensor_dataset = TensorDataset(input_ids, attention_mask)

    # Create the DataLoader that provides batches during training, validation, or inference.
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle) 

    # Return the prepared DataLoader.
    return loader

def train_model(model,
    train_data,
    dev_data,
    device,
    pos_weight,
    loss_fn,
    epochs,
    patience,
    learning_rate,
    weight_decay,
    warmup_ratio,
    gamma_neg,
    gamma_pos,
    clip,
    disable_tqdm=False):
    """
    Train the model with early stopping based on development Macro MCC.

    The model checkpoint with the highest development MCC is saved and restored before returning
    the trained model.
    """

    # Initialize the AdamW optimizer.
    # AdamW is recommended for Transformer-based models because it decouples
    # weight decay from the gradient update, leading to better regularization.
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Total number of training steps
    total_steps = len(train_data) * epochs

    # Use warmup_ratio of the training steps for warm-up
    warmup_steps = int(warmup_ratio * total_steps)

    # Linear warm-up followed by linear learning-rate decay
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if loss_fn == "BCE":

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device)
        )

    elif loss_fn == "ASL":

        criterion = AsymmetricLossMultiLabel(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip
        )

    else:
    
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Track the best development MCC seen so far
    best_dev_mcc = -float("inf")

    # Count epochs since the last improvement.
    epochs_without_improvement = 0

    # Path where the best model will be saved.
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    best_model_path = f"best_paraphrase_detection_bart_model_job{slurm_job_id}.pt"

    # Put the model into training mode.
    # This enables training-specific layers such as Dropout.
    model.train()

    # Loop over all training epochs.
    for epoch in range(epochs):

        # Ensure the model is in training mode at the beginning of every epoch.
        model.train()

        # Variable used to accumulate the loss across all batches.
        total_loss = 0

        # Iterate through every mini-batch in the training dataset.
        # tqdm provides a progress bar during training.
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", disable=disable_tqdm):

            # Unpack the batch into input tensors and target labels.            
            input_ids, attention_mask, labels = batch

            # Move all tensors from CPU to the selected device (GPU or CPU).
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Clear previously accumulated gradients.
            # PyTorch accumulates gradients by default, so this step is required
            # before every optimization step.
            optimizer.zero_grad()

            # Perform a forward pass through the BART model.
            # The model returns one raw logit for each of the 26 paraphrase types.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Compute Binary Cross Entropy loss directly from the raw logits
            # and the ground-truth binary label vectors.
            # BCEWithLogitsLoss applies the sigmoid internally.
            loss = criterion(outputs, labels)

            # Perform backpropagation.
            # Gradients of the loss with respect to every trainable parameter are computed automatically.
            loss.backward()

            # Clip gradients to prevent excessively large updates.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update the model parameters using the optimizer.
            optimizer.step()

            # Update the learning rate according to the scheduler.
            scheduler.step()

            # Accumulate the loss value for computing the average epoch loss.
            total_loss += loss.item()

        # Compute the average training loss over all batches.
        avg_loss = total_loss / len(train_data)

        # Evaluate model performance on the training dataset.
        # This provides an indication of how well the model has learned the training examples.
        train_acc, train_mcc = evaluate_model(model, train_data, device)

        # Calculate the thresholds
        optimal_thresholds = find_optimal_thresholds(
            model,
            train_data,
            device,
            num_labels=26
        )

        # Evaluate model performance on the validation (development) dataset.
        # The dev accuracy is useful for monitoring generalization and detecting overfitting during training.
        dev_acc, dev_mcc = evaluate_model(model, dev_data, device, thresholds=optimal_thresholds)

        # Display training statistics after each epoch.
        print(
            f"Epoch {epoch+1}: "
            f"Loss={avg_loss:.4f}, "
            f"Train Acc={train_acc:.4f}, "
            f"Train MCC={train_mcc:.4f}, "
            f"Dev Acc={dev_acc:.4f}, "
            f"Dev MCC={dev_mcc:.4f}"
        )

        # --------------------------------------------------
        # Check whether this is the best model so far
        # --------------------------------------------------

        if dev_mcc > best_dev_mcc:

            best_dev_mcc = dev_mcc
            epochs_without_improvement = 0

            # Save the model with the best development MCC.
            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"  -> New best Dev MCC: {best_dev_mcc:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  -> No improvement in Dev MCC "
                f"({epochs_without_improvement}/{patience})"
            )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping triggered after "
                f"{epoch + 1} epochs."
            )

            break

    # ------------------------------------------------------
    # Restore the best model
    # ------------------------------------------------------

    print(
        f"Restoring best model with Dev MCC={best_dev_mcc:.4f}"
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    # Return the fully trained model so it can later be evaluated
    # or used to generate predictions on the test dataset.
    return model

def test_model(model, test_data, test_ids, device, thresholds, disable_tqdm=False):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO

    # raise NotImplementedError

    # Switch the model to evaluation mode.
    # This disables training-specific operations such as Dropout and BatchNorm updates,
    # ensuring deterministic predictions during inference.
    model.eval()

    # Store the predicted binary vectors for every sentence pair.
    predictions = []

    # Disable gradient computation during inference.
    # This reduces memory consumption and speeds up prediction since
    # gradients are not needed when the model is only making predictions.
    with torch.no_grad():

        # Iterate through every batch in the test dataset.
        # tqdm displays a progress bar while processing the batches.
        for batch in tqdm(test_data, desc="Testing", disable=disable_tqdm):
            
            # Each batch from the test set contains only the model inputs.
            # Unlike the training set, no labels are available.
            input_ids, attention_mask = batch

            # Move the input tensors to the selected device (GPU or CPU).
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Perform a forward pass through the trained model.
            # The model returns a raw logit for each of the 26 paraphrase types.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Convert predicted probabilities into binary predictions.
            # Apply the class-specific optimized thresholds:
            # probability > threshold[class] -> 1
            # probability <= threshold[class] -> 0
            #
            # The resulting tensor is converted to integers,
            # moved back to the CPU, transformed into a NumPy array,
            # and finally converted into a Python list.

            # Convert logits to probabilities.
            probs = torch.sigmoid(outputs)

            # Convert the 26 thresholds into a tensor.
            threshold_tensor = torch.tensor(
                thresholds,
                dtype=probs.dtype,
                device=probs.device
            )

            # Apply a different threshold to each class.
            preds = (probs > threshold_tensor).int()

            # Append the predictions from the current batch to the
            # overall prediction list while preserving the original order.
            predictions.extend(preds.tolist())

    # Create the final submission DataFrame.
    #
    # The "id" column contains the original example identifiers from the test set.
    # The "Predicted_Paraphrase_Types" column contains a binary vector of length 26
    # representing the predicted paraphrase types for each sentence pair.
    df = pd.DataFrame({
        "id": test_ids,
        "Predicted_Paraphrase_Types": predictions
    })

    # Return the DataFrame so it can be saved as a CSV file
    return df

def evaluate_model(model, test_data, device, thresholds=None):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the 26 paraphrase types the model has predicted correctly for each data point..
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)

            # --------------------------------------------------
            # Prediction threshold
            # --------------------------------------------------

            if thresholds is None:

                # Original baseline behaviour:
                # every class uses threshold 0.5
                predicted_labels = (
                    probs > 0.5
                ).int()

            else:

                # Convert thresholds to a PyTorch tensor.
                threshold_tensor = torch.tensor(
                    thresholds,
                    dtype=probs.dtype,
                    device=probs.device
                )

                # Each class gets its own threshold.
                predicted_labels = (
                    probs > threshold_tensor
                ).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx])
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars"
    )

    # ------------------------------------------------------------
    # Training hyperparameters
    # ---------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of epochs without improvement before early stopping"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.5e-5,
        help="Learning rate for AdamW"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW"
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.10,
        help="Fraction of training steps used for learning-rate warmup"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token sequence length"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training/evaluation batch size"
    )

    # ------------------------------------------------------------
    # Model hyperparameters
    # ------------------------------------------------------------

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability for the classification head"
    )

    parser.add_argument(
        "--classifier_hidden_size",
        type=int,
        default=1024,
        help="Hidden size of the additional classification layer"
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["cls", "mean", "max", "eos"],
        help="Pooling strategy for the classification head"
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    parser.add_argument(
        "--loss",
        type=str,
        choices=["BCE", "ASL"],
        default="BCE",
        help="Loss function: BCE (BCEWithLogitsLoss) or ASL (AsymmetricLoss)"
    )

    parser.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=0.0,
        help="ASL positive focusing parameter"
    )

    parser.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=4.0,
        help="ASL negative focusing parameter"
    )

    parser.add_argument(
        "--asl_clip",
        type=float,
        default=0.05,
        help="ASL probability clipping value"
    )

    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Enable contrastive pretraining before classification training"
    )

    # ------------------------------------------------------------
    # Contrastive learning
    # ------------------------------------------------------------

    args = parser.parse_args()
    return args


def finetune_paraphrase_detection(args):

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # ------------------------------------------------------------
    # Load ETPC
    # ------------------------------------------------------------

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # ------------------------------------------------------------
    # Create 26-dimensional multi-label targets for stratified split
    # BEFORE splitting the data.
    # ------------------------------------------------------------

    labels = create_binary_labels(train_dataset)

    print(f"Label matrix shape: {labels.shape}")

    # ------------------------------------------------------------
    # Multi-label stratified 80/20 train/dev split
    # ------------------------------------------------------------

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=args.seed
    )

    train_idx, dev_idx = next(
        msss.split(train_dataset, labels)
    )

    # Keep the original dataframe intact
    train_df = train_dataset

    train_dataset = train_df.iloc[train_idx].copy()
    dev_dataset = train_df.iloc[dev_idx].copy()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")

    # Create BART classifier.
    model = BartWithClassifier(
        dropout=args.dropout, 
        pooling=args.pooling,
        classifier_hidden_size=args.classifier_hidden_size
    )
    model.to(device)

    if args.contrastive:

        # ============================================================
        # STAGE 1: CONTRASTIVE PRETRAINING
        # ============================================================

        print("\n" + "=" * 60)
        print("STAGE 1: CONTRASTIVE PRETRAINING")
        print("=" * 60)

        contrastive_model = BartSentenceEncoder(projection_dim=256, pooling=args.pooling)

        contrastive_model.to(device)

        # Use ONLY the training split.
        #
        # Do not contrastively train on dev data, because that would
        # leak development information into the model.
        contrastive_data = transform_contrastive_data(
            train_dataset,
            max_length=256
        )

        contrastive_model = train_contrastive(
            model=contrastive_model,
            train_data=contrastive_data,
            device=device,
            epochs=3,
            lr=1e-5,
            temperature=0.05,
            disable_tqdm=args.disable_tqdm
        )

        # ------------------------------------------------------------
        # Transfer contrastively pretrained BART weights
        # ------------------------------------------------------------

        model.bart.load_state_dict(
            contrastive_model.bart.state_dict()
        )

        print("Transferred contrastively pretrained BART weights to the ETPC classifier.")

        # ============================================================
        # STAGE 2: ETPC CLASSIFICATION
        # ============================================================

        print("\n" + "=" * 60)
        print("STAGE 2: ETPC CLASSIFICATION")
        print("=" * 60)

    # ------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------

    # Calculate class weights from Training Data ONLY
    pos_weight = calculate_pos_weights(train_dataset)

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)

    # ------------------------------------------------------------
    # Method 2: Split train into train/dev (80/20)
    # ------------------------------------------------------------
    # Split the original training dataset into:
    #   - 80% training data
    #   - 20% development (validation) data
    #
    # random_state ensures that the split is reproducible,
    # producing exactly the same train/dev partition every run.
    """
    train_dataset, dev_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        random_state=args.seed
    )
    """

    # ------------------------------------------------------------
    # Method 3 (Alternative)
    # ------------------------------------------------------------
    # Randomly sample 80% of the dataset for training,
    # then use the remaining 20% as validation.
    #train_dataset = train_dataset.sample(frac=0.8, random_state=args.seed)
    #dev_dataset = train_dataset.drop(train_split.index)

    # ------------------------------------------------------------
    # Method 4 (Alternative)
    # ------------------------------------------------------------
    # Shuffle the dataset first, then manually split it
    # according to an 80/20 ratio.
    #split_idx = int(0.8 * len(train_dataset))
    #train_df = train_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True) # Shuffle
    #dev_dataset = train_df.iloc[split_idx:]
    #train_dataset = train_df.iloc[:split_idx]

    # ------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------   

    train_data = transform_data(train_dataset, max_length=args.max_length, batch_size=args.batch_size)
    dev_data = transform_data(dev_dataset, max_length=args.max_length, batch_size=args.batch_size)
    test_data = transform_data(test_dataset, max_length=args.max_length, batch_size=args.batch_size)

    print(f"Loaded {len(train_dataset)} training samples.")

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    model = train_model(
        model=model,
        train_data=train_data,
        dev_data=dev_data,
        device=device,
        pos_weight=pos_weight,
        loss_fn=args.loss,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gamma_pos=args.asl_gamma_pos,
        gamma_neg=args.asl_gamma_neg,
        clip=args.asl_clip,
        disable_tqdm=args.disable_tqdm
    )

    print("Training finished.")

    # ------------------------------------------------------------
    # Find class-specific optimal thresholds using Train set
    # ------------------------------------------------------------

    optimal_thresholds = find_optimal_thresholds(
        model,
        train_data,
        device,
        num_labels=26
    )

    # ------------------------------------------------------------
    # Evaluate DEV using optimized thresholds obtained from Train set
    # ------------------------------------------------------------

    accuracy, matthews_corr = evaluate_model(model, dev_data, device, thresholds=optimal_thresholds)
    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    # ------------------------------------------------------------
    # Find class-specific optimal thresholds using Train + Dev set
    # ------------------------------------------------------------

    # Combine all labeled data
    combined_dataset = pd.concat(
        [train_dataset, dev_dataset],
        ignore_index=True
    )

    combined_train_dev_data = transform_data(combined_dataset, max_length=args.max_length, batch_size=args.batch_size)

    optimal_thresholds = find_optimal_thresholds(
        model,
        combined_train_dev_data,
        device,
        num_labels=26
    )

    # ------------------------------------------------------------
    # Test - Using optimized thresholds obtained from Train + Dev set
    # ------------------------------------------------------------

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
