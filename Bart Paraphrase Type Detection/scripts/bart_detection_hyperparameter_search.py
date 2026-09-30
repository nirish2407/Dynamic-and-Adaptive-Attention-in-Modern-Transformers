import optuna
import subprocess
import re

def objective(trial):

    # ------------------------------------------------------------
    # Ask Optuna for hyperparameters
    # ------------------------------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        5e-6,
        3e-5,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-4,
        5e-2,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.05,
        0.5
    )

    classifier_hidden_size = trial.suggest_categorical(
        "classifier_hidden_size",
        [512, 768, 1024, 1536]
    )

    warmup_ratio = trial.suggest_float(
        "warmup_ratio",
        0.0,
        0.2
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [4, 8, 16]
    )

    # ------------------------------------------------------------
    # Build command for your ORIGINAL training script
    # ------------------------------------------------------------

    command = [
        "python",
        "-u",
        "bart_detection.py",

        "--use_gpu",

        "--learning_rate",
        str(learning_rate),

        "--weight_decay",
        str(weight_decay),

        "--dropout",
        str(dropout),

        "--classifier_hidden_size",
        str(classifier_hidden_size),

        "--warmup_ratio",
        str(warmup_ratio),

        "--batch_size",
        str(batch_size),

        # Keep other settings fixed
        "--epochs",
        "20",

        "--patience",
        "5",

        "--loss",
        "BCE",

        "--pooling",
        "mean",

        "--max_length",
        "512",

        "--seed",
        "11711",

        "--disable_tqdm"
    ]

    print("\nRunning:")
    print(" ".join(command))

    # ------------------------------------------------------------
    # Run ORIGINAL program
    # ------------------------------------------------------------

    result = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []

    for line in result.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    result.wait()

    output = "".join(output_lines)

    # ------------------------------------------------------------
    # Check whether training succeeded
    # ------------------------------------------------------------

    if result.returncode != 0:

        raise RuntimeError(
            f"Training failed for trial {trial.number}"
        )

    # ------------------------------------------------------------
    # Extract Dev Accuracy from printed output
    # ------------------------------------------------------------

    accuracy_match = re.search(
        r"The accuracy of the model is:\s*([0-9.eE+-]+)",
        output
    )

    if accuracy_match is None:

        raise RuntimeError(
            "Could not find Dev Accuracy in training output."
        )

    dev_accuracy = float(
        accuracy_match.group(1)
    )

    # ------------------------------------------------------------
    # Store Dev Accuracy as an Optuna user attribute
    # ------------------------------------------------------------

    trial.set_user_attr(
        "dev_accuracy",
        dev_accuracy
    )

    # ------------------------------------------------------------
    # Extract Dev MCC from printed output
    # ------------------------------------------------------------

    mcc_match = re.search(
        r"Matthews Correlation Coefficient of the model is:\s*([0-9.eE+-]+)",
        output
    )

    if mcc_match is None:

        raise RuntimeError(
            "Could not find Dev MCC in training output."
        )

    dev_mcc = float(mcc_match.group(1))

    # ------------------------------------------------------------
    # Print both metrics
    # ------------------------------------------------------------

    print(
        f"Trial {trial.number} "
        f"Dev MCC = {dev_mcc:.4f} "
        f"Dev Accuracy = {dev_accuracy:.4f}"
    )

    # ------------------------------------------------------------
    # Optuna still optimizes Dev MCC
    # ------------------------------------------------------------

    return dev_mcc

# ================================================================
# Save results after every completed trial
# ================================================================

def save_results(study, trial):
    study.trials_dataframe().to_csv(
        "optuna_results_1.csv",
        index=False
    )

# ================================================================
# Create Optuna study
# ================================================================

study = optuna.create_study(
    direction="maximize",
    study_name="bart_etpc_search_1",
    storage="sqlite:///bart_etpc_search_1.db",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(
        seed=11711
    )
)

study.optimize(
    objective,
    n_trials=12,
    callbacks=[save_results]
)

# ================================================================
# Results
# ================================================================

print("\n" + "=" * 60)
print("BEST TRIAL")
print("=" * 60)

print(
    f"Best Dev MCC: {study.best_value:.4f}"
)

best_trial = study.best_trial

best_dev_accuracy = best_trial.user_attrs.get(
    "dev_accuracy",
    None
)

if best_dev_accuracy is not None:
    print(
        f"Dev Accuracy: {best_dev_accuracy:.4f}"
    )

print("\nBest hyperparameters:")

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")

# Final save
study.trials_dataframe().to_csv(
    "optuna_results_1.csv",
    index=False
)