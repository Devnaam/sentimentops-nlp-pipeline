"""
DistilBERT fine-tuning for 3-class sentiment classification.

Why DistilBERT over full BERT:
    DistilBERT retains ~97% of BERT-base's language understanding capacity
    while being 60% faster at inference and 40% smaller in parameter count
    (66M vs 110M). For a downstream 3-class sentiment task the marginal
    accuracy gain from full BERT rarely justifies the doubled training time
    and GPU memory footprint. On a small dataset like ours (~900 samples),
    the regularization benefit of a smaller model actually helps — there are
    fewer parameters to overfit.

Architecture:
    distilbert-base-uncased → DistilBertForSequenceClassification (3 labels)
    Training via HuggingFace Trainer API with early-stopping-like behavior
    (best model saved at end of each epoch via `load_best_model_at_end`).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import mlflow

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_WARMUP_STEPS,
    BERT_WEIGHT_DECAY,
    CLEANED_DATASET_PATH,
    LABEL_TO_ID,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    NUM_LABELS,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    TEST_SIZE,
    setup_logging,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Dataset wrapper — Trainer expects a torch Dataset with __getitem__
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Wraps tokenized inputs into a PyTorch Dataset for the Trainer API.

    Args:
        encodings: Dict of tokenizer outputs (input_ids, attention_mask).
        labels: Integer label array.
    """

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Metrics callback for Trainer
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """Compute F1, accuracy, and AUC-ROC during evaluation.

    Args:
        eval_pred: EvalPrediction from HuggingFace Trainer (logits, labels).

    Returns:
        Dict of metric names to values.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Softmax for AUC-ROC
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    # AUC-ROC (one-vs-rest)
    try:
        labels_bin = label_binarize(labels, classes=list(range(NUM_LABELS)))
        auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="weighted")
    except ValueError:
        # Can happen if a class is missing from a small eval batch
        auc = 0.0

    return {"f1_weighted": f1, "accuracy": acc, "auc_roc_weighted": auc}


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_bert() -> None:
    """Fine-tune DistilBERT and log results to MLflow."""

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    logger.info("Loading cleaned dataset from %s", CLEANED_DATASET_PATH)
    df = pd.read_csv(CLEANED_DATASET_PATH)

    texts = df["review_text"].tolist()
    labels = df["sentiment"].map(LABEL_TO_ID).values

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    logger.info("Tokenizing with %s (max_length=%d)", BERT_MODEL_NAME, BERT_MAX_LENGTH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    train_encodings = tokenizer(
        X_train, truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
    )
    test_encodings = tokenizer(
        X_test, truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
    )

    train_dataset = SentimentDataset(train_encodings, y_train)
    test_dataset = SentimentDataset(test_encodings, y_test)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading %s for %d-class classification", BERT_MODEL_NAME, NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=SENTIMENT_LABELS,
        label2id=LABEL_TO_ID,
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    # Output dir for checkpoints — cleaned up after training
    output_dir = MODELS_DIR / "bert_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_steps=BERT_WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        seed=RANDOM_SEED,
        # CPU-specific: disable pin_memory to avoid warnings on non-CUDA systems
        dataloader_pin_memory=False,
        # Disable default MLflow integration — we handle logging ourselves
        # to maintain consistent experiment structure across models
        report_to="none",
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning for %d epochs ...", BERT_EPOCHS)
    train_result = trainer.train()
    logger.info("Training complete. Total steps: %d", train_result.global_step)

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set ...")
    eval_results = trainer.evaluate()

    # Also get per-sample predictions for detailed report
    start = time.perf_counter()
    predictions = trainer.predict(test_dataset)
    inference_time_ms = (time.perf_counter() - start) * 1000

    preds = np.argmax(predictions.predictions, axis=-1)
    target_names = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]
    report = classification_report(y_test, preds, target_names=target_names)

    logger.info("Eval results: %s", eval_results)
    logger.info("Inference time: %.2f ms (full test set)", inference_time_ms)
    logger.info("\n%s", report)

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentimentops")

    with mlflow.start_run(run_name="distilbert_sentiment"):
        mlflow.log_params({
            "model": BERT_MODEL_NAME,
            "max_length": BERT_MAX_LENGTH,
            "learning_rate": BERT_LEARNING_RATE,
            "weight_decay": BERT_WEIGHT_DECAY,
            "epochs": BERT_EPOCHS,
            "batch_size": BERT_BATCH_SIZE,
            "warmup_steps": BERT_WARMUP_STEPS,
            "test_size": TEST_SIZE,
        })
        mlflow.log_metrics({
            "f1_weighted": eval_results.get("eval_f1_weighted", 0),
            "accuracy": eval_results.get("eval_accuracy", 0),
            "auc_roc_weighted": eval_results.get("eval_auc_roc_weighted", 0),
            "inference_time_ms": inference_time_ms,
            "train_loss": train_result.training_loss,
        })
        mlflow.log_text(report, "classification_report.txt")

    logger.info("Metrics logged to MLflow")

    # ------------------------------------------------------------------
    # Save model + tokenizer
    # ------------------------------------------------------------------
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))
    logger.info("Model saved to %s", BERT_MODEL_DIR)

    # Clean up checkpoints to save disk space
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleaned up checkpoint directory")


if __name__ == "__main__":
    train_bert()
