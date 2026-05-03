"""
Model evaluation and comparison module.

Loads both saved models (baseline LR + DistilBERT), runs them on the
same test set, and produces a side-by-side comparison table with F1,
AUC-ROC, and inference time. Also generates confusion matrices.
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import mlflow

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    BASELINE_MODEL_PATH,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    CLEANED_DATASET_PATH,
    LABEL_TO_ID,
    MLFLOW_TRACKING_URI,
    MODELS_FIGURES_DIR,
    NUM_LABELS,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    TEST_SIZE,
    setup_logging,
)

logger = setup_logging()


def _get_test_data() -> tuple:
    """Load and split data to get the same test set used during training.

    Returns:
        Tuple of (X_test texts, y_test labels).
    """
    df = pd.read_csv(CLEANED_DATASET_PATH)
    texts = df["review_text"].tolist()
    labels = df["sentiment"].map(LABEL_TO_ID).values

    _, X_test, _, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    return X_test, y_test


def _evaluate_baseline(X_test: list, y_test: np.ndarray) -> dict:
    """Evaluate the baseline TF-IDF + LogReg model.

    Args:
        X_test: List of test review texts.
        y_test: Array of true labels.

    Returns:
        Dict with predictions, probabilities, metrics, and timing.
    """
    with open(BASELINE_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]

    X_tfidf = vectorizer.transform(X_test)

    start = time.perf_counter()
    y_pred = classifier.predict(X_tfidf)
    y_proba = classifier.predict_proba(X_tfidf)
    inference_ms = (time.perf_counter() - start) * 1000

    y_test_bin = label_binarize(y_test, classes=list(range(NUM_LABELS)))
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")

    return {
        "model": "Baseline (TF-IDF + LR)",
        "y_pred": y_pred,
        "y_proba": y_proba,
        "f1_weighted": round(f1, 4),
        "auc_roc": round(auc, 4),
        "inference_ms": round(inference_ms, 2),
    }


def _evaluate_bert(X_test: list, y_test: np.ndarray) -> dict:
    """Evaluate the fine-tuned DistilBERT model.

    Args:
        X_test: List of test review texts.
        y_test: Array of true labels.

    Returns:
        Dict with predictions, probabilities, metrics, and timing.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
    model.eval()

    # Tokenize all at once
    encodings = tokenizer(
        X_test, truncation=True, padding=True,
        max_length=BERT_MAX_LENGTH, return_tensors="pt",
    )

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()
    inference_ms = (time.perf_counter() - start) * 1000

    y_pred = np.argmax(probs, axis=-1)

    y_test_bin = label_binarize(y_test, classes=list(range(NUM_LABELS)))
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test_bin, probs, multi_class="ovr", average="weighted")

    return {
        "model": "DistilBERT",
        "y_pred": y_pred,
        "y_proba": probs,
        "f1_weighted": round(f1, 4),
        "auc_roc": round(auc, 4),
        "inference_ms": round(inference_ms, 2),
    }


def _save_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                            model_name: str, save_dir: Path) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        model_name: Name for the plot title and filename.
        save_dir: Directory to save the PNG.
    """
    labels = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_LABELS)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save_path = save_dir / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)


def evaluate() -> None:
    """Run full model evaluation and comparison."""

    MODELS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test data ...")
    X_test, y_test = _get_test_data()
    logger.info("Test set size: %d", len(X_test))

    results = []

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    if BASELINE_MODEL_PATH.exists():
        logger.info("Evaluating baseline model ...")
        baseline = _evaluate_baseline(X_test, y_test)
        results.append(baseline)
        _save_confusion_matrix(y_test, baseline["y_pred"], baseline["model"], MODELS_FIGURES_DIR)

        target_names = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]
        report = classification_report(y_test, baseline["y_pred"], target_names=target_names)
        logger.info("Baseline report:\n%s", report)
    else:
        logger.warning("Baseline model not found, skipping")

    # ------------------------------------------------------------------
    # DistilBERT
    # ------------------------------------------------------------------
    if BERT_MODEL_DIR.exists():
        logger.info("Evaluating DistilBERT model ...")
        bert = _evaluate_bert(X_test, y_test)
        results.append(bert)
        _save_confusion_matrix(y_test, bert["y_pred"], bert["model"], MODELS_FIGURES_DIR)

        target_names = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]
        report = classification_report(y_test, bert["y_pred"], target_names=target_names)
        logger.info("DistilBERT report:\n%s", report)
    else:
        logger.warning("DistilBERT model not found, skipping")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    if results:
        comparison = pd.DataFrame([
            {
                "Model": r["model"],
                "F1 (weighted)": r["f1_weighted"],
                "AUC-ROC": r["auc_roc"],
                "Inference Time (ms)": r["inference_ms"],
            }
            for r in results
        ])
        logger.info("\n--- Model Comparison ---\n%s", comparison.to_string(index=False))

        # Save comparison table
        comparison.to_csv(MODELS_FIGURES_DIR / "model_comparison.csv", index=False)
        logger.info("Saved comparison table to %s", MODELS_FIGURES_DIR / "model_comparison.csv")

    # ------------------------------------------------------------------
    # Log comparison to MLflow
    # ------------------------------------------------------------------
    if results:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("sentimentops")

        with mlflow.start_run(run_name="model_comparison"):
            for r in results:
                import re
                prefix = re.sub(r'[^a-z0-9_]', '_', r["model"].lower())
                prefix = re.sub(r'_+', '_', prefix).strip('_')  # collapse multiple underscores
                mlflow.log_metrics({
                    f"{prefix}_f1": r["f1_weighted"],
                    f"{prefix}_auc": r["auc_roc"],
                    f"{prefix}_inference_ms": r["inference_ms"],
                })
            # Log confusion matrix images as artifacts
            for png in MODELS_FIGURES_DIR.glob("confusion_matrix_*.png"):
                mlflow.log_artifact(str(png))

        logger.info("Comparison logged to MLflow")


if __name__ == "__main__":
    evaluate()
