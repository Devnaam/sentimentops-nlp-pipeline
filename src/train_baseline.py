"""
TF-IDF + Logistic Regression baseline for sentiment classification.

This serves as the performance floor. Any neural model that can't beat a
well-tuned TF-IDF + LR baseline on a dataset this small isn't worth the
inference cost. LogReg is fast, interpretable, and surprisingly competitive
on text classification when paired with good n-gram features.
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    BASELINE_MODEL_PATH,
    CLEANED_DATASET_PATH,
    LABEL_TO_ID,
    LR_MAX_ITER,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    NUM_LABELS,
    RANDOM_SEED,
    SENTIMENT_LABELS,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    setup_logging,
)

logger = setup_logging()


def train_baseline() -> None:
    """Train a TF-IDF + LogReg pipeline and log results to MLflow."""

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading cleaned dataset from %s", CLEANED_DATASET_PATH)
    df = pd.read_csv(CLEANED_DATASET_PATH)

    X = df["review_text"].values
    y = df["sentiment"].map(LABEL_TO_ID).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # TF-IDF vectorization
    # ------------------------------------------------------------------
    logger.info(
        "Fitting TF-IDF (max_features=%d, ngram_range=%s)",
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    )
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,  # apply log normalization — reduces the impact of very frequent terms
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    # class_weight='balanced' compensates for the heavy positive-class skew
    # (84% positive vs ~8% each for neutral/negative). Without it the model
    # would learn to predict "positive" for everything.
    clf = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    logger.info("Training Logistic Regression ...")
    clf.fit(X_train_tfidf, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    start = time.perf_counter()
    y_pred = clf.predict(X_test_tfidf)
    inference_time_ms = (time.perf_counter() - start) * 1000

    y_proba = clf.predict_proba(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # AUC-ROC requires one-hot encoding for multiclass
    y_test_bin = label_binarize(y_test, classes=list(range(NUM_LABELS)))
    auc_roc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")

    target_names = [SENTIMENT_LABELS[i] for i in range(NUM_LABELS)]
    report = classification_report(y_test, y_pred, target_names=target_names)

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("F1 (weighted): %.4f", f1)
    logger.info("AUC-ROC (weighted): %.4f", auc_roc)
    logger.info("Inference time: %.2f ms (full test set)", inference_time_ms)
    logger.info("\n%s", report)

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sentimentops")

    with mlflow.start_run(run_name="baseline_lr"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": str(TFIDF_NGRAM_RANGE),
            "test_size": TEST_SIZE,
            "class_weight": "balanced",
            "max_iter": LR_MAX_ITER,
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_weighted": f1,
            "auc_roc_weighted": auc_roc,
            "inference_time_ms": inference_time_ms,
        })
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(clf, "model")

    logger.info("Metrics logged to MLflow")

    # ------------------------------------------------------------------
    # Save model artifacts
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
    }
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    logger.info("Model saved to %s", BASELINE_MODEL_PATH)


if __name__ == "__main__":
    train_baseline()
