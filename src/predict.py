"""
Unified prediction interface for SentimentOps.

This module is the single entry point for all inference — both the FastAPI
endpoint and the Streamlit dashboard call `predict()` here. It runs the
DistilBERT model first, checks confidence, and falls back to the LLM if
needed. If the BERT model isn't available (e.g., first run before training),
it gracefully degrades to the baseline LogReg model.

This layered approach ensures we always return a prediction, even if the
best model isn't available yet.
"""

import sys
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    BASELINE_MODEL_PATH,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    CONFIDENCE_THRESHOLD,
    SENTIMENT_LABELS,
    setup_logging,
)
from src.llm_fallback import classify_with_llm, should_use_fallback

logger = setup_logging()

# ---------------------------------------------------------------------------
# Model loading — cached at module level to avoid reloading on every request
# ---------------------------------------------------------------------------

_bert_model = None
_bert_tokenizer = None
_baseline_model = None


def _load_bert():
    """Load the fine-tuned DistilBERT model and tokenizer.

    Returns:
        Tuple of (model, tokenizer) or (None, None) if unavailable.
    """
    global _bert_model, _bert_tokenizer

    if _bert_model is not None:
        return _bert_model, _bert_tokenizer

    if not BERT_MODEL_DIR.exists():
        logger.warning("DistilBERT model not found at %s", BERT_MODEL_DIR)
        return None, None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _bert_tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        _bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        _bert_model.eval()
        logger.info("Loaded DistilBERT model from %s", BERT_MODEL_DIR)
        return _bert_model, _bert_tokenizer
    except Exception as e:
        logger.error("Failed to load DistilBERT: %s", e)
        return None, None


def _load_baseline():
    """Load the baseline TF-IDF + LogReg model.

    Returns:
        Dict with 'vectorizer' and 'classifier' keys, or None.
    """
    global _baseline_model

    if _baseline_model is not None:
        return _baseline_model

    if not BASELINE_MODEL_PATH.exists():
        logger.warning("Baseline model not found at %s", BASELINE_MODEL_PATH)
        return None

    try:
        with open(BASELINE_MODEL_PATH, "rb") as f:
            _baseline_model = pickle.load(f)
        logger.info("Loaded baseline model from %s", BASELINE_MODEL_PATH)
        return _baseline_model
    except Exception as e:
        logger.error("Failed to load baseline model: %s", e)
        return None


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------

def _predict_bert(text: str) -> Optional[dict]:
    """Run inference with the DistilBERT model.

    Args:
        text: Cleaned review text.

    Returns:
        Dict with sentiment, confidence, and model_used, or None if model
        is unavailable.
    """
    model, tokenizer = _load_bert()
    if model is None:
        return None

    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=BERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    sentiment = SENTIMENT_LABELS[predicted_class]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "model_used": "distilbert",
        "all_probs": {SENTIMENT_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }


def _predict_baseline(text: str) -> Optional[dict]:
    """Run inference with the baseline TF-IDF + LogReg model.

    Args:
        text: Cleaned review text.

    Returns:
        Dict with sentiment, confidence, and model_used, or None.
    """
    model_artifact = _load_baseline()
    if model_artifact is None:
        return None

    vectorizer = model_artifact["vectorizer"]
    classifier = model_artifact["classifier"]

    X = vectorizer.transform([text])
    probs = classifier.predict_proba(X)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    sentiment = SENTIMENT_LABELS[predicted_class]

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "model_used": "baseline_lr",
        "all_probs": {SENTIMENT_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))},
    }


def predict(text: str) -> dict:
    """Classify a review's sentiment using the best available model.

    Pipeline:
        1. Try DistilBERT → if confidence >= threshold, return result
        2. If confidence < threshold, try LLM fallback
        3. If BERT unavailable, fall back to baseline LogReg
        4. If nothing works, return a safe default

    Args:
        text: The review text to classify.

    Returns:
        Dict containing:
            - sentiment: 'positive', 'negative', or 'neutral'
            - confidence: float between 0 and 1
            - model_used: which model produced the final answer
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "model_used": "none",
            "error": "Empty input text",
        }

    # Clean the text minimally (the models were trained on cleaned text)
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"[^a-zA-Z\s]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip().lower()

    if not clean:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "model_used": "none",
            "error": "Text became empty after cleaning",
        }

    # --- Try DistilBERT first ---
    bert_result = _predict_bert(clean)

    if bert_result is not None:
        if not should_use_fallback(bert_result["confidence"]):
            return bert_result

        # Confidence is low — try LLM fallback
        logger.info(
            "BERT confidence %.2f < %.2f threshold, trying LLM fallback",
            bert_result["confidence"], CONFIDENCE_THRESHOLD,
        )
        llm_result = classify_with_llm(text)  # send original text, not cleaned
        if llm_result["model_used"] != "fallback_default":
            return {
                "sentiment": llm_result["sentiment"],
                "confidence": bert_result["confidence"],  # keep BERT's confidence for reference
                "model_used": llm_result["model_used"],
                "reason": llm_result.get("reason", ""),
            }

        # LLM fallback unavailable — return BERT result anyway
        return bert_result

    # --- DistilBERT unavailable, try baseline ---
    baseline_result = _predict_baseline(clean)
    if baseline_result is not None:
        return baseline_result

    # --- Nothing available ---
    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "model_used": "none",
        "error": "No models available. Run train_baseline.py or train_bert.py first.",
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """Classify a batch of reviews.

    Args:
        texts: List of review texts.

    Returns:
        List of prediction dicts (same format as predict()).
    """
    return [predict(text) for text in texts]


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "This is an amazing product, I love it!",
        "Terrible quality, broke after one day.",
        "It's okay, nothing special.",
        "",
    ]
    for text in test_texts:
        result = predict(text)
        logger.info("'%s' → %s (%.2f, %s)", text[:50], result["sentiment"], result["confidence"], result["model_used"])
