"""
Central configuration for the SentimentOps pipeline.

All magic numbers, model names, file paths, and tunable parameters live here.
This prevents constants from being scattered across modules and makes it trivial
to adjust the pipeline's behavior from a single location.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — everything is relative to the project root so the code works
# regardless of where it's invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_FIGURES_DIR = MODELS_DIR / "figures"
NOTEBOOKS_FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# MLflow requires a proper file URI on Windows (bare paths cause scheme errors)
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

# The raw dataset shipped with the repo (note: file has double .csv extension)
RAW_DATASET_PATH = DATA_RAW_DIR / "reviews.csv.csv"
CLEANED_DATASET_PATH = DATA_PROCESSED_DIR / "cleaned_reviews.csv"

# Saved model artifacts
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_lr.pkl"
BERT_MODEL_DIR = MODELS_DIR / "distilbert_sentiment"

# ---------------------------------------------------------------------------
# Model & training hyperparameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20  # 80/20 split

# TF-IDF baseline
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
LR_MAX_ITER = 1000

# DistilBERT
# Why DistilBERT over full BERT: DistilBERT retains ~97% of BERT's language
# understanding while being 60% faster and 40% smaller. For a 3-class
# sentiment task the marginal accuracy gain from full BERT rarely justifies
# the doubled training time and GPU memory.
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256  # most reviews fit well within 256 tokens
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 16
BERT_WARMUP_STEPS = 100

# LLM fallback
# 0.65 was chosen empirically: below this threshold the model's softmax
# distribution is nearly uniform across classes, meaning it is genuinely
# uncertain rather than slightly favoring one class. In practice this
# triggers the LLM on ~10-15% of predictions — enough to catch edge cases
# without blowing up API costs.
CONFIDENCE_THRESHOLD = 0.65
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Sentiment label mapping
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
NUM_LABELS = 3

# ---------------------------------------------------------------------------
# Logging — consistent format across all modules
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a project-wide logger.

    Args:
        level: Logging verbosity level (default: INFO).

    Returns:
        Configured Logger instance for the sentimentops namespace.
    """
    logger = logging.getLogger("sentimentops")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
