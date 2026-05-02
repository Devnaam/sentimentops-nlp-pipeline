"""
Data preprocessing pipeline for SentimentOps.

Responsibilities:
    1. Load the raw Amazon product reviews CSV.
    2. Clean review text (HTML tags, special characters, whitespace).
    3. Map star ratings → three sentiment classes (Positive / Neutral / Negative).
    4. Persist the cleaned dataframe for downstream consumers.

Design notes:
    - We intentionally keep preprocessing as a standalone script rather than
      burying it inside a training loop. This lets us inspect the cleaned data,
      cache it, and feed multiple models from the same artifact.
    - The HuggingFace tokenizer is NOT run here. DistilBERT tokenization is
      deferred to `train_bert.py` because the Trainer API expects raw strings
      and applies tokenization inside its own DataLoader. Running it twice
      would waste memory for no benefit.
"""

import re
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `src.config` resolves when
# this script is executed directly (python src/preprocess.py).
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import (
    CLEANED_DATASET_PATH,
    DATA_PROCESSED_DIR,
    RAW_DATASET_PATH,
    setup_logging,
)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove HTML tags, special characters, and collapse whitespace.

    Args:
        text: Raw review string, possibly containing HTML artifacts.

    Returns:
        Lowercased string with only alphabetic characters and single spaces.
    """
    if not isinstance(text, str):
        return ""

    # Strip HTML tags (the dataset contains <br />, <b>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep only letters and spaces — punctuation doesn't help TF-IDF much
    # and DistilBERT's tokenizer handles raw text better when we give it
    # clean input (its WordPiece vocab doesn't benefit from stray symbols).
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


# ---------------------------------------------------------------------------
# Rating → sentiment mapping
# ---------------------------------------------------------------------------

def map_rating_to_sentiment(rating: int) -> str:
    """Convert a 1-5 star rating to a sentiment label.

    Mapping rationale:
        - 4-5 stars → positive  (clear satisfaction)
        - 3 stars   → neutral   (ambivalent / mixed)
        - 1-2 stars → negative  (clear dissatisfaction)

    Args:
        rating: Integer star rating between 1 and 5.

    Returns:
        One of 'positive', 'neutral', or 'negative'.
    """
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def load_and_clean(input_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load raw CSV, clean text, and derive sentiment labels.

    Args:
        input_path: Path to the raw reviews CSV file.

    Returns:
        DataFrame with columns: review_text, review_title, rating,
        sentiment, review_date, product_category, product_name.
    """
    logger.info("Loading raw dataset from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Raw dataset shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Select and rename only the columns we actually need.
    # This keeps the cleaned dataset lean and easy to reason about.
    # ------------------------------------------------------------------
    column_mapping = {
        "reviews.text": "review_text",
        "reviews.title": "review_title",
        "reviews.rating": "rating",
        "reviews.date": "review_date",
        "categories": "product_category",
        "name": "product_name",
    }

    # Keep only columns that exist in the raw data
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df[list(available_cols.keys())].rename(columns=available_cols)

    # ------------------------------------------------------------------
    # Drop rows without usable text or rating
    # ------------------------------------------------------------------
    initial_count = len(df)
    df = df.dropna(subset=["review_text", "rating"])
    df["rating"] = df["rating"].astype(int)
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with missing text or rating", dropped)

    # ------------------------------------------------------------------
    # Clean text
    # ------------------------------------------------------------------
    logger.info("Cleaning review text ...")
    df["review_text"] = df["review_text"].apply(clean_text)

    # Drop any rows that became empty after cleaning
    df = df[df["review_text"].str.len() > 0]

    # ------------------------------------------------------------------
    # Map ratings → sentiment
    # ------------------------------------------------------------------
    df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)

    # ------------------------------------------------------------------
    # Clean up dates (best-effort parse, NaT for unparseable)
    # ------------------------------------------------------------------
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    # ------------------------------------------------------------------
    # Deduplicate — some entries in the raw data are exact copies
    # ------------------------------------------------------------------
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["review_text"], keep="first")
    dupes = before_dedup - len(df)
    if dupes > 0:
        logger.info("Removed %d duplicate reviews", dupes)

    df = df.reset_index(drop=True)
    logger.info("Cleaned dataset shape: %s", df.shape)

    return df


def save_cleaned(df: pd.DataFrame, output_path: Path = CLEANED_DATASET_PATH) -> None:
    """Persist the cleaned dataframe to CSV.

    Args:
        df: Cleaned DataFrame to save.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)


def print_class_distribution(df: pd.DataFrame) -> None:
    """Log the class distribution of the sentiment column.

    Args:
        df: DataFrame with a 'sentiment' column.
    """
    dist = df["sentiment"].value_counts()
    total = len(df)
    logger.info("--- Class Distribution ---")
    for label, count in dist.items():
        pct = count / total * 100
        logger.info("  %s: %d (%.1f%%)", label, count, pct)
    logger.info("--------------------------")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full preprocessing pipeline."""
    df = load_and_clean()
    print_class_distribution(df)
    save_cleaned(df)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
