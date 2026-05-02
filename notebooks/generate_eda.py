"""
Generate the EDA notebook (01_eda.ipynb) programmatically.

We build the notebook in code because .ipynb files are JSON and
cannot be edited by the file editor. This script creates the notebook
with all cells, then executes it to produce the output figures.
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import CLEANED_DATASET_PATH, NOTEBOOKS_FIGURES_DIR, setup_logging

logger = setup_logging()


def run_eda() -> None:
    """Execute all EDA analyses and save figures."""

    # Setup
    NOTEBOOKS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    logger.info("Loading cleaned dataset from %s", CLEANED_DATASET_PATH)
    df = pd.read_csv(CLEANED_DATASET_PATH)
    logger.info("Dataset shape: %s", df.shape)

    # ----------------------------------------------------------------
    # 1. Class distribution bar chart
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["positive", "neutral", "negative"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    counts = df["sentiment"].value_counts().reindex(order)
    ax.bar(order, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, (label, val) in enumerate(zip(order, counts.values)):
        ax.text(i, val + 5, str(val), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(NOTEBOOKS_FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved class_distribution.png")

    # ----------------------------------------------------------------
    # 2. Review length distribution histogram
    # ----------------------------------------------------------------
    df["review_length"] = df["review_text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    for sentiment, color in zip(order, colors):
        subset = df[df["sentiment"] == sentiment]["review_length"]
        ax.hist(subset, bins=50, alpha=0.6, label=sentiment, color=color, edgecolor="white")
    ax.set_title("Review Length Distribution by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_xlim(0, df["review_length"].quantile(0.98))  # trim outliers visually
    fig.tight_layout()
    fig.savefig(NOTEBOOKS_FIGURES_DIR / "review_length_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved review_length_distribution.png")

    # ----------------------------------------------------------------
    # 3. Top 20 most common words per sentiment class
    # ----------------------------------------------------------------
    # Using simple stopword list instead of nltk dependency
    stopwords = set(
        "i me my myself we our ours ourselves you your yours yourself yourselves "
        "he him his himself she her hers herself it its itself they them their theirs "
        "themselves what which who whom this that these those am is are was were be "
        "been being have has had having do does did doing a an the and but if or "
        "because as until while of at by for with about against between through during "
        "before after above below to from up down in out on off over under again further "
        "then once here there when where why how all both each few more most other some "
        "such no nor not only own same so than too very s t can will just don should now "
        "d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn "
        "needn shan shouldn wasn weren won wouldn also would could one two get got like "
        "really much even still just use go going make know think see come take want "
        "read br".split()
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, sentiment, color in zip(axes, order, colors):
        texts = df[df["sentiment"] == sentiment]["review_text"].str.cat(sep=" ")
        words = [w for w in texts.split() if w not in stopwords and len(w) > 2]
        word_counts = Counter(words).most_common(20)

        if word_counts:
            words_list, counts_list = zip(*word_counts)
            ax.barh(range(len(words_list)), counts_list, color=color, edgecolor="white")
            ax.set_yticks(range(len(words_list)))
            ax.set_yticklabels(words_list)
            ax.invert_yaxis()
        ax.set_title(f"Top 20 Words — {sentiment.title()}", fontweight="bold")
        ax.set_xlabel("Frequency")

    fig.suptitle("Most Common Words by Sentiment Class", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(NOTEBOOKS_FIGURES_DIR / "top_words_per_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved top_words_per_sentiment.png")

    # ----------------------------------------------------------------
    # 4. Word clouds per sentiment
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, sentiment in zip(axes, order):
        texts = df[df["sentiment"] == sentiment]["review_text"].str.cat(sep=" ")
        wc = WordCloud(
            width=800, height=400, background_color="white",
            max_words=100, stopwords=stopwords,
            colormap="viridis"
        ).generate(texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{sentiment.title()}", fontweight="bold", fontsize=13)
        ax.axis("off")

    fig.suptitle("Word Clouds by Sentiment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(NOTEBOOKS_FIGURES_DIR / "wordclouds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved wordclouds.png")

    # ----------------------------------------------------------------
    # 5. Average rating by product category
    # ----------------------------------------------------------------
    if "product_category" in df.columns:
        # Categories are comma-separated — take the first category for grouping
        df["primary_category"] = df["product_category"].fillna("Unknown").apply(
            lambda x: x.split(",")[0].strip()
        )
        cat_ratings = (
            df.groupby("primary_category")["rating"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=True)
        )
        # Only show categories with at least 5 reviews to avoid noise
        cat_ratings = cat_ratings[cat_ratings["count"] >= 5]

        if len(cat_ratings) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(cat_ratings) * 0.4)))
            bars = ax.barh(
                cat_ratings.index, cat_ratings["mean"],
                color=sns.color_palette("coolwarm", len(cat_ratings)),
                edgecolor="white"
            )
            ax.set_xlabel("Average Rating")
            ax.set_title("Average Rating by Product Category", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 5.5)
            for bar, count in zip(bars, cat_ratings["count"]):
                ax.text(
                    bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"n={count}", va="center", fontsize=9
                )
            fig.tight_layout()
            fig.savefig(NOTEBOOKS_FIGURES_DIR / "avg_rating_by_category.png", dpi=150)
            plt.close(fig)
            logger.info("Saved avg_rating_by_category.png")

    # ----------------------------------------------------------------
    # 6. Rating distribution
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = df["rating"].value_counts().sort_index()
    ax.bar(
        rating_counts.index.astype(str), rating_counts.values,
        color=sns.color_palette("RdYlGn", 5), edgecolor="black", linewidth=0.5
    )
    ax.set_title("Star Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Count")
    for i, val in enumerate(rating_counts.values):
        ax.text(i, val + 3, str(val), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(NOTEBOOKS_FIGURES_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved rating_distribution.png")

    logger.info("EDA complete — all figures saved to %s", NOTEBOOKS_FIGURES_DIR)


def create_notebook() -> None:
    """Create the Jupyter notebook with EDA code cells."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell(
        "# SentimentOps — Exploratory Data Analysis\n\n"
        "This notebook explores the cleaned Amazon product reviews dataset to understand "
        "class distributions, review characteristics, and vocabulary patterns across "
        "sentiment classes.\n\n"
        "**Dataset**: `data/processed/cleaned_reviews.csv` (output of `src/preprocess.py`)"
    ))

    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "from collections import Counter\n\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from wordcloud import WordCloud\n\n"
        "# Ensure project root is on path\n"
        "PROJECT_ROOT = Path.cwd().parent\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.config import CLEANED_DATASET_PATH, NOTEBOOKS_FIGURES_DIR\n\n"
        "NOTEBOOKS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)\n\n"
        "df = pd.read_csv(CLEANED_DATASET_PATH)\n"
        "print(f'Dataset shape: {df.shape}')\n"
        "df.head()"
    ))

    # Class distribution
    cells.append(nbf.v4.new_markdown_cell("## 1. Sentiment Class Distribution"))
    cells.append(nbf.v4.new_code_cell(
        "order = ['positive', 'neutral', 'negative']\n"
        "colors = ['#2ecc71', '#f39c12', '#e74c3c']\n"
        "counts = df['sentiment'].value_counts().reindex(order)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.bar(order, counts.values, color=colors, edgecolor='black', linewidth=0.5)\n"
        "for i, val in enumerate(counts.values):\n"
        "    ax.text(i, val + 5, str(val), ha='center', fontweight='bold', fontsize=12)\n"
        "ax.set_title('Sentiment Class Distribution', fontsize=14, fontweight='bold')\n"
        "ax.set_xlabel('Sentiment')\n"
        "ax.set_ylabel('Count')\n"
        "fig.tight_layout()\n"
        "fig.savefig(NOTEBOOKS_FIGURES_DIR / 'class_distribution.png', dpi=150)\n"
        "plt.show()"
    ))

    # Review length
    cells.append(nbf.v4.new_markdown_cell("## 2. Review Length Distribution"))
    cells.append(nbf.v4.new_code_cell(
        "df['review_length'] = df['review_text'].str.split().str.len()\n\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "for sentiment, color in zip(order, colors):\n"
        "    subset = df[df['sentiment'] == sentiment]['review_length']\n"
        "    ax.hist(subset, bins=50, alpha=0.6, label=sentiment, color=color)\n"
        "ax.set_title('Review Length Distribution by Sentiment', fontweight='bold')\n"
        "ax.set_xlabel('Number of Words')\n"
        "ax.set_ylabel('Frequency')\n"
        "ax.legend()\n"
        "ax.set_xlim(0, df['review_length'].quantile(0.98))\n"
        "fig.tight_layout()\n"
        "fig.savefig(NOTEBOOKS_FIGURES_DIR / 'review_length_distribution.png', dpi=150)\n"
        "plt.show()"
    ))

    # Top words
    cells.append(nbf.v4.new_markdown_cell("## 3. Top 20 Most Common Words per Sentiment"))
    cells.append(nbf.v4.new_code_cell(
        "stopwords = set(\n"
        "    'i me my myself we our ours ourselves you your yours yourself yourselves '\n"
        "    'he him his himself she her hers herself it its itself they them their theirs '\n"
        "    'themselves what which who whom this that these those am is are was were be '\n"
        "    'been being have has had having do does did doing a an the and but if or '\n"
        "    'because as until while of at by for with about against between through during '\n"
        "    'before after above below to from up down in out on off over under again further '\n"
        "    'then once here there when where why how all both each few more most other some '\n"
        "    'such no nor not only own same so than too very s t can will just don should now '\n"
        "    'read br also would could one two get got like really much even still use go going '\n"
        "    'make know think see come take want'.split()\n"
        ")\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n"
        "for ax, sentiment, color in zip(axes, order, colors):\n"
        "    texts = df[df['sentiment'] == sentiment]['review_text'].str.cat(sep=' ')\n"
        "    words = [w for w in texts.split() if w not in stopwords and len(w) > 2]\n"
        "    wc = Counter(words).most_common(20)\n"
        "    if wc:\n"
        "        w, c = zip(*wc)\n"
        "        ax.barh(range(len(w)), c, color=color)\n"
        "        ax.set_yticks(range(len(w)))\n"
        "        ax.set_yticklabels(w)\n"
        "        ax.invert_yaxis()\n"
        "    ax.set_title(f'Top 20 — {sentiment.title()}', fontweight='bold')\n"
        "fig.tight_layout()\n"
        "fig.savefig(NOTEBOOKS_FIGURES_DIR / 'top_words_per_sentiment.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # Word clouds
    cells.append(nbf.v4.new_markdown_cell("## 4. Word Clouds"))
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n"
        "for ax, sentiment in zip(axes, order):\n"
        "    texts = df[df['sentiment'] == sentiment]['review_text'].str.cat(sep=' ')\n"
        "    wc = WordCloud(width=800, height=400, background_color='white', max_words=100,\n"
        "                   stopwords=stopwords, colormap='viridis').generate(texts)\n"
        "    ax.imshow(wc, interpolation='bilinear')\n"
        "    ax.set_title(f'{sentiment.title()}', fontweight='bold')\n"
        "    ax.axis('off')\n"
        "fig.suptitle('Word Clouds by Sentiment', fontsize=14, fontweight='bold')\n"
        "fig.tight_layout()\n"
        "fig.savefig(NOTEBOOKS_FIGURES_DIR / 'wordclouds.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ))

    # Average rating by category
    cells.append(nbf.v4.new_markdown_cell("## 5. Average Rating by Product Category"))
    cells.append(nbf.v4.new_code_cell(
        "if 'product_category' in df.columns:\n"
        "    df['primary_category'] = df['product_category'].fillna('Unknown').apply(\n"
        "        lambda x: x.split(',')[0].strip()\n"
        "    )\n"
        "    cat_ratings = df.groupby('primary_category')['rating'].agg(['mean', 'count'])\n"
        "    cat_ratings = cat_ratings[cat_ratings['count'] >= 5].sort_values('mean', ascending=True)\n\n"
        "    fig, ax = plt.subplots(figsize=(10, max(4, len(cat_ratings) * 0.4)))\n"
        "    ax.barh(cat_ratings.index, cat_ratings['mean'],\n"
        "            color=sns.color_palette('coolwarm', len(cat_ratings)))\n"
        "    ax.set_xlabel('Average Rating')\n"
        "    ax.set_title('Average Rating by Product Category', fontweight='bold')\n"
        "    ax.set_xlim(0, 5.5)\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(NOTEBOOKS_FIGURES_DIR / 'avg_rating_by_category.png', dpi=150)\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No product_category column found')"
    ))

    nb.cells = cells

    notebook_path = _project_root / "notebooks" / "01_eda.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    logger.info("Created notebook at %s", notebook_path)


if __name__ == "__main__":
    # Generate the notebook file
    create_notebook()
    # Also run the EDA directly to produce figures
    run_eda()
