"""
Streamlit dashboard for SentimentOps.

Provides an interactive UI for:
    - Uploading a CSV of reviews and running batch predictions
    - Visualizing sentiment distribution across uploaded reviews
    - Showing sentiment trends over time (if a date column exists)
    - Displaying color-coded predictions table with confidence scores
    - Tracking what percentage of predictions used the LLM fallback
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.predict import predict, predict_batch
from src.config import CONFIDENCE_THRESHOLD, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentimentOps Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a cleaner look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .sentiment-positive { color: #2ecc71; font-weight: bold; }
    .sentiment-negative { color: #e74c3c; font-weight: bold; }
    .sentiment-neutral { color: #f39c12; font-weight: bold; }
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 SentimentOps Dashboard")
st.markdown(
    "Upload a CSV of product reviews to run batch sentiment analysis. "
    f"Predictions with confidence < **{CONFIDENCE_THRESHOLD:.0%}** trigger the LLM fallback."
)

# ---------------------------------------------------------------------------
# Sidebar — single review prediction
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Quick Predict")
    st.markdown("Test a single review instantly.")
    single_text = st.text_area("Enter a review:", height=120, placeholder="Type a product review here...")

    if st.button("Predict", type="primary"):
        if single_text.strip():
            with st.spinner("Analyzing ..."):
                result = predict(single_text)

            sentiment_colors = {
                "positive": "🟢", "negative": "🔴", "neutral": "🟡"
            }
            emoji = sentiment_colors.get(result["sentiment"], "⚪")

            st.markdown(f"### {emoji} {result['sentiment'].title()}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.caption(f"Model: `{result['model_used']}`")

            if result.get("reason"):
                st.info(f"LLM Reason: {result['reason']}")
        else:
            st.warning("Please enter some text.")

# ---------------------------------------------------------------------------
# Main area — CSV upload and batch analysis
# ---------------------------------------------------------------------------
st.header("📁 Batch Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a text column (e.g., 'review_text', 'text', 'reviews.text')",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown(f"**Uploaded:** {len(df)} rows, {len(df.columns)} columns")

    # Auto-detect the text column
    text_col_candidates = ["review_text", "text", "reviews.text", "review", "comment", "body"]
    text_col = None
    for candidate in text_col_candidates:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        # Let the user pick
        text_col = st.selectbox("Select the text column:", df.columns.tolist())

    # Auto-detect date column
    date_col = None
    date_candidates = ["review_date", "date", "reviews.date", "timestamp", "created_at"]
    for candidate in date_candidates:
        if candidate in df.columns:
            date_col = candidate
            break

    if st.button("🚀 Run Batch Predictions", type="primary"):
        texts = df[text_col].fillna("").tolist()

        # Progress bar
        progress = st.progress(0, text="Running predictions ...")
        results = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = predict_batch(batch)
            results.extend(batch_results)
            progress.progress(
                min((i + batch_size) / len(texts), 1.0),
                text=f"Processed {min(i + batch_size, len(texts))}/{len(texts)} reviews",
            )

        progress.empty()

        # Add results to dataframe
        df["sentiment"] = [r["sentiment"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["model_used"] = [r["model_used"] for r in results]

        st.success(f"✅ Completed {len(results)} predictions!")

        # ----- Metrics row -----
        col1, col2, col3, col4 = st.columns(4)

        n_positive = (df["sentiment"] == "positive").sum()
        n_negative = (df["sentiment"] == "negative").sum()
        n_neutral = (df["sentiment"] == "neutral").sum()
        n_fallback = sum(1 for r in results if "groq" in r.get("model_used", ""))
        fallback_pct = n_fallback / len(results) * 100 if results else 0

        col1.metric("🟢 Positive", n_positive)
        col2.metric("🟡 Neutral", n_neutral)
        col3.metric("🔴 Negative", n_negative)
        col4.metric("🤖 LLM Fallback", f"{fallback_pct:.1f}%")

        # ----- Charts -----
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Sentiment Distribution")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
            counts = df["sentiment"].value_counts()
            bars = ax.bar(
                counts.index, counts.values,
                color=[colors.get(s, "#95a5a6") for s in counts.index],
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                        str(val), ha="center", fontweight="bold")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
            plt.close(fig)

        with chart_col2:
            # Sentiment trend if date column exists
            if date_col and date_col in df.columns:
                st.subheader("Sentiment Trend Over Time")
                df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
                df_dated = df.dropna(subset=["_date"]).copy()

                if len(df_dated) > 0:
                    # Group by month and sentiment
                    df_dated["_month"] = df_dated["_date"].dt.to_period("M").astype(str)
                    trend = df_dated.groupby(["_month", "sentiment"]).size().unstack(fill_value=0)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    for sentiment in ["positive", "neutral", "negative"]:
                        if sentiment in trend.columns:
                            ax.plot(
                                trend.index, trend[sentiment],
                                label=sentiment, color=colors.get(sentiment, "#95a5a6"),
                                marker="o", linewidth=2,
                            )
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Trend")
                    ax.legend()
                    plt.xticks(rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No valid dates found for trend chart.")
            else:
                st.subheader("Confidence Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df["confidence"], bins=20, color="#3498db", edgecolor="white", alpha=0.8)
                ax.axvline(CONFIDENCE_THRESHOLD, color="#e74c3c", linestyle="--",
                           label=f"Threshold ({CONFIDENCE_THRESHOLD})")
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Count")
                ax.set_title("Prediction Confidence Distribution")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

        # ----- Predictions table -----
        st.subheader("📋 Prediction Results")

        # Color-code sentiment
        def style_sentiment(val):
            colors_map = {
                "positive": "background-color: rgba(46, 204, 113, 0.2); color: #2ecc71;",
                "negative": "background-color: rgba(231, 76, 60, 0.2); color: #e74c3c;",
                "neutral": "background-color: rgba(243, 156, 18, 0.2); color: #f39c12;",
            }
            return colors_map.get(val, "")

        display_df = df[[text_col, "sentiment", "confidence", "model_used"]].copy()
        display_df.columns = ["Review Text", "Sentiment", "Confidence", "Model"]
        display_df["Review Text"] = display_df["Review Text"].str[:200] + "..."
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")

        styled = display_df.style.map(style_sentiment, subset=["Sentiment"])
        st.dataframe(styled, use_container_width=True, height=400)

        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "sentiment_predictions.csv",
            "text/csv",
        )

else:
    st.info("👆 Upload a CSV file to get started, or use the sidebar for single predictions.")

    # Show a quick demo
    st.markdown("---")
    st.subheader("📝 How it works")
    st.markdown("""
    1. **Upload** a CSV with product reviews
    2. **Select** the text column containing reviews
    3. **Run** batch predictions — each review is classified as Positive, Neutral, or Negative
    4. **Explore** the results with interactive charts and tables

    The pipeline uses **DistilBERT** as the primary model. When the model's confidence
    is below the threshold, it automatically falls back to an **LLM (via Groq)** for
    a second opinion.
    """)
