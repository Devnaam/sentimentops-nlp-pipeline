"""
LLM fallback module using LangChain + Groq API.

This module provides a secondary classification path for reviews where the
DistilBERT model's confidence is below the threshold. Rather than returning
a low-confidence prediction to the user, we ask a large language model to
classify the review and explain its reasoning.

Design decisions:
    - We use Groq because it offers sub-second inference on large models
      (Llama 3) at a fraction of the cost of OpenAI, making it practical
      to use as a fallback on ~10-15% of predictions.
    - The confidence threshold of 0.65 was chosen because below that value
      the model's softmax distribution is nearly uniform across 3 classes.
      A uniform distribution would give ~0.33 per class, so 0.65 represents
      roughly double the "random guess" confidence — a reasonable boundary
      between "the model has a clear preference" and "the model is guessing."
      In practice this triggers the LLM on the genuinely ambiguous cases
      (sarcasm, mixed reviews, very short texts) without blowing up API costs.
"""

import json
import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import CONFIDENCE_THRESHOLD, GROQ_MODEL_NAME, setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Lazy imports — LangChain + Groq are only imported when actually needed,
# so the rest of the pipeline doesn't break if they aren't installed or
# the API key isn't set.
# ---------------------------------------------------------------------------

_llm_chain = None


def _build_chain():
    """Lazily construct the LangChain chain for sentiment classification.

    Returns:
        A LangChain chain, or None if dependencies/API key are missing.
    """
    global _llm_chain
    if _llm_chain is not None:
        return _llm_chain

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, relying on system env vars")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.warning(
            "GROQ_API_KEY not set or is placeholder — LLM fallback is disabled. "
            "Set it in .env to enable LLM fallback."
        )
        return None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError:
        logger.warning(
            "langchain-groq not installed — LLM fallback is disabled. "
            "Install with: pip install langchain langchain-groq"
        )
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a sentiment analysis expert. Classify the following product "
            "review into exactly one of these categories: positive, negative, or neutral.\n\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            '{{"sentiment": "positive/negative/neutral", "reason": "brief explanation"}}\n\n'
            "Do not include any text outside the JSON object."
        )),
        ("human", "Review: {review_text}"),
    ])

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        temperature=0,  # deterministic for classification
        max_tokens=150,
    )

    _llm_chain = prompt | llm | StrOutputParser()
    logger.info("LLM fallback chain initialized with model: %s", GROQ_MODEL_NAME)
    return _llm_chain


def _parse_llm_response(response: str) -> dict:
    """Parse the LLM's JSON response, handling malformed output gracefully.

    Args:
        response: Raw string response from the LLM.

    Returns:
        Dict with 'sentiment' and 'reason' keys. Falls back to 'neutral'
        if parsing fails, since that's the least harmful default — it
        doesn't falsely attribute strong positive or negative sentiment.
    """
    fallback = {"sentiment": "neutral", "reason": "LLM response could not be parsed"}

    if not response:
        return fallback

    # Try to extract JSON from the response (LLMs sometimes wrap it in markdown)
    response = response.strip()
    if response.startswith("```"):
        # Strip markdown code fences
        lines = response.split("\n")
        response = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Last resort: try to find JSON-like content in the response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response: %s", response[:200])
                return fallback
        else:
            logger.warning("No JSON found in LLM response: %s", response[:200])
            return fallback

    sentiment = parsed.get("sentiment", "").lower().strip()
    valid_sentiments = {"positive", "negative", "neutral"}
    if sentiment not in valid_sentiments:
        logger.warning("Invalid sentiment '%s' from LLM, defaulting to neutral", sentiment)
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "reason": parsed.get("reason", "No reason provided"),
    }


def classify_with_llm(review_text: str) -> dict:
    """Classify a single review using the LLM fallback.

    Args:
        review_text: The review text to classify.

    Returns:
        Dict with 'sentiment', 'reason', and 'model_used' keys.
        Returns a neutral default if the LLM is unavailable.
    """
    chain = _build_chain()

    if chain is None:
        return {
            "sentiment": "neutral",
            "reason": "LLM fallback unavailable (API key not configured)",
            "model_used": "fallback_default",
        }

    try:
        response = chain.invoke({"review_text": review_text})
        result = _parse_llm_response(response)
        result["model_used"] = f"groq_{GROQ_MODEL_NAME}"
        return result
    except Exception as e:
        logger.error("LLM fallback failed: %s", str(e))
        return {
            "sentiment": "neutral",
            "reason": f"LLM fallback error: {str(e)}",
            "model_used": "fallback_default",
        }


def should_use_fallback(confidence: float) -> bool:
    """Determine whether to trigger the LLM fallback.

    Args:
        confidence: The maximum softmax probability from the primary model.

    Returns:
        True if confidence is below the threshold and fallback should be used.
    """
    return confidence < CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    # Quick smoke test
    logger.info("Testing LLM fallback module ...")
    logger.info("Confidence threshold: %.2f", CONFIDENCE_THRESHOLD)
    logger.info("Should use fallback at 0.50 confidence: %s", should_use_fallback(0.50))
    logger.info("Should use fallback at 0.80 confidence: %s", should_use_fallback(0.80))

    test_review = "This product is okay, nothing special but it works."
    result = classify_with_llm(test_review)
    logger.info("Test result: %s", result)
