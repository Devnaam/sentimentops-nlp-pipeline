"""
FastAPI application for the SentimentOps prediction service.

Endpoints:
    POST /predict — classify a review's sentiment
    GET  /health  — liveness check

The API is intentionally thin — all prediction logic lives in src/predict.py
so it can be reused by the Streamlit dashboard without circular imports.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.predict import predict
from src.config import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SentimentOps API",
    description="Sentiment classification with DistilBERT + LLM fallback",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input schema for the /predict endpoint."""
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only strings at validation time."""
        if not v or not v.strip():
            raise ValueError("Review text must not be empty")
        return v


class PredictResponse(BaseModel):
    """Output schema for the /predict endpoint."""
    sentiment: str
    confidence: float
    model_used: str
    reason: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest) -> PredictResponse:
    """Classify the sentiment of a product review.

    The prediction pipeline tries DistilBERT first. If its confidence
    is below the threshold, it falls back to an LLM via Groq API.
    If DistilBERT isn't available, it falls back to the baseline
    TF-IDF + Logistic Regression model.

    Args:
        request: JSON body with a 'text' field.

    Returns:
        JSON with sentiment, confidence score, and which model was used.
    """
    try:
        result = predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple liveness probe for load balancers and monitoring.

    Returns:
        JSON with status and API version.
    """
    return HealthResponse(status="healthy", version="1.0.0")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
