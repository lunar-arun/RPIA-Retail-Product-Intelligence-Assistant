"""
Central configuration: file paths and app-wide settings.

Keeping paths in one place means the rest of the app never hard-codes
"data/..." strings, so the project can be run from any working directory
(e.g. `streamlit run app/main.py` from the repo root, or from a deployment
container).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_REVIEWS_PATH = RAW_DIR / "reviews.csv"
CLEAN_REVIEWS_PATH = PROCESSED_DIR / "clean_reviews.csv"
SENTIMENT_REVIEWS_PATH = PROCESSED_DIR / "reviews_with_sentiment.csv"
RETRIEVAL_INDEX_PATH = MODELS_DIR / "retrieval_index.pkl"

# --- Future API integration -------------------------------------------------
# This project runs fully offline today (no keys required). If/when a real
# marketplace or LLM API is introduced, its key(s) should be read from the
# environment (never hard-coded) and the relevant service class swapped in
# behind the interfaces in src/llm/answer_service.py and
# src/search/retrieval_service.py. Example:
#
#   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
USE_REMOTE_ANSWER_SERVICE = os.getenv("RPIA_USE_REMOTE_ANSWER_SERVICE", "false").lower() == "true"

# --- Optional local Hugging Face sentiment model ----------------------------
# When True, tag_sentiment() uses distilbert-base-uncased-finetuned-sst-2-english
# (downloaded once to the local HF cache, ~260 MB, then run fully offline).
# When False (default) the lightweight lexicon-based scorer is used instead —
# no download, no network access, fully deterministic.
# Enable with:  RPIA_USE_HF_SENTIMENT_MODEL=true python scripts/build_pipeline.py
USE_HF_SENTIMENT_MODEL: bool = (
    os.getenv("RPIA_USE_HF_SENTIMENT_MODEL", "false").lower() == "true"
)
HF_SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

