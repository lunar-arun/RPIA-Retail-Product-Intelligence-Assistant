"""
Local sentiment analysis.

Two scoring modes are available, selected by ``src.config.USE_HF_SENTIMENT_MODEL``:

1. **Lexicon scorer** (default, ``USE_HF_SENTIMENT_MODEL=False``)
   - No download, no external calls, fully deterministic.
   - Uses small keyword sets to vote POSITIVE / NEGATIVE / NEUTRAL.

2. **HF model scorer** (opt-in, ``USE_HF_SENTIMENT_MODEL=True``)
   - Uses ``distilbert-base-uncased-finetuned-sst-2-english`` via
     ``transformers.pipeline``, downloaded once to the local HF cache
     (~260 MB) and then run fully offline with no network calls at
     inference time.
   - Star rating is the *primary* signal; the model output is used only
     to resolve ambiguous 3-star cases (NEUTRAL → model decides).

Callers only ever see ``tag_sentiment(df) -> df`` — the mode is an
implementation detail hidden behind the config flag.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass  # avoid circular imports at type-check time

# ---------------------------------------------------------------------------
# Lexicon scorer (mode 1 — always available, no downloads)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "love", "loved", "good", "fast",
    "smooth", "premium", "impressive", "recommend", "easy", "solid",
    "quick", "value", "worth", "best", "perfect", "reliable", "comfortable",
}

NEGATIVE_WORDS = {
    "bad", "slow", "poor", "cheap", "defect", "defective", "issue",
    "issues", "problem", "problems", "drains", "worse", "bug", "bugs",
    "broken", "disappointing", "damaged", "lag", "lags", "waste", "slow",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _score_text(text: str) -> tuple[str, float]:
    """Return (label, confidence) for a single piece of review text."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    pos_hits = len(words & POSITIVE_WORDS)
    neg_hits = len(words & NEGATIVE_WORDS)

    if pos_hits == neg_hits:
        return "NEUTRAL", 0.5
    if pos_hits > neg_hits:
        confidence = min(0.95, 0.6 + 0.1 * (pos_hits - neg_hits))
        return "POSITIVE", confidence
    confidence = min(0.95, 0.6 + 0.1 * (neg_hits - pos_hits))
    return "NEGATIVE", confidence


# ---------------------------------------------------------------------------
# HF model scorer (mode 2 — optional, requires transformers + torch)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_hf_pipeline():
    """Load the HF sentiment pipeline exactly once per process.

    Downloaded on first call; subsequent calls return the cached object
    with no network access.  Returns None if transformers is not installed
    so the caller can fall back gracefully.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        from src.config import HF_SENTIMENT_MODEL

        return hf_pipeline(
            "sentiment-analysis",
            model=HF_SENTIMENT_MODEL,
            device=-1,          # CPU-only — no GPU required
            truncation=True,
            max_length=512,
        )
    except Exception:  # ImportError, OSError, network down, etc.
        return None


def _hf_label_to_rpia(hf_label: str) -> str:
    """Map transformers output labels to RPIA's three-category labels."""
    label = hf_label.upper()
    if label in ("POSITIVE", "LABEL_1"):
        return "POSITIVE"
    if label in ("NEGATIVE", "LABEL_0"):
        return "NEGATIVE"
    return "NEUTRAL"


def _score_text_hf(text: str) -> tuple[str, float] | None:
    """Run the HF pipeline on *text*.  Returns None if model unavailable."""
    pipe = _get_hf_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])  # guard against very long texts
        label = _hf_label_to_rpia(result[0]["label"])
        score = float(result[0]["score"])
        return label, score
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Add ``sentiment`` and ``sentiment_score`` columns to a copy of *df*.

    Combination logic when ``USE_HF_SENTIMENT_MODEL`` is True:
    - star_rating > 3  → POSITIVE  (rating is primary signal)
    - star_rating < 3  → NEGATIVE
    - star_rating == 3 → use HF model output; fall back to lexicon / NEUTRAL

    When ``USE_HF_SENTIMENT_MODEL`` is False (default), the lexicon scorer
    is used with the same star-rating fallback as the original version,
    and no model download is required.
    """
    from src.config import USE_HF_SENTIMENT_MODEL

    df = df.copy()
    labels: list[str] = []
    scores: list[float] = []

    has_rating = "star_rating" in df.columns

    for _, row in df.iterrows():
        text = str(row[text_column])
        rating = float(row["star_rating"]) if has_rating else None

        if USE_HF_SENTIMENT_MODEL:
            # --- Star rating is primary ---------------------------------
            if rating is not None and rating > 3:
                label, score = "POSITIVE", 0.85
            elif rating is not None and rating < 3:
                label, score = "NEGATIVE", 0.85
            else:
                # Ambiguous (3-star or no rating) — ask the HF model
                hf_result = _score_text_hf(text)
                if hf_result is not None:
                    label, score = hf_result
                else:
                    # Model unavailable — fall back to lexicon
                    label, score = _score_text(text)
                    if label == "NEUTRAL" and rating is not None:
                        label, score = "NEUTRAL", 0.5
        else:
            # --- Lexicon scorer with star-rating fallback (original) ----
            label, score = _score_text(text)
            if label == "NEUTRAL" and has_rating:
                rating_val = row["star_rating"]
                if rating_val >= 4:
                    label, score = "POSITIVE", 0.6
                elif rating_val <= 2:
                    label, score = "NEGATIVE", 0.6

        labels.append(label)
        scores.append(score)

    df["sentiment"] = labels
    df["sentiment_score"] = scores
    return df


def main() -> None:
    """CLI entry point kept for parity with the original standalone script."""
    from src.config import CLEAN_REVIEWS_PATH, SENTIMENT_REVIEWS_PATH

    df = pd.read_csv(CLEAN_REVIEWS_PATH)
    df = tag_sentiment(df)
    df.to_csv(SENTIMENT_REVIEWS_PATH, index=False)
    print(df[["review_text", "sentiment"]])


if __name__ == "__main__":
    main()
