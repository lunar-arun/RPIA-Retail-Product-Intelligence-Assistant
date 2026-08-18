"""
Local sentiment analysis.

The original version of this module called `transformers.pipeline(
"sentiment-analysis")`, which downloads a multi-hundred-MB model from the
Hugging Face Hub on first use. That makes the project depend on network
access and an external service just to run a demo, and is unnecessary given
this app's scale.

This version uses a small, transparent lexicon-based scorer instead:
- No download, no external calls, fully deterministic.
- Easy to swap out later for a heavier model (transformers, a hosted NLP
  API, etc.) without touching any other part of the app -- callers only
  ever see `tag_sentiment(df) -> df` from this module.
"""

from __future__ import annotations

import re

import pandas as pd

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


def tag_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """Add `sentiment` and `sentiment_score` columns to a copy of `df`.

    Falls back to the review's star rating when the text itself is too
    short/ambiguous to score (e.g. no lexicon hits), so results stay
    sensible even for terse reviews.
    """
    df = df.copy()
    labels = []
    scores = []
    for _, row in df.iterrows():
        label, score = _score_text(str(row[text_column]))
        if label == "NEUTRAL" and "star_rating" in df.columns:
            rating = row["star_rating"]
            if rating >= 4:
                label, score = "POSITIVE", 0.6
            elif rating <= 2:
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
