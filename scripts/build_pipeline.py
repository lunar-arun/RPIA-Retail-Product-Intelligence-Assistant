"""
Runs the full local data pipeline:

    data/raw/reviews.csv
        -> clean                (src.nlp / basic pandas cleaning)
        -> sentiment-tag         (src.nlp.sentiment_analyzer)
        -> data/processed/reviews_with_sentiment.csv
        -> build retrieval index (src.search.retrieval_service)
        -> models/retrieval_index.pkl

No external API calls are made anywhere in this pipeline.

Run with:  python scripts/build_pipeline.py
(Run scripts/generate_sample_data.py first if data/raw/reviews.csv does not exist.)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd

from src.config import RAW_REVIEWS_PATH, CLEAN_REVIEWS_PATH, SENTIMENT_REVIEWS_PATH
from src.nlp.sentiment_analyzer import tag_sentiment
from src.search.retrieval_service import TfidfRetrievalService


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Basic, dependency-free cleaning of the raw review data."""
    df = df.dropna(subset=["review_text", "product_name"]).copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["product_name", "review_text", "review_date"])
    return df.reset_index(drop=True)


def main() -> None:
    if not RAW_REVIEWS_PATH.exists():
        raise SystemExit(
            f"Raw data not found at {RAW_REVIEWS_PATH}.\n"
            "Run `python scripts/generate_sample_data.py` first."
        )

    print(f"Loading raw reviews from {RAW_REVIEWS_PATH} ...")
    raw_df = pd.read_csv(RAW_REVIEWS_PATH)

    print("Cleaning ...")
    clean_df = clean_reviews(raw_df)
    CLEAN_REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_REVIEWS_PATH, index=False)
    print(f"  saved {len(clean_df)} rows -> {CLEAN_REVIEWS_PATH}")

    print("Tagging sentiment (local lexicon-based analyzer) ...")
    sentiment_df = tag_sentiment(clean_df)
    sentiment_df.to_csv(SENTIMENT_REVIEWS_PATH, index=False)
    print(f"  saved -> {SENTIMENT_REVIEWS_PATH}")

    print("Building local TF-IDF retrieval index ...")
    retrieval_service = TfidfRetrievalService()
    retrieval_service.build(sentiment_df)
    retrieval_service.save()
    print("  retrieval index saved.")

    print("\nPipeline complete. The app can now run fully offline.")


if __name__ == "__main__":
    main()
