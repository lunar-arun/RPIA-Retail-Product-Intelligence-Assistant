"""
Data repository: the single place that knows how review data is stored.

Everything else in the app (analytics, retrieval, UI) asks this module for
data and never touches a CSV path directly. That means swapping the storage
backend later (a real database, a marketplace API, etc.) only requires
changes here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import SENTIMENT_REVIEWS_PATH


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    """Load the processed, sentiment-tagged review dataset.

    Raises FileNotFoundError with a clear message if the local pipeline
    hasn't been run yet, so the UI can show a helpful setup instruction
    instead of a raw traceback.
    """
    if not SENTIMENT_REVIEWS_PATH.exists():
        raise FileNotFoundError(
            f"Processed review data not found at {SENTIMENT_REVIEWS_PATH}.\n"
            "Run `python scripts/generate_sample_data.py` then "
            "`python scripts/build_pipeline.py` to create it."
        )
    df = pd.read_csv(SENTIMENT_REVIEWS_PATH)
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


def list_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["category"].dropna().unique().tolist())


def list_products(df: pd.DataFrame, category: str | None = None) -> list[str]:
    if category and category != "All":
        df = df[df["category"] == category]
    return sorted(df["product_name"].dropna().unique().tolist())


def filter_reviews(
    df: pd.DataFrame,
    category: str | None = None,
    product: str | None = None,
    sentiment: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> pd.DataFrame:
    result = df
    if category and category != "All":
        result = result[result["category"] == category]
    if product and product != "All":
        result = result[result["product_name"] == product]
    if sentiment and sentiment != "All":
        result = result[result["sentiment"] == sentiment]
    if min_price is not None:
        result = result[result["price"] >= min_price]
    if max_price is not None:
        result = result[result["price"] <= max_price]
    return result


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product: latest price, avg rating, sentiment mix, review count."""
    latest = (
        df.sort_values("review_date")
        .groupby("product_name")
        .tail(1)[["product_name", "category", "brand", "price"]]
        .rename(columns={"price": "latest_price"})
    )

    agg = df.groupby("product_name").agg(
        avg_rating=("star_rating", "mean"),
        review_count=("review_text", "count"),
        positive_pct=("sentiment", lambda s: (s == "POSITIVE").mean() * 100),
        negative_pct=("sentiment", lambda s: (s == "NEGATIVE").mean() * 100),
    ).reset_index()

    summary = latest.merge(agg, on="product_name")
    summary["avg_rating"] = summary["avg_rating"].round(2)
    summary["positive_pct"] = summary["positive_pct"].round(1)
    summary["negative_pct"] = summary["negative_pct"].round(1)
    return summary.sort_values("avg_rating", ascending=False).reset_index(drop=True)
