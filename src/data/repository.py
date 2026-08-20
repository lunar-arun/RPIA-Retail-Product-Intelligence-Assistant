"""
Data repository: the single place that knows how review data is stored.

Everything else in the app (analytics, retrieval, UI) asks this module for
data and never touches a CSV path directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR, SENTIMENT_REVIEWS_PATH

USER_REVIEWS_PATH = PROCESSED_DIR / "user_uploaded_reviews.csv"


def load_reviews() -> pd.DataFrame | None:
    """Load the active review dataset.

    Priority:
    1. Active DataFrame in ``st.session_state["active_df"]``.
    2. User-uploaded CSV at ``USER_REVIEWS_PATH``.
    3. Returns ``None`` if no user dataset has been uploaded yet (no auto default).
    """
    if "active_df" in st.session_state and st.session_state["active_df"] is not None:
        return st.session_state["active_df"]

    if USER_REVIEWS_PATH.exists():
        try:
            df = pd.read_csv(USER_REVIEWS_PATH)
            if "review_date" in df.columns:
                df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
            st.session_state["active_df"] = df
            return df
        except Exception:
            pass

    return None


def save_reviews(df: pd.DataFrame) -> None:
    """Persist *df* as the active dataset and update in-memory state.

    Called by the Upload Dataset flow after the user confirms their upload.
    The DataFrame is saved to ``USER_REVIEWS_PATH`` and ``SENTIMENT_REVIEWS_PATH``.
    """
    df = df.copy()
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(USER_REVIEWS_PATH, index=False)
    df.to_csv(SENTIMENT_REVIEWS_PATH, index=False)

    st.session_state["active_df"] = df


def load_sample_data() -> pd.DataFrame | None:
    """Explicit fallback loader if the user chooses to test with mock sample data."""
    if SENTIMENT_REVIEWS_PATH.exists():
        try:
            df = pd.read_csv(SENTIMENT_REVIEWS_PATH)
            if "review_date" in df.columns:
                df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
            save_reviews(df)
            return df
        except Exception:
            pass
    return None


def list_categories(df: pd.DataFrame) -> list[str]:
    if "category" not in df.columns:
        return []
    return sorted(df["category"].dropna().unique().tolist())


def list_products(df: pd.DataFrame, category: str | None = None) -> list[str]:
    if "product_name" not in df.columns:
        return []
    if category and category != "All" and "category" in df.columns:
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
    if category and category != "All" and "category" in result.columns:
        result = result[result["category"] == category]
    if product and product != "All" and "product_name" in result.columns:
        result = result[result["product_name"] == product]
    if sentiment and sentiment != "All" and "sentiment" in result.columns:
        result = result[result["sentiment"] == sentiment]
    if min_price is not None and "price" in result.columns:
        result = result[result["price"] >= min_price]
    if max_price is not None and "price" in result.columns:
        result = result[result["price"] <= max_price]
    return result


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product with available aggregations. Safe for any uploaded CSV."""
    if df.empty or "product_name" not in df.columns:
        return pd.DataFrame()

    has_date = "review_date" in df.columns
    if has_date:
        df = df.copy()
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        sort_col = "review_date"
    else:
        sort_col = df.columns[0]

    latest_cols = ["product_name"]
    for c in ["category", "brand", "price"]:
        if c in df.columns:
            latest_cols.append(c)

    latest = (
        df.sort_values(sort_col)
        .groupby("product_name")
        .tail(1)[latest_cols]
    )
    if "price" in latest.columns:
        latest = latest.rename(columns={"price": "latest_price"})

    agg_dict = {}
    if "star_rating" in df.columns:
        agg_dict["avg_rating"] = ("star_rating", "mean")
    if "review_text" in df.columns:
        agg_dict["review_count"] = ("review_text", "count")
    else:
        agg_dict["review_count"] = (df.columns[0], "count")

    if "sentiment" in df.columns:
        agg_dict["positive_pct"] = ("sentiment", lambda s: (s == "POSITIVE").mean() * 100)
        agg_dict["negative_pct"] = ("sentiment", lambda s: (s == "NEGATIVE").mean() * 100)

    agg = df.groupby("product_name").agg(**agg_dict).reset_index()

    summary = latest.merge(agg, on="product_name")

    for col in summary.columns:
        if pd.api.types.is_float_dtype(summary[col]):
            summary[col] = summary[col].round(2)

    sort_col = "avg_rating" if "avg_rating" in summary.columns else "product_name"
    ascending = False if sort_col == "avg_rating" else True
    return summary.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
