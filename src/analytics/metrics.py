"""Analytics helpers used by the dashboard and comparison pages."""

from __future__ import annotations

import pandas as pd


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    return counts


def category_overview(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")
        .agg(
            products=("product_name", "nunique"),
            reviews=("review_text", "count"),
            avg_price=("price", "mean"),
            avg_rating=("star_rating", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("reviews", ascending=False)
    )


def price_trend(df: pd.DataFrame, product_names: list[str]) -> pd.DataFrame:
    """Daily price points for the given products, for a trend line chart."""
    subset = df[df["product_name"].isin(product_names)]
    trend = (
        subset.sort_values("review_date")[["product_name", "review_date", "price"]]
        .rename(columns={"review_date": "date"})
    )
    return trend


def sentiment_trend(df: pd.DataFrame, product_names: list[str]) -> pd.DataFrame:
    """Weekly positive vs negative review counts for the given products."""
    subset = df[df["product_name"].isin(product_names)].copy()
    subset["week"] = subset["review_date"].dt.to_period("W").dt.start_time
    trend = (
        subset.groupby(["week", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    return trend


def compare_products(df: pd.DataFrame, product_names: list[str]) -> pd.DataFrame:
    from src.data.repository import product_summary

    summary = product_summary(df)
    return summary[summary["product_name"].isin(product_names)].reset_index(drop=True)
