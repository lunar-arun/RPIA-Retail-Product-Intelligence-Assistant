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


def sentiment_mover(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Compute week-over-week positive-sentiment delta per product.

    Returns a DataFrame with columns:
        product_name, latest_pos_pct, prev_pos_pct, delta
    sorted by |delta| descending, limited to *top_n* rows.

    Used by the PDF report to highlight the biggest sentiment movers.
    """
    if df.empty or "sentiment" not in df.columns or "review_date" not in df.columns:
        return pd.DataFrame(columns=["product_name", "latest_pos_pct", "prev_pos_pct", "delta"])

    tmp = df.copy()
    tmp["week"] = tmp["review_date"].dt.to_period("W")

    weekly = (
        tmp.groupby(["product_name", "week"])
        .apply(lambda g: (g["sentiment"] == "POSITIVE").mean() * 100, include_groups=False)
        .reset_index(name="pos_pct")
    )

    results = []
    for product, grp in weekly.groupby("product_name"):
        grp = grp.sort_values("week")
        if len(grp) < 2:
            continue
        latest = float(grp.iloc[-1]["pos_pct"])
        prev = float(grp.iloc[-2]["pos_pct"])
        results.append(
            {"product_name": product, "latest_pos_pct": round(latest, 1),
             "prev_pos_pct": round(prev, 1), "delta": round(latest - prev, 1)}
        )

    if not results:
        return pd.DataFrame(columns=["product_name", "latest_pos_pct", "prev_pos_pct", "delta"])

    out = pd.DataFrame(results)
    out["abs_delta"] = out["delta"].abs()
    return out.sort_values("abs_delta", ascending=False).drop(columns="abs_delta").head(top_n).reset_index(drop=True)

