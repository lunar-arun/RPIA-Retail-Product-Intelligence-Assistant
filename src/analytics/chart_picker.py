"""
Automatic chart-type selection for arbitrary DataFrame columns.

``pick_chart`` is the main entry point: given a DataFrame and a column name
it inspects the dtype and returns a ready-to-render Plotly figure.

``suggest_chart_for_query`` is the heuristic used by the Ask Assistant page:
it looks at the retrieved reviews and picks the most informative chart to
show alongside the text answer.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Colour map shared with the rest of the app
_SENTIMENT_COLOURS = {
    "POSITIVE": "#2E6F5E",
    "NEGATIVE": "#C1553B",
    "NEUTRAL":  "#B8AE9C",
}


def pick_chart(
    df: pd.DataFrame,
    column: str,
    group_by: str | None = "product_name",
) -> go.Figure:
    """Return an appropriate Plotly figure for *column* in *df*.

    Rules:
    - ``sentiment`` → stacked bar chart grouped by week.
    - Numeric column → line/bar trend chart over ``review_date``
      (one trace per value of *group_by* if that column exists).
    - Categorical column → grouped bar chart (value_counts per group).

    Args:
        df: The DataFrame to visualise.
        column: The column to chart.
        group_by: Optional grouping column (default ``"product_name"``).
                  Set to ``None`` to treat the whole DataFrame as one group.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart``.
    """
    if df.empty or column not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for '{column}'",
            xaxis_visible=False, yaxis_visible=False,
        )
        return fig

    col_data = df[column]

    # ── Sentiment column ────────────────────────────────────────────────────
    if column == "sentiment":
        return _sentiment_bar(df, group_by)

    # ── Numeric column ──────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(col_data):
        return _numeric_trend(df, column, group_by)

    # ── Categorical / object column ─────────────────────────────────────────
    return _categorical_bar(df, column, group_by)


# ---------------------------------------------------------------------------
# Heuristic for Ask Assistant
# ---------------------------------------------------------------------------

def suggest_chart_for_query(df: pd.DataFrame) -> go.Figure | None:
    """Pick the most informative chart for a set of retrieved reviews.

    Priority:
    1. Sentiment distribution (always available when ``sentiment`` column exists).
    2. Price trend if a ``price`` column and ``review_date`` are present.

    Returns ``None`` if *df* is empty or neither column exists.
    """
    if df.empty:
        return None

    if "sentiment" in df.columns:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.bar(
            counts,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map=_SENTIMENT_COLOURS,
            title="Sentiment of retrieved reviews",
            labels={"count": "Reviews", "sentiment": "Sentiment"},
        )
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        return fig

    if "price" in df.columns and "review_date" in df.columns:
        trend = df.sort_values("review_date")[["review_date", "price"]].copy()
        fig = px.line(
            trend, x="review_date", y="price",
            title="Price trend of retrieved reviews",
            labels={"review_date": "Date", "price": "Price ($)"},
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        return fig

    return None


# ---------------------------------------------------------------------------
# Internal chart builders
# ---------------------------------------------------------------------------

def _sentiment_bar(df: pd.DataFrame, group_by: str | None) -> go.Figure:
    """Stacked bar of sentiment counts, optionally grouped."""
    has_date = "review_date" in df.columns
    has_group = group_by and group_by in df.columns

    if has_date:
        plot_df = df.copy()
        plot_df["week"] = pd.to_datetime(plot_df["review_date"]).dt.to_period("W").dt.start_time
        agg = (
            plot_df.groupby(["week", "sentiment"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            agg, x="week", y="count", color="sentiment",
            barmode="stack",
            color_discrete_map=_SENTIMENT_COLOURS,
            title="Sentiment over time",
            labels={"week": "Week", "count": "Reviews", "sentiment": "Sentiment"},
        )
    else:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.bar(
            counts, x="sentiment", y="count", color="sentiment",
            color_discrete_map=_SENTIMENT_COLOURS,
            title="Sentiment distribution",
            labels={"count": "Reviews", "sentiment": "Sentiment"},
        )

    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend_title_text="")
    return fig


def _numeric_trend(df: pd.DataFrame, column: str, group_by: str | None) -> go.Figure:
    """Line chart of a numeric column over time, grouped if possible."""
    has_date = "review_date" in df.columns
    has_group = group_by and group_by in df.columns

    if has_date:
        plot_df = df.sort_values("review_date").copy()
        if has_group:
            fig = px.line(
                plot_df, x="review_date", y=column, color=group_by,
                markers=True,
                title=f"{column} trend",
                labels={"review_date": "Date", column: column, group_by: group_by},
            )
        else:
            fig = px.line(
                plot_df, x="review_date", y=column,
                markers=True,
                title=f"{column} trend",
                labels={"review_date": "Date", column: column},
            )
    else:
        # No date column — fall back to a bar chart of mean values per group
        if has_group:
            agg = df.groupby(group_by)[column].mean().reset_index()
            fig = px.bar(
                agg, x=group_by, y=column,
                title=f"Average {column} by {group_by}",
                labels={group_by: group_by, column: f"Avg {column}"},
            )
        else:
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")

    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend_title_text="")
    return fig


def _categorical_bar(df: pd.DataFrame, column: str, group_by: str | None) -> go.Figure:
    """Grouped bar chart of value counts for a categorical column."""
    has_group = group_by and group_by in df.columns

    if has_group:
        agg = (
            df.groupby([group_by, column])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            agg, x=group_by, y="count", color=column,
            barmode="group",
            title=f"{column} breakdown by {group_by}",
            labels={group_by: group_by, "count": "Count", column: column},
        )
    else:
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(
            counts, x=column, y="count",
            title=f"{column} distribution",
            labels={column: column, "count": "Count"},
        )

    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), legend_title_text="")
    return fig
