import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.metrics import category_overview, sentiment_distribution
from src.data.repository import list_categories, product_summary
from src.ui_support import get_data

st.set_page_config(
    page_title="Retail Product Intelligence Assistant",
    page_icon="\U0001F6CD\uFE0F",
    layout="wide",
)

st.title("\U0001F6CD\uFE0F Retail Product Intelligence Assistant")
st.caption(
    "Track competitive product performance, customer sentiment, and pricing "
    "trends -- powered entirely by local sample data, no external API required."
)

df = get_data()

# --- Filters -----------------------------------------------------------
categories = ["All"] + list_categories(df)
selected_category = st.selectbox("Category", categories)
scoped_df = df if selected_category == "All" else df[df["category"] == selected_category]

# --- KPI row -------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Products tracked", scoped_df["product_name"].nunique())
col2.metric("Reviews analyzed", len(scoped_df))
col3.metric("Avg. star rating", f"{scoped_df['star_rating'].mean():.2f}")
pos_pct = (scoped_df["sentiment"] == "POSITIVE").mean() * 100
col4.metric("Positive sentiment", f"{pos_pct:.0f}%")

st.divider()

# --- Charts ---------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Sentiment distribution")
    dist = sentiment_distribution(scoped_df)
    color_map = {"POSITIVE": "#2E6F5E", "NEGATIVE": "#C1553B", "NEUTRAL": "#B8AE9C"}
    fig = px.pie(
        dist, names="sentiment", values="count", hole=0.55,
        color="sentiment", color_discrete_map=color_map,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Category overview")
    overview = category_overview(scoped_df if selected_category == "All" else df)
    fig2 = px.bar(
        overview, x="category", y="avg_rating",
        color="avg_price", color_continuous_scale="Teal",
        labels={"avg_rating": "Avg. rating", "avg_price": "Avg. price ($)"},
    )
    fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Product leaderboard ---------------------------------------------------
st.subheader("Product leaderboard")
st.caption("Ranked by average customer rating within the selected category.")
summary = product_summary(scoped_df)
st.dataframe(
    summary.rename(columns={
        "product_name": "Product", "category": "Category", "brand": "Brand",
        "latest_price": "Latest price ($)", "avg_rating": "Avg. rating",
        "review_count": "Reviews", "positive_pct": "% positive", "negative_pct": "% negative",
    }),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Use the sidebar to explore individual products, compare products head-to-head, "
    "or ask the assistant a free-form question about the reviews.",
    icon="\U0001F449",
)
