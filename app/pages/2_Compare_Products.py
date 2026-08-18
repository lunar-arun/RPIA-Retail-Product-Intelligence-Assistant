import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.metrics import compare_products, price_trend, sentiment_trend
from src.data.repository import list_products
from src.ui_support import get_data

st.set_page_config(page_title="Compare Products", page_icon="\u2696\uFE0F", layout="wide")
st.title("\u2696\uFE0F Compare Products")
st.caption("Pick two or more products to compare pricing, ratings, and sentiment side by side.")

df = get_data()

selected = st.multiselect(
    "Products to compare",
    options=list_products(df),
    default=list_products(df)[:2] if len(list_products(df)) >= 2 else list_products(df),
    max_selections=5,
)

if len(selected) < 2:
    st.info("Select at least two products to compare.", icon="\U0001F449")
    st.stop()

summary = compare_products(df, selected)
st.subheader("Head-to-head summary")
st.dataframe(
    summary.rename(columns={
        "product_name": "Product", "category": "Category", "brand": "Brand",
        "latest_price": "Latest price ($)", "avg_rating": "Avg. rating",
        "review_count": "Reviews", "positive_pct": "% positive", "negative_pct": "% negative",
    }),
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price trend")
    trend = price_trend(df, selected)
    fig = px.line(trend, x="date", y="price", color="product_name", markers=True)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Price ($)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Sentiment over time")
    s_trend = sentiment_trend(df, selected)
    fig2 = px.bar(
        s_trend, x="week", y="count", color="sentiment", barmode="stack",
        color_discrete_map={"POSITIVE": "#2E6F5E", "NEGATIVE": "#C1553B", "NEUTRAL": "#B8AE9C"},
    )
    fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)
