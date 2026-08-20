import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.chart_picker import pick_chart
from src.analytics.metrics import compare_products, price_trend, sentiment_trend
from src.data.repository import list_categories, list_products
from src.ui_support import get_data

st.set_page_config(page_title="Compare Products", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Products")
st.caption("Pick a category and select two or more products to compare pricing, ratings, sentiment, and any custom columns side by side.")

df = get_data()

if df is None or df.empty:
    st.info("No dataset active yet. Please upload a CSV dataset on the home page to get started.", icon="📂")
    st.stop()

prod_col = "product_name" if "product_name" in df.columns else df.columns[0]

# ---------------------------------------------------------------------------
# 1. Category selector (FIRST dropdown as requested)
# ---------------------------------------------------------------------------
categories = list_categories(df)
if categories:
    selected_category = st.selectbox("Category", ["All"] + categories)
else:
    selected_category = "All"

# Filter available products based on selected category
available_products = list_products(df, category=selected_category)

if len(available_products) < 2 and selected_category != "All":
    st.warning(f"Category '{selected_category}' has fewer than 2 products. Select 'All' or another category.", icon="⚠️")
    st.stop()

if not available_products:
    st.warning("No products found in the dataset.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------------
# 2. Product selector (Filtered by category)
# ---------------------------------------------------------------------------
selected_products = st.multiselect(
    "Products to compare",
    options=available_products,
    default=available_products[:min(2, len(available_products))],
    max_selections=5,
)

if len(selected_products) < 2:
    st.info("Select at least two products to compare.", icon="👉")
    st.stop()

# ---------------------------------------------------------------------------
# 3. Dynamic Column selector — ALL columns in uploaded dataset
# ---------------------------------------------------------------------------
excluded_from_charts = {prod_col}
selectable_cols = [c for c in df.columns if c not in excluded_from_charts]

# Standard defaults if present, else first 3 available columns
default_cols = [c for c in ["price", "star_rating", "sentiment"] if c in selectable_cols]
if not default_cols:
    default_cols = selectable_cols[:min(3, len(selectable_cols))]

selected_cols = st.multiselect(
    "Columns to compare",
    options=selectable_cols,
    default=default_cols,
    help="Select any columns from your uploaded dataset. Numeric columns plot trend/mean, categorical plot bar charts.",
)

st.divider()

# ---------------------------------------------------------------------------
# 4. Dynamic per-column visual comparisons (Feature 3)
# ---------------------------------------------------------------------------
subset = df[df[prod_col].isin(selected_products)]

if selected_cols:
    st.subheader("Visual Comparisons")
    col_pairs = [selected_cols[i:i + 2] for i in range(0, len(selected_cols), 2)]
    for pair in col_pairs:
        cols = st.columns(len(pair))
        for ui_col, data_col in zip(cols, pair):
            with ui_col:
                st.subheader(data_col.replace("_", " ").title())
                fig = pick_chart(subset, data_col, group_by=prod_col)
                st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Head-to-head summary table
# ---------------------------------------------------------------------------
st.subheader("Head-to-head summary table")
summary = compare_products(df, selected_products)
if not summary.empty:
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# 6. Trends over time if review_date & price/sentiment are present
# ---------------------------------------------------------------------------
if "review_date" in df.columns and ("price" in df.columns or "sentiment" in df.columns):
    st.divider()
    st.subheader("Trends over time")
    c1, c2 = st.columns(2)
    if "price" in df.columns:
        with c1:
            st.markdown("**Price trend**")
            trend = price_trend(df, selected_products)
            if not trend.empty:
                fig = px.line(trend, x="date", y="price", color="product_name", markers=True)
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Price ($)", legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
    if "sentiment" in df.columns:
        with c2:
            st.markdown("**Sentiment over time**")
            s_trend = sentiment_trend(df, selected_products)
            if not s_trend.empty:
                fig2 = px.bar(
                    s_trend, x="week", y="count", color="sentiment", barmode="stack",
                    color_discrete_map={"POSITIVE": "#2E6F5E", "NEGATIVE": "#C1553B", "NEUTRAL": "#B8AE9C"},
                )
                fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
                st.plotly_chart(fig2, use_container_width=True)
