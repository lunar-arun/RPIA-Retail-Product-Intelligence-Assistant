import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.metrics import category_overview, sentiment_distribution
from src.data.repository import list_categories, product_summary
from src.ui_support import get_data, render_upload_widget

st.set_page_config(
    page_title="Retail Product Intelligence Assistant",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Product Intelligence Assistant")
st.caption(
    "Track competitive product performance, customer sentiment, and pricing "
    "trends -- powered entirely by local data, no external API required."
)

df = get_data()

# ---------------------------------------------------------------------------
# Opening View: If no dataset is uploaded yet, prompt CSV Upload immediately
# ---------------------------------------------------------------------------
if df is None or df.empty:
    st.subheader("Welcome! Get started by uploading your dataset")
    render_upload_widget()
    st.stop()

# ---------------------------------------------------------------------------
# Active Dataset Header & Option to Change Dataset
# ---------------------------------------------------------------------------
with st.expander("📂 Active Dataset & Upload New CSV", expanded=False):
    st.write(f"Currently active: **{len(df):,} rows × {len(df.columns)} columns**")

    if st.button("🗑️ Clear active dataset"):
        if "active_df" in st.session_state:
            del st.session_state["active_df"]
        from src.data.repository import USER_REVIEWS_PATH
        if USER_REVIEWS_PATH.exists():
            USER_REVIEWS_PATH.unlink()
        st.rerun()

    st.divider()
    render_upload_widget()

st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
categories = list_categories(df)
if categories:
    selected_category = st.selectbox("Category", ["All"] + categories)
    scoped_df = df if selected_category == "All" else df[df["category"] == selected_category]
else:
    selected_category = "All"
    scoped_df = df

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

prod_col = "product_name" if "product_name" in scoped_df.columns else scoped_df.columns[0]
col1.metric("Products tracked", scoped_df[prod_col].nunique())
col2.metric("Reviews analyzed", len(scoped_df))

if "star_rating" in scoped_df.columns:
    col3.metric("Avg. star rating", f"{scoped_df['star_rating'].mean():.2f}")
else:
    col3.metric("Avg. star rating", "N/A")

if "sentiment" in scoped_df.columns:
    pos_pct = (scoped_df["sentiment"] == "POSITIVE").mean() * 100
    col4.metric("Positive sentiment", f"{pos_pct:.0f}%")
else:
    col4.metric("Positive sentiment", "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Sentiment distribution")
    if "sentiment" in scoped_df.columns:
        dist = sentiment_distribution(scoped_df)
        color_map = {"POSITIVE": "#2E6F5E", "NEGATIVE": "#C1553B", "NEUTRAL": "#B8AE9C"}
        fig = px.pie(
            dist, names="sentiment", values="count", hole=0.55,
            color="sentiment", color_discrete_map=color_map,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No `sentiment` column found in active dataset.")

with right:
    st.subheader("Category overview")
    if "category" in scoped_df.columns:
        overview = category_overview(scoped_df if selected_category == "All" else df)
        color_col = "avg_price" if "avg_price" in overview.columns else None
        fig2 = px.bar(
            overview, x="category", y="avg_rating" if "avg_rating" in overview.columns else "reviews",
            color=color_col, color_continuous_scale="Teal",
            labels={"avg_rating": "Avg. rating", "avg_price": "Avg. price ($)", "reviews": "Reviews"},
        )
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No `category` column found in active dataset.")

st.divider()

# ---------------------------------------------------------------------------
# Product leaderboard
# ---------------------------------------------------------------------------
if "product_name" in scoped_df.columns:
    st.subheader("Product leaderboard")
    summary = product_summary(scoped_df)
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "Use the sidebar to explore individual products, compare products head-to-head, "
    "ask the assistant questions, or export a PDF report.",
    icon="👉",
)
