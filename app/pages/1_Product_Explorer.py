import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.metrics import price_trend
from src.data.repository import filter_reviews, list_categories, list_products
from src.ui_support import get_data

st.set_page_config(page_title="Product Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Product Explorer")
st.caption("Browse products, filter by category / sentiment / price, and drill into individual reviews.")

df = get_data()

if df is None or df.empty:
    st.info("No dataset active yet. Please upload a CSV dataset on the home page or Upload page.", icon="📂")
    st.stop()

has_price = "price" in df.columns
has_sentiment = "sentiment" in df.columns
has_category = "category" in df.columns

with st.sidebar:
    st.header("Filters")
    categories = list_categories(df) if has_category else []
    category = st.selectbox("Category", ["All"] + categories) if categories else "All"
    product = st.selectbox("Product", ["All"] + list_products(df, category))
    sentiment = st.selectbox("Sentiment", ["All", "POSITIVE", "NEUTRAL", "NEGATIVE"]) if has_sentiment else "All"

    if has_price and not df["price"].dropna().empty:
        price_min, price_max = float(df["price"].min()), float(df["price"].max())
        if price_min < price_max:
            price_range = st.slider(
                "Price range ($)", min_value=round(price_min, 2), max_value=round(price_max, 2),
                value=(round(price_min, 2), round(price_max, 2)),
            )
        else:
            price_range = (price_min, price_max)
    else:
        price_range = (None, None)

filtered = filter_reviews(
    df, category=category, product=product, sentiment=sentiment,
    min_price=price_range[0], max_price=price_range[1],
)

st.subheader(f"{len(filtered)} matching review(s)")

if filtered.empty:
    st.warning("No reviews match these filters. Try widening your criteria.")
else:
    if product != "All" and has_price and "review_date" in df.columns:
        st.subheader(f"Price history: {product}")
        trend = price_trend(df, [product])
        if not trend.empty:
            fig = px.line(trend, x="date", y="price", markers=True)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )
