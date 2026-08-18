import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import streamlit as st

from src.analytics.metrics import price_trend
from src.data.repository import filter_reviews, list_categories, list_products
from src.ui_support import get_data

st.set_page_config(page_title="Product Explorer", page_icon="\U0001F50D", layout="wide")
st.title("\U0001F50D Product Explorer")
st.caption("Browse products, filter by category / sentiment / price, and drill into individual reviews.")

df = get_data()

with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Category", ["All"] + list_categories(df))
    product = st.selectbox("Product", ["All"] + list_products(df, category))
    sentiment = st.selectbox("Sentiment", ["All", "POSITIVE", "NEUTRAL", "NEGATIVE"])
    price_min, price_max = float(df["price"].min()), float(df["price"].max())
    price_range = st.slider(
        "Price range ($)", min_value=round(price_min, 2), max_value=round(price_max, 2),
        value=(round(price_min, 2), round(price_max, 2)),
    )

filtered = filter_reviews(
    df, category=category, product=product, sentiment=sentiment,
    min_price=price_range[0], max_price=price_range[1],
)

st.subheader(f"{len(filtered)} matching review(s)")

if filtered.empty:
    st.warning("No reviews match these filters. Try widening your criteria.")
else:
    if product != "All":
        st.subheader(f"Price history: {product}")
        trend = price_trend(df, [product])
        fig = px.line(trend, x="date", y="price", markers=True)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    display_cols = ["product_name", "category", "brand", "review_text", "star_rating", "price", "sentiment", "review_date"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "product_name": "Product", "category": "Category", "brand": "Brand",
            "review_text": "Review", "star_rating": "Stars", "price": "Price ($)",
            "sentiment": "Sentiment", "review_date": "Date",
        }).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
