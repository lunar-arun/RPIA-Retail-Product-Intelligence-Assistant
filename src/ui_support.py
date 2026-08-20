"""Small shared helper used by every Streamlit page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import RETRIEVAL_INDEX_PATH
from src.data.cleaning import REQUIRED_COLUMNS, clean_reviews, validate_columns
from src.data.repository import load_reviews, load_sample_data, save_reviews
from src.nlp.sentiment_analyzer import tag_sentiment
from src.search.retrieval_service import TfidfRetrievalService


def get_data() -> pd.DataFrame | None:
    """Load review data, or return None if no dataset is active."""
    return load_reviews()


def render_upload_widget() -> pd.DataFrame | None:
    """Render the upload CSV UI workflow directly on any page."""
    st.info("📂 Please upload a CSV dataset to get started.", icon="📂")

    col_left, col_right = st.columns([3, 1])
    with col_right:
        if st.button("🧪 Test with sample data"):
            with st.spinner("Loading built-in sample data…"):
                df = load_sample_data()
                if df is not None:
                    st.success("Loaded sample dataset!")
                    st.rerun()

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Recommended columns: product_name, category, brand, review_text, star_rating, price, review_date",
    )

    if uploaded is None:
        return None

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        return None

    st.write(f"**{len(raw_df):,} rows × {len(raw_df.columns)} columns** loaded.")

    missing = validate_columns(raw_df)
    if missing:
        st.warning(
            f"⚠️ Missing recommended columns: `{'`, `'.join(missing)}`.\n"
            "You can still proceed, but some specific charts (like price trend or rating mix) may be disabled if those columns are missing.",
            icon="⚠️",
        )

    clean_df = clean_reviews(raw_df)
    st.write(f"Cleaned dataset: **{len(clean_df):,} rows**.")

    all_cols = list(clean_df.columns)
    selected_cols = st.multiselect(
        "Columns to include in active dataset",
        options=all_cols,
        default=all_cols,
    )

    if not selected_cols:
        st.warning("Please select at least one column.")
        return None

    if st.button("✅ Activate Uploaded Dataset", type="primary"):
        selected_df = clean_df[selected_cols].copy()
        with st.spinner("Processing sentiment and index…"):
            if "review_text" in selected_df.columns:
                try:
                    selected_df = tag_sentiment(selected_df)
                except Exception:
                    pass
                try:
                    svc = TfidfRetrievalService()
                    svc.build(selected_df)
                    svc.save(RETRIEVAL_INDEX_PATH)
                except Exception:
                    pass

            save_reviews(selected_df)
            st.success("✅ Dataset activated successfully!")
            st.rerun()

    return None


def bootstrap_path() -> None:
    """Ensure the repo root is importable."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
