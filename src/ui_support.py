"""Small shared helper used by every Streamlit page."""

from __future__ import annotations

import streamlit as st

from src.data.repository import load_reviews


def get_data():
    """Load review data, or stop the current page with setup instructions."""
    try:
        return load_reviews()
    except FileNotFoundError as exc:
        st.error("No local data found yet.")
        st.markdown(
            "This app runs entirely on local sample data -- no API keys needed. "
            "Generate it once with:"
        )
        st.code(
            "python scripts/generate_sample_data.py\n"
            "python scripts/build_pipeline.py",
            language="bash",
        )
        with st.expander("Details"):
            st.text(str(exc))
        st.stop()


def bootstrap_path() -> None:
    """Ensure the repo root is importable. Call once at the top of each page."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
