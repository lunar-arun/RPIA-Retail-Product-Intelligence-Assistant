import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.reports.pdf_report import generate_pdf_report
from src.ui_support import get_data

st.set_page_config(page_title="Download Report", page_icon="📄", layout="wide")
st.title("📄 Download Report")
st.caption(
    "Generate a PDF summary of the active dataset — sentiment distribution, "
    "category overview, price trends, product leaderboard, and sentiment movers. "
    "No external API call; the report is built entirely on local data."
)

df = get_data()

if df is None or df.empty:
    st.info("No dataset active yet. Please upload a CSV dataset on the home page or Upload page.", icon="📂")
    st.stop()

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
prod_col = "product_name" if "product_name" in df.columns else df.columns[0]
col1.metric("Products", df[prod_col].nunique())
col2.metric("Reviews", f"{len(df):,}")
col3.metric(
    "Avg. rating",
    f"{df['star_rating'].mean():.2f}" if "star_rating" in df.columns else "N/A",
)
pos_pct = (df["sentiment"] == "POSITIVE").mean() * 100 if "sentiment" in df.columns else 0
col4.metric("Positive sentiment", f"{pos_pct:.0f}%" if "sentiment" in df.columns else "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Report scope filter (optional)
# ---------------------------------------------------------------------------
st.subheader("Report scope (optional)")
all_categories = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []

if all_categories:
    selected_cats = st.multiselect(
        "Limit report to these categories (leave empty = all categories)",
        options=all_categories,
        default=[],
    )
    report_df = df[df["category"].isin(selected_cats)] if selected_cats else df
else:
    report_df = df
    st.info("No `category` column found — report will include all rows.", icon="ℹ️")

st.write(f"Report will cover **{len(report_df):,} rows** across **{report_df[prod_col].nunique()} products**.")

st.divider()

# ---------------------------------------------------------------------------
# Generate & download
# ---------------------------------------------------------------------------
st.subheader("Generate PDF")
st.write(
    "Click the button below to build the report in memory. "
    "The download will start immediately — no file is saved on the server."
)

if st.button("🔄 Generate PDF report", type="primary"):
    if report_df.empty:
        st.warning("The filtered dataset is empty. Widen your category filter.", icon="⚠️")
    else:
        with st.spinner("Building PDF report… this may take a few seconds."):
            try:
                pdf_bytes = generate_pdf_report(report_df)
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
                st.stop()

        st.success("✅ Report ready!", icon="✅")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="rpia_report.pdf",
            mime="application/pdf",
            type="primary",
        )
