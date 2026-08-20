"""
Data cleaning helpers shared by the CLI pipeline and the Upload Dataset UI page.

Extracting ``clean_reviews`` here (from scripts/build_pipeline.py) means both
callers use the exact same logic — no duplication, no drift.
"""

from __future__ import annotations

import pandas as pd

# Columns the rest of the application expects to exist.
# The UI upload page will warn (but not crash) if any of these are absent.
REQUIRED_COLUMNS: list[str] = [
    "product_name",
    "category",
    "brand",
    "review_text",
    "star_rating",
    "price",
    "review_date",
]


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of required column names that are missing from *df*.

    An empty list means the DataFrame is valid.
    Column comparison is case-insensitive (matching what ``clean_reviews``
    does to column names).
    """
    lowered = {c.lower().strip() for c in df.columns}
    return [col for col in REQUIRED_COLUMNS if col not in lowered]


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Basic, dependency-free cleaning of raw review data.

    Steps:
    - Drop rows with missing ``review_text`` or ``product_name``.
    - Normalise column names to lowercase + strip whitespace.
    - Strip leading/trailing whitespace from text columns.
    - Remove empty ``review_text`` rows.
    - Drop exact duplicates on (product_name, review_text, review_date).
    - Reset index.

    This function is intentionally minimal — it produces a clean DataFrame
    ready for sentiment tagging; no domain logic is applied here.
    """
    df = df.copy()
    # Normalise column names first so downstream checks are reliable
    df.columns = [c.lower().strip() for c in df.columns]

    # Drop rows missing the two essential text columns
    required_for_clean = [c for c in ["review_text", "product_name"] if c in df.columns]
    if required_for_clean:
        df = df.dropna(subset=required_for_clean)

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].astype(str).str.strip()
        df = df[df["review_text"] != ""]

    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].astype(str).str.strip()

    # Deduplicate
    dedup_cols = [c for c in ["product_name", "review_text", "review_date"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)

    return df.reset_index(drop=True)
