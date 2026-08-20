"""
Local semantic-ish retrieval over review text using scikit-learn's TF-IDF.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RETRIEVAL_INDEX_PATH


class RetrievalService(ABC):
    """Interface for "find reviews relevant to this query"."""

    @abstractmethod
    def build(self, df: pd.DataFrame) -> None:
        """Fit the index over a DataFrame."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return the top_k most relevant rows, with a `relevance` column."""


class TfidfRetrievalService(RetrievalService):
    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._metadata: pd.DataFrame | None = None
        self._text_col: str = "review_text"

    def _find_text_column(self, df: pd.DataFrame) -> str:
        for col in ["review_text", "review", "customer_review", "user_review", "comment", "text", "description", "feedback", "summary"]:
            if col in df.columns:
                return col
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                return col
        return df.columns[0]

    def _build_search_corpus(self, df: pd.DataFrame) -> pd.Series:
        """Combine metadata (category, brand, product_name) and text into a single searchable string per row."""
        cols_to_combine = [c for c in ["category", "brand", "product_name", "product", self._text_col] if c in df.columns]
        if not cols_to_combine:
            return df[self._text_col].astype(str)

        combined = df[cols_to_combine[0]].fillna("").astype(str)
        for col in cols_to_combine[1:]:
            combined = combined + " " + df[col].fillna("").astype(str)
        return combined

    def build(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self._metadata = df.reset_index(drop=True)
        self._text_col = self._find_text_column(self._metadata)
        corpus = self._build_search_corpus(self._metadata)

        # Unigram + bigram TF-IDF with sublinear scaling
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def save(self, path=None) -> None:
        path = path or RETRIEVAL_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                    "metadata": self._metadata,
                    "text_col": self._text_col,
                },
                f,
            )

    def load(self, path=None) -> bool:
        path = path or RETRIEVAL_INDEX_PATH
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._vectorizer = state["vectorizer"]
            self._matrix = state["matrix"]
            self._metadata = state["metadata"]
            self._text_col = state.get("text_col", "review_text")
            return True
        except Exception:
            return False

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self._vectorizer is None or self._matrix is None or self._metadata is None:
            raise RuntimeError("Retrieval index not built. Please load a dataset first.")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        results = self._metadata.copy()
        results["relevance"] = scores.round(3)
        sorted_results = results.sort_values("relevance", ascending=False)

        # Filter for positive relevance, or fallback to top_k if no positive match
        positive_matches = sorted_results[sorted_results["relevance"] > 0]
        if not positive_matches.empty:
            return positive_matches.head(top_k).reset_index(drop=True)

        return sorted_results.head(top_k).reset_index(drop=True)
