"""
Local semantic-ish retrieval over review text.

The original implementation used `sentence-transformers` + FAISS. Both work
fine, but sentence-transformers downloads a ~90MB model from the Hugging
Face Hub the first time it runs -- an external dependency this project is
meant to avoid, and unnecessary given the dataset size here.

This module keeps the same *shape* of the pipeline (fit a vectorizer once,
persist it, search by cosine similarity) using scikit-learn's TF-IDF, which
is pure local computation with no downloads. `RetrievalService` is an
abstract interface so a heavier embedding backend (or a hosted vector-search
API) can be swapped in later without touching any calling code.
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
        """Fit the index over a DataFrame with a `review_text` column."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return the top_k most relevant rows, with a `relevance` column."""


class TfidfRetrievalService(RetrievalService):
    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._metadata: pd.DataFrame | None = None

    def build(self, df: pd.DataFrame) -> None:
        self._metadata = df.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._matrix = self._vectorizer.fit_transform(self._metadata["review_text"].astype(str))

    def save(self, path=None) -> None:
        path = path or RETRIEVAL_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._matrix, "metadata": self._metadata},
                f,
            )

    def load(self, path=None) -> bool:
        path = path or RETRIEVAL_INDEX_PATH
        if not path.exists():
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._vectorizer = state["vectorizer"]
        self._matrix = state["matrix"]
        self._metadata = state["metadata"]
        return True

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Retrieval index not built/loaded. Call build() or load() first.")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        results = self._metadata.copy()
        results["relevance"] = scores
        results = results.sort_values("relevance", ascending=False).head(top_k)
        return results[results["relevance"] > 0].reset_index(drop=True)
