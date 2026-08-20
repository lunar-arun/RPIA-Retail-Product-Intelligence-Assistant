"""
Answer generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AnswerResult:
    summary: str
    positive_count: int
    negative_count: int
    neutral_count: int
    sources: pd.DataFrame = field(repr=False)
    suggested_chart: Any = field(default=None, repr=False)


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_reviews: pd.DataFrame) -> AnswerResult:
        """Produce an answer grounded in the given retrieved reviews."""


class LocalAnswerGenerator(AnswerGenerator):
    """Rule-based synthesis: no external calls, fully offline."""

    def generate(self, query: str, retrieved_reviews: pd.DataFrame) -> AnswerResult:
        suggested_chart = None
        try:
            from src.analytics.chart_picker import suggest_chart_for_query
            suggested_chart = suggest_chart_for_query(retrieved_reviews)
        except Exception:
            pass

        if retrieved_reviews.empty:
            return AnswerResult(
                summary=(
                    "I couldn't find any reviews closely related to that question. "
                    "Try rephrasing, or ask about a specific product, feature, or category."
                ),
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                sources=retrieved_reviews,
                suggested_chart=suggested_chart,
            )

        # Handle sentiment counting safely
        if "sentiment" in retrieved_reviews.columns:
            pos = int((retrieved_reviews["sentiment"] == "POSITIVE").sum())
            neg = int((retrieved_reviews["sentiment"] == "NEGATIVE").sum())
            neu = int((retrieved_reviews["sentiment"] == "NEUTRAL").sum())
        elif "star_rating" in retrieved_reviews.columns:
            pos = int((retrieved_reviews["star_rating"] > 3).sum())
            neg = int((retrieved_reviews["star_rating"] < 3).sum())
            neu = int((retrieved_reviews["star_rating"] == 3).sum())
        else:
            pos, neg, neu = 0, 0, len(retrieved_reviews)

        total = len(retrieved_reviews)

        # Handle product name column safely
        prod_col = "product_name" if "product_name" in retrieved_reviews.columns else (
            "product" if "product" in retrieved_reviews.columns else None
        )

        if prod_col:
            products = retrieved_reviews[prod_col].unique().tolist()
            product_phrase = (
                f"across {', '.join(str(p) for p in products[:3])}"
                if len(products) > 1
                else f"for {products[0]}"
            )
        else:
            product_phrase = "in the retrieved dataset"

        if pos > neg:
            lean = "generally positive"
        elif neg > pos:
            lean = "generally negative"
        else:
            lean = "mixed"

        summary_lines = [
            f"Based on {total} matching review(s) {product_phrase}, sentiment is **{lean}** "
            f"({pos} positive / {neg} negative / {neu} neutral)."
        ]

        # Top relevant review snippet
        top = retrieved_reviews.iloc[0]
        text_col = next(
            (c for c in ["review_text", "review", "comment", "text", "description"] if c in top.index),
            None,
        )
        if text_col and pd.notna(top[text_col]):
            prod_info = f"{top[prod_col]}, " if prod_col and pd.notna(top[prod_col]) else ""
            sent_info = f"{str(top['sentiment']).lower()}" if "sentiment" in top.index and pd.notna(top['sentiment']) else "mention"
            summary_lines.append(
                f"Most relevant mention ({prod_info}{sent_info}): "
                f"\u201c{top[text_col]}\u201d"
            )

        return AnswerResult(
            summary="\n\n".join(summary_lines),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            sources=retrieved_reviews,
            suggested_chart=suggested_chart,
        )


def get_answer_generator() -> AnswerGenerator:
    """Factory so callers never need to know which implementation is active."""
    from src.config import USE_REMOTE_ANSWER_SERVICE

    if USE_REMOTE_ANSWER_SERVICE:
        raise NotImplementedError(
            "Remote answer generation is not configured in this version. "
            "Implement RemoteAnswerGenerator in src/llm/answer_service.py "
            "and wire it in here when a real API is introduced."
        )
    return LocalAnswerGenerator()
