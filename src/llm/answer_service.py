"""
Answer generation.

`AnswerGenerator` is the abstraction the rest of the app talks to. Today the
only implementation is `LocalAnswerGenerator`, which synthesizes an answer
from retrieved reviews using simple, transparent rules -- no external LLM
API call, no API key required.

Future API integration
-----------------------
To add a real LLM later (e.g. the Anthropic or OpenAI API), implement a new
class that satisfies the same interface, for example:

    class RemoteAnswerGenerator(AnswerGenerator):
        def __init__(self, api_key: str):
            ...
        def generate(self, query, retrieved_reviews) -> AnswerResult:
            # call the hosted API with `retrieved_reviews` as context
            ...

and select it in `get_answer_generator()` based on `src.config`
(`USE_REMOTE_ANSWER_SERVICE`) -- no other file needs to change, since every
caller only depends on the `AnswerGenerator` interface below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnswerResult:
    summary: str
    positive_count: int
    negative_count: int
    neutral_count: int
    sources: pd.DataFrame = field(repr=False)


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_reviews: pd.DataFrame) -> AnswerResult:
        """Produce an answer grounded in the given retrieved reviews."""


class LocalAnswerGenerator(AnswerGenerator):
    """Rule-based synthesis: no external calls, fully offline."""

    def generate(self, query: str, retrieved_reviews: pd.DataFrame) -> AnswerResult:
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
            )

        pos = int((retrieved_reviews["sentiment"] == "POSITIVE").sum())
        neg = int((retrieved_reviews["sentiment"] == "NEGATIVE").sum())
        neu = int((retrieved_reviews["sentiment"] == "NEUTRAL").sum())
        total = len(retrieved_reviews)
        products = retrieved_reviews["product_name"].unique().tolist()

        if pos > neg:
            lean = "generally positive"
        elif neg > pos:
            lean = "generally negative"
        else:
            lean = "mixed"

        product_phrase = (
            f"across {', '.join(products[:3])}"
            if len(products) > 1
            else f"for {products[0]}"
        )

        summary_lines = [
            f"Based on {total} matching review(s) {product_phrase}, sentiment is **{lean}** "
            f"({pos} positive / {neg} negative / {neu} neutral)."
        ]

        top = retrieved_reviews.iloc[0]
        summary_lines.append(
            f"Most relevant mention ({top['product_name']}, {top['sentiment'].lower()}): "
            f"\u201c{top['review_text']}\u201d"
        )

        return AnswerResult(
            summary="\n\n".join(summary_lines),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            sources=retrieved_reviews,
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
