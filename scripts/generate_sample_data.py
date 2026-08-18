"""
Generates the RAW sample dataset used by RPIA.

This is MOCK / SYNTHETIC data. It exists so the app is fully usable without
any external API, marketplace scraper, or paid service. The generator is
seeded, so output is reproducible.

Design notes for future API integration:
    The rest of the application only ever reads
    `data/raw/reviews.csv`  (via src/data/repository.py).
    To plug in a real marketplace API later, replace this script's output
    with a real ingestion job that writes a CSV/DataFrame with the same
    columns:
        product_name, category, brand, review_text, star_rating,
        price, review_date

Run with:  python scripts/generate_sample_data.py
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _stable_seed(text: str) -> int:
    """Deterministic seed from a string, independent of PYTHONHASHSEED.

    Python's built-in hash() is randomized per-process for strings, so it
    can't be used to seed a reproducible RNG across runs/machines.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)

RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "reviews.csv"

CATEGORIES = {
    "Smartphones": ["Nova", "Pulse", "Zenith", "Orbit", "Halo", "Vertex"],
    "Laptops": ["AeroBook", "TitanPro", "SlimEdge", "PowerLine", "CoreMax"],
    "Headphones": ["EchoSound", "BassWave", "AirTone", "PureBeat"],
    "Smartwatches": ["PulseFit", "TimeSync", "ActiveTrack"],
}

POSITIVE_SNIPPETS = [
    "Battery life easily lasts a full day of heavy use.",
    "Build quality feels premium for the price.",
    "Performance is smooth even with multiple apps open.",
    "Camera/display quality exceeded my expectations.",
    "Customer support resolved my issue quickly.",
    "Great value for money compared to competitors.",
    "Setup was simple and worked right out of the box.",
    "Sound/screen quality is genuinely impressive.",
    "Shipping was fast and packaging was solid.",
    "Would definitely recommend this to a friend.",
]

NEGATIVE_SNIPPETS = [
    "Battery drains much faster than advertised.",
    "Started having issues within the first two weeks.",
    "Build quality feels cheap for this price point.",
    "Customer support was slow to respond.",
    "Performance lags noticeably under normal use.",
    "Arrived with a minor defect out of the box.",
    "Not worth the price compared to alternatives.",
    "Software updates introduced more bugs than fixes.",
    "Noticeably worse than the previous model.",
    "Packaging was damaged during shipping.",
]

NEUTRAL_SNIPPETS = [
    "It does the job, nothing special either way.",
    "Average product, matches what was described.",
    "Some features are great, others feel unfinished.",
    "Decent for the price but there are trade-offs.",
    "Works as expected, no major complaints.",
]


def _make_product_catalog() -> list[dict]:
    random.seed(RANDOM_SEED)
    products = []
    for category, brands in CATEGORIES.items():
        for brand in brands:
            base_price = {
                "Smartphones": random.uniform(250, 950),
                "Laptops": random.uniform(500, 1800),
                "Headphones": random.uniform(30, 300),
                "Smartwatches": random.uniform(80, 450),
            }[category]
            products.append(
                {
                    "product_name": f"{brand} {random.choice(['X', 'S', 'Pro', 'Lite', '2'])}",
                    "category": category,
                    "brand": brand,
                    "base_price": round(base_price, 2),
                }
            )
    return products


def _generate_reviews_for_product(product: dict, today: datetime) -> list[dict]:
    random.seed(_stable_seed(product["product_name"]))
    num_reviews = random.randint(5, 9)
    reviews = []

    # Slight price drift over time (mock "price trend" data)
    price = product["base_price"]

    for i in range(num_reviews):
        days_ago = random.randint(0, 180)
        review_date = today - timedelta(days=days_ago)

        sentiment_roll = random.random()
        if sentiment_roll < 0.55:
            snippet = random.choice(POSITIVE_SNIPPETS)
            star_rating = random.choice([4, 5, 5])
        elif sentiment_roll < 0.85:
            snippet = random.choice(NEGATIVE_SNIPPETS)
            star_rating = random.choice([1, 2, 2])
        else:
            snippet = random.choice(NEUTRAL_SNIPPETS)
            star_rating = 3

        # small random walk on price to simulate discounts/markups over time
        price = max(5.0, price * random.uniform(0.97, 1.03))

        reviews.append(
            {
                "product_name": product["product_name"],
                "category": product["category"],
                "brand": product["brand"],
                "review_text": snippet,
                "star_rating": star_rating,
                "price": round(price, 2),
                "review_date": review_date.strftime("%Y-%m-%d"),
            }
        )

    return reviews


def generate() -> pd.DataFrame:
    today = datetime(2026, 8, 17)
    products = _make_product_catalog()

    rows: list[dict] = []
    for product in products:
        rows.extend(_generate_reviews_for_product(product, today))

    df = pd.DataFrame(rows)
    df = df.sort_values(["product_name", "review_date"]).reset_index(drop=True)
    return df


def main() -> None:
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} mock reviews across {df['product_name'].nunique()} products.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
