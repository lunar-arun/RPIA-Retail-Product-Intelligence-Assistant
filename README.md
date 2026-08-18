# Retail Product Intelligence Assistant (RPIA)

Retailers and product managers struggle to track competitive product
performance, customer sentiment, and pricing trends across marketplaces.
RPIA gives them a single dashboard to explore that — **entirely on local
sample data, with no external API keys or paid services required.**

## Features

- **Dashboard** — KPIs, sentiment distribution, category breakdown, and a
  product leaderboard ranked by rating.
- **Product Explorer** — filter reviews by category, product, sentiment, and
  price; view a product's price history over time.
- **Compare Products** — pick 2–5 products and compare price trends,
  sentiment trends, and ratings side by side.
- **Ask the Assistant** — ask a free-form question; the app retrieves the
  most relevant reviews (local TF-IDF search) and synthesizes an answer with
  cited sources — no LLM API call involved.

## Tech stack

- **UI:** Streamlit (multi-page app) + Plotly charts
- **Data:** pandas, local CSV files
- **Retrieval:** scikit-learn TF-IDF + cosine similarity (no model download)
- **Sentiment:** local lexicon-based scorer (no model download)
- **Answer generation:** local, rule-based synthesis over retrieved reviews

No `sentence-transformers`, `faiss`, `transformers`, or LLM SDK is required
to run the current version.

## Project structure

```
app/
  main.py                    # Dashboard (entry point)
  pages/
    1_Product_Explorer.py
    2_Compare_Products.py
    3_Ask_Assistant.py
src/
  config.py                  # paths & settings
  ui_support.py              # shared Streamlit helper
  data/repository.py         # single place that reads processed data
  analytics/metrics.py       # dashboard/comparison calculations
  nlp/sentiment_analyzer.py  # local sentiment tagging
  search/retrieval_service.py# local TF-IDF retrieval (RetrievalService interface)
  llm/answer_service.py      # local answer synthesis (AnswerGenerator interface)
scripts/
  generate_sample_data.py    # creates data/raw/reviews.csv (mock data)
  build_pipeline.py          # raw -> clean -> sentiment-tagged -> retrieval index
data/
  raw/reviews.csv                        # generated
  processed/clean_reviews.csv            # generated
  processed/reviews_with_sentiment.csv   # generated
models/
  retrieval_index.pkl        # generated (TF-IDF index)
notebooks/                   # exploratory notebooks mirroring the scripts
```

## Getting started

```bash
pip install -r requirements.txt

# Regenerate the local sample data + retrieval index (already included in
# this package, but re-run any time you want a fresh dataset):
python scripts/generate_sample_data.py
python scripts/build_pipeline.py

# Launch the app
streamlit run app/main.py
```

The dashboard opens at `http://localhost:8501`. Use the sidebar to move
between the four pages.

## How the mock data works

`scripts/generate_sample_data.py` creates a reproducible (seeded) dataset of
18 products across 4 categories (Smartphones, Laptops, Headphones,
Smartwatches), each with 5–9 reviews spread over the last 6 months with a
realistic price random-walk. This is enough to demonstrate genuine price and
sentiment trend charts, which the original single-review-per-product dataset
could not support.

`scripts/build_pipeline.py` then cleans that data, tags sentiment locally,
and builds the TF-IDF retrieval index — mirroring what a production
ingestion job (pulling from a real marketplace API) would eventually do.

## Adding a real API later

The architecture is deliberately layered so a future version can plug in
real data or a real LLM without a rewrite:

- **Real marketplace data:** point `src/data/repository.py` at a live data
  source instead of the CSV — nothing else in the app touches file paths
  directly.
- **Real retrieval (e.g. hosted embeddings/vector DB):** implement a new
  class that satisfies the `RetrievalService` interface in
  `src/search/retrieval_service.py`.
- **Real LLM-generated answers:** implement a new class that satisfies the
  `AnswerGenerator` interface in `src/llm/answer_service.py` (a
  `RemoteAnswerGenerator` stub and wiring point is documented there), read
  any API key from the environment (see `.env.example`), and flip
  `RPIA_USE_REMOTE_ANSWER_SERVICE=true`.

## Known limitations

- The dataset is synthetic/sample data, not live marketplace data.
- Sentiment analysis is a lightweight lexicon-based scorer, not a trained ML
  model — good enough for demo purposes but not production-grade accuracy.
- "Price trend" reflects the mock random-walk price history, not real
  historical pricing.
- No authentication/multi-user support — this is a single-user local app.
