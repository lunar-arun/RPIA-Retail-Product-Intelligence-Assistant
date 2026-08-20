# Retail Product Intelligence Assistant (RPIA)

Retailers and product managers struggle to track competitive product
performance, customer sentiment, and pricing trends across marketplaces.
RPIA gives them a single dashboard to explore that — **entirely on local
data, with no external API keys or paid services required.**

> **No API keys. No paid services. No cloud calls — ever.**
> Every feature, including the optional Hugging Face sentiment model,
> runs 100 % locally on your machine.

## Features

- **Upload Dataset** — upload any CSV of product reviews; the app cleans
  it, lets you choose which columns to keep, tags sentiment, and rebuilds
  the search index automatically. Falls back to built-in sample data if
  nothing is uploaded.
- **Dashboard** — KPIs, sentiment distribution, category breakdown, and a
  product leaderboard ranked by rating.
- **Product Explorer** — filter reviews by category, product, sentiment,
  and price; view a product's price history over time.
- **Compare Products** — pick 2–5 products and compare any combination of
  columns (price, star rating, sentiment, or any custom columns from an
  uploaded dataset). Charts are auto-selected by data type: numeric →
  trend lines, categorical / sentiment → bar charts.
- **Ask the Assistant** — ask a free-form question; the app retrieves the
  most relevant reviews (local TF-IDF search) and synthesizes an answer
  with cited sources **plus an auto-generated chart** relevant to the
  retrieved data — no LLM API call involved.
- **Download Report** — one-click PDF export with section headings, KPI
  table, sentiment distribution chart, category overview, price trend,
  product leaderboard, and a sentiment-mover insights paragraph. Served
  as an in-memory download — no server-side file is written.

## Tech stack

| Layer | Library / Approach |
|---|---|
| UI | Streamlit (multi-page) + Plotly charts |
| Data | pandas, local CSV files |
| Cleaning | `src/data/cleaning.py` — shared by CLI pipeline and Upload page |
| Retrieval | scikit-learn TF-IDF + cosine similarity (no model download) |
| Sentiment (default) | Local lexicon-based scorer — no download, fully deterministic |
| Sentiment (optional) | `distilbert-base-uncased-finetuned-sst-2-english` via `transformers` — see below |
| Answer generation | Local, rule-based synthesis over retrieved reviews |
| PDF export | ReportLab (Platypus + native Drawing — no kaleido required) |

No LLM SDK, no hosted API, no remote vector DB is required to run the app.

## Project structure

```
app/
  main.py                     # Dashboard & CSV Upload (entry point)
  pages/
    1_Product_Explorer.py
    2_Compare_Products.py     # Dynamic column + chart-type selection
    3_Ask_Assistant.py        # Chart shown alongside each answer
    4_Download_Report.py      # One-click PDF export
src/
  config.py                   # paths, flags (USE_HF_SENTIMENT_MODEL etc.)
  ui_support.py               # shared Streamlit helpers
  data/
    cleaning.py               # NEW — clean_reviews(), validate_columns() (shared)
    repository.py             # UPDATED — save_reviews() added
  analytics/
    metrics.py                # UPDATED — sentiment_mover() added
    chart_picker.py           # NEW — pick_chart(), suggest_chart_for_query()
  nlp/sentiment_analyzer.py   # UPDATED — optional HF model branch
  search/retrieval_service.py # local TF-IDF retrieval (RetrievalService interface)
  llm/answer_service.py       # UPDATED — AnswerResult.suggested_chart field
  reports/
    pdf_report.py             # NEW — generate_pdf_report(df) -> bytes
scripts/
  generate_sample_data.py     # creates data/raw/reviews.csv (mock data)
  build_pipeline.py           # raw -> clean -> sentiment-tagged -> retrieval index
data/
  raw/reviews.csv                        # generated
  processed/clean_reviews.csv            # generated
  processed/reviews_with_sentiment.csv   # generated (also written by Upload page)
models/
  retrieval_index.pkl         # generated (TF-IDF index)
notebooks/                    # exploratory notebooks mirroring the scripts
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
between the six pages.

### Using your own data

1. Open **Upload Dataset** in the sidebar.
2. Upload a CSV with at least these columns:
   `product_name, category, brand, review_text, star_rating, price, review_date`
3. Preview, select columns, and click **Activate this dataset**.
4. All other pages will immediately use your data.

The built-in sample data is always the fallback if no file is uploaded.

## Optional: local Hugging Face sentiment model

By default the app uses a lightweight lexicon-based scorer — zero downloads,
fully deterministic, works offline immediately.

To opt in to the more accurate `distilbert-base-uncased-finetuned-sst-2-english`
model (~260 MB, CPU-friendly, runs fully offline after the first download):

```bash
# 1. Install CPU-only torch (much smaller than the GPU build):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install transformers:
pip install transformers>=4.40

# 3. Run the pipeline with the flag set:
RPIA_USE_HF_SENTIMENT_MODEL=true python scripts/build_pipeline.py
```

The model is cached in your local Hugging Face cache directory
(`~/.cache/huggingface/hub/` on Linux/macOS,
`%USERPROFILE%\.cache\huggingface\hub\` on Windows).
After the first download, **no network access is needed at inference time**.

The flag is read from the environment variable `RPIA_USE_HF_SENTIMENT_MODEL`.
The app itself always runs offline regardless of this setting — it is purely
a choice of which local algorithm tags the sentiment.

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

- The built-in dataset is synthetic/sample data, not live marketplace data.
  Upload your own CSV on the Upload Dataset page to use real data.
- Default sentiment analysis is a lightweight lexicon-based scorer — good
  enough for demo purposes. Enable the HF model flag for better accuracy
  on ambiguous reviews (see above).
- "Price trend" in the built-in data reflects a mock random-walk, not real
  historical pricing.
- No authentication/multi-user support — this is a single-user local app.
- PDF chart rendering uses ReportLab's native vector drawing (no browser
  required), so chart styling differs from the interactive Plotly charts in
  the UI.
- The HF sentiment model one-time download requires an internet connection
  (~260 MB). After that, inference is fully offline. The download is
  **never triggered automatically** — it only happens when you explicitly
  set `RPIA_USE_HF_SENTIMENT_MODEL=true`.
