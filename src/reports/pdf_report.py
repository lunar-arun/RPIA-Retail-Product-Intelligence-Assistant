"""
PDF report generator for RPIA.

``generate_pdf_report(df) -> bytes`` builds a self-contained PDF using
ReportLab's Platypus (high-level layout) and Drawing (vector charts).

No external API calls, no file written to disk — the bytes are returned
directly for Streamlit's ``st.download_button``.

Sections
--------
1. Title / KPI summary
2. Sentiment distribution bar chart
3. Category overview table
4. Price trend line chart (top products by review count)
5. Top-5 / Bottom-5 products table
6. Sentiment movers table + insights paragraph
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd

# ReportLab imports — reportlab must be in requirements.txt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, PolyLine
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.lib.colors import HexColor

# ---------------------------------------------------------------------------
# Colour palette (matches the Streamlit app)
# ---------------------------------------------------------------------------
COLOUR_POSITIVE = HexColor("#2E6F5E")
COLOUR_NEGATIVE = HexColor("#C1553B")
COLOUR_NEUTRAL  = HexColor("#B8AE9C")
COLOUR_ACCENT   = HexColor("#1F4E79")
COLOUR_LIGHT    = HexColor("#F5F5F5")
COLOUR_BORDER   = HexColor("#CCCCCC")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _styles():
    s = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RpiaTitle",
        parent=s["Title"],
        fontSize=22,
        textColor=COLOUR_ACCENT,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "RpiaH1",
        parent=s["Heading1"],
        fontSize=14,
        textColor=COLOUR_ACCENT,
        spaceBefore=14,
        spaceAfter=4,
        borderPad=0,
    )
    h2 = ParagraphStyle(
        "RpiaH2",
        parent=s["Heading2"],
        fontSize=11,
        textColor=COLOUR_ACCENT,
        spaceBefore=10,
        spaceAfter=2,
    )
    body = ParagraphStyle(
        "RpiaBody",
        parent=s["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=4,
    )
    caption = ParagraphStyle(
        "RpiaCaption",
        parent=s["Normal"],
        fontSize=7,
        textColor=colors.grey,
        leading=10,
    )
    return title_style, h1, h2, body, caption


def _df_to_table(df: pd.DataFrame, col_widths: list[float] | None = None) -> Table:
    """Convert a DataFrame to a styled ReportLab Table."""
    header = list(df.columns)
    rows = [header] + [list(r) for r in df.itertuples(index=False, name=None)]
    # Stringify all cells
    rows = [[str(cell) for cell in row] for row in rows]

    available = PAGE_W - 2 * MARGIN
    if col_widths is None:
        col_widths = [available / len(header)] * len(header)

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  COLOUR_ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 7),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [COLOUR_LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, COLOUR_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Chart builders (pure ReportLab — no kaleido, no PNG conversion)
# ---------------------------------------------------------------------------

def _sentiment_bar_chart(df: pd.DataFrame) -> Drawing:
    """Vertical bar chart of sentiment counts."""
    counts = df["sentiment"].value_counts()
    labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    values = [int(counts.get(lbl, 0)) for lbl in labels]
    chart_colours = [COLOUR_POSITIVE, COLOUR_NEGATIVE, COLOUR_NEUTRAL]

    d = Drawing(380, 180)
    bc = VerticalBarChart()
    bc.x = 40
    bc.y = 20
    bc.width = 320
    bc.height = 140
    bc.data = [values]
    bc.strokeColor = None
    bc.bars[0].fillColor = COLOUR_POSITIVE  # will be overridden per bar below

    # Colour each bar individually
    for i, c in enumerate(chart_colours):
        bc.bars[(0, i)].fillColor = c

    bc.categoryAxis.categoryNames = labels
    bc.categoryAxis.labels.fontSize = 8
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(values) * 1.2 if max(values) > 0 else 10
    bc.valueAxis.labels.fontSize = 7

    d.add(bc)
    return d


def _price_trend_chart(df: pd.DataFrame, max_products: int = 5) -> Drawing | None:
    """Line plot of price over time for the most-reviewed products."""
    if "price" not in df.columns or "review_date" not in df.columns:
        return None

    top_products = (
        df.groupby("product_name")["review_text"]
        .count()
        .nlargest(max_products)
        .index.tolist()
    )
    if not top_products:
        return None

    subset = df[df["product_name"].isin(top_products)].copy()
    subset["review_date"] = pd.to_datetime(subset["review_date"])
    subset = subset.sort_values("review_date")

    # Convert dates to ordinal floats for LinePlot
    min_date = subset["review_date"].min()

    line_colours = [
        HexColor("#1F4E79"), HexColor("#2E6F5E"), HexColor("#C1553B"),
        HexColor("#E68C3A"), HexColor("#7B4F9E"),
    ]

    d = Drawing(380, 180)
    lp = LinePlot()
    lp.x = 50
    lp.y = 20
    lp.width = 310
    lp.height = 140

    series_data = []
    for product in top_products:
        pdata = subset[subset["product_name"] == product]
        pts = [
            (float((row["review_date"] - min_date).days), float(row["price"]))
            for _, row in pdata.iterrows()
        ]
        if pts:
            series_data.append(pts)

    if not series_data:
        return None

    lp.data = series_data
    all_prices = [p for pts in series_data for _, p in pts]
    lp.yValueAxis.valueMin = max(0, min(all_prices) * 0.9)
    lp.yValueAxis.valueMax = max(all_prices) * 1.1
    lp.yValueAxis.labels.fontSize = 7
    lp.xValueAxis.labels.fontSize = 7
    lp.xValueAxis.labelTextFormat = lambda v: ""  # suppress crowded date labels

    for i, c in enumerate(line_colours[: len(series_data)]):
        lp.lines[i].strokeColor = c
        lp.lines[i].strokeWidth = 1.5

    d.add(lp)

    # Legend
    legend_y = 165
    for i, product in enumerate(top_products[: len(series_data)]):
        x_offset = 50 + i * 72
        if x_offset + 70 > 380:
            break
        r = Rect(x_offset, legend_y, 10, 6, fillColor=line_colours[i], strokeColor=None)
        d.add(r)
        d.add(String(x_offset + 13, legend_y, product[:12], fontSize=6, fillColor=colors.black))

    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf_report(df: pd.DataFrame) -> bytes:
    """Build a PDF report for the active dataset and return it as bytes.

    No files are written to disk.  Pass the returned bytes directly to
    ``st.download_button``.
    """
    from src.analytics.metrics import category_overview, sentiment_distribution, sentiment_mover
    from src.data.repository import product_summary

    buf = io.BytesIO()

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )
    frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    title_style, h1, h2, body, caption = _styles()
    story: list[Any] = []

    # ── Title ────────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph("Retail Product Intelligence Assistant", title_style))
    story.append(Paragraph("Automated Analysis Report", h2))
    story.append(Paragraph(f"Generated: {now} &nbsp;|&nbsp; Rows: {len(df):,} &nbsp;|&nbsp; Products: {df['product_name'].nunique()}", caption))
    story.append(Spacer(1, 0.4 * cm))

    # ── KPI Summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("Key Performance Indicators", h1))

    pos_pct = (df["sentiment"] == "POSITIVE").mean() * 100 if "sentiment" in df.columns else 0
    neg_pct = (df["sentiment"] == "NEGATIVE").mean() * 100 if "sentiment" in df.columns else 0
    avg_rating = df["star_rating"].mean() if "star_rating" in df.columns else 0
    avg_price  = df["price"].mean() if "price" in df.columns else 0

    kpi_data = [
        ["Metric", "Value"],
        ["Total Reviews", f"{len(df):,}"],
        ["Unique Products", f"{df['product_name'].nunique()}"],
        ["Avg. Star Rating", f"{avg_rating:.2f} / 5"],
        ["Avg. Price", f"${avg_price:.2f}"],
        ["Positive Sentiment", f"{pos_pct:.1f}%"],
        ["Negative Sentiment", f"{neg_pct:.1f}%"],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[8 * cm, 6 * cm])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), COLOUR_ACCENT),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLOUR_LIGHT, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.3, COLOUR_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── Sentiment Distribution ───────────────────────────────────────────────
    if "sentiment" in df.columns:
        story.append(Paragraph("Sentiment Distribution", h1))
        chart = _sentiment_bar_chart(df)
        story.append(KeepTogether([chart, Spacer(1, 0.2 * cm)]))

    # ── Category Overview ────────────────────────────────────────────────────
    if "category" in df.columns:
        story.append(Paragraph("Category Overview", h1))
        cat_df = category_overview(df).round(2)
        cat_df.columns = ["Category", "Products", "Reviews", "Avg Price ($)", "Avg Rating"]
        story.append(_df_to_table(cat_df))
        story.append(Spacer(1, 0.3 * cm))

    # ── Price Trend ──────────────────────────────────────────────────────────
    if "price" in df.columns and "review_date" in df.columns:
        story.append(Paragraph("Price Trend (Top Products by Review Count)", h1))
        price_chart = _price_trend_chart(df)
        if price_chart:
            story.append(KeepTogether([price_chart, Spacer(1, 0.2 * cm)]))
            story.append(Paragraph("Each line represents a product's price over the review period.", caption))

    # ── Top / Bottom Products ────────────────────────────────────────────────
    story.append(Paragraph("Product Leaderboard", h1))
    summary = product_summary(df)

    top5 = summary.head(5)[["product_name", "avg_rating", "positive_pct", "review_count"]].copy()
    top5.columns = ["Product", "Avg Rating", "% Positive", "Reviews"]

    story.append(Paragraph("Top 5 Products (by Average Rating)", h2))
    story.append(_df_to_table(top5))
    story.append(Spacer(1, 0.3 * cm))

    bottom5 = summary.tail(5)[["product_name", "avg_rating", "positive_pct", "review_count"]].copy()
    bottom5.columns = ["Product", "Avg Rating", "% Positive", "Reviews"]
    story.append(Paragraph("Bottom 5 Products (by Average Rating)", h2))
    story.append(_df_to_table(bottom5))
    story.append(Spacer(1, 0.3 * cm))

    # ── Sentiment Movers ─────────────────────────────────────────────────────
    movers = sentiment_mover(df, top_n=5)
    if not movers.empty:
        story.append(Paragraph("Biggest Sentiment Movers (Week-over-Week)", h1))
        movers_display = movers.copy()
        movers_display.columns = ["Product", "Latest % Pos", "Prev % Pos", "Δ %"]
        story.append(_df_to_table(movers_display))
        story.append(Spacer(1, 0.3 * cm))

        # Insights paragraph
        top_gainer = movers[movers["delta"] == movers["delta"].max()].iloc[0]
        top_loser  = movers[movers["delta"] == movers["delta"].min()].iloc[0]
        insight_text = (
            f"<b>Insights:</b> "
            f"The biggest positive mover this period is <b>{top_gainer['product_name']}</b> "
            f"(+{top_gainer['delta']:.1f}% positive sentiment week-over-week). "
        )
        if top_loser["delta"] < 0:
            insight_text += (
                f"The biggest negative mover is <b>{top_loser['product_name']}</b> "
                f"({top_loser['delta']:.1f}% week-over-week). "
                "Consider reviewing recent reviews for that product to identify concerns."
            )
        else:
            insight_text += "All tracked products show stable or improving sentiment this period."

        story.append(Paragraph(insight_text, body))

    # ── Footer note ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "This report was generated entirely offline using local data — "
        "no external API calls or paid services were used.",
        caption,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
