import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.config import RETRIEVAL_INDEX_PATH
from src.llm.answer_service import get_answer_generator
from src.search.retrieval_service import TfidfRetrievalService
from src.ui_support import get_data

st.set_page_config(page_title="Ask the Assistant", page_icon="\U0001F4AC", layout="wide")
st.title("\U0001F4AC Ask the Assistant")
st.caption(
    "Ask a free-form question about the reviews. Answers are generated locally by "
    "retrieving the most relevant reviews and summarizing them -- no external API call."
)

df = get_data()


@st.cache_resource(show_spinner=False)
def _load_retrieval_service() -> TfidfRetrievalService:
    service = TfidfRetrievalService()
    if not service.load(RETRIEVAL_INDEX_PATH):
        # Index wasn't pre-built (e.g. build_pipeline.py not run yet) -- build on the fly.
        service.build(df)
    return service


retrieval_service = _load_retrieval_service()
answer_generator = get_answer_generator()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input(
    "What do you want to know?",
    placeholder="e.g. What are people saying about battery life?",
)
top_k = st.slider("Number of reviews to consider", min_value=3, max_value=15, value=5)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Searching reviews..."):
        retrieved = retrieval_service.search(query, top_k=top_k)
        result = answer_generator.generate(query, retrieved)
    st.session_state.chat_history.append((query, result))

if not st.session_state.chat_history:
    st.info(
        "Try asking things like *\"How is the battery life on smartphones?\"* or "
        "*\"Are people happy with laptop build quality?\"*",
        icon="\U0001F4A1",
    )

for asked_query, result in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(asked_query)
    with st.chat_message("assistant"):
        st.markdown(result.summary)
        if not result.sources.empty:
            with st.expander(f"Sources ({len(result.sources)} reviews)"):
                st.dataframe(
                    result.sources[
                        ["product_name", "review_text", "sentiment", "star_rating", "relevance"]
                    ].rename(columns={
                        "product_name": "Product", "review_text": "Review",
                        "sentiment": "Sentiment", "star_rating": "Stars", "relevance": "Match score",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
