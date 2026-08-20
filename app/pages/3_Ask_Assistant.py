import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.llm.answer_service import get_answer_generator
from src.search.retrieval_service import TfidfRetrievalService
from src.ui_support import get_data

st.set_page_config(page_title="Ask the Assistant", page_icon="💬", layout="wide")
st.title("💬 Ask the Assistant")
st.caption(
    "Ask a free-form question about the reviews. Answers are generated locally by "
    "retrieving the most relevant reviews and summarizing them — no external API call."
)

df = get_data()

if df is None or df.empty:
    st.info("No dataset active yet. Please upload a CSV dataset on the home page.", icon="📂")
    st.stop()


# Build retrieval service directly on active DataFrame in memory
@st.cache_resource(show_spinner=False)
def _get_retrieval_service(_dataset) -> TfidfRetrievalService:
    service = TfidfRetrievalService()
    service.build(_dataset)
    return service


try:
    retrieval_service = _get_retrieval_service(df)
except Exception:
    retrieval_service = TfidfRetrievalService()
    retrieval_service.build(df)

answer_generator = get_answer_generator()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input(
    "What do you want to know?",
    placeholder="e.g. What are people saying about battery life?",
)
top_k = st.slider("Number of reviews to consider", min_value=3, max_value=15, value=5)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Searching reviews…"):
        try:
            retrieved = retrieval_service.search(query, top_k=top_k)
            result = answer_generator.generate(query, retrieved)
            st.session_state.chat_history.append((query, result))
        except Exception as exc:
            st.error(f"Search failed: {exc}")

if not st.session_state.chat_history:
    st.info(
        "Try asking things like *\"How is the battery life on smartphones?\"* or "
        "*\"Are people happy with build quality?\"*",
        icon="💡",
    )

for asked_query, result in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(asked_query)
    with st.chat_message("assistant"):
        st.markdown(result.summary)

        if result.suggested_chart is not None:
            st.plotly_chart(result.suggested_chart, use_container_width=True)

        if not result.sources.empty:
            with st.expander(f"Sources ({len(result.sources)} reviews)"):
                st.dataframe(
                    result.sources,
                    use_container_width=True,
                    hide_index=True,
                )
