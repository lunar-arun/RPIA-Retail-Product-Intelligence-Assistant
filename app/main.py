import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.rag_pipeline import query_rag_system

import streamlit as st

st.set_page_config(page_title="Retail Product Intelligence Assistant")

st.title("🛍️ Retail Product Intelligence Assistant")
st.markdown("Ask questions based on real customer reviews.")

query = st.text_input("What do you want to know?")

if query:
    with st.spinner("Thinking..."):
        answer = query_rag_system(query)
        st.success(answer)
