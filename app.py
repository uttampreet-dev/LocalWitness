"""Keptra — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import streamlit as st

from keptra.query.answer import ask_llm

st.set_page_config(page_title="Keptra", page_icon="🧠", layout="wide")

with st.sidebar:
    st.title("🧠 Keptra")
    st.caption("Everything you kept, recalled — entirely on your device.")
    st.divider()
    st.markdown("**100% local** — no data ever leaves this machine.")

upload_tab, library_tab, ask_tab, metrics_tab = st.tabs(
    ["Upload", "Library", "Ask", "Metrics"]
)

with upload_tab:
    st.subheader("Upload")
    st.info("Coming soon: drop in voice notes, documents, and images.")

with library_tab:
    st.subheader("Library")
    st.info("Coming soon: everything you've indexed, in one place.")

with ask_tab:
    st.subheader("Ask")
    st.caption("Raw local-LLM passthrough for now — proves on-device inference works.")
    question = st.text_input("Your question", placeholder="Say hello to your local LLM…")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking locally…"):
            st.markdown(ask_llm(question))

with metrics_tab:
    st.subheader("Metrics")
    st.info("Coming soon: model sizes, tokens/sec, ms/query — all measured live.")
