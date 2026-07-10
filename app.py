"""Keptra — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from keptra.ingest.audio import transcribe
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
    audio_file = st.file_uploader(
        "Voice note (.mp3 / .wav / .m4a)", type=["mp3", "wav", "m4a"]
    )
    if audio_file is not None:
        suffix = Path(audio_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Transcribing locally (first run loads Whisper)…"):
            result = transcribe(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        st.success(
            f"Transcribed **{audio_file.name}** "
            f"({result['duration']:.0f}s of audio, fully offline)"
        )
        for seg in result["segments"]:
            st.markdown(f"`[{seg['start']}–{seg['end']}]` {seg['text']}")

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
