"""Keptra — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from keptra.index.chunk import chunk_segments, chunk_text
from keptra.index.store import add_chunks
from keptra.ingest.audio import transcribe
from keptra.ingest.documents import extract_text
from keptra.query.answer import ask_llm

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}

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
    uploaded = st.file_uploader(
        "Voice note or document (.mp3 / .wav / .m4a / .pdf / .txt / .md)",
        type=["mp3", "wav", "m4a", "pdf", "txt", "md"],
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        created_at = datetime.now().isoformat(timespec="seconds")
        if suffix in AUDIO_EXTENSIONS:
            with st.spinner("Transcribing locally (first run loads Whisper)…"):
                result = transcribe(tmp_path)
            st.success(
                f"Transcribed **{uploaded.name}** "
                f"({result['duration']:.0f}s of audio, fully offline)"
            )
            for seg in result["segments"]:
                st.markdown(f"`[{seg['start']}–{seg['end']}]` {seg['text']}")
            chunks = chunk_segments(
                result["segments"],
                {
                    "source_type": "audio",
                    "source_name": uploaded.name,
                    "created_at": created_at,
                },
            )
        else:
            with st.spinner("Extracting text locally…"):
                items = extract_text(tmp_path)
            st.success(f"Extracted **{uploaded.name}** ({len(items)} part(s))")
            chunks = []
            for item in items:
                label = f"Page {item['page']}" if item["page"] else "Full text"
                with st.expander(label, expanded=True):
                    st.text(item["text"])
                chunks.extend(
                    chunk_text(
                        item["text"],
                        {
                            "source_type": "document",
                            "source_name": uploaded.name,
                            "page": item["page"],
                            "created_at": created_at,
                        },
                    )
                )
        with st.spinner("Indexing locally (chunk → embed → store)…"):
            indexed = add_chunks(chunks)
        st.toast(f"Indexed {indexed} chunk(s) from {uploaded.name} 🧠", icon="✅")
        Path(tmp_path).unlink(missing_ok=True)

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
