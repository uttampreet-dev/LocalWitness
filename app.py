"""Keptra — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from keptra.index.chunk import chunk_segments, chunk_text
from keptra.index.store import (
    add_chunks,
    clear,
    count,
    delete_source,
    list_sources,
    query,
)
from keptra.metrics import MODEL_SPECS, get_metrics, latest
from keptra.ingest.audio import transcribe
from keptra.ingest.documents import extract_text
from keptra.ingest.images import caption
from keptra.query.answer import OLLAMA_ERRORS, answer_stream
from keptra.query.retrieve import cite, retrieve

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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
        "Voice note, document, or photo "
        "(.mp3 / .wav / .m4a / .pdf / .txt / .md / .jpg / .png)",
        type=["mp3", "wav", "m4a", "pdf", "txt", "md", "jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        created_at = datetime.now().isoformat(timespec="seconds")
        cache_key = f"{uploaded.name}:{uploaded.size}"
        if suffix in AUDIO_EXTENSIONS:
            transcript_cache = st.session_state.setdefault("transcript_cache", {})
            result = transcript_cache.get(cache_key)
            if result is None:
                with st.spinner("Transcribing locally (first run loads Whisper)…"):
                    result = transcribe(tmp_path)
                transcript_cache[cache_key] = result
            st.success(
                f"Transcribed **{uploaded.name}** "
                f"({result['duration']:.0f}s of audio, fully offline)"
            )
            for seg in result["segments"]:
                st.markdown(f"`[{seg['start']}–{seg['end']}]` {seg['text']}")
            transcript = "\n".join(
                f"[{seg['start']}-{seg['end']}] {seg['text']}"
                for seg in result["segments"]
            )
            redact_export = st.toggle(
                "Redact PII in transcript export",
                help="Replace names, emails, phones, and IDs with typed tags "
                "like [PERSON] — detected locally by Presidio.",
            )
            if redact_export:
                from keptra.privacy.redact import redact

                with st.spinner("Redacting locally (first run loads spaCy)…"):
                    transcript = redact(transcript)
                st.markdown(f"```\n{transcript}\n```")
            st.download_button(
                "⬇️ Export transcript (.txt)",
                transcript,
                file_name=f"{Path(uploaded.name).stem}_transcript.txt",
            )
            chunks = chunk_segments(
                result["segments"],
                {
                    "source_type": "audio",
                    "source_name": uploaded.name,
                    "created_at": created_at,
                },
            )
        elif suffix in IMAGE_EXTENSIONS:
            caption_cache = st.session_state.setdefault("caption_cache", {})
            described = caption_cache.get(cache_key)
            if described is None:
                with st.spinner("Captioning locally (Moondream)…"):
                    try:
                        described = caption(tmp_path)
                        caption_cache[cache_key] = described
                    except OLLAMA_ERRORS as exc:
                        st.error(
                            "Couldn't caption: is Ollama running and moondream "
                            f"pulled (`ollama pull moondream`)?\n\nDetails: {exc}"
                        )
            chunks = []
            if described:
                st.image(tmp_path, width=360)
                st.success(f"Captioned **{uploaded.name}** (fully offline)")
                st.markdown(f"> {described}")
                chunks = chunk_text(
                    described,
                    {
                        "source_type": "image",
                        "source_name": uploaded.name,
                        "created_at": created_at,
                    },
                )
                if st.button(
                    "🛡️ Export (privacy-safe)",
                    help="Save a copy to exports/ with detected people blurred (YOLOv8n, local).",
                ):
                    from keptra.privacy.blur import blur_people

                    exports_dir = Path("exports")
                    exports_dir.mkdir(exist_ok=True)
                    export_path = exports_dir / f"{Path(uploaded.name).stem}_blurred{suffix}"
                    with st.spinner("Detecting and blurring people locally…"):
                        n_blurred = blur_people(tmp_path, str(export_path))
                    st.success(
                        f"Exported **{export_path}** — {n_blurred} region(s) blurred."
                    )
                    st.image(str(export_path), width=360)
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
        replaced = indexed = 0
        indexed_keys = st.session_state.setdefault("indexed_keys", set())
        if chunks and cache_key not in indexed_keys:
            with st.spinner("Indexing locally (chunk → embed → store)…"):
                replaced = delete_source(uploaded.name)
                indexed = add_chunks(chunks)
            indexed_keys.add(cache_key)
        if replaced:
            st.toast(
                f"Re-indexed {uploaded.name}: {indexed} chunk(s) "
                f"(replaced {replaced} old)",
                icon="♻️",
            )
        elif indexed:
            st.toast(f"Indexed {indexed} chunk(s) from {uploaded.name} 🧠", icon="✅")
        Path(tmp_path).unlink(missing_ok=True)

    st.divider()
    if st.button(
        "🗑️ Reset index",
        help="Delete every indexed chunk from the local Chroma store (files are untouched).",
    ):
        removed = clear()
        st.toast(f"Index cleared — {removed} chunk(s) removed.", icon="🗑️")

with library_tab:
    st.subheader("Library")
    search = st.text_input(
        "Semantic search", placeholder="Try something vague — e.g. 'money stuff'…"
    )
    if search.strip():
        hits = query(search, k=5)
        if not hits:
            st.warning("No matches — is anything indexed yet?")
        for hit in hits:
            similarity = 1 - hit["distance"]
            st.markdown(f"**{cite(hit['metadata'])}** · similarity `{similarity:.2f}`")
            st.markdown(f"> {hit['text']}")
        st.divider()

    sources = list_sources()
    if not sources:
        st.info("Nothing indexed yet — add voice notes or documents in Upload.")
    else:
        st.caption(f"{len(sources)} source(s), {sum(s['chunks'] for s in sources)} chunk(s) — all stored locally.")
        st.dataframe(
            [
                {
                    "Source": s["source_name"],
                    "Type": s["source_type"],
                    "Added": s["created_at"],
                    "Chunks": s["chunks"],
                }
                for s in sources
            ],
            use_container_width=True,
            hide_index=True,
        )

with ask_tab:
    st.subheader("Ask")
    st.caption("Answers come only from your indexed notes — every claim cited.")
    question = st.text_input(
        "Your question",
        placeholder="What did she say the deadline was, and which document mentions the payment?",
    )
    redact_answer = st.toggle(
        "Redact PII in answer",
        help="Replace names, emails, phones, and IDs with typed tags like "
        "[PERSON] — detected locally by Presidio.",
    )
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Searching your notes…"):
            hits = retrieve(question)
        if not hits:
            st.warning("Nothing indexed yet — upload a voice note or document first.")
        elif redact_answer:
            # No streaming here: PII must never flash on screen before the
            # redaction pass runs over the completed answer.
            from keptra.privacy.redact import redact

            with st.spinner("Answering + redacting locally…"):
                full_answer = "".join(answer_stream(question, hits))
                st.markdown(redact(full_answer))
        else:
            st.write_stream(answer_stream(question, hits))
            with st.expander(f"Sources — {len(hits)} chunk(s) used", expanded=False):
                for hit in hits:
                    st.markdown(f"**{cite(hit['metadata'])}**")
                    st.markdown(f"> {hit['text']}")
                    st.divider()

with metrics_tab:
    st.subheader("Metrics")
    st.markdown(
        "**All inference runs locally on the user's own device — no cloud, no GPU server.**"
    )
    st.caption(
        "Tested on Apple Silicon (M-series MacBook Air); runs anywhere with "
        "Python + Ollama."
    )

    sources = list_sources()
    col1, col2, col3 = st.columns(3)
    col1.metric("Items indexed", len(sources))
    col2.metric("Chunks stored", count())
    col3.metric(
        "Questions answered", get_metrics()["counters"].get("questions_answered", 0)
    )

    rows = []
    for spec in MODEL_SPECS:
        perf = latest(spec["perf_key"])
        rows.append(
            {
                "Model": spec["name"],
                "Task": spec["task"],
                "Source": spec["source"],
                "License": spec["license"],
                "Approx size": spec["size"],
                "Measured performance": (
                    f"{perf:.1f} {spec['perf_label']}" if perf is not None else "—"
                ),
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )
    st.caption(
        "Performance numbers are measured live in this session — a “—” means "
        "that stage hasn't run yet (upload something or ask a question)."
    )
