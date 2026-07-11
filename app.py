"""Keptra — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from keptra import ui
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
from keptra.query.retrieve import retrieve

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

st.set_page_config(page_title="Keptra", page_icon="🧠", layout="wide")
ui.inject_css()

page = ui.rail()


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


if page == "Upload":
    ui.heading(
        "Upload",
        "Voice notes, documents, and photos — transcribed, described, "
        "and indexed on this machine.",
    )
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
            pipe = ui.Pipeline(
                [
                    "Transcribing (Whisper)",
                    "Chunking",
                    "Embedding + indexing (MiniLM)",
                ]
            )
            pipe.begin(0)
            transcript_cache = st.session_state.setdefault("transcript_cache", {})
            result = transcript_cache.get(cache_key)
            if result is None:
                result = transcribe(tmp_path)
                transcript_cache[cache_key] = result
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
                "Export transcript (.txt)",
                transcript,
                file_name=f"{Path(uploaded.name).stem}_transcript.txt",
            )
            pipe.begin(1)
            chunks = chunk_segments(
                result["segments"],
                {
                    "source_type": "audio",
                    "source_name": uploaded.name,
                    "created_at": created_at,
                },
            )
        elif suffix in IMAGE_EXTENSIONS:
            pipe = ui.Pipeline(
                [
                    "Describing + reading text (Moondream)",
                    "Chunking",
                    "Embedding + indexing (MiniLM)",
                ]
            )
            pipe.begin(0)
            caption_cache = st.session_state.setdefault("caption_cache", {})
            described = caption_cache.get(cache_key)
            if described is None:
                try:
                    described = caption(tmp_path)
                    caption_cache[cache_key] = described
                except OLLAMA_ERRORS as exc:
                    pipe.halt()
                    st.error(
                        "Couldn't caption: is Ollama running and moondream "
                        f"pulled (`ollama pull moondream`)?\n\nDetails: {exc}"
                    )
            chunks = []
            if described:
                st.image(tmp_path, width=360)
                st.markdown(f"> {described}")
                pipe.begin(1)
                chunks = chunk_text(
                    described,
                    {
                        "source_type": "image",
                        "source_name": uploaded.name,
                        "created_at": created_at,
                    },
                )
                if st.button(
                    "Export privacy-safe copy",
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
            pipe = ui.Pipeline(
                ["Extracting text", "Chunking", "Embedding + indexing (MiniLM)"]
            )
            pipe.begin(0)
            items = extract_text(tmp_path)
            pipe.begin(1)
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
        replaced = 0
        indexed_keys = st.session_state.setdefault("indexed_keys", set())
        if chunks:
            pipe.begin(2)
            if cache_key not in indexed_keys:
                replaced = delete_source(uploaded.name)
                add_chunks(chunks)
                indexed_keys.add(cache_key)
            summary = [f"indexed {len(chunks)} chunk(s)"]
            if replaced:
                summary.append(f"replaced {replaced} old")
            summary.append("stored locally")
            pipe.finish(" · ".join(summary))
        elif suffix not in IMAGE_EXTENSIONS:
            # Nothing indexable came out of the file (image errors already halt)
            pipe.halt()
            st.warning(f"No indexable text found in {uploaded.name}.")
        Path(tmp_path).unlink(missing_ok=True)

    recent = list_sources()
    if recent:
        st.divider()
        ui.caption("recently indexed")
        st.markdown(ui.recent_rows(recent), unsafe_allow_html=True)

    st.divider()
    if st.button(
        "Reset index",
        key="reset_index",
        help="Delete every indexed chunk from the local Chroma store (files are untouched).",
    ):
        removed = clear()
        st.toast(
            f"Index cleared — {removed} chunk(s) removed.", icon=":material/delete:"
        )

elif page == "Library":
    ui.heading("Library", "Everything indexed so far — stored locally in ./chroma_db.")
    search = st.text_input(
        "Semantic search", placeholder="Try something vague — e.g. 'money stuff'…"
    )
    if search.strip():
        hits = query(search, k=5)
        if not hits:
            st.warning("No matches — is anything indexed yet?")
        else:
            st.markdown(ui.hit_rows(hits), unsafe_allow_html=True)
        st.divider()

    sources = list_sources()
    if not sources:
        st.info("Nothing indexed yet — add voice notes or documents in Upload.")
    else:
        ui.caption(
            f"{len(sources)} source(s) · {sum(s['chunks'] for s in sources)} "
            "chunk(s) · all stored locally"
        )
        st.markdown(
            ui.table_html(
                headers=[
                    ("Source", ""),
                    ("Type", ""),
                    ("Added", ""),
                    ("Chunks", "num"),
                ],
                rows=[
                    [
                        {
                            "text": s["source_name"],
                            "class": "mono",
                            "icon": s["source_type"],
                        },
                        {"text": s["source_type"], "class": "dim"},
                        {
                            "text": s["created_at"].replace("T", " "),
                            "class": "mono dim",
                        },
                        {"text": s["chunks"], "class": "num"},
                    ]
                    for s in sources
                ],
            ),
            unsafe_allow_html=True,
        )

elif page == "Ask":
    ui.heading(
        "Ask", "Answers come only from your indexed notes — every claim cited."
    )
    # Single-shot flow: once an answer lands, the input clears (flag consumed
    # here, before the widget instantiates) so the next question needs no
    # manual delete. The last answer + evidence stay rendered below.
    if st.session_state.pop("ask_q_clear", False):
        st.session_state["ask_q"] = ""
    question = st.text_input(
        "Your question",
        key="ask_q",
        placeholder="What did she say the deadline was, and which document mentions the payment?",
    )

    def _fill_question(text: str) -> None:
        st.session_state["ask_q"] = text

    if "ask_result" not in st.session_state:
        suggestions = [
            "What's the deadline and the fee?",
            "Which theme did they choose?",
            "What's in the contract?",
            "Summarize the project",
        ]
        for row_index in range(0, len(suggestions), 2):
            pair = suggestions[row_index : row_index + 2]
            for col, text in zip(st.columns(len(pair)), pair):
                col.button(
                    text,
                    key=f"chip_{row_index + pair.index(text)}",
                    use_container_width=True,
                    on_click=_fill_question,
                    args=(text,),
                )

    ask_col, toggle_col = st.columns([1, 4], vertical_alignment="center")
    with toggle_col:
        redact_answer = st.toggle(
            "Redact PII in answer",
            help="Replace names, emails, phones, and IDs with typed tags like "
            "[PERSON] — detected locally by Presidio.",
        )
    with ask_col:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    def _show_evidence(hits: list[dict]) -> None:
        st.markdown(ui.evidence_block(hits), unsafe_allow_html=True)

    def _remember_question(text: str) -> None:
        recents = st.session_state.setdefault("recent_questions", [])
        if text in recents:
            recents.remove(text)
        recents.insert(0, text)
        del recents[5:]

    def _rerun_recent(text: str) -> None:
        st.session_state["ask_q"] = text
        st.session_state["ask_auto"] = True

    ask_auto = st.session_state.pop("ask_auto", False)

    if (ask_clicked or ask_auto) and question.strip():
        with st.spinner("Searching your notes…"):
            hits = retrieve(question)
        if not hits:
            st.warning("Nothing indexed yet — upload a voice note or document first.")
        elif redact_answer:
            # No streaming here: PII must never flash on screen before the
            # redaction pass runs over the completed answer.
            from keptra.privacy.redact import redact

            with st.spinner("Answering + redacting locally…"):
                full_answer = redact("".join(answer_stream(question, hits)))
            st.session_state["ask_result"] = {
                "question": question,
                "answer": full_answer,
                "hits": hits,
                "redacted": True,
            }
            _remember_question(question)
            st.session_state["ask_q_clear"] = True
            st.rerun()
        else:
            slot = st.empty()
            answer = ""
            for token in answer_stream(question, hits):
                answer += token
                slot.markdown(
                    ui.styled_answer(answer, question, streaming=True),
                    unsafe_allow_html=True,
                )
            st.session_state["ask_result"] = {
                "question": question,
                "answer": answer,
                "hits": hits,
                "redacted": False,
            }
            _remember_question(question)
            st.session_state["ask_q_clear"] = True
            st.rerun()
    if "ask_result" in st.session_state:
        result = st.session_state["ask_result"]
        st.markdown(
            ui.styled_answer(result["answer"], result["question"]),
            unsafe_allow_html=True,
        )
        if result["redacted"]:
            ui.caption(
                "evidence hidden while redaction is on — raw excerpts may contain PII"
            )
        else:
            _show_evidence(result["hits"])

    # Lightweight shortcuts, not a transcript: question strings only.
    recents = st.session_state.get("recent_questions", [])
    if recents:
        st.markdown(ui.label("Recent"), unsafe_allow_html=True)
        for index, recent_question in enumerate(recents):
            st.button(
                recent_question,
                key=f"recent_{index}",
                on_click=_rerun_recent,
                args=(recent_question,),
            )

elif page == "Metrics":
    ui.heading(
        "Metrics",
        "Tested on Apple Silicon (M-series MacBook Air); runs anywhere with "
        "Python + Ollama.",
    )
    sources = list_sources()
    st.markdown(
        ui.stats_html(
            [
                ("items indexed", len(sources)),
                ("chunks stored", count()),
                (
                    "questions answered",
                    get_metrics()["counters"].get("questions_answered", 0),
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    stack = [
        ("Whisper", "speech"),
        ("MiniLM", "embeddings"),
        ("Moondream", "vision"),
        ("YOLOv8n", "detection"),
        ("Presidio", "redaction"),
        ("Qwen2.5", "language"),
    ]
    # Honest figures only: sizes summed from the (approximate) spec table,
    # runtime is true by construction, latency only once actually measured.
    size_pattern = re.compile(r"~(\d+(?:\.\d+)?)\s*(MB|GB)")
    total_gb = sum(
        float(m.group(1)) * (1 if m.group(2) == "GB" else 1 / 1024)
        for spec in MODEL_SPECS
        if (m := size_pattern.search(spec["size"]))
    )
    figures = [
        f"~{total_gb:.1f} GB of models on disk",
        "inference: on-device (CPU + Metal)",
    ]
    last_query_ms = latest("retrieval_ms")
    if last_query_ms is not None:
        figures.append(f"last query {last_query_ms:.0f} ms")
    st.markdown(ui.stack_strip(stack, figures), unsafe_allow_html=True)

    ui.caption(
        "All inference runs locally on the user's own device — "
        "no cloud, no GPU server."
    )
    rows = []
    for spec in MODEL_SPECS:
        perf = latest(spec["perf_key"])
        rows.append(
            [
                {"text": spec["name"], "class": "mono"},
                {"text": spec["task"], "class": "dim"},
                {"text": spec["license"], "class": "mono dim"},
                {"text": spec["size"], "class": "num dim"},
                {
                    "text": (
                        f"{perf:.1f} {spec['perf_label']}"
                        if perf is not None
                        else "—"
                    ),
                    "class": "num" if perf is not None else "num dim",
                },
                {
                    "text": _domain(spec["source"]),
                    "class": "mono",
                    "href": spec["source"],
                },
            ]
        )
    st.markdown(
        ui.table_html(
            headers=[
                ("Model", ""),
                ("Task", ""),
                ("License", ""),
                ("Size", "num"),
                ("Measured", "num"),
                ("Source", ""),
            ],
            rows=rows,
        ),
        unsafe_allow_html=True,
    )
    ui.caption(
        "Measured live in this session — “—” means that stage hasn't run yet "
        "(upload something or ask a question)."
    )
