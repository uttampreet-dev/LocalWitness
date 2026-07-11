"""LocalWitness — a local, offline, private second brain.

Streamlit entry point. Run with: streamlit run app.py
"""

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from localwitness import ui
from localwitness.index.chunk import chunk_segments, chunk_text
from localwitness.index.store import (
    add_chunks,
    clear,
    count,
    delete_source,
    list_sources,
    query,
)
from localwitness.metrics import MODEL_SPECS, get_metrics, latest
from localwitness.ingest.audio import transcribe
from localwitness.ingest.documents import extract_text
from localwitness.ingest.images import caption
from localwitness.query.answer import OLLAMA_ERRORS, answer_stream
from localwitness.query.retrieve import retrieve

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Local vault of original uploads so evidence rows can open the real source
# (kept on-device, gitignored). Sources indexed before this existed fall back
# to sample_data/, then degrade to the stored snippet.
SOURCES_DIR = Path(__file__).resolve().parent / "sources"


def resolve_source_file(source_name: str) -> Path | None:
    for base in (SOURCES_DIR, Path(__file__).resolve().parent / "sample_data"):
        candidate = base / source_name
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False, max_entries=32)
def pdf_page_png(path_str: str, page_number: int, needle: str, mtime: float) -> bytes:
    """Render the cited PDF page as PNG, outlining the cited passage in
    accent where PyMuPDF's text search finds it (best effort — no fake
    highlight if the search misses). Cached per (file, page, needle, mtime)."""
    import fitz

    with fitz.open(path_str) as doc:
        page = doc[page_number - 1]
        if needle:
            for rect in page.search_for(needle):
                page.draw_rect(rect, color=(0.302, 0.831, 0.769), width=1.2)
        return page.get_pixmap(dpi=120).tobytes("png")


def text_context_window(path: Path, chunk_text: str, window: int = 650):
    """(pre, match, post) around the cited chunk in the source text, using
    the same whitespace normalization the indexer used so the chunk is an
    exact substring. Returns None if the passage can't be located."""
    from localwitness.ingest.documents import _normalize

    text = _normalize(path.read_text(encoding="utf-8", errors="replace"))
    needle = chunk_text.strip()
    index = text.find(needle)
    if index == -1:  # fall back to the chunk's head if edges were trimmed
        needle = needle[:120]
        index = text.find(needle)
    if index == -1:
        return None
    start = max(0, index - window)
    end = min(len(text), index + len(needle) + window)
    if start > 0:
        start = text.find(" ", start, index)
        start = 0 if start == -1 else start + 1
    if end < len(text):
        cut = text.rfind(" ", index + len(needle), end)
        end = end if cut == -1 else cut
    pre = ("… " if start > 0 else "") + text[start:index]
    post = text[index + len(needle) : end] + (" …" if end < len(text) else "")
    return pre, needle, post

st.set_page_config(page_title="LocalWitness", page_icon="🧠", layout="wide")
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
                from localwitness.privacy.redact import redact

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
                    from localwitness.privacy.blur import blur_people

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
            # Keep the original in the local vault so evidence rows can open
            # the real source later. On-device only; gitignored.
            SOURCES_DIR.mkdir(exist_ok=True)
            shutil.copy2(tmp_path, SOURCES_DIR / uploaded.name)
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

    def _render_context(group: dict) -> None:
        """The 'view in context' body for one evidence source. Lazy: only
        runs while the row's toggle is on; presentation only."""
        path = resolve_source_file(group["name"])
        if path is None:
            ui.caption(
                "source file not in the local vault — re-upload it once to "
                "enable the context view"
            )
            return
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            window = text_context_window(path, group["text"])
            if window is None:
                ui.caption("couldn't locate the cited passage in the file")
            else:
                st.markdown(ui.context_html(*window), unsafe_allow_html=True)
        elif suffix == ".pdf":
            page_number = int(group["page"]) if group["page"].isdigit() else 1
            needle = " ".join(group["text"].split())[:60]
            png = pdf_page_png(
                str(path), page_number, needle, path.stat().st_mtime
            )
            st.image(png, width=680)
            ui.caption(f"{group['name']} — page {page_number}, cited passage outlined")
        elif suffix in AUDIO_EXTENSIONS:
            timestamp = group["timestamp"] or "00:00"
            try:
                minutes, seconds = timestamp.split(":")
                start_at = int(minutes) * 60 + int(seconds)
            except ValueError:
                start_at = 0
            st.audio(str(path), start_time=start_at)
            ui.caption(f"cued to the cited moment — @ {timestamp}")
        elif suffix in IMAGE_EXTENSIONS:
            st.image(str(path), width=420)
            ui.caption(f"{group['name']} — what Moondream read from it is quoted above")
        else:
            ui.caption(f"no context view for {suffix} sources")

    def _show_evidence(hits: list[dict]) -> None:
        groups = ui.group_evidence(hits)
        plural = "source" if len(groups) == 1 else "sources"
        st.markdown(
            ui.label(f"Evidence · {len(groups)} {plural}"), unsafe_allow_html=True
        )
        for group in groups:
            with st.container(key=f"evgrp_{group['name']}"):
                st.markdown(ui.exhibit_row_html(group), unsafe_allow_html=True)
                if st.toggle("view in context", key=f"evctx_{group['name']}"):
                    _render_context(group)

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
            from localwitness.privacy.redact import redact

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
