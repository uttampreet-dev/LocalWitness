"""Keptra UI: design-system CSS injection + HTML fragment builders.

Presentation only — no model, retrieval, or privacy logic lives here.
"""

import html
import re
from pathlib import Path

import streamlit as st

from keptra.query.retrieve import cite, clean_value

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

NAV_ITEMS = ["Upload", "Library", "Ask", "Metrics"]

SNIPPET_CHARS = 220


def inject_css() -> None:
    css = (ASSETS_DIR / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def rail() -> str:
    """Left rail: wordmark, nav, and the pinned offline status. Returns the
    selected page name."""
    with st.sidebar:
        # Logo mark: an angular bracket-K — stem + chevron — thin stroke,
        # currentColor so it takes the accent from CSS. Inline SVG, offline.
        mark = (
            '<span class="kp-mark"><svg viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true">'
            "<path d='M5 3v18'/><path d='M19 3l-8.5 9 8.5 9'/></svg></span>"
        )
        st.markdown(
            f'<p class="kp-wordmark">{mark}KEPTRA</p>'
            '<p class="kp-tagline">Your memory, on-device.</p>'
            '<p class="kp-tagsub">PRIVATE · SEARCHABLE · OFFLINE</p>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation", NAV_ITEMS, key="nav", label_visibility="collapsed"
        )
        st.markdown(
            '<div class="kp-offline">'
            '<span class="dot">●</span><span class="state">AIR-GAPPED</span>'
            '<div class="sub">no outbound requests</div>'
            '<div class="sub">inference: on-device</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    return page


class Pipeline:
    """Staged pipeline readout, driven by the real ingest calls.

    `begin(i)` marks stage i active (earlier stages done) — call it right
    before the actual work for that stage, so timing is honest, never faked.
    `finish(message)` marks every stage done with a quiet summary line.
    `halt()` clears the readout when a stage fails.
    """

    def __init__(self, stages: list[str]):
        self._stages = stages
        self._slot = st.empty()
        self._render(active=None)

    def _render(self, active: int | None, all_done: bool = False, message: str = "") -> None:
        parts = []
        for index, name in enumerate(self._stages):
            if all_done or (active is not None and index < active):
                state, glyph = "done", "●"
            elif active == index:
                state, glyph = "active", "◐"
            else:
                state, glyph = "", "○"
            parts.append(
                f'<div class="kp-pipe-row {state}">'
                f'<span class="glyph">{glyph}</span>{html.escape(name)}</div>'
            )
            if state == "active":
                parts.append('<div class="kp-pipe-track"></div>')
        if message:
            parts.append(
                '<div class="kp-pipe-done-line"><span class="check">✓</span>'
                f"{html.escape(message)}</div>"
            )
        self._slot.markdown(
            f'<div class="kp-pipe">{"".join(parts)}</div>', unsafe_allow_html=True
        )

    def begin(self, index: int) -> None:
        self._render(active=index)

    def finish(self, message: str) -> None:
        self._render(active=None, all_done=True, message=message)

    def halt(self) -> None:
        self._slot.empty()


def heading(title: str, sub: str | None = None) -> None:
    st.subheader(title)
    if sub:
        st.markdown(f'<p class="kp-sub">{html.escape(sub)}</p>', unsafe_allow_html=True)


def caption(text: str) -> None:
    """Quiet mono caption line."""
    st.markdown(f'<p class="kp-caption">{html.escape(text)}</p>', unsafe_allow_html=True)


# Thin-line source-type icons (stroke = currentColor, colored by CSS).
_ICON_PATHS = {
    "audio": (
        "<path d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z'/>"
        "<path d='M19 10v2a7 7 0 0 1-14 0v-2'/><path d='M12 19v3'/>"
    ),
    "document": (
        "<path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 "
        "2-2V8l-6-6z'/><path d='M14 2v6h6'/>"
        "<path d='M16 13H8'/><path d='M16 17H8'/>"
    ),
    "image": (
        "<rect x='3' y='3' width='18' height='18' rx='2'/>"
        "<circle cx='8.5' cy='8.5' r='1.5'/><path d='M21 15l-5-5L5 21'/>"
    ),
    "default": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='1'/>",
}

STAGGER_MS = 80
STAGGER_CAP = 8  # rows past this share the last delay


def icon(kind: str) -> str:
    paths = _ICON_PATHS.get(kind, _ICON_PATHS["default"])
    return (
        '<svg class="kp-ic" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


def _delay(index: int) -> str:
    return f'style="animation-delay:{min(index, STAGGER_CAP) * STAGGER_MS}ms"'


_CITE_RE = re.compile(r"\[[^\[\]\n]{1,120}\]")


def label(text: str, accent: bool = False) -> str:
    cls = "kp-label kp-label-accent" if accent else "kp-label"
    return f'<p class="{cls}">{html.escape(text)}</p>'


def styled_answer(text: str, question: str = "", streaming: bool = False) -> str:
    """Answer block with inline citations set in mono accent; optional
    blinking caret while the answer is still streaming."""
    body = _CITE_RE.sub(
        lambda m: f'<span class="kp-cite">{m.group(0)}</span>', html.escape(text)
    )
    caret = '<span class="kp-caret">▍</span>' if streaming else ""
    q_line = (
        f'<p class="kp-caption">Q · {html.escape(question)}</p>' if question else ""
    )
    return f'{q_line}{label("Answer", accent=True)}<div class="kp-answer">{body}{caret}</div>'


def rows_html(rows: list[tuple[str, str, str, str]]) -> str:
    """Hairline-separated rows: (icon kind, name, right-aligned meta, snippet).

    Rows fade/stagger in; the icon is colored by CSS (dim, accent on hover).
    """
    items = []
    for index, (kind, name, meta, snippet) in enumerate(rows):
        snippet_html = (
            f'<div class="kp-row-snippet">{html.escape(snippet)}</div>'
            if snippet
            else ""
        )
        items.append(
            f'<div class="kp-row" {_delay(index)}><div class="kp-row-head">'
            f'<span class="kp-row-name">{icon(kind)}{html.escape(name)}</span>'
            f'<span class="kp-row-meta">{html.escape(meta)}</span>'
            f"</div>{snippet_html}</div>"
        )
    return f'<div class="kp-rows">{"".join(items)}</div>'


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > SNIPPET_CHARS:
        text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def evidence_block(hits: list[dict]) -> str:
    """EVIDENCE label + retrieved chunks grouped per real source, styled as
    pulled exhibits. Each source appears once, with the locations of every
    chunk it contributed and its best (measured) retrieval similarity."""
    groups: dict[str, dict] = {}
    for hit in hits:
        meta = hit.get("metadata") or {}
        name = clean_value(meta.get("source_name")) or "unknown source"
        group = groups.setdefault(
            name,
            {
                "type": clean_value(meta.get("source_type")),
                "locations": [],
                "snippet": _snippet(hit["text"]),
                "chunks": 0,
                "sim": None,
            },
        )
        group["chunks"] += 1
        if hit.get("distance") is not None:
            similarity = 1 - hit["distance"]
            group["sim"] = max(group["sim"] or similarity, similarity)
        timestamp = clean_value(meta.get("timestamp"))
        page = clean_value(meta.get("page"))
        location = f"@ {timestamp}" if timestamp else (f"p.{page}" if page else "")
        if location and location not in group["locations"]:
            group["locations"].append(location)
    rows = []
    for name, group in groups.items():
        meta_bits = [bit for bit in (group["type"], ", ".join(group["locations"])) if bit]
        meta_bits.append(f"{group['chunks']} chunk{'s' if group['chunks'] != 1 else ''}")
        if group["sim"] is not None:
            meta_bits.append(f"sim {group['sim']:.2f}")
        rows.append((group["type"], name, " · ".join(meta_bits), group["snippet"]))
    plural = "source" if len(groups) == 1 else "sources"
    return (
        label(f"Evidence · {len(groups)} {plural}")
        + f'<div class="kp-evidence">{rows_html(rows)}</div>'
    )


def hit_rows(hits: list[dict]) -> str:
    """Semantic-search results as hairline rows with similarity readouts."""
    return rows_html(
        [
            (
                clean_value((hit.get("metadata") or {}).get("source_type")),
                cite(hit.get("metadata")),
                f"sim {1 - hit['distance']:.2f}",
                _snippet(hit["text"]),
            )
            for hit in hits
        ]
    )


def recent_rows(sources: list[dict], limit: int = 4) -> str:
    """Compact 'recently indexed' strip: newest sources, no snippets."""
    rows = [
        (
            source["source_type"],
            source["source_name"],
            f"{source['chunks']} chunk{'s' if source['chunks'] != 1 else ''}"
            f" · {source['created_at'].replace('T', ' ')[:16]}",
            "",
        )
        for source in sources[:limit]
    ]
    return f'<div class="kp-recent">{rows_html(rows)}</div>'


def stats_html(stats: list[tuple[str, object]]) -> str:
    """Flat mono stat figures: [(label, value), ...]."""
    items = "".join(
        f'<div {_delay(index)}>'
        f'<div class="kp-stat-value">{html.escape(str(value))}</div>'
        f'<div class="kp-stat-label">{html.escape(label)}</div></div>'
        for index, (label, value) in enumerate(stats)
    )
    return f'<div class="kp-stats">{items}</div>'


def stack_strip(models: list[tuple[str, str]], figures: list[str]) -> str:
    """LOCAL AI STACK strip: mono model items with accent dots, then a quiet
    caption of honest figures (only pass figures that are actually measured
    or true by construction)."""
    items = "".join(
        f'<span class="kp-stack-item"><span class="dot">●</span>'
        f'{html.escape(name)}<span class="role">{html.escape(role)}</span></span>'
        for name, role in models
    )
    figures_html = (
        f'<p class="kp-caption">{html.escape(" · ".join(figures))}</p>'
        if figures
        else ""
    )
    return (
        label("Local AI stack")
        + f'<div class="kp-stack">{items}</div>{figures_html}'
    )


def table_html(headers: list[tuple[str, str]], rows: list[list[dict]]) -> str:
    """Instrument-readout table.

    headers: [(label, css_class)], rows: cells as
    {"text": str, "class": str, "href": optional url, "icon": optional kind}.
    """
    head = "".join(
        f'<th class="{cls}">{html.escape(label)}</th>' for label, cls in headers
    )
    body = []
    for index, row in enumerate(rows):
        cells = []
        for cell in row:
            text = html.escape(str(cell.get("text", "")))
            if cell.get("href"):
                href = html.escape(cell["href"], quote=True)
                text = f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
            if cell.get("icon"):
                text = icon(cell["icon"]) + text
            cells.append(f'<td class="{cell.get("class", "")}">{text}</td>')
        body.append(f"<tr {_delay(index)}>{''.join(cells)}</tr>")
    return (
        f'<table class="kp-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )
