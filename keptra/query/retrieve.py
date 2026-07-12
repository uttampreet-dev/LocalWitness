"""Retrieval: top-k chunk lookup + citation labels for a question."""

from keptra.index.store import query

TOP_K = 5


def clean_value(value: object) -> str:
    """Metadata value as a display string; '' for missing/None-ish values."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("none", "null") else text


def cite(meta: dict | None) -> str:
    """Human-readable citation for a chunk: source plus timestamp/page."""
    if not meta:
        return "unknown source"
    name = clean_value(meta.get("source_name")) or "unknown source"
    timestamp = clean_value(meta.get("timestamp"))
    if timestamp:
        return f"{name} @ {timestamp}"
    page = clean_value(meta.get("page"))
    if page:
        return f"{name}, page {page}"
    return name


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Top-k most relevant chunks for the question, best-first."""
    return query(question, k=k)
