"""Retrieval: top-k chunk lookup + citation labels for a question."""

from keptra.index.store import query

TOP_K = 5


def cite(meta: dict | None) -> str:
    """Human-readable citation for a chunk: source plus timestamp/page."""
    if not meta:
        return "unknown source"
    name = meta.get("source_name", "unknown source")
    if meta.get("timestamp"):
        return f"{name} @ {meta['timestamp']}"
    if meta.get("page"):
        return f"{name}, page {meta['page']}"
    return name


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Top-k most relevant chunks for the question, best-first."""
    return query(question, k=k)
