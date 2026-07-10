"""Retrieval: top-k chunk lookup + citation labels for a question."""

from keptra.index.store import query

TOP_K = 5


def cite(meta: dict) -> str:
    """Human-readable citation for a chunk: source plus timestamp/page."""
    if meta.get("timestamp"):
        return f"{meta['source_name']} @ {meta['timestamp']}"
    if meta.get("page"):
        return f"{meta['source_name']}, page {meta['page']}"
    return meta["source_name"]


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Top-k most relevant chunks for the question, best-first."""
    return query(question, k=k)
