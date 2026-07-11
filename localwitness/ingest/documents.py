"""Document text extraction: PDF (per page), .txt, and .md — all local."""

import time
from pathlib import Path

from pypdf import PdfReader

from localwitness import metrics

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _normalize(text: str) -> str:
    """Collapse layout-padding whitespace runs while keeping line breaks."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


def extract_text(path: str) -> list[dict]:
    """Extract text from a document.

    Returns a list of {"text": str, "page": int | None} items — one per PDF
    page (1-indexed, blank pages skipped), or a single item with page=None
    for .txt/.md.
    """
    file = Path(path)
    ext = file.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {ext!r} "
            f"(expected one of {sorted(SUPPORTED_EXTENSIONS)})"
        )

    start = time.perf_counter()
    if ext == ".pdf":
        reader = PdfReader(file)
        items = [
            {"text": _normalize(page.extract_text() or ""), "page": number}
            for number, page in enumerate(reader.pages, start=1)
        ]
        items = [item for item in items if item["text"]]
    else:
        text = file.read_text(encoding="utf-8", errors="replace").strip()
        items = [{"text": text, "page": None}]
    metrics.record_timing("doc_extract_s", time.perf_counter() - start)
    metrics.increment("documents_extracted")
    return items
