"""Chunking: split content into ~500-char overlapping chunks with metadata."""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    source_meta: dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping chunks, each carrying source_meta.

    source_meta should hold source_type, source_name, timestamp or page,
    and created_at. Breaks at whitespace where possible.
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + chunk_size // 2, end + 1)
            newline = text.rfind("\n", start + chunk_size // 2, end + 1)
            cut = max(space, newline)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append({"text": piece, **source_meta})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_segments(
    segments: list[dict],
    source_meta: dict,
    chunk_size: int = CHUNK_SIZE,
) -> list[dict]:
    """Group Whisper segments into ~chunk_size chunks.

    Each chunk's timestamp is the start of its first segment, so citations
    can point at the right moment in the audio.
    """
    chunks = []
    current: list[str] = []
    current_start = None

    def flush():
        if current:
            chunks.append(
                {"text": " ".join(current), "timestamp": current_start, **source_meta}
            )

    for seg in segments:
        if current and sum(len(t) + 1 for t in current) + len(seg["text"]) > chunk_size:
            flush()
            current = []
            current_start = None
        if current_start is None:
            current_start = seg["start"]
        current.append(seg["text"])
    flush()
    return chunks
