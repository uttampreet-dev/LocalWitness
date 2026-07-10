"""Persistent ChromaDB vector store wrapper — local, at ./chroma_db."""

import logging
import time
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

# Telemetry is disabled below, but chromadb 0.5.x still logs a (harmlessly
# failing) send attempt on startup — keep that noise out of the UI/logs.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from keptra import metrics
from keptra.index.embed import embed

CHROMA_PATH = Path(__file__).resolve().parents[2] / "chroma_db"
COLLECTION_NAME = "keptra"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            # Privacy: Chroma's telemetry phones home by default — never allowed here.
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def add_chunks(chunks: list[dict]) -> int:
    """Embed and persist chunks (dicts with 'text' + metadata keys)."""
    if not chunks:
        return 0
    collection = _get_collection()
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {k: v for k, v in chunk.items() if k != "text" and v is not None}
        for chunk in chunks
    ]
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=texts,
        embeddings=embed(texts),
        metadatas=metadatas,
    )
    metrics.increment("chunks_indexed", len(chunks))
    return len(chunks)


def query(text: str, k: int = 5) -> list[dict]:
    """Semantic search: returns [{id, text, metadata, distance}] best-first."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    start = time.perf_counter()
    result = collection.query(
        query_embeddings=embed([text]), n_results=min(k, collection.count())
    )
    metrics.record_timing("retrieval_ms", (time.perf_counter() - start) * 1000)
    return [
        {
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i] or {},
            "distance": result["distances"][0][i],
        }
        for i in range(len(result["ids"][0]))
    ]


def count() -> int:
    return _get_collection().count()


def clear() -> int:
    """Delete every indexed chunk from the local store; returns how many."""
    collection = _get_collection()
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_sources() -> list[dict]:
    """All indexed sources grouped by source_name, newest first.

    Returns [{source_name, source_type, created_at, chunks}].
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    sources: dict[str, dict] = {}
    for meta in result["metadatas"]:
        entry = sources.setdefault(
            meta["source_name"],
            {
                "source_name": meta["source_name"],
                "source_type": meta.get("source_type", "unknown"),
                "created_at": meta.get("created_at", ""),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
        entry["created_at"] = min(entry["created_at"], meta.get("created_at", "")) or meta.get("created_at", "")
    return sorted(sources.values(), key=lambda s: s["created_at"], reverse=True)
