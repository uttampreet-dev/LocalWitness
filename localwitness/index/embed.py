"""Embeddings with sentence-transformers all-MiniLM-L6-v2 — fully local."""

import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from localwitness import metrics

EMBED_MODEL = "all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        start = time.perf_counter()
        try:
            # Offline-first: never touch the network once weights are cached
            # (the HF hub otherwise phones home to revalidate on every load).
            _model = SentenceTransformer(
                EMBED_MODEL,
                device=device,
                cache_folder=str(MODELS_DIR),
                local_files_only=True,
            )
        except Exception:
            # First run only: weights not downloaded yet.
            _model = SentenceTransformer(
                EMBED_MODEL, device=device, cache_folder=str(MODELS_DIR)
            )
        metrics.record_timing("embed_load_s", time.perf_counter() - start)
        metrics.record_model(
            f"sentence-transformers ({EMBED_MODEL})",
            {
                "task": "text embeddings (384-dim)",
                "runtime": f"PyTorch, device={device}",
                "approx_size": "~90 MB",
                "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                "license": "Apache-2.0",
            },
        )
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; records ms-per-text in metrics."""
    model = _get_model()
    start = time.perf_counter()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    metrics.record_timing("embed_ms_per_text", elapsed * 1000 / max(len(texts), 1))
    return vectors.tolist()
