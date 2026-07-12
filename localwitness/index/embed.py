"""Embeddings with all-MiniLM-L6-v2 — fully local.

The app runs the **INT8-quantized ONNX** build of the model (4x smaller,
~3.5x faster per query — see scripts/quantize_embeddings.py, which produces
it). If that artifact hasn't been built yet, we transparently fall back to
the PyTorch model so a fresh clone still works. Either way the vectors are
384-dim, mean-pooled and L2-normalized, so both backends are interchangeable
against the same index.
"""

import time
from pathlib import Path

import numpy as np

from localwitness import metrics

EMBED_MODEL = "all-MiniLM-L6-v2"
HF_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
INT8_MODEL = MODELS_DIR / "minilm_onnx_int8" / "model_quantized.onnx"

# ("onnx", session, tokenizer) or ("torch", model, None)
_backend: tuple | None = None


def _load_onnx() -> tuple | None:
    """INT8 ONNX Runtime session + tokenizer, or None if not built/available."""
    if not INT8_MODEL.exists():
        return None
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            HF_ID, cache_dir=str(MODELS_DIR), local_files_only=True
        )
        session = ort.InferenceSession(
            str(INT8_MODEL), providers=["CPUExecutionProvider"]
        )
        return session, tokenizer
    except Exception:
        return None


def _load_torch():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        # Offline-first: never touch the network once weights are cached
        # (the HF hub otherwise phones home to revalidate on every load).
        model = SentenceTransformer(
            EMBED_MODEL,
            device=device,
            cache_folder=str(MODELS_DIR),
            local_files_only=True,
        )
    except Exception:
        # First run only: weights not downloaded yet.
        model = SentenceTransformer(
            EMBED_MODEL, device=device, cache_folder=str(MODELS_DIR)
        )
    return model, device


def _get_backend() -> tuple:
    global _backend
    if _backend is not None:
        return _backend

    start = time.perf_counter()
    onnx = _load_onnx()
    if onnx is not None:
        session, tokenizer = onnx
        _backend = ("onnx", session, tokenizer)
        size_mb = INT8_MODEL.stat().st_size / 1e6
        runtime = "ONNX Runtime, INT8 (dynamic quantization), CPU"
        approx_size = f"~{size_mb:.0f} MB (INT8)"
    else:
        model, device = _load_torch()
        _backend = ("torch", model, None)
        runtime = f"PyTorch, device={device}"
        approx_size = "~90 MB (fp32)"

    metrics.record_timing("embed_load_s", time.perf_counter() - start)
    metrics.record_model(
        f"all-MiniLM-L6-v2 ({_backend[0]})",
        {
            "task": "text embeddings (384-dim)",
            "runtime": runtime,
            "approx_size": approx_size,
            "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            "license": "Apache-2.0",
        },
    )
    return _backend


def active_backend() -> str:
    """'onnx' (INT8) or 'torch' (fp32) — whichever the app actually loaded."""
    return _get_backend()[0]


def _embed_onnx(session, tokenizer, texts: list[str]) -> np.ndarray:
    """Mean-pool + L2-normalize, matching sentence-transformers' pipeline."""
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="np"
    )
    inputs = {
        name.name: tokens[name.name]
        for name in session.get_inputs()
        if name.name in tokens
    }
    hidden = session.run(None, inputs)[0]  # (batch, seq, 384)
    mask = tokens["attention_mask"][..., None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-9, None)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; records ms-per-text in metrics."""
    kind, model, tokenizer = _get_backend()
    start = time.perf_counter()
    if kind == "onnx":
        vectors = _embed_onnx(model, tokenizer, texts)
    else:
        vectors = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
    elapsed = time.perf_counter() - start
    metrics.record_timing("embed_ms_per_text", elapsed * 1000 / max(len(texts), 1))
    return np.asarray(vectors).tolist()
