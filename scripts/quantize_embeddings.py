"""Quantize the embedding model to ONNX INT8; report footprint + latency.

Optional optimization experiment: exports all-MiniLM-L6-v2 to ONNX, applies
dynamic INT8 quantization (arm64 config), and measures model size and
single-query embedding latency before vs after. Everything stays in models/
(gitignored) and runs locally.

Usage: python scripts/quantize_embeddings.py
"""

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "what did she say the deadline was?"
RUNS = 50


def tree_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def pytorch_baseline() -> tuple[float, float]:
    """Return (size_mb, ms_per_query) for the PyTorch model the app uses."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, device=device, cache_folder=str(MODELS_DIR))
    model.encode([QUERY])  # warm-up
    start = time.perf_counter()
    for _ in range(RUNS):
        model.encode([QUERY])
    ms = (time.perf_counter() - start) * 1000 / RUNS

    # HF-hub cache layout: actual weights live under blobs/.
    cache = MODELS_DIR / f"models--{MODEL_ID.replace('/', '--')}" / "blobs"
    return tree_size_mb(cache), ms


def onnx_int8() -> tuple[float, float]:
    """Export + quantize (cached across runs); return (size_mb, ms_per_query)."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    fp32_dir = MODELS_DIR / "minilm_onnx"
    int8_dir = MODELS_DIR / "minilm_onnx_int8"
    int8_file = int8_dir / "model_quantized.onnx"
    if not int8_file.exists():
        ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True).save_pretrained(fp32_dir)
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        quantizer.quantize(
            save_dir=int8_dir,
            quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=False),
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForFeatureExtraction.from_pretrained(
        int8_dir, file_name="model_quantized.onnx"
    )

    def embed_once() -> np.ndarray:
        tokens = tokenizer(QUERY, return_tensors="pt")
        hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling
        return pooled.detach().numpy()

    embed_once()  # warm-up
    start = time.perf_counter()
    for _ in range(RUNS):
        embed_once()
    ms = (time.perf_counter() - start) * 1000 / RUNS
    return int8_file.stat().st_size / 1e6, ms


def main() -> None:
    pt_size, pt_ms = pytorch_baseline()
    q_size, q_ms = onnx_int8()
    print("\n============= OPTIMIZATION REPORT =============")
    print(f"{'':24} {'size':>10} {'ms/query':>10}")
    print(f"{'PyTorch (app default)':24} {pt_size:>8.1f}MB {pt_ms:>10.1f}")
    print(f"{'ONNX INT8 (CPU)':24} {q_size:>8.1f}MB {q_ms:>10.1f}")
    print(f"footprint: {pt_size / q_size:.1f}x smaller | "
          f"latency: {pt_ms / q_ms:.1f}x faster (single query)")


if __name__ == "__main__":
    main()
