"""Metrics: timing, model, and usage measurements for Keptra.

A simple module-level singleton. Every model records its load time, per-call
timings, and identity here; the Metrics tab renders whatever has accumulated.
"""

from collections import defaultdict

_metrics: dict = {
    "timings": defaultdict(list),   # name -> [seconds, ...]
    "models": {},                   # model name -> info dict (size, runtime, ...)
    "counters": defaultdict(int),   # name -> count
}


def record_timing(name: str, seconds: float) -> None:
    _metrics["timings"][name].append(round(seconds, 3))


def record_model(name: str, info: dict) -> None:
    _metrics["models"][name] = info


def increment(name: str, by: int = 1) -> None:
    _metrics["counters"][name] += by


def get_metrics() -> dict:
    return _metrics


def latest(name: str) -> float | None:
    """Most recent recorded timing for name, or None if never measured."""
    values = _metrics["timings"].get(name)
    return values[-1] if values else None


# The Phase-1 stack. perf_key/perf_label map each row to its live timing.
MODEL_SPECS = [
    {
        "name": "faster-whisper (base, int8)",
        "task": "speech-to-text",
        "source": "https://github.com/SYSTRAN/faster-whisper",
        "license": "MIT",
        "size": "~145 MB",
        "perf_key": "whisper_s_per_audio_min",
        "perf_label": "s per min of audio",
    },
    {
        "name": "all-MiniLM-L6-v2",
        "task": "text embeddings (384-dim)",
        "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "license": "Apache-2.0",
        "size": "~90 MB",
        "perf_key": "embed_ms_per_text",
        "perf_label": "ms per text embedded",
    },
    {
        "name": "ChromaDB (persistent, HNSW)",
        "task": "vector search",
        "source": "https://github.com/chroma-core/chroma",
        "license": "Apache-2.0",
        "size": "n/a (library)",
        "perf_key": "retrieval_ms",
        "perf_label": "ms per query (incl. embedding)",
    },
    {
        "name": "Moondream2 via Ollama",
        "task": "image captioning + text-in-image (VLM)",
        "source": "https://ollama.com/library/moondream",
        "license": "Apache-2.0",
        "size": "~1.7 GB",
        "perf_key": "caption_ms_per_image",
        "perf_label": "ms per image",
    },
    {
        "name": "qwen2.5:3b (INT4 GGUF) via Ollama",
        "task": "RAG answering (local LLM)",
        "source": "https://ollama.com/library/qwen2.5",
        "license": "Apache-2.0",
        "size": "~1.9 GB",
        "perf_key": "llm_tokens_per_s",
        "perf_label": "tokens/sec",
    },
]
