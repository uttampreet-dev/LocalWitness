"""Speech-to-text with faster-whisper — fully local."""

import time
from pathlib import Path

from faster_whisper import WhisperModel

from localwitness import metrics

WHISPER_MODEL = "base"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_model: WhisperModel | None = None


def _load(local_only: bool) -> WhisperModel:
    return WhisperModel(
        WHISPER_MODEL,
        device="auto",
        compute_type="int8",
        download_root=str(MODELS_DIR),
        local_files_only=local_only,
    )


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        start = time.perf_counter()
        # Privacy: offline-first. Without local_files_only, faster-whisper
        # resolves huggingface.co on EVERY load even when the weights are
        # already cached. We only reach the network if the model is genuinely
        # missing (the documented one-time first-run download).
        try:
            _model = _load(local_only=True)
        except Exception:
            _model = _load(local_only=False)
        load_s = time.perf_counter() - start
        metrics.record_timing("whisper_load_s", load_s)
        metrics.record_model(
            f"faster-whisper ({WHISPER_MODEL})",
            {
                "task": "speech-to-text",
                "runtime": "CTranslate2, int8",
                "approx_size": "~145 MB",
                "source": "https://github.com/SYSTRAN/faster-whisper",
                "license": "MIT",
            },
        )
    return _model


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe(path: str) -> dict:
    """Transcribe an audio file locally.

    Returns {"segments": [{"text", "start", "end"}, ...], "text": full text,
    "duration": audio length in seconds}. start/end are mm:ss strings.
    """
    model = _get_model()
    start = time.perf_counter()
    raw_segments, info = model.transcribe(str(path))
    segments = [
        {"text": seg.text.strip(), "start": _mmss(seg.start), "end": _mmss(seg.end)}
        for seg in raw_segments
    ]
    elapsed = time.perf_counter() - start
    metrics.record_timing("whisper_transcribe_s", elapsed)
    if info.duration:
        metrics.record_timing("whisper_s_per_audio_min", elapsed / (info.duration / 60))
    metrics.increment("audio_files_transcribed")
    return {
        "segments": segments,
        "text": " ".join(seg["text"] for seg in segments),
        "duration": info.duration,
    }
