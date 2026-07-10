"""Speech-to-text with faster-whisper — fully local."""

import time
from pathlib import Path

from faster_whisper import WhisperModel

from keptra import metrics

WHISPER_MODEL = "base"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        start = time.perf_counter()
        _model = WhisperModel(
            WHISPER_MODEL,
            device="auto",
            compute_type="int8",
            download_root=str(MODELS_DIR),
        )
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
