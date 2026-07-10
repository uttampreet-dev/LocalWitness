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
