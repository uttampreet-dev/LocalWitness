"""Ollama RAG answering — grounded, cited, fully local."""

import time
from collections.abc import Iterator

import httpx
import ollama

# Errors that actually mean "the Ollama server is unreachable/unhappy" —
# anything else is a bug in our code and must surface as itself.
OLLAMA_ERRORS = (httpx.HTTPError, ollama.RequestError, ollama.ResponseError, ConnectionError)

from keptra import metrics
from keptra.query.retrieve import cite

LLM_MODEL = "qwen2.5:3b"

OLLAMA_HELP = (
    "Couldn't reach the local Ollama server. Make sure it is running "
    "(`ollama serve` or the Ollama menu-bar app) and the model is pulled "
    f"(`ollama pull {LLM_MODEL}`), then try again."
)

SYSTEM_PROMPT = (
    "You are Keptra, a private, offline second brain. Answer ONLY using the "
    "provided context. Every claim must carry a citation in [brackets], "
    "copying the source label exactly as given, e.g. [interview_note.m4a @ "
    "00:00] or [vendor_contract.pdf, page 2] — no sentence without one. When "
    "a sentence combines facts from more than one source, chain a citation "
    "for each, e.g. [interview_note.m4a @ 00:00][vendor_contract.pdf, page "
    "2]. If "
    "the answer isn't in the context, reply with exactly \"That's not in my "
    "notes.\" and nothing else — no citation. Never use outside knowledge."
)

FALLBACK = "That's not in my notes."

_model_registered = False


def _register_model() -> None:
    global _model_registered
    if not _model_registered:
        metrics.record_model(
            f"Ollama ({LLM_MODEL})",
            {
                "task": "RAG answering (local LLM)",
                "runtime": "Ollama / llama.cpp, Metal, INT4 GGUF",
                "approx_size": "~1.9 GB",
                "source": "https://ollama.com/library/qwen2.5",
                "license": "Apache-2.0 (Qwen2.5 3B)",
            },
        )
        _model_registered = True


def build_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the RAG prompt: labeled context chunks + the question."""
    context = "\n\n".join(f"[{cite(hit['metadata'])}]\n{hit['text']}" for hit in hits)
    example = cite(hits[0]["metadata"]) if hits else "source"
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer using only the context above. End every sentence with the "
        f"citation of the chunk it came from, like [{example}]; if a "
        "sentence draws on several chunks, chain one citation per chunk, "
        f"like [{example}][another label]. Valid "
        "citations are ONLY the bracketed labels that precede each context "
        "chunk above — copy them exactly, never invent one, and never cite a "
        "file name that merely appears inside the text."
    )


def answer_stream(question: str, hits: list[dict]) -> Iterator[str]:
    """Stream a grounded answer; records tokens/sec + total latency."""
    _register_model()
    start = time.perf_counter()
    try:
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, hits)},
            ],
            stream=True,
            options={"temperature": 0},
        )
        eval_count = eval_duration = 0
        # Buffer the first few tokens: if the model is refusing, emit the bare
        # fallback and drop whatever spurious citation it tacks on after it.
        buffer = ""
        buffering = True
        for part in stream:
            content = part["message"]["content"]
            if content:
                if buffering:
                    buffer += content
                    if buffer.strip().startswith(FALLBACK):
                        yield FALLBACK
                        break
                    if len(buffer) > len(FALLBACK):
                        yield buffer
                        buffering = False
                else:
                    yield content
            if getattr(part, "done", False):
                eval_count = getattr(part, "eval_count", 0) or 0
                eval_duration = getattr(part, "eval_duration", 0) or 0
        if buffering and not buffer.strip().startswith(FALLBACK):
            yield buffer
    except OLLAMA_ERRORS as exc:
        yield f"{OLLAMA_HELP}\n\nDetails: {exc}"
        return
    metrics.record_timing("llm_total_s", time.perf_counter() - start)
    if eval_count and eval_duration:
        metrics.record_timing("llm_tokens_per_s", eval_count / (eval_duration / 1e9))
    metrics.increment("questions_answered")


def ask_llm(prompt: str) -> str:
    """Raw (non-RAG) passthrough to the local LLM; kept for sanity checks."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except OLLAMA_ERRORS as exc:
        return f"{OLLAMA_HELP}\n\nDetails: {exc}"
