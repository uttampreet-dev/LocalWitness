"""Ollama RAG answering (raw LLM passthrough for now; RAG comes later)."""

import ollama

LLM_MODEL = "qwen2.5:3b"

OLLAMA_HELP = (
    "⚠️ Couldn't reach the local Ollama server. Make sure it is running "
    "(`ollama serve` or the Ollama menu-bar app) and the model is pulled "
    f"(`ollama pull {LLM_MODEL}`), then try again."
)


def ask_llm(prompt: str) -> str:
    """Send a prompt to the local Ollama model and return its reply."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as exc:
        return f"{OLLAMA_HELP}\n\nDetails: {exc}"
