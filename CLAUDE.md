# CLAUDE.md — Keptra

> Working name: **Keptra**. If renamed, this file's name stays `CLAUDE.md`; just update the product name below.

## What this project is
Keptra is a **private, fully offline, on-device multimodal "second brain."** A user drops in voice notes, documents, and images; the app transcribes, describes, and indexes them **locally**, then answers natural-language questions with **citations** — all with no internet required. Built solo for the OSDHack 2026 hackathon (theme: On-Device AI). Full spec is in `Keptra-PRD.md` — read it once at the start of a session if you need architecture context.

## THE ONE HARD RULE (non-negotiable)
**All AI inference must run locally.** Never add a call to a cloud AI/LLM API (OpenAI, Anthropic, Gemini, Hugging Face Inference API, etc.) for any core feature. The LLM is local via **Ollama**; speech-to-text, embeddings, captioning, detection, and PII all run on-device. If a task seems to need a cloud model, stop and flag it instead of adding it. Cloud is allowed ONLY for optional, non-AI support (e.g. saving an export to a folder) and is off by default.

## How we work (follow strictly)
1. **One step at a time.** I paste prompts from `Keptra-ClaudeCode-Prompts.md` in order. Do exactly the current step. Do NOT jump ahead, scaffold future features, or "helpfully" build the whole app at once.
2. **Verify, then commit.** After each step, make sure it runs, then `git add -A && git commit -m "<step>"`. Frequent commits are required — they prove the project was built during the event window.
3. **`v1-core` is sacred.** Once the transcribe → index → ask → cited-answer loop works (Phase 1), it gets tagged `v1-core`. After that, never break it. Phase-2 features go on separate commits; if one breaks the core and can't be fixed fast, revert it.
4. **Small and flawless beats big and broken.** Judging is on a recorded demo video + the repo. A working 3-model app is a winning submission; don't jeopardize it chasing bonus models.

## Tech stack (do not swap without asking)
- Python 3.11, **Streamlit** UI (renders in browser, runs locally)
- **faster-whisper** (`base`) — speech-to-text
- **sentence-transformers** `all-MiniLM-L6-v2` — embeddings
- **ChromaDB** (persistent, `./chroma_db`) — vector store
- **Ollama** + `qwen2.5:3b` — local LLM for RAG answers
- Phase 2 only: **Moondream** (via Ollama) captioning · **Ultralytics YOLOv8n** face-blur · **Presidio** PII redaction
- License: **MIT** (`LICENSE` at root, present from day 1)

## Code conventions
- Package code under `keptra/` with the structure in the PRD (`ingest/`, `index/`, `query/`, `privacy/`, `metrics.py`). `app.py` is the Streamlit entry.
- Lazy-load every model once (module-level cache); never reload per request.
- Record timings (whisper s/min, embed ms, retrieval ms/query, LLM tokens/sec) into `keptra/metrics.py` as we go — the Metrics tab is a scored differentiator.
- RAG answers must be grounded: instruct the LLM to answer only from retrieved context, cite `source_name` + timestamp/page, and say "That's not in my notes" if absent. Always show the retrieved chunks alongside the answer.
- Keep dependencies in `requirements.txt`, pinned. Gitignore `.venv/`, `models/`, `chroma_db/`, `__pycache__/`, `.DS_Store`.

## Environment
- macOS on Apple Silicon. Use MPS if available, else CPU. No GPU server, no CUDA.
- Ollama runs at `http://localhost:11434`; assume it's already running.

## When unsure
Ask a short clarifying question rather than guessing or expanding scope. Prefer the simplest thing that makes the current step work.