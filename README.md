# Keptra

## Short description

## Problem

## Solution

## On-Device AI usage

**All inference runs locally on the user's own device — no cloud, no GPU
server.** The whole AI core (speech-to-text, embeddings, vector search, LLM
answering) works with Wi-Fi off. Nothing ever leaves the machine: no cloud AI
APIs, ChromaDB telemetry is explicitly disabled, and the HuggingFace hub is
only touched once to download weights (loads are offline-first afterwards).

Keptra is cross-platform: anything that runs Python + Ollama can run it.

**Tested on:** Apple Silicon (M-series MacBook Air), CPU/MPS/Metal — the
numbers below are from that machine:

| Stage | Model | Runtime | License | Approx size | Measured |
|---|---|---|---|---|---|
| Speech-to-text | [faster-whisper `base`](https://github.com/SYSTRAN/faster-whisper) (int8) | CTranslate2 | MIT | ~145 MB | ~5–7 s per minute of audio |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim) | PyTorch on MPS | Apache-2.0 | ~90 MB | ~90 ms per chunk |
| Vector search | [ChromaDB](https://github.com/chroma-core/chroma) (persistent, HNSW) | local, on-disk | Apache-2.0 | n/a (library) | ~330 ms per query (incl. query embedding) |
| Answering | [qwen2.5:3b](https://ollama.com/library/qwen2.5) (INT4 GGUF) via Ollama | llama.cpp / Metal | Apache-2.0 | ~1.9 GB | ~46 tokens/sec, 1–3.5 s per answer |

The in-app **Metrics tab** measures all of these live on whatever machine the
app is running on.

## Tech stack

## Setup

Prereqs: Python 3.11 and [Ollama](https://ollama.com) installed and running.

```bash
git clone <your-repo> && cd keptra
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b      # local LLM for cited answers
ollama pull moondream       # local VLM for image captioning
streamlit run app.py
```

## Usage

## Demo

## Screenshots

## License

## Limitations & future scope

## Credits
