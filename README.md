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

### Customization: fine-tuning YOLOv8n

The privacy-blur detector can be fine-tuned to protect *your* sensitive class
(e.g. ID cards, badges, license plates) with a few hundred images —
[scripts/finetune_yolo.py](scripts/finetune_yolo.py) is the whole loop:

```bash
# prove the loop end-to-end on a tiny synthetic dataset:
python scripts/finetune_yolo.py --make-sample-data 60
python scripts/finetune_yolo.py --epochs 10 --imgsz 320
```

Pretrained YOLOv8n knows 80 COCO classes and cannot detect a novel class at
all, so fine-tuning takes its mAP from **0.000** to a working detector:

| | mAP50 | mAP50-95 |
|---|---|---|
| Before (pretrained, class unknown) | 0.000 | 0.000 |
| After (fine-tuned, tiny sample run) | 0.995 | 0.949 |

*(Measured: 10 epochs, imgsz 320, on Apple Silicon MPS — a ~2-minute run on
the 60-image synthetic sample; exported weights are 6.2 MB.)*

For a real 200–400-image dataset, run the same script on Google Colab (free
T4 GPU), then copy the exported weights from `models/` back — where you train
doesn't matter, **inference stays 100% local**.

### Privacy receipts

We didn't just claim privacy — we audited the dependency chain for network
calls and closed every leak we found. Even our dependencies kept trying to
call home, and we shut every one of them off:

1. **ChromaDB** ships with anonymized telemetry enabled — disabled explicitly
   in our client settings ([keptra/index/store.py](keptra/index/store.py)).
2. **HuggingFace hub** revalidates cached models over the network on every
   load — loads are forced offline-first once weights are downloaded
   ([keptra/index/embed.py](keptra/index/embed.py)).
3. **Ultralytics** ships with usage analytics ("sync") enabled — disabled at
   import time ([keptra/privacy/blur.py](keptra/privacy/blur.py)).

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

- **Privacy blur has a detection floor:** tiny faces cropped at the image
  border can fall below what YOLOv8n (a nano model) can detect — one such
  bystander face survived blurring in our test photo. Review exports before
  sharing; a larger detection model would raise the floor at the cost of
  speed.

## Credits

Every model and library Keptra uses, with licenses:

- **Object detection uses [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics), licensed AGPL-3.0.**
  (Keptra's own code remains MIT — we use the AGPL library, we don't relicense it.)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) — speech-to-text, Whisper `base` weights by OpenAI (MIT)
- [sentence-transformers](https://www.sbert.net/) / [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0) — embeddings
- [ChromaDB](https://github.com/chroma-core/chroma) (Apache-2.0) — local vector store
- [Ollama](https://ollama.com) (MIT) with [Qwen2.5 3B](https://ollama.com/library/qwen2.5) (Apache-2.0) — local LLM
- [Moondream2](https://ollama.com/library/moondream) (Apache-2.0) — image captioning
- [Streamlit](https://streamlit.io) (Apache-2.0) — UI · [pypdf](https://github.com/py-pdf/pypdf) (BSD-3-Clause) · [python-docx](https://github.com/python-openxml/python-docx) (MIT) · [OpenCV](https://opencv.org) (Apache-2.0) · [NumPy](https://numpy.org) (BSD-3-Clause)
