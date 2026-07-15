# Technical Report — LocalWitness

All figures below are **measured on the tested device**, not estimated. Anyone
can reproduce them: `scripts/verify_offline.py`, `scripts/evaluate.py`, and the
in-app **Metrics** tab re-measure live.

## Tested device

| | |
|---|---|
| Machine | Apple Silicon MacBook Air |
| Chip | Apple Silicon (M-series), 10-core |
| Memory | 16 GB unified |
| OS | macOS 26.5 |
| Python | 3.11.0 |
| Accelerators used | CPU + Apple GPU (Metal). **No NPU, no CUDA, no discrete GPU, no cloud.** |

LocalWitness is device-agnostic: anything that runs Python + Ollama can run it.
The numbers below are from the machine above.

## Models, runtimes, sizes

| Stage | Model | Format / runtime | On-disk size | Processor |
|---|---|---|---|---|
| Speech-to-text | faster-whisper `base` | INT8, CTranslate2 | ~145 MB | CPU |
| Embeddings | all-MiniLM-L6-v2 | **INT8 ONNX** | **~23 MB** (fp32: 90 MB) | CPU |
| Vector search | ChromaDB (HNSW) | library, on-disk | n/a | CPU |
| Image captioning | Moondream2 | Ollama / llama.cpp | ~1.3 GB (GPU-resident) | Apple GPU (Metal) |
| Answering | qwen2.5:3b | **INT4 GGUF**, Ollama | ~2.2 GB (GPU-resident) | Apple GPU (Metal) |
| PII redaction | Presidio + spaCy `en_core_web_lg` | spaCy NER + rules | ~590 MB | CPU |
| Privacy blur | YOLOv8n | Ultralytics + OpenCV | ~6 MB | CPU |

`ollama ps` confirms Qwen2.5 and Moondream run **100% on the GPU** (Metal).

## Quantization / optimization

`scripts/quantize_embeddings.py` exports the embedding model to ONNX and applies
dynamic INT8 quantization. Measured, single-query:

| all-MiniLM-L6-v2 | Size | Latency |
|---|---|---|
| PyTorch (fp32) | 91.6 MB | 4.7 ms |
| **ONNX INT8** | **22.9 MB** | **1.4 ms** |
| **Improvement** | **4.0× smaller** | **3.5× faster** |

INT8 is lossy (~0.97 cosine to fp32), but because the app embeds both index and
queries with the same backend, retrieval ranking is unaffected. The LLM is
likewise INT4-quantized GGUF (~6 GB fp16 → ~2.2 GB on disk).

## Inference latency (measured, models warm)

| Operation | Time |
|---|---|
| Speech-to-text (53 s of audio) | ~2.8 s |
| Document extraction (PDF) | <0.1 s |
| Image captioning (2 VLM passes) | ~1.8–4.5 s |
| Embed one chunk (INT8) | ~1.4–3 ms |
| Retrieval (incl. query embedding) | ~0.3 s |
| **Question → cited answer (end-to-end)** | **~3 s** |
| LLM throughput | ~46 tokens/sec |
| PII redaction | ~1.5–3 s (first call loads spaCy) |

## Compute & memory usage

- **CPU:** Whisper (CTranslate2 INT8), MiniLM embeddings (ONNX INT8), ChromaDB,
  Presidio/spaCy, YOLOv8n.
- **Apple GPU (Metal):** Qwen2.5:3b and Moondream2 via Ollama (llama.cpp), 100% GPU.
- **NPU (Apple Neural Engine):** not used by these runtimes.
- **Peak memory, app process** (full pipeline in `verify_offline.py`): **~837 MB
  RSS**. The LLM/VLM live in the separate Ollama process (~2.2 GB Qwen + ~1.3 GB
  Moondream, GPU-resident, loaded on demand). Total working set comfortably fits
  16 GB; the app alone runs in well under 1 GB.

## Evaluation

Method: a fixed golden set over the shipped `sample_data` corpus —
9 questions the corpus **can** answer (each paired with the source that holds the
fact) and 5 it **cannot**. Run: `python scripts/evaluate.py`.

| Metric | Result |
|---|---|
| Retrieval hit-rate | **9/9 (100%)** — correct source among retrieved chunks |
| Citation accuracy | **7/9 (78%)** — answer cites the source holding the fact |
| Refusal rate | **5/5 (100%)** — refused every unanswerable question |
| Hallucinations | **0** |

**Baseline / what "good" means here.** A conventional RAG chatbot answers ~100%
of questions — including the ones it *cannot* support — by design. Against that
baseline, LocalWitness deliberately trades coverage for trust: its failure mode
is **silence, not fabrication**.

**Known failure cases (both are refusals, never wrong answers):**
- *"How many rounds of revisions are included?"* — the source sentence *"Two
  rounds of design revisions are includ|ed in the base fee"* is split across a
  chunk boundary, so the retrieved half omits the word "two"; the model declines
  rather than guess.
- *"Did they ask for an extra landing page?"* — the sources genuinely conflict
  (notes say *"a maybe"*, the later voice note says *"after all"*); it declines
  rather than pick a side.

Both point at the same future-scope fix: sentence-aware chunk boundaries.

## Reproducibility

```bash
python -m spacy download en_core_web_lg
ollama pull qwen2.5:3b && ollama pull moondream
python scripts/verify_offline.py   # → VERDICT: PASS, zero outbound
python scripts/evaluate.py         # → 9/9 retrieval, 5/5 refusal, 0 hallucinations
streamlit run app.py
```
