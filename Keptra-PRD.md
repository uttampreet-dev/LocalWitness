# Keptra — Product Requirements & Build Blueprint

> **Working name:** Keptra *(verify it's unused on GitHub / npm / domain / Google before committing — see naming note. If taken, find-and-replace "Keptra" everywhere.)*
>
> **Tagline:** *Everything you kept, recalled — entirely on your device.*
>
> **Event:** OSDHack 2026 · Theme: On-Device AI · Solo build · MacBook Air (Apple Silicon) · 5-day window (10–15 July 2026)

---

## 1. One-line pitch

Keptra is a **private, fully offline "second brain."** You drop in voice notes, documents, and photos; Keptra transcribes, describes, and indexes them locally, then lets you **ask questions in natural language and get answers with citations** — with Wi-Fi turned off. Nothing ever leaves your machine.

## 2. Why it matters (the "problem" for the README)

People accumulate voice memos, PDFs, screenshots, and notes containing sensitive personal, medical, legal, or journalistic information. Cloud "second brain" and AI-search tools require uploading all of it to a server — an unacceptable privacy trade-off for exactly the people who need search most (journalists protecting sources, clinicians, researchers, lawyers, ordinary privacy-conscious users). **Keptra makes on-device the whole point, not a feature:** the AI that transcribes, understands, and answers runs locally, so private data never has to be trusted to anyone.

This framing hits three things OSDHack rewards at once: **privacy-focused**, **offline-first**, and **genuinely useful**.

## 3. The On-Device rule — honest mapping

Per the rulebook, the **core AI must run locally**; cloud is allowed only for support features. In Keptra:

| Runs locally (the AI core) | Allowed cloud/support (optional, off by default) |
|---|---|
| Speech-to-text (Whisper) | Nothing required. Optionally: encrypted backup/export to a folder or drive |
| Embeddings + semantic search | — |
| LLM question-answering (Ollama) | — |
| Image captioning, object detection, PII redaction | — |

**Demo proof:** the money shot is turning Wi-Fi **off** on camera and running a full transcribe → index → ask → cited-answer flow. Also show the OS network monitor / an empty request log.

## 4. Target user & primary use narrative

Build the entire demo around **one story**, not a feature menu:

> "I recorded a 4-minute interview and photographed two documents. Offline, I ask Keptra: *'What did she say the deadline was, and which document mentions the payment?'* — and it answers in plain language, citing the exact voice note (with timestamp) and the exact document."

Every model added must serve *this* story.

## 5. Architecture

```
                ┌────────────────────────────────────────────┐
                │              Streamlit UI (local)           │
                │  Upload · Library · Search · Ask · Metrics  │
                └───────────────┬────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────────────┐
        │ INGEST                 │ QUERY                          │
        ▼                        ▼                                │
  ┌───────────┐          ┌──────────────┐   ┌──────────────────┐ │
  │ Audio →   │          │ user question│──▶│ embed question   │ │
  │ Whisper   │          └──────────────┘   │ (MiniLM/BGE)     │ │
  │ (faster-  │                              └─────────┬────────┘ │
  │  whisper) │                                        ▼          │
  └─────┬─────┘                              ┌──────────────────┐ │
        │        ┌───────────┐               │ vector search    │ │
  ┌─────▼─────┐  │ Docs →    │               │ (Chroma, local)  │ │
  │ chunk +   │  │ text/PDF  │──┐            └─────────┬────────┘ │
  │ embed     │◀─┤ parser    │  │                      ▼          │
  │ (MiniLM)  │  └───────────┘  │            ┌──────────────────┐ │
  └─────┬─────┘                 │            │ top-k chunks     │ │
        │        ┌───────────┐  │            └─────────┬────────┘ │
        │        │ Images →  │  │                      ▼          │
        │        │ caption   │──┘            ┌──────────────────┐ │
        │        │ (Moondream)│              │ LLM answer w/    │ │
        ▼        └───────────┘               │ citations        │ │
  ┌──────────────────────────┐               │ (Ollama qwen2.5) │ │
  │ Chroma persistent store   │◀─────────────┴──────────────────┘ │
  │ (vectors + metadata +     │                                   │
  │  source file + timestamp) │                                   │
  └──────────────────────────┘                                   │
        ▲                                                          │
        │  Phase-2 privacy layer on export:                        │
        │  YOLOv8n face-blur  ·  Presidio PII redaction ───────────┘
```

## 6. Tech stack (Apple-Silicon-specific, all local & open source)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| UI | **Streamlit** | Renders in a browser tab → clean screenshots + demo video, but 100% local |
| Speech-to-text | **faster-whisper** (`base` or `small`) | Fast on Apple Silicon (CTranslate2). Alt: `whisper.cpp` w/ Metal |
| Embeddings | **sentence-transformers** — `all-MiniLM-L6-v2` (or `bge-small-en-v1.5`) | Runs on MPS/CPU |
| Vector store | **ChromaDB** (persistent) | Simplest local store w/ metadata. Alt: FAISS |
| Local LLM | **Ollama** + `qwen2.5:3b` (INT4 GGUF) | Alt: `gemma2:2b`, `llama3.2:3b`. Metal-accelerated |
| PDF/text parsing | `pypdf` / `python-docx` | |
| **[P2]** Image caption/VLM | **Moondream2** via `ollama pull moondream` | Makes images searchable; light OCR |
| **[P2]** Object detection | **Ultralytics YOLOv8n** | Face/person detect → auto-blur on export; **fine-tune hook** |
| **[P2]** PII redaction | **Microsoft Presidio** (local, spaCy) | Names/emails/phones/IDs. Alt: `dslim/bert-base-NER` |
| License | **MIT** or **Apache-2.0** (OSI) | Add `LICENSE` file at repo root |

**[P2] = Phase 2 bonus. Only build after Phase 1 is flawless.**

## 7. Data model (Chroma document metadata)

Each indexed chunk stores:
```json
{
  "id": "uuid",
  "text": "chunk text",
  "source_type": "audio | document | image",
  "source_name": "interview_01.m4a",
  "timestamp": "00:01:24",        // audio only
  "page": 3,                       // document only
  "created_at": "2026-07-11T10:22:00",
  "embedding": [ ... ]
}
```
Citations in answers are built from `source_name` + `timestamp`/`page`.

## 8. Feature scope — phased, with hard checkpoints

### Phase 1 — CORE (must ship; this alone is a winning submission)
Models used: **Whisper + Embeddings + LLM (3).** Covers audio + embeddings + local LLM + quantization + RAG — already most of the Resource Guide.

- [ ] Project scaffold, venv, requirements, README stub, LICENSE
- [ ] Upload audio → transcribe with Whisper → show transcript
- [ ] Upload PDF/txt/md → extract text
- [ ] Chunk + embed all content → persist in Chroma
- [ ] Library view: list everything indexed
- [ ] Semantic search box → returns matching chunks with source
- [ ] Ask panel: question → retrieve top-k → Ollama RAG answer **with citations**
- [ ] Metrics panel (model names, sizes, tokens/sec, ms/query, # items) — start it early
- [ ] Offline verification documented

**CHECKPOINT — end of Day 3:** If the full transcribe → index → ask → cited-answer loop works, **you have a submittable project. Freeze it on a git tag `v1-core`. Everything after is bonus.**

### Phase 2 — BONUS (only if ahead; each on its own commit, never breaking v1)
Add in this priority order; stop whenever time runs low:

- [ ] **Model 4 — Moondream2:** caption uploaded images → index captions so photos are searchable. *(Adds the vision modality — highest value bonus.)*
- [ ] **Model 5 — YOLOv8n:** detect faces/people in images; **auto-blur on export**. Optional: fine-tune YOLOv8n on ~200–400 images of one custom class (e.g. ID cards) and **report before/after mAP** — this is your headline fine-tuning story.
- [ ] **Model 6 — Presidio:** PII redaction pass on exported transcripts/answers (names, emails, phones, IDs).
- [ ] **Optimization report:** quantize embedding or NER model to ONNX INT8; document footprint reduction.

**CHECKPOINT — Day 4 evening:** Stop adding models. Whatever works, works. Spend Day 5 on polish, README, and the demo video.

## 9. The rubric-winning layer (do NOT skip — most teams will)

The Resource Guide practically hands you the scoring criteria. Surface these **in the app (a "Metrics" tab) and in the README**:

- Model name, source link, license, file format, approx size for **every** model.
- Before/after size for anything quantized/fine-tuned (e.g. "base ~3 GB → INT4 ~2 GB").
- Live performance numbers: **tokens/sec** (LLM), **ms/query** (retrieval), **s per minute of audio** (Whisper), **ms/image** (caption/detect).
- Target hardware line: "Runs locally on Apple Silicon (M-series), CPU/Metal, no GPU server."
- "# items indexed, fully offline" counter.

## 10. Repository structure

```
keptra/
├── app.py                    # Streamlit entry
├── keptra/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── audio.py          # Whisper transcription
│   │   ├── documents.py      # PDF/text parsing
│   │   └── images.py         # [P2] Moondream captioning
│   ├── index/
│   │   ├── chunk.py
│   │   ├── embed.py          # sentence-transformers
│   │   └── store.py          # Chroma wrapper
│   ├── query/
│   │   ├── retrieve.py
│   │   └── answer.py         # Ollama RAG + citations
│   ├── privacy/
│   │   ├── redact.py         # [P2] Presidio PII
│   │   └── blur.py           # [P2] YOLOv8n face blur
│   └── metrics.py            # timing + model info
├── models/                   # (gitignored weights, documented in README)
├── sample_data/              # a few demo audio/docs/images
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 11. Setup (what goes in README so judges can run it)

```bash
# Prereqs: Python 3.11, Ollama (https://ollama.com)
git clone <your-repo> && cd keptra
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b          # local LLM
# ollama pull moondream         # [P2] image captioning
streamlit run app.py
```

## 12. Licensing & fair-play compliance checklist

- [ ] OSI license (MIT/Apache-2.0) + visible `LICENSE` file
- [ ] Public repo at submission
- [ ] Attribution for every model/library used (README "Credits" section w/ licenses)
- [ ] All commits inside the 10–15 July window; commit **often** so history proves it was built during the event
- [ ] No copied/pre-built project; scaffold from empty on Day 1
- [ ] Responsible-use: privacy tool built for defensive/beneficial purposes — state this explicitly

## 13. Submission-checklist mapping (straight from the rulebook)

| Required item | How Keptra satisfies it |
|---|---|
| Public repo + source | GitHub, MIT-licensed |
| README (all 12 checklist fields) | Section 14 below is the outline |
| Demo video (2–5 min) | Script in Section 15 |
| Screenshots | Upload, Library, Search, Ask-with-citations, Metrics tab |
| On-Device AI explanation | Section 3 table + Metrics tab |
| "How others can run it" | Section 11 setup block |

## 14. README outline (fill these — matches rulebook Appendix B)

Project name · Short description · Problem · Solution · **On-Device AI usage** (what runs locally, which model/runtime/device) · Tech stack · Setup · Usage · Demo video link · Screenshots · License · Known limitations & future scope · Credits/attribution.

## 15. Demo video script (2–5 min, single narrative)

1. **(0:00–0:20)** Hook: "This is a second brain that never sends your data anywhere. Watch — I'm turning Wi-Fi off now." *(Show it go off.)*
2. **(0:20–1:00)** Upload a voice note → Whisper transcribes it live. Upload two documents.
3. **(1:00–1:40)** Show the Library + semantic search: type a vague query, get the right chunk.
4. **(1:40–2:40)** The Ask panel: ask the cross-source question → answer appears **with citations** (voice note @ timestamp + document @ page). This is the wow moment.
5. **(2:40–3:20)** *(If P2 built)* Upload a photo → auto-caption makes it searchable → export with faces blurred + PII redacted.
6. **(3:20–4:00)** Metrics tab: model names, sizes, tokens/sec, ms/query. "All of this, offline, on a MacBook Air, no GPU server." Close.

## 16. Risks & escape hatches

- **Ollama not installed on judge's machine** → README makes it step 1; also record the video so judging never depends on their setup.
- **Whisper slow on long audio** → default to `base` model; cap demo clips at ~4 min.
- **RAG answer ungrounded/hallucinated** → strong system prompt: "Answer ONLY from provided context; cite sources; say 'not in my notes' if absent." Show retrieved chunks alongside the answer for transparency.
- **Phase 2 eating Phase 1** → the `v1-core` git tag is your safety net; you can always ship that.
- **Time collapse** → Phase 1 minus images is still a complete, coherent, winning submission. Protect it.

## 17. Day-by-day plan (solo + Claude Code)

- **Day 1 (10 Jul, eve):** Scaffold, venv, Streamlit skeleton, Ollama working, LICENSE, README stub, first commits.
- **Day 2:** Audio→Whisper→transcript; docs→text; chunk+embed+Chroma; Library view.
- **Day 3:** Semantic search; RAG Ask panel with citations; Metrics panel. **→ tag `v1-core`.**
- **Day 4:** Phase 2 in priority order (Moondream → YOLO blur → Presidio), each its own commit. Stop by evening.
- **Day 5:** Polish UI, write full README, record + edit demo video, take screenshots, final submit on Unstop well before 6 PM IST.

---

*Build one story flawlessly. Report the numbers nobody else reports. Turn Wi-Fi off on camera.*
