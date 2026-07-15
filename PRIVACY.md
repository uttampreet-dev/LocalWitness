# Privacy & Safety — LocalWitness

LocalWitness is a privacy tool, so it is held to a privacy tool's standard: the
claims below are enforced by an executable test, not just stated.

## Data handling

- **Everything stays on the device.** Ingested audio, documents, and images are
  processed by local models only. No content — raw, embedded, redacted, or
  otherwise — is ever transmitted off the machine.
- **No cloud AI.** There is no API key anywhere in the codebase. No OpenAI,
  Anthropic, Gemini, or hosted-inference call exists in any code path.
- **Proven, not asserted.** [`scripts/verify_offline.py`](scripts/verify_offline.py)
  blocks every non-loopback socket and runs the full pipeline: **zero outbound
  connections**. The only sockets opened are loopback (`127.0.0.1:11434`, the
  local Ollama server).

## Permissions

- The app needs **local filesystem access** only: to read files you upload, to
  read/write the local index (`./chroma_db`) and the original-file vault
  (`./sources`), and to write model weights to `./models` on first run.
- **No network permission is required to operate.** After the one-time model
  download, it runs fully offline.
- No microphone/camera capture — you supply files; the app does not record.

## Storage

| What | Where | Notes |
|---|---|---|
| Vector index | `./chroma_db` | local, gitignored |
| Original uploads | `./sources` | local vault so citations can reopen sources; gitignored |
| Model weights | `./models` | downloaded once, gitignored |
| Privacy-safe exports | `./exports` | blurred image copies you explicitly export |

Nothing is encrypted at rest in this version (see risks). Everything lives under
your control on your disk and can be deleted at any time; the in-app **Reset
index** clears the vector store.

## Safety features (on-device)

- **PII redaction** — Presidio + spaCy detect and replace names, emails, phones,
  and IDs with typed tags (`[PERSON]`, `[EMAIL_ADDRESS]`, …) before an answer or
  transcript is shown. When redaction is on, the raw-excerpt evidence panel is
  deliberately hidden (excerpts can contain PII).
- **Privacy-safe image export** — YOLOv8n detects people and Gaussian-blurs them
  in exported copies.
- **Grounded answers** — the model refuses (*"That's not in my notes"*) rather
  than fabricate; measured zero hallucinations on the unanswerable test set.

## Limitations & potential risks (stated honestly)

- **Filenames are preserved for citation integrity.** A citation like
  `[sarah_followup_call.m4a @ 00:12]` still exposes a name through the *filename*
  even when the answer body is redacted. Rename sensitive files before indexing
  anything you plan to share redacted.
- **Redaction is model-based, not a guarantee.** Presidio catches common PII
  types well but is not infallible; review before sharing redacted output.
- **Blur has a detection floor.** Tiny faces at the image border can fall below
  what a nano detector (YOLOv8n) catches. Review exports before sharing.
- **No at-rest encryption yet.** The local index and vault are plaintext on disk;
  they are only as protected as the device itself. Encrypted local storage is
  named in future scope.
- **Small local model.** qwen2.5:3b is grounded by design but limited on complex
  multi-step reasoning versus large cloud models — the deliberate trade for full
  privacy.

## Responsible use

LocalWitness is a defensive, privacy-preserving tool: it exists so sensitive
personal, legal, medical, and journalistic material can be searched **without**
uploading it to anyone's cloud. It performs no surveillance, collects no
telemetry (third-party telemetry in dependencies is explicitly disabled), and
transmits nothing off the device.
