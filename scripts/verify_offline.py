"""Prove the core claim: LocalWitness makes zero outbound network connections.

LocalWitness tells you never to trust an answer without a receipt. This script
is the project's own receipt.

It blocks every non-loopback socket connection at the Python level, then runs
the FULL real pipeline — speech-to-text, image captioning, document
extraction, embedding, indexing, retrieval, LLM answering, PII redaction, and
privacy blur — using the actual application code. If any model or dependency
tries to reach the internet, the attempt is recorded and the run FAILS.

Loopback (127.0.0.1) is deliberately allowed: the local LLM and VLM are served
by Ollama on 127.0.0.1:11434. That traffic never leaves the machine, and we
count it openly rather than pretending it doesn't exist.

Run:  python scripts/verify_offline.py
Exit: 0 = no outbound connections, 1 = a leak was detected.
"""

import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "sample_data"
AUDIO = SAMPLES / "sarah_followup_call.m4a"
IMAGE = SAMPLES / "whiteboard_sprint.png"
DOCUMENT = SAMPLES / "northwind_contract.pdf"

# A source name used only by this script, deleted from the index afterwards so
# verification never pollutes the user's real library.
PROBE_SOURCE = "__offline_verification_probe__"

ACCENT, DIM, BOLD, RED, GREEN, RESET = (
    "\033[38;2;77;212;196m",
    "\033[38;2;139;149;165m",
    "\033[1m",
    "\033[38;2;229;115;91m",
    "\033[38;2;77;212;196m",
    "\033[0m",
)


class OutboundBlocked(RuntimeError):
    """Raised when code under verification tries to reach a non-loopback host."""


blocked_attempts: list[str] = []
loopback_connections: list[str] = []

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo
_real_create_connection = socket.create_connection


def _host_of(address) -> str:
    if isinstance(address, tuple):
        return str(address[0])
    return str(address)


def _is_loopback(host: str) -> bool:
    host = host.strip("[]").lower()
    return (
        host in {"localhost", "::1", "0.0.0.0", ""}
        or host.startswith("127.")
        or host.endswith(".localhost")
    )


def _guard(host: str, via: str) -> None:
    """Record the connection; raise if it would leave the machine."""
    if _is_loopback(host):
        loopback_connections.append(f"{host} ({via})")
        return
    attempt = f"{host} (via {via})"
    blocked_attempts.append(attempt)
    raise OutboundBlocked(f"outbound connection to {attempt} blocked")


def _install_network_block() -> None:
    """Block every non-loopback connection and name resolution."""

    def connect(self, address):
        _guard(_host_of(address), "socket.connect")
        return _real_connect(self, address)

    def connect_ex(self, address):
        _guard(_host_of(address), "socket.connect_ex")
        return _real_connect_ex(self, address)

    def create_connection(address, *args, **kwargs):
        _guard(_host_of(address), "socket.create_connection")
        return _real_create_connection(address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        # DNS resolution of a remote host is itself a packet leaving the box.
        _guard(str(host), "DNS/getaddrinfo")
        return _real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo


def _stage(index: int, total: int, name: str):
    print(f"  {DIM}[{index}/{total}]{RESET} {name:<38}", end="", flush=True)
    return time.perf_counter()


def _ok(started: float, detail: str = "") -> None:
    elapsed = time.perf_counter() - started
    print(f"{ACCENT}ok{RESET}   {DIM}{elapsed:5.1f}s  {detail}{RESET}")


def main() -> int:
    print(f"\n{BOLD}LocalWitness — offline verification{RESET}")
    print(
        f"{DIM}network policy: every non-loopback connection is BLOCKED at the\n"
        f"socket layer. Loopback is allowed and counted (Ollama serves the local\n"
        f"LLM/VLM on 127.0.0.1:11434 — that traffic never leaves the machine).{RESET}\n"
    )

    _install_network_block()

    # Imported *after* the block is installed, so even import-time phone-homes
    # are caught.
    from localwitness.index.chunk import chunk_text
    from localwitness.index.embed import embed
    from localwitness.index.store import add_chunks, delete_source, query
    from localwitness.ingest.audio import transcribe
    from localwitness.ingest.documents import extract_text
    from localwitness.ingest.images import caption
    from localwitness.privacy.blur import blur_people
    from localwitness.privacy.redact import redact
    from localwitness.query.answer import answer_stream
    from localwitness.query.retrieve import retrieve

    total = 9
    try:
        t = _stage(1, total, "Speech-to-text (faster-whisper)")
        result = transcribe(str(AUDIO))
        _ok(t, f"{len(result['segments'])} segments")

        t = _stage(2, total, "Document extraction (pypdf)")
        items = extract_text(str(DOCUMENT))
        _ok(t, f"{len(items)} page(s)")

        t = _stage(3, total, "Image captioning (Moondream/Ollama)")
        described = caption(str(IMAGE))
        _ok(t, f"{len(described)} chars")

        t = _stage(4, total, "Embeddings (MiniLM, INT8 ONNX)")
        vectors = embed([items[0]["text"][:400]])
        _ok(t, f"dim {len(vectors[0])}")

        t = _stage(5, total, "Indexing (ChromaDB, local)")
        chunks = chunk_text(
            items[0]["text"],
            {"source_type": "document", "source_name": PROBE_SOURCE},
        )
        added = add_chunks(chunks)
        _ok(t, f"{added} chunk(s)")

        t = _stage(6, total, "Retrieval (semantic search)")
        hits = retrieve("What is the payment schedule?")
        _ok(t, f"{len(hits)} hit(s)")

        t = _stage(7, total, "RAG answer (qwen2.5:3b / Ollama)")
        answer = "".join(answer_stream("What is the payment schedule?", hits))
        _ok(t, f"{len(answer)} chars")

        t = _stage(8, total, "PII redaction (Presidio + spaCy)")
        redacted = redact("Call Sarah Chen at sarah@northwind.com or 555-0142.")
        _ok(t, redacted[:34])

        t = _stage(9, total, "Privacy blur (YOLOv8n)")
        with tempfile.TemporaryDirectory() as tmp:
            regions = blur_people(str(IMAGE), str(Path(tmp) / "blurred.png"))
        _ok(t, f"{regions} region(s)")

    except OutboundBlocked as exc:
        print(f"{RED}LEAK{RESET}\n\n  {RED}{exc}{RESET}")
    except Exception as exc:  # a real failure, not a network leak
        print(f"{RED}error{RESET}\n\n  {RED}{type(exc).__name__}: {exc}{RESET}")
        return 1
    finally:
        try:
            delete_source(PROBE_SOURCE)  # never pollute the real library
        except Exception:
            pass

    # ---- Receipt -----------------------------------------------------------
    unique_loopback = sorted(set(loopback_connections))
    print(f"\n{BOLD}RECEIPT{RESET}")
    print(
        f"  loopback connections (allowed, on-device):  "
        f"{ACCENT}{len(loopback_connections)}{RESET} {DIM}→ {', '.join(unique_loopback) or 'none'}{RESET}"
    )
    print(
        f"  outbound connection attempts (non-loopback): "
        f"{RED if blocked_attempts else ACCENT}{len(blocked_attempts)}{RESET}"
    )
    for attempt in sorted(set(blocked_attempts)):
        print(f"      {RED}✗ {attempt}{RESET}")

    if blocked_attempts:
        print(
            f"\n  {RED}{BOLD}VERDICT: FAIL{RESET} — something tried to reach the "
            f"internet.\n"
        )
        return 1

    print(
        f"\n  {GREEN}{BOLD}VERDICT: PASS{RESET} — all six models ran, start to "
        f"finish,\n  with {BOLD}zero non-loopback connections{RESET}. "
        f"Nothing left this device.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
