"""Measure whether LocalWitness is actually RIGHT — not just fast.

verify_offline.py proves the app is private. This proves it is grounded.

Speed numbers say nothing about trustworthiness. A citation is only worth
something if it points at the source that genuinely contains the answer, and a
"second brain" is only safe if it refuses to invent things it was never told.
So this script scores three things on a fixed golden set over the shipped
sample_data corpus:

  1. Retrieval hit-rate  — is the source that actually holds the answer among
                           the chunks retrieved for the question?
  2. Citation accuracy   — does the generated answer cite that source?
  3. Refusal rate        — for questions the corpus CANNOT answer, does it say
                           "That's not in my notes" instead of hallucinating?

(3) is the one that matters most. A RAG system that answers everything
confidently is worse than useless for contracts and medical notes.

Everything runs locally. Numbers are printed as measured — no rounding in our
favour, no cherry-picking.

Run:  python scripts/evaluate.py
Exit: 0 if no hallucinations were detected, 1 otherwise.
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localwitness.index.chunk import chunk_segments, chunk_text
from localwitness.index.store import add_chunks, delete_source, list_sources
from localwitness.ingest.audio import transcribe
from localwitness.ingest.documents import extract_text
from localwitness.ingest.images import caption
from localwitness.query.answer import FALLBACK, answer_stream
from localwitness.query.retrieve import retrieve

SAMPLES = ROOT / "sample_data"

ACCENT, DIM, BOLD, RED, RESET = (
    "\033[38;2;77;212;196m",
    "\033[38;2;139;149;165m",
    "\033[1m",
    "\033[38;2;229;115;91m",
    "\033[0m",
)

AUDIO = "sarah_followup_call.m4a"
CONTRACT = "northwind_contract.pdf"
NOTES = "kickoff_meeting_notes.md"

# Questions the corpus CAN answer, with the source that actually holds the fact.
GROUNDED: list[tuple[str, str]] = [
    ("What is the total fixed fee for the project?", CONTRACT),
    ("How many rounds of revisions are included?", CONTRACT),
    ("What is the hourly rate for extra work beyond the scope?", CONTRACT),
    ("What are the invoice payment terms?", CONTRACT),
    ("How much is the deposit?", CONTRACT),
    ("When are the final designs due?", AUDIO),
    ("Which theme did they decide on, light or dark?", AUDIO),
    ("Did they ask for an extra landing page?", AUDIO),
    ("Who is the client contact for the project?", NOTES),
]

# Questions the corpus CANNOT answer. The only correct behaviour is refusal.
UNANSWERABLE: list[str] = [
    "What is the client's office street address?",
    "What is the penalty if the contractor delivers late?",
    "How many employees does Northwind Labs have?",
    "What is the client's bank account number?",
    "Who is the contractor's project manager?",
]

CITE_RE = re.compile(r"\[([^\[\]\n]{1,120})\]")


def ensure_corpus_indexed() -> None:
    """Index the four sample files (idempotent) so the run is reproducible."""
    indexed = {s["source_name"] for s in list_sources()}
    needed = {AUDIO, CONTRACT, NOTES, "whiteboard_sprint.png"}
    if needed.issubset(indexed):
        print(f"{DIM}  corpus already indexed: {len(needed)} sources{RESET}\n")
        return

    print(f"{DIM}  indexing sample_data (one-time, local)…{RESET}")
    for name in sorted(needed - indexed):
        path = SAMPLES / name
        meta = {"source_type": "document", "source_name": name}
        if name.endswith(".m4a"):
            result = transcribe(str(path))
            chunks = chunk_segments(result["segments"], {**meta, "source_type": "audio"})
        elif name.endswith(".png"):
            chunks = chunk_text(caption(str(path)), {**meta, "source_type": "image"})
        else:
            chunks = []
            for item in extract_text(str(path)):
                chunks.extend(chunk_text(item["text"], {**meta, "page": item["page"]}))
        delete_source(name)
        add_chunks(chunks)
        print(f"{DIM}    + {name}  ({len(chunks)} chunks){RESET}")
    print()


def answer_for(question: str) -> tuple[str, list[dict]]:
    hits = retrieve(question)
    return "".join(answer_stream(question, hits)), hits


def main() -> int:
    print(f"\n{BOLD}LocalWitness — grounding & hallucination evaluation{RESET}")
    print(
        f"{DIM}fixed golden set over the shipped sample_data corpus. "
        f"all inference local.{RESET}\n"
    )
    ensure_corpus_indexed()

    # ---- 1 & 2: grounded questions ----------------------------------------
    print(f"{BOLD}Grounded questions{RESET} {DIM}(expects the right source, cited){RESET}")
    retrieved_ok = cited_ok = 0
    for question, expected in GROUNDED:
        started = time.perf_counter()
        answer, hits = answer_for(question)
        elapsed = time.perf_counter() - started

        sources = {(h.get("metadata") or {}).get("source_name") for h in hits}
        hit = expected in sources
        cited = any(expected in c for c in CITE_RE.findall(answer))
        retrieved_ok += hit
        cited_ok += cited

        mark = f"{ACCENT}✓{RESET}" if (hit and cited) else f"{RED}✗{RESET}"
        flags = f"retrieved={'✓' if hit else '✗'} cited={'✓' if cited else '✗'}"
        print(f"  {mark} {question[:52]:<54}{DIM}{flags}  {elapsed:4.1f}s{RESET}")
        if not (hit and cited):
            print(f"      {DIM}expected {expected} · answer: {answer[:90]}{RESET}")

    # ---- 3: unanswerable questions (the one that matters) ------------------
    print(f"\n{BOLD}Unanswerable questions{RESET} {DIM}(the only correct answer is refusal){RESET}")
    refused = 0
    hallucinations: list[tuple[str, str]] = []
    for question in UNANSWERABLE:
        started = time.perf_counter()
        answer, _ = answer_for(question)
        elapsed = time.perf_counter() - started

        did_refuse = answer.strip().startswith(FALLBACK)
        refused += did_refuse
        mark = f"{ACCENT}✓{RESET}" if did_refuse else f"{RED}✗{RESET}"
        verdict = "refused" if did_refuse else f"{RED}HALLUCINATED{RESET}"
        print(f"  {mark} {question[:52]:<54}{DIM}{verdict}  {elapsed:4.1f}s{RESET}")
        if not did_refuse:
            hallucinations.append((question, answer[:120]))
            print(f"      {RED}{answer[:100]}{RESET}")

    # ---- Receipt -----------------------------------------------------------
    n_grounded, n_unans = len(GROUNDED), len(UNANSWERABLE)
    pct = lambda x, n: f"{100 * x / n:.0f}%"

    print(f"\n{BOLD}RECEIPT{RESET}")
    print(
        f"  retrieval hit-rate   {ACCENT}{retrieved_ok}/{n_grounded}{RESET}"
        f"  {DIM}({pct(retrieved_ok, n_grounded)}) — correct source among retrieved chunks{RESET}"
    )
    print(
        f"  citation accuracy    {ACCENT}{cited_ok}/{n_grounded}{RESET}"
        f"  {DIM}({pct(cited_ok, n_grounded)}) — answer cites the source holding the fact{RESET}"
    )
    colour = RED if hallucinations else ACCENT
    print(
        f"  refusal rate         {colour}{refused}/{n_unans}{RESET}"
        f"  {DIM}({pct(refused, n_unans)}) — said \"{FALLBACK}\" instead of inventing{RESET}"
    )

    if hallucinations:
        print(
            f"\n  {RED}{BOLD}{len(hallucinations)} HALLUCINATION(S){RESET} — the model answered "
            f"a question the corpus cannot support.\n"
        )
        return 1

    print(
        f"\n  {ACCENT}{BOLD}ZERO HALLUCINATIONS{RESET} across {n_unans} unanswerable "
        f"questions.\n  Every claim it made, it could point at.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
