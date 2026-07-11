"""Image captioning with Moondream via Ollama — fully local. [Phase 2]"""

import time

import ollama

from localwitness import metrics

VISION_MODEL = "moondream"

DESCRIBE_PROMPT = "Describe this image in detail."
TRANSCRIBE_PROMPT = "Transcribe all text visible in this image, line by line."


def _ask(prompt: str, path: str) -> str:
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [str(path)]}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def caption(path: str) -> str:
    """Describe an image and transcribe its visible text with the local VLM.

    Two passes: Moondream surfaces far more in-image text when asked to
    transcribe explicitly than when asked to describe-and-transcribe at once.
    """
    start = time.perf_counter()
    description = _ask(DESCRIBE_PROMPT, path)
    visible_text = _ask(TRANSCRIBE_PROMPT, path)
    metrics.record_timing("caption_ms_per_image", (time.perf_counter() - start) * 1000)
    metrics.increment("images_captioned")
    return f"{description}\n\nVisible text: {visible_text}"
