"""YOLOv8n person detection + Gaussian blur for privacy-safe export. [Phase 2]"""

import os
import time
from pathlib import Path

import cv2

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
WEIGHTS = MODELS_DIR / "yolov8n.pt"

# Privacy: Ultralytics probes Cloudflare/Google DNS (1.1.1.1, 8.8.8.8) at
# IMPORT time to set an ONLINE flag — before you call a single function.
# YOLO_OFFLINE is the only way to stop it and must be set before the import
# below, so the order here is load-bearing.
#
# Offline-first, not offline-only: YOLO_OFFLINE also blocks weight downloads,
# so we set it only once the weights are cached. A fresh clone is allowed the
# documented one-time fetch; every run after that touches no network at all.
if WEIGHTS.exists():
    os.environ.setdefault("YOLO_OFFLINE", "True")

from ultralytics import YOLO, settings  # noqa: E402

from localwitness import metrics  # noqa: E402

# Privacy: Ultralytics ships with usage analytics ("sync") on — never allowed here.
settings.update({"sync": False})
PERSON_CLASS = 0  # COCO class id for "person"
# Deliberately low: for a privacy blur, a false positive (blurring a bit too
# much) is far cheaper than a miss (exporting a recognizable bystander).
CONFIDENCE = 0.2

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        start = time.perf_counter()
        # Weights live in models/ (gitignored); downloaded there on first use.
        _model = YOLO(str(MODELS_DIR / "yolov8n.pt"))
        metrics.record_timing("yolo_load_s", time.perf_counter() - start)
    return _model


def blur_people(src_path: str, dst_path: str) -> int:
    """Write a copy of src with every detected person Gaussian-blurred.

    Returns the number of blurred regions.
    """
    model = _get_model()
    start = time.perf_counter()
    image = cv2.imread(str(src_path))
    if image is None:
        raise ValueError(f"Could not read image: {src_path}")
    results = model.predict(
        image, classes=[PERSON_CLASS], conf=CONFIDENCE, verbose=False
    )
    blurred = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        region = image[y1:y2, x1:x2]
        if region.size:
            # Kernel scales with region size (and must be odd) so large
            # subjects are as unrecognizable as small ones.
            kernel = max(31, ((x2 - x1) // 3) | 1)
            image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
            blurred += 1
    cv2.imwrite(str(dst_path), image)
    metrics.record_timing("blur_ms_per_image", (time.perf_counter() - start) * 1000)
    metrics.increment("people_blurred", blurred)
    return blurred
