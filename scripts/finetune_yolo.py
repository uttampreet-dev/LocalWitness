"""Fine-tune YOLOv8n on a small custom single-class dataset (e.g. "id_card").

The "before/after mAP" story: pretrained YOLOv8n knows 80 COCO classes and
has never seen your custom class, so its before-mAP on it is 0.000 by
construction. A few epochs on a few hundred images teaches it the class;
this script measures and prints both numbers.

Usage:
  # 1) (optional) generate a tiny synthetic sample dataset to prove the loop
  python scripts/finetune_yolo.py --make-sample-data 60

  # 2) fine-tune + report before/after mAP
  python scripts/finetune_yolo.py --epochs 15

Runs on Apple Silicon (MPS) or CPU for small datasets. For a real 200-400
image dataset, run this same script on Google Colab (free T4 GPU), then copy
the exported weights from models/ back to your machine — training location
doesn't matter, inference stays 100% local.
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "idcard_sample"
MODELS_DIR = ROOT / "models"
CLASS_NAME = "id_card"


def make_sample_data(n: int, root: Path = DATASET_DIR) -> Path:
    """Generate a synthetic single-class detection dataset in YOLO format.

    Draws an ID-card-like rounded rectangle (text lines + photo box) at a
    random size/position over a noisy background, so the bounding box is
    exact by construction. Good enough to prove the training loop end to
    end; swap in real photos for a real model.
    """
    from PIL import Image, ImageDraw

    rng = random.Random(42)
    n_train = max(1, int(n * 0.8))
    for split, count in (("train", n_train), ("val", n - n_train)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            side = 640
            img = Image.new(
                "RGB",
                (side, side),
                (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)),
            )
            draw = ImageDraw.Draw(img)
            for _ in range(rng.randint(8, 20)):  # background clutter
                x, y = rng.randint(0, side), rng.randint(0, side)
                w, h = rng.randint(20, 160), rng.randint(20, 160)
                draw.rectangle(
                    [x, y, x + w, y + h],
                    fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)),
                )
            # the "ID card"
            cw = rng.randint(180, 360)
            ch = int(cw * 0.63)
            cx = rng.randint(0, side - cw)
            cy = rng.randint(0, side - ch)
            card = (rng.randint(230, 255),) * 3
            draw.rounded_rectangle(
                [cx, cy, cx + cw, cy + ch], radius=12, fill=card, outline=(60, 60, 60), width=3
            )
            draw.rectangle(  # photo box
                [cx + 12, cy + 12, cx + 12 + ch // 2, cy + 12 + ch // 2],
                fill=(120, 140, 170),
            )
            for line in range(3):  # text lines
                ly = cy + 16 + line * (ch // 5)
                draw.rectangle(
                    [cx + ch // 2 + 24, ly, cx + cw - 14, ly + ch // 12],
                    fill=(70, 70, 70),
                )
            img.save(img_dir / f"card_{i:03d}.jpg", quality=90)
            # YOLO label: class cx cy w h, normalized
            (lbl_dir / f"card_{i:03d}.txt").write_text(
                f"0 {(cx + cw / 2) / side:.6f} {(cy + ch / 2) / side:.6f} "
                f"{cw / side:.6f} {ch / side:.6f}\n"
            )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: {CLASS_NAME}\n"
    )
    print(f"Wrote {n} images to {root} (data: {data_yaml})")
    return data_yaml


def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-sample-data", type=int, metavar="N", default=0,
                        help="generate N synthetic sample images and exit")
    parser.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default=None, help="mps / cpu / 0 (default: auto)")
    args = parser.parse_args()

    if args.make_sample_data:
        make_sample_data(args.make_sample_data)
        return

    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no analytics, ever
    device = args.device or pick_device()
    base_weights = str(MODELS_DIR / "yolov8n.pt")

    # BEFORE: pretrained YOLOv8n has no notion of the custom class.
    print(f"BEFORE fine-tune: pretrained YOLOv8n has no '{CLASS_NAME}' class "
          "-> mAP50 = 0.000 on this dataset by construction.")

    model = YOLO(base_weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        project=str(ROOT / "runs"),
        name="finetune_idcard",
        exist_ok=True,
        verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"

    # AFTER: evaluate the fine-tuned weights on the val split.
    tuned = YOLO(str(best))
    metrics = tuned.val(data=args.data, device=device, verbose=False)

    exported = MODELS_DIR / f"yolov8n_{CLASS_NAME}.pt"
    shutil.copy(best, exported)

    print("\n================ RESULTS ================")
    print(f"device: {device} | epochs: {args.epochs} | imgsz: {args.imgsz}")
    print(f"BEFORE  mAP50: 0.000   mAP50-95: 0.000   (class unknown to COCO model)")
    print(f"AFTER   mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"exported weights: {exported} ({exported.stat().st_size / 1e6:.1f} MB)")
    print("Load locally with: YOLO('models/yolov8n_id_card.pt')")


if __name__ == "__main__":
    main()
