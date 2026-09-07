"""
classify_to_datasets.py
-----------------------
Classifies all existing images using the trained YOLO model
and saves them INTO datasets/ train/val/test folders IMMEDIATELY
as each image is processed (no buffering — you see files appear live).

Split:  70% train  |  20% val  |  10% test  (deterministic hash-based)
"""

import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
AI_DIR  = PROJECT / "ai"

DATASETS_DIR = PROJECT / "datasets"

BEST_PT  = PROJECT / "runs" / "detect" / "road_damage_ai" / "weights" / "best.pt"
NANO_PT  = PROJECT / "yolov8n.pt"
MODEL_PATH = str(BEST_PT) if BEST_PT.exists() else str(NANO_PT)

# All source folders to scan for images
SOURCE_DIRS = [
    AI_DIR / "dataset" / "merged_pool"    / "images",
    AI_DIR / "dataset" / "Data Y12 Final" / "train"  / "images",
    AI_DIR / "dataset" / "Data Y12 Final" / "valid"  / "images",
    AI_DIR / "dataset" / "kaggle_raw",
]

CONF_THRESH = 0.20
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = {0: "Crack", 1: "Pothole", 2: "Surface_Erosion"}


# ── Deterministic split assignment based on filename hash ─────────────────────
def get_split(path: Path) -> str:
    """
    Assign train/val/test deterministically from filename hash.
    No shuffling needed — reproducible across runs.
    70 / 20 / 10
    """
    h = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 100
    if   h < 70: return "train"
    elif h < 90: return "val"
    else:        return "test"


# ── Folder name safe for filesystem ──────────────────────────────────────────
def cls_folder(cls_id: int, img_path: Path) -> str:
    name = CLASS_NAMES.get(cls_id)
    if name:
        return name
    # Fallback: infer from folder name
    p = str(img_path).lower()
    if "pothole" in p: return "Pothole"
    if "crack"   in p: return "Crack"
    return "Surface_Erosion"


# ── Collect unique images ────────────────────────────────────────────────────
def collect(sources):
    seen, imgs = set(), []
    for src in sources:
        src = Path(src)
        if not src.exists():
            continue
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if p.name not in seen:
                    seen.add(p.name)
                    imgs.append(p)
    return imgs


# ── Ensure output dirs exist ──────────────────────────────────────────────────
def make_dirs():
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            (DATASETS_DIR / split / "images" / cls).mkdir(parents=True, exist_ok=True)
            (DATASETS_DIR / split / "labels" / cls).mkdir(parents=True, exist_ok=True)


# ── Save one image + label immediately ───────────────────────────────────────
def save_image(img_path: Path, cls_id: int, yolo_lines: list, counters: dict):
    split    = get_split(img_path)
    cls_name = cls_folder(cls_id, img_path)

    dest_img = DATASETS_DIR / split / "images" / cls_name / img_path.name
    dest_lbl = DATASETS_DIR / split / "labels" / cls_name / (img_path.stem + ".txt")

    # Handle filename collisions
    if dest_img.exists():
        stem = img_path.stem
        ext  = img_path.suffix.lower()
        n = 1
        while dest_img.exists():
            dest_img = DATASETS_DIR / split / "images" / cls_name / f"{stem}_{n}{ext}"
            dest_lbl = DATASETS_DIR / split / "labels" / cls_name / f"{stem}_{n}.txt"
            n += 1

    shutil.copy2(img_path, dest_img)
    dest_lbl.write_text("\n".join(yolo_lines))

    counters[split][cls_name] += 1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    make_dirs()

    print("=" * 60)
    print("  Smart Civic AI — Dataset Classifier (live save)")
    print(f"  Model : {Path(MODEL_PATH).name}")
    print(f"  Output: {DATASETS_DIR}")
    print("=" * 60)

    model  = YOLO(MODEL_PATH)
    images = collect(SOURCE_DIRS)
    print(f"\n[SCAN] {len(images):,} unique images found\n")

    if not images:
        print("[ERROR] No images found.")
        return

    counters = defaultdict(lambda: defaultdict(int))
    skipped  = 0

    for img_path in tqdm(images, desc="Classifying", unit="img"):
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=640,
                conf=CONF_THRESH,
                verbose=False
            )
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                # No detection — infer from folder name
                p = str(img_path).lower()
                if   "pothole" in p: cls_id = 1
                elif "crack"   in p: cls_id = 0
                elif "erosion" in p or "surface" in p: cls_id = 2
                else:
                    skipped += 1
                    continue
                yolo_lines = []
            else:
                # Most-frequent detected class wins
                from collections import Counter
                cls_counts = Counter(int(b.cls.item()) for b in r.boxes)
                cls_id     = cls_counts.most_common(1)[0][0]

                # Build YOLO label lines
                ih, iw = r.orig_shape
                yolo_lines = []
                for box in r.boxes:
                    cid  = int(box.cls.item())
                    conf = float(box.conf.item())
                    if conf < CONF_THRESH:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / iw
                    cy = ((y1 + y2) / 2) / ih
                    bw = (x2 - x1) / iw
                    bh = (y2 - y1) / ih
                    yolo_lines.append(
                        f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    )

            # ── Save immediately ──────────────────────────────────────────
            save_image(img_path, cls_id, yolo_lines, counters)

        except Exception as e:
            tqdm.write(f"[WARN] {img_path.name}: {e}")
            skipped += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  Classification complete!")
    print(f"  {'Split':<8} {'Class':<22} {'Images':>7}")
    print("  " + "-" * 40)
    grand = 0
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            n = counters[split][cls]
            if n:
                print(f"  {split:<8} {cls:<22} {n:>7,}")
                grand += n
    print(f"\n  Skipped (no class): {skipped:,}")
    print(f"  Grand total saved : {grand:,}")
    print(f"\n  Saved to: {DATASETS_DIR}")

    # Write data.yaml
    yaml = DATASETS_DIR / "data.yaml"
    yaml.write_text(
        f"path: {DATASETS_DIR.as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 3\nnames: ['Crack', 'Pothole', 'Surface_Erosion']\n"
    )
    print(f"  YAML : {yaml}")
    print("=" * 60)


if __name__ == "__main__":
    main()
