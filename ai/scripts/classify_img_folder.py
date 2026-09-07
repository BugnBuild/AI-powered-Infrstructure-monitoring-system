"""
classify_img_folder.py
-----------------------
Classifies all images from the img/ folder using YOLO,
saves them into datasets/train|val|test/images/<Class>/
and at the END prints a full summary table.

The img/data/ folder contains:
  - images/train/images/    (5843 pre-split images)
  - images/valid/images/    (1288 pre-split images)
  - images/test/images/     (1263 pre-split images)
  - Road Issues/Pothole Issues/           (3348)
  - Road Issues/Damaged Road issues/      (677)
  - Road Issues/Broken Road Sign Issues/  (1789)
  - Road Issues/Illegal Parking Issues/   (104)
  - Road Issues/Mixed Issues/             (191)
  - Public Cleanliness + Environmental/   (3543)

Split strategy:
  - pre-labelled train/valid/test → preserve their split
  - all other images → 70/20/10 hash-based split

Run AFTER classify_to_datasets.py finishes.
"""

import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).resolve().parents[2]
IMG_DIR   = PROJECT / "img" / "data"
DATASETS  = PROJECT / "datasets"

BEST_PT  = PROJECT / "runs" / "detect" / "road_damage_ai" / "weights" / "best.pt"
NANO_PT  = PROJECT / "yolov8n.pt"
MODEL_PATH = str(BEST_PT) if BEST_PT.exists() else str(NANO_PT)

CONF_THRESH = 0.20
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = {0: "Crack", 1: "Pothole", 2: "Surface_Erosion"}

# Pre-labelled splits in img/data/images/
PRE_SPLIT_MAP = {
    IMG_DIR / "images" / "train" / "images": "train",
    IMG_DIR / "images" / "valid" / "images": "val",
    IMG_DIR / "images" / "test"  / "images": "test",
}


def get_split_hash(path: Path) -> str:
    h = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 100
    if   h < 70: return "train"
    elif h < 90: return "val"
    else:        return "test"


def folder_hint_class(img_path: Path) -> int | None:
    p = str(img_path).lower()
    if "pothole" in p: return 1
    if "crack"   in p: return 0
    if "erosion" in p or "surface" in p or "abrasion" in p: return 2
    if "road sign" in p or "parking" in p: return None   # not road damage
    if "garbage" in p or "litter" in p or "vandalism" in p: return None
    return None


def collect_img_folder():
    """
    Returns list of (img_path, forced_split_or_None).
    forced_split is set for the pre-labelled train/valid/test subsets.
    """
    seen, items = set(), []
    # 1. Pre-labelled
    for folder, forced_split in PRE_SPLIT_MAP.items():
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if p.name not in seen:
                    seen.add(p.name)
                    items.append((p, forced_split))
    # 2. Category subfolders
    for p in IMG_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if p.name not in seen:
                # Skip videos_without_audio folder
                if "videos_without_audio" in str(p):
                    continue
                seen.add(p.name)
                items.append((p, None))
    return items


def save_image(img_path: Path, cls_id: int, yolo_lines: list,
               forced_split, counters: dict):
    split    = forced_split or get_split_hash(img_path)
    cls_name = CLASS_NAMES[cls_id]
    dest_img = DATASETS / split / "images" / cls_name / img_path.name
    dest_lbl = DATASETS / split / "labels" / cls_name / (img_path.stem + ".txt")

    if dest_img.exists():
        stem, ext = img_path.stem, img_path.suffix.lower()
        n = 1
        while dest_img.exists():
            dest_img = DATASETS / split / "images" / cls_name / f"{stem}_{n}{ext}"
            dest_lbl = DATASETS / split / "labels" / cls_name / f"{stem}_{n}.txt"
            n += 1

    shutil.copy2(img_path, dest_img)
    dest_lbl.write_text("\n".join(yolo_lines))
    counters[split][cls_name] += 1


def print_summary(counters: dict, skipped: int, total_seen: int):
    print("\n" + "=" * 62)
    print("  FINAL DATASET SUMMARY")
    print("=" * 62)
    print(f"  {'Split':<8} {'Class':<22} {'Count':>8}")
    print("  " + "-" * 42)

    split_totals = {}
    grand = 0
    for split in ("train", "val", "test"):
        s_total = 0
        for cls in sorted(CLASS_NAMES.values()):
            n = counters[split][cls]
            if n:
                print(f"  {split:<8} {cls:<22} {n:>8,}")
                s_total += n
                grand   += n
        split_totals[split] = s_total
        print(f"  {'':8} {'--- SUBTOTAL ---':<22} {s_total:>8,}")
        print()

    print("=" * 62)
    print(f"  {'TRAIN total':<30} {split_totals.get('train',0):>8,}")
    print(f"  {'VAL   total':<30} {split_totals.get('val',  0):>8,}")
    print(f"  {'TEST  total':<30} {split_totals.get('test', 0):>8,}")
    print(f"  {'GRAND TOTAL':<30} {grand:>8,}")
    print(f"  {'Skipped (no class)':<30} {skipped:>8,}")
    print(f"  {'Source images scanned':<30} {total_seen:>8,}")
    print("=" * 62)
    print(f"\n  Saved to: {DATASETS}")


def main():
    # Ensure output dirs exist
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            (DATASETS / split / "images" / cls).mkdir(parents=True, exist_ok=True)
            (DATASETS / split / "labels" / cls).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Smart Civic AI — img/ Folder Classifier")
    print(f"  Model : {Path(MODEL_PATH).name}")
    print(f"  Source: {IMG_DIR}")
    print("=" * 62)

    model = YOLO(MODEL_PATH)
    items = collect_img_folder()
    print(f"\n[SCAN] {len(items):,} unique images in img/\n")

    counters = defaultdict(lambda: defaultdict(int))
    skipped  = 0

    for img_path, forced_split in tqdm(items, desc="Classifying img/", unit="img"):
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=640,
                conf=CONF_THRESH,
                verbose=False
            )
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                # No YOLO detection — try folder name hint
                cls_id = folder_hint_class(img_path)
                if cls_id is None:
                    skipped += 1
                    continue
                yolo_lines = []
            else:
                from collections import Counter
                cls_counts = Counter(int(b.cls.item()) for b in r.boxes)
                cls_id     = cls_counts.most_common(1)[0][0]

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

            save_image(img_path, cls_id, yolo_lines, forced_split, counters)

        except Exception as e:
            tqdm.write(f"[WARN] {img_path.name}: {e}")
            skipped += 1

    # ── Read already-saved counts from datasets/ (from previous classify run) ──
    # Merge with existing counters
    existing = defaultdict(lambda: defaultdict(int))
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            imgs = list((DATASETS / split / "images" / cls).glob("*"))
            existing[split][cls] = len(imgs)

    # Print final combined summary
    print("\n\n" + "=" * 62)
    print("  COMBINED datasets/ SUMMARY (all sources)")
    print_summary(existing, skipped, len(items))

    # Update data.yaml
    yaml = DATASETS / "data.yaml"
    yaml.write_text(
        f"path: {DATASETS.as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 3\nnames: ['Crack', 'Pothole', 'Surface_Erosion']\n"
    )


if __name__ == "__main__":
    main()
