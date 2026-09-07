"""
classify_all_images.py  (v2 — classify ALL 18,046 img/ images)
--------------------------------------------------------------
Every image from img/data/ goes into datasets/ — no skipping.
If YOLO detects road damage → use that class.
If YOLO finds nothing     → use folder name hint.
If folder has no hint     → default to Pothole (most common in dataset).

Split rules:
  img/data/images/train/  → train   (preserve original split)
  img/data/images/valid/  → val     (preserve original split)
  img/data/images/test/   → test    (preserve original split)
  All other folders       → 70/20/10 hash-based split

Also re-processes ai/dataset sources.
Skips filenames already saved in datasets/.
Prints EXACT final counts at the end.
"""

import hashlib
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).resolve().parents[2]
DATASETS  = PROJECT / "datasets"
IMG_DATA  = PROJECT / "img" / "data"
AI_DATA   = PROJECT / "ai" / "dataset"

BEST_PT   = PROJECT / "runs" / "detect" / "road_damage_ai" / "weights" / "best.pt"
NANO_PT   = PROJECT / "yolov8n.pt"
MODEL_PATH = str(BEST_PT) if BEST_PT.exists() else str(NANO_PT)

IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp"}
CONF_THRESH = 0.20
CLASS_NAMES = {0: "Crack", 1: "Pothole", 2: "Surface_Erosion"}

# ── Source dirs with (path, forced_split) ────────────────────────────────────
# forced_split = "train"/"val"/"test" to preserve original labelling
# forced_split = None → use hash-based 70/20/10
SOURCE_DIRS = [
    # Pre-labelled img/ splits — preserve their split
    (IMG_DATA / "images" / "train" / "images",  "train"),
    (IMG_DATA / "images" / "valid" / "images",  "val"),
    (IMG_DATA / "images" / "test"  / "images",  "test"),
    # Road damage category folders → auto-split
    (IMG_DATA / "Road Issues" / "Pothole Issues",           None),
    (IMG_DATA / "Road Issues" / "Damaged Road issues",      None),
    (IMG_DATA / "Road Issues" / "Mixed Issues",             None),
    (IMG_DATA / "Road Issues" / "Broken Road Sign Issues",  None),
    (IMG_DATA / "Road Issues" / "Illegal Parking Issues",   None),
    # Public cleanliness — YOLO will detect if there's road damage
    (IMG_DATA / "Public Cleanliness + Environmental Issues", None),
    # ai/ dataset sources
    (AI_DATA  / "merged_pool" / "images",  None),
    (AI_DATA  / "kaggle_raw",              None),
]

# Folder-name → class hint (when YOLO finds nothing)
FOLDER_CLASS = {
    "pothole issues":          1,
    "pothole":                 1,
    "damaged road issues":     2,
    "damaged road":            2,
    "mixed issues":            1,   # mixed → default pothole
    "broken road sign issues": 0,   # signs are on cracked roads
    "illegal parking issues":  1,
    "crack":                   0,
    "erosion":                 2,
    "surface":                 2,
    "littering garbage":       1,   # no road hint → pothole default
    "vandalism":               0,   # often cracked surfaces
    # pre-labelled train/valid/test default
    "images":                  1,   # default → pothole (most common)
}

DEFAULT_CLASS = 1   # Pothole — fallback when nothing else matches


# ── Helpers ───────────────────────────────────────────────────────────────────
def split_by_hash(path: Path) -> str:
    h = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 100
    if   h < 70: return "train"
    elif h < 90: return "val"
    else:        return "test"


def get_folder_hint(img_path: Path) -> int:
    """Return class from folder name, default Pothole."""
    parts = [p.lower() for p in img_path.parts]
    for part in reversed(parts):
        for key, cls in FOLDER_CLASS.items():
            if key in part:
                return cls
    return DEFAULT_CLASS


def make_dirs():
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            (DATASETS / split / "images" / cls).mkdir(parents=True, exist_ok=True)
            (DATASETS / split / "labels" / cls).mkdir(parents=True, exist_ok=True)


def collect_all():
    """Collect all unique images. Returns list of (path, forced_split)."""
    seen, items = set(), []
    for src_dir, forced_split in SOURCE_DIRS:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            continue
        for p in src_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if "videos_without_audio" in str(p):
                continue
            if p.name not in seen:
                seen.add(p.name)
                items.append((p, forced_split))
    return items


def already_saved() -> set:
    """Filenames already in datasets/."""
    done = set()
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            folder = DATASETS / split / "images" / cls
            if folder.exists():
                for f in folder.iterdir():
                    done.add(f.name)
    return done


def save_image(img_path, cls_id, yolo_lines, forced_split, counters):
    split    = forced_split or split_by_hash(img_path)
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
    dest_lbl.write_text("\n".join(yolo_lines) if yolo_lines else "")
    counters[split][cls_name] += 1


def print_summary(grand_before: int):
    final = defaultdict(lambda: defaultdict(int))
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES.values():
            final[split][cls] = len(list(
                (DATASETS / split / "images" / cls).glob("*")
            ))

    print("\n\n" + "=" * 60)
    print("  EXACT FINAL DATASET COUNTS")
    print("=" * 60)
    print(f"  {'Split':<8} {'Class':<22} {'Images':>8}  {'Labels':>8}")
    print("  " + "-" * 48)

    split_totals = {}
    grand = 0
    for split in ("train", "val", "test"):
        s = 0
        for cls in sorted(CLASS_NAMES.values()):
            imgs = final[split][cls]
            lbls = len(list((DATASETS / split / "labels" / cls).glob("*.txt")))
            print(f"  {split:<8} {cls:<22} {imgs:>8,}  {lbls:>8,}")
            s    += imgs
            grand += imgs
        split_totals[split] = s
        print(f"  {'':8} {'── SUBTOTAL ──':<22} {s:>8,}")
        print()

    print("=" * 60)
    print(f"  {'TRAIN  images (70%)':<35} {split_totals['train']:>8,}")
    print(f"  {'VAL    images (20%)':<35} {split_totals['val']:>8,}")
    print(f"  {'TEST   images (10%)':<35} {split_totals['test']:>8,}")
    print(f"  {'-'*45}")
    print(f"  {'GRAND TOTAL':<35} {grand:>8,}")
    print(f"  {'Added this run':<35} {grand - grand_before:>8,}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    make_dirs()

    print("=" * 60)
    print("  Smart Civic AI — Classify ALL Images (v2)")
    print(f"  Model : {Path(MODEL_PATH).name}")
    print("=" * 60)

    model = YOLO(MODEL_PATH)
    items = collect_all()
    print(f"\n[SCAN]  {len(items):,} unique source images")

    done      = already_saved()
    pending   = [(p, s) for p, s in items if p.name not in done]
    already_n = len(items) - len(pending)

    # Count what's already there
    grand_before = sum(
        len(list((DATASETS / sp / "images" / cls).glob("*")))
        for sp in ("train","val","test")
        for cls in CLASS_NAMES.values()
    )

    print(f"[DONE]  {already_n:,} already classified (skipping)")
    print(f"[TODO]  {len(pending):,} images to classify now\n")

    counters = defaultdict(lambda: defaultdict(int))

    for img_path, forced_split in tqdm(pending, desc="Classifying", unit="img"):
        try:
            results = model.predict(
                source=str(img_path),
                imgsz=640,
                conf=CONF_THRESH,
                verbose=False
            )
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                # No detection → use folder hint (never skip)
                cls_id     = get_folder_hint(img_path)
                yolo_lines = []
            else:
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
            continue

    print_summary(grand_before)

    # Update data.yaml
    (DATASETS / "data.yaml").write_text(
        f"path: {DATASETS.as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 3\nnames: ['Crack', 'Pothole', 'Surface_Erosion']\n"
    )
    print(f"\n  data.yaml → {DATASETS / 'data.yaml'}")
    print("  Done!")


if __name__ == "__main__":
    main()
