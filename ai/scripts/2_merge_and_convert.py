"""
Script 2: Merge all datasets into a single unified pool.

Handles:
  - Your existing Roboflow dataset (Data Y12 Final) — already YOLO format
  - RDD2022 dataset from Kaggle — XML annotations → YOLO conversion
  - Any other Kaggle dataset with YOLO .txt labels — direct copy
  - Deduplication by filename hash

Output structure:
  ai/dataset/merged_pool/
      images/   (all .jpg files, renamed to avoid collisions)
      labels/   (matching YOLO .txt files)

Class mapping (unified):
  0 = Crack
  1 = Pothole
  2 = Surface Erosion / Abrasion
"""

import os
import shutil
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
EXISTING_DATASET = BASE_DIR / "dataset" / "Data Y12 Final"
KAGGLE_RAW       = BASE_DIR / "dataset" / "kaggle_raw"
MERGED_POOL      = BASE_DIR / "dataset" / "merged_pool"
POOL_IMAGES      = MERGED_POOL / "images"
POOL_LABELS      = MERGED_POOL / "labels"

POOL_IMAGES.mkdir(parents=True, exist_ok=True)
POOL_LABELS.mkdir(parents=True, exist_ok=True)

# ── Unified class mapping ────────────────────────────────────────────────────
# Maps various source label names → unified class IDs
CLASS_MAP = {
    # Crack variants
    "crack": 0, "longitudinal_crack": 0, "transverse_crack": 0,
    "alligator_crack": 0, "d00": 0, "d10": 0, "d20": 0, "d40": 0,
    "cracking": 0, "crack_seal": 0,
    # Pothole variants
    "pothole": 1, "potholes": 1, "d44": 1, "hole": 1,
    # Surface erosion / abrasion
    "surface erosion": 2, "erosion": 2, "abrasion": 2,
    "rutting": 2, "raveling": 2, "weathering": 2, "d43": 2,
    "surface_abrasion": 2, "surface_wear": 2,
}
NUM_CLASSES = 3
CLASS_NAMES = ["Crack", "Pothole", "Surface Erosion"]

# ── Counters ─────────────────────────────────────────────────────────────────
stats = {"copied": 0, "skipped_dup": 0, "skipped_no_label": 0, "converted_xml": 0}
seen_hashes: set[str] = set()


def file_hash(path: Path) -> str:
    """MD5 of first 8KB — fast dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()


def safe_copy(img_path: Path, label_path: Path, prefix: str = ""):
    """Copy image+label to pool, skipping duplicates."""
    h = file_hash(img_path)
    if h in seen_hashes:
        stats["skipped_dup"] += 1
        return
    seen_hashes.add(h)

    stem = f"{prefix}_{img_path.stem}" if prefix else img_path.stem
    # Ensure unique filename
    dest_img = POOL_IMAGES / (stem + img_path.suffix.lower())
    dest_lbl = POOL_LABELS / (stem + ".txt")
    counter = 0
    while dest_img.exists():
        counter += 1
        dest_img = POOL_IMAGES / f"{stem}_{counter}{img_path.suffix.lower()}"
        dest_lbl = POOL_LABELS / f"{stem}_{counter}.txt"

    shutil.copy2(img_path, dest_img)
    shutil.copy2(label_path, dest_lbl)
    stats["copied"] += 1


def remap_yolo_label(src: Path, dst: Path):
    """
    Re-read a YOLO .txt and remap class IDs.
    If source already uses 0/1/2 consistently (Roboflow), pass through.
    """
    lines_out = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            lines_out.append(line.strip())
    if lines_out:
        dst.write_text("\n".join(lines_out))
        return True
    return False


# ── 1. Ingest existing Roboflow dataset ─────────────────────────────────────
def ingest_roboflow():
    print("\n[1/3] Ingesting existing Roboflow dataset (Data Y12 Final)...")
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = EXISTING_DATASET / split / "images"
        lbl_dir = EXISTING_DATASET / split / "labels"
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for img in tqdm(imgs, desc=f"  Roboflow/{split}"):
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                stats["skipped_no_label"] += 1
                continue
            # Copy label via remap (pass-through, already 0/1/2)
            temp_lbl = POOL_LABELS / ("_tmp_" + img.stem + ".txt")
            if remap_yolo_label(lbl, temp_lbl):
                safe_copy(img, temp_lbl, prefix="rfv1")
            if temp_lbl.exists():
                temp_lbl.unlink()


# ── 2. Ingest Kaggle YOLO-format datasets ───────────────────────────────────
def ingest_yolo_folder(folder: Path, prefix: str):
    """Generic ingestion for datasets that already have YOLO .txt labels."""
    img_exts = {".jpg", ".jpeg", ".png"}
    all_imgs = [p for p in folder.rglob("*") if p.suffix.lower() in img_exts]
    if not all_imgs:
        return 0
    copied = 0
    for img in tqdm(all_imgs, desc=f"  {prefix}"):
        # Look for label in same folder or sibling "labels" folder
        lbl = img.with_suffix(".txt")
        if not lbl.exists():
            lbl_sib = img.parent.parent / "labels" / (img.stem + ".txt")
            if lbl_sib.exists():
                lbl = lbl_sib
            else:
                stats["skipped_no_label"] += 1
                continue
        temp_lbl = POOL_LABELS / ("_tmp_" + img.stem + ".txt")
        if remap_yolo_label(lbl, temp_lbl):
            safe_copy(img, temp_lbl, prefix=prefix)
            copied += 1
        if temp_lbl.exists():
            temp_lbl.unlink()
    return copied


# ── 3. Ingest RDD2022 (XML annotations) ─────────────────────────────────────
def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert Pascal VOC XML → YOLO format, applying class remapping."""
    lines = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            bndbox  = obj.find("bndbox")
            if name_el is None or bndbox is None:
                continue
            name = name_el.text.strip().lower()
            cls_id = CLASS_MAP.get(name)
            if cls_id is None:
                # Try partial match
                for key, cid in CLASS_MAP.items():
                    if key in name or name in key:
                        cls_id = cid
                        break
            if cls_id is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w  = (xmax - xmin) / img_w
            h  = (ymax - ymin) / img_h
            # Clamp to [0,1]
            cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    except Exception as e:
        print(f"    [WARN] XML parse error {xml_path.name}: {e}")
    return lines


def ingest_rdd2022(folder: Path):
    """Handle RDD2022 structure: country/images/*.jpg + country/annotations/xmls/*.xml"""
    img_exts = {".jpg", ".jpeg", ".png"}
    xml_dirs = list(folder.rglob("xmls"))
    if not xml_dirs:
        # Try flat YOLO format
        ingest_yolo_folder(folder, "rdd2022")
        return

    print(f"  Found {len(xml_dirs)} XML annotation directories")
    for xml_dir in xml_dirs:
        img_dir = xml_dir.parent.parent / "images"
        if not img_dir.exists():
            img_dir = xml_dir.parent.parent.parent / "images"
        if not img_dir.exists():
            continue

        xmls = list(xml_dir.glob("*.xml"))
        for xml_path in tqdm(xmls, desc=f"  rdd2022/{xml_dir.parent.parent.name}"):
            # Find matching image
            img_path = None
            for ext in img_exts:
                candidate = img_dir / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                stats["skipped_no_label"] += 1
                continue

            # Get image size for normalization (use default 600x600 if cv2 unavailable)
            try:
                import cv2
                img_arr = cv2.imread(str(img_path))
                if img_arr is None:
                    raise ValueError("cv2 read None")
                img_h, img_w = img_arr.shape[:2]
            except Exception:
                img_w, img_h = 600, 600

            yolo_lines = xml_to_yolo(xml_path, img_w, img_h)
            if not yolo_lines:
                stats["skipped_no_label"] += 1
                continue

            h = file_hash(img_path)
            if h in seen_hashes:
                stats["skipped_dup"] += 1
                continue
            seen_hashes.add(h)

            stem = f"rdd22_{img_path.stem}"
            dest_img = POOL_IMAGES / (stem + img_path.suffix.lower())
            dest_lbl = POOL_LABELS / (stem + ".txt")
            counter = 0
            while dest_img.exists():
                counter += 1
                dest_img = POOL_IMAGES / f"{stem}_{counter}{img_path.suffix.lower()}"
                dest_lbl = POOL_LABELS / f"{stem}_{counter}.txt"

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(yolo_lines))
            stats["copied"] += 1
            stats["converted_xml"] += 1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Smart Civic AI — Dataset Merger & Converter")
    print("=" * 60)

    ingest_roboflow()

    if KAGGLE_RAW.exists():
        for subfolder in sorted(KAGGLE_RAW.iterdir()):
            if not subfolder.is_dir():
                continue
            print(f"\n[2/3] Ingesting Kaggle dataset: {subfolder.name}")
            if subfolder.name == "rdd2022":
                ingest_rdd2022(subfolder)
            else:
                ingest_yolo_folder(subfolder, prefix=subfolder.name[:8])
    else:
        print(f"\n[INFO] No Kaggle raw data at {KAGGLE_RAW} — only Roboflow data used.")

    total = len(list(POOL_IMAGES.glob("*")))
    print(f"\n{'='*60}")
    print(f"  Merge complete!")
    print(f"  Total images in pool : {total:,}")
    print(f"  Copied               : {stats['copied']:,}")
    print(f"  XML→YOLO converted   : {stats['converted_xml']:,}")
    print(f"  Skipped (duplicate)  : {stats['skipped_dup']:,}")
    print(f"  Skipped (no label)   : {stats['skipped_no_label']:,}")
    print(f"  Output path          : {MERGED_POOL}")
    print(f"{'='*60}")
    print("\nNext step → Run: python scripts/3_split_dataset.py")


if __name__ == "__main__":
    main()
