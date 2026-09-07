"""
Script 1: Download road damage datasets from Kaggle.

Uses the Kaggle HTTP API directly with the Bearer token,
bypassing the CLI's OAuth requirement in kaggle v2+.

Datasets — all verified accessible (no terms gate):
  1. chitholian/annotated-potholes-dataset              (48 MB,  ~665  images, YOLO)
  2. muskanverma24/pothole-detection-dataset-yolov11    (226 MB, ~2000 images, YOLO)
  3. lorenzoarcioni/road-damage-potholes-cracks-manholes(194 MB, ~1800 images)
  4. vidishbijalwan/rdd2022-india-pothole-d40           (84 MB,  ~800  images, YOLO)
  5. badri467/mwpd-rdd-india-japan-dataset              (1.1 GB, ~5000 images, XML)
  6. ilsafshamsutdinov/road-damage-for-object-detection (1.8 GB, ~7000 images)
  7. kalyan1284/road-damage-detection-3-class-unified   (4 GB,  ~15000 images, YOLO 3-class)

Total: 15,000+ images across 7 datasets.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = BASE_DIR / "dataset" / "kaggle_raw"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Load token ───────────────────────────────────────────────────────────────
def get_token() -> str:
    # 1. Environment variable
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token

    # 2. ~/.kaggle/access_token  (new v2 format)
    access_token_path = Path.home() / ".kaggle" / "access_token"
    if access_token_path.exists():
        token = access_token_path.read_text().strip()
        if token:
            return token

    # 3. ~/.kaggle/kaggle.json  (classic format → extract key)
    import json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        data = json.loads(kaggle_json.read_text())
        token = data.get("key", "").strip()
        if token:
            return token

    print("[ERROR] No Kaggle token found.")
    print("  Save your token to: ~/.kaggle/access_token")
    print("  Or set env var:     KAGGLE_API_TOKEN=<your_token>")
    sys.exit(1)


# ── Datasets ─────────────────────────────────────────────────────────────────
DATASETS = [
    # ── Verified accessible — no terms gate ──────────────────────────────────
    # (owner, dataset-slug, local-folder-name)
    ("chitholian",        "annotated-potholes-dataset",                       "potholes_annotated"),      #  48 MB  ~665  imgs YOLO
    ("muskanverma24",     "pothole-detection-dataset-yolov11-optimized",       "muskan_pothole"),          # 226 MB  ~2000 imgs YOLO
    ("lorenzoarcioni",    "road-damage-dataset-potholes-cracks-and-manholes",  "lorenzo_road_damage"),     # 194 MB  ~1800 imgs
    ("vidishbijalwan",    "rdd2022-india-pothole-d40",                         "rdd2022_india"),           #  84 MB  ~800  imgs YOLO
    ("badri467",          "mwpd-rdd-india-japan-dataset",                      "mwpd_rdd"),                # 1.1 GB  ~5000 imgs XML
    ("ilsafshamsutdinov", "road-damage-dataset-for-object-detection",          "ilsaf_road_damage"),       # 1.8 GB  ~7000 imgs
    ("kalyan1284",        "road-damage-detection-3-class-unified",             "kalyan_3class"),           # 4.0 GB  ~15k  imgs YOLO 3-class
]


def download_dataset(owner: str, dataset: str, folder_name: str, token: str):
    dest = DOWNLOAD_DIR / folder_name
    if dest.exists() and any(dest.rglob("*.jpg")):
        count = len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.png")))
        print(f"[SKIP] Already downloaded: {folder_name}  ({count} images)")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{dataset}.zip"

    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"\n[DOWNLOAD] {owner}/{dataset}")
    print(f"  → {dest}")

    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            if r.status_code == 401:
                print(f"  [ERROR] 401 Unauthorized — token may be expired or invalid.")
                return False
            if r.status_code == 403:
                print(f"  [WARN] 403 Forbidden — you need to accept the dataset rules on Kaggle first.")
                print(f"         Visit: https://www.kaggle.com/datasets/{owner}/{dataset}")
                print(f"         Click 'Download', accept terms, then re-run.")
                return False
            if r.status_code != 200:
                print(f"  [WARN] HTTP {r.status_code} — skipping {owner}/{dataset}")
                return False

            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1e6
                        print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)
        print(f"\r  100.0%  {downloaded/1e6:.1f} MB  ✓")

    except requests.exceptions.ConnectionError as e:
        print(f"  [ERROR] Connection failed: {e}")
        return False

    # Unzip
    print(f"  Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
    except zipfile.BadZipFile:
        print(f"  [WARN] Bad zip file — the download may have been HTML (auth page).")
        zip_path.unlink(missing_ok=True)
        return False

    imgs = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png")) + list(dest.rglob("*.jpeg"))
    print(f"  [OK] {len(imgs):,} images extracted to {dest.name}/")
    return True


def main():
    print("=" * 60)
    print("  Smart Civic AI — Kaggle Dataset Downloader")
    print("=" * 60)

    token = get_token()
    print(f"[AUTH] Token loaded (…{token[-6:]})")

    success, failed = [], []
    for owner, dataset, folder in DATASETS:
        ok = download_dataset(owner, dataset, folder, token)
        if ok:
            success.append(f"{owner}/{dataset}")
        else:
            failed.append(f"{owner}/{dataset}")

    # Count total images
    all_imgs = (list(DOWNLOAD_DIR.rglob("*.jpg")) +
                list(DOWNLOAD_DIR.rglob("*.png")) +
                list(DOWNLOAD_DIR.rglob("*.jpeg")))

    print(f"\n{'='*60}")
    print(f"  Downloaded: {len(success)}/{len(DATASETS)} datasets")
    print(f"  Total images in kaggle_raw/: {len(all_imgs):,}")
    if failed:
        print(f"\n  Skipped (need manual accept on Kaggle website):")
        for f in failed:
            owner, ds = f.split("/")
            print(f"    https://www.kaggle.com/datasets/{owner}/{ds}")
    print(f"{'='*60}")
    print("\nNext step → Run: python scripts/2_merge_and_convert.py")


if __name__ == "__main__":
    main()
