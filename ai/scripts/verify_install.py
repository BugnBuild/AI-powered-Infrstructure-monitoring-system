"""
Quick installation & environment verification.
Run this before starting the pipeline.
"""
import sys
from pathlib import Path

print("=" * 55)
print("  Smart Civic AI — Environment Check")
print("=" * 55)

errors = []

# Python version
import sys
py = sys.version_info
print(f"[OK]  Python {py.major}.{py.minor}.{py.micro}")
if py < (3, 9):
    errors.append("Python 3.9+ required")

# PyTorch & CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"[OK]  PyTorch {torch.__version__}  (CUDA: {cuda})")
    if cuda:
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"      VRAM: {vram:.1f} GB")
        if vram < 4:
            print("      [WARN] < 4GB VRAM — reduce batch size in script 4")
except ImportError:
    errors.append("PyTorch not installed: pip install torch")

# Ultralytics
try:
    import ultralytics
    print(f"[OK]  Ultralytics {ultralytics.__version__}")
except ImportError:
    errors.append("Ultralytics not installed: pip install ultralytics")

# Kaggle CLI
try:
    import kaggle
    print(f"[OK]  Kaggle {kaggle.__version__}")
    # Check credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"[OK]  kaggle.json found")
    else:
        errors.append(f"kaggle.json missing — download from https://www.kaggle.com/settings and save to {kaggle_json}")
except ImportError:
    errors.append("Kaggle CLI not installed: pip install kaggle")

# OpenCV
try:
    import cv2
    print(f"[OK]  OpenCV {cv2.__version__}")
except ImportError:
    errors.append("OpenCV not installed: pip install opencv-python-headless")

# tqdm
try:
    import tqdm
    print(f"[OK]  tqdm {tqdm.__version__}")
except ImportError:
    errors.append("tqdm not installed: pip install tqdm")

# Dataset check
base = Path(__file__).resolve().parents[1]
yaml = base / "dataset" / "Data Y12 Final" / "data.yaml"
if yaml.exists():
    print(f"[OK]  Existing dataset found: {yaml.parent.name}")
else:
    print(f"[WARN] Existing dataset not at expected path: {yaml}")

# Summary
print("\n" + "=" * 55)
if errors:
    print(f"  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    ✗  {e}")
    print("\n  Fix the issues above then re-run this script.")
    sys.exit(1)
else:
    print("  All checks passed! Ready to run the pipeline.")
    print("  Start with: python scripts/1_download_kaggle_datasets.py")
print("=" * 55)
