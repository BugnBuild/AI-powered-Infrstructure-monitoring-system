"""
Master pipeline runner — executes all 5 steps sequentially.

Usage:
    python scripts/run_full_pipeline.py

Or run individual steps:
    python scripts/1_download_kaggle_datasets.py
    python scripts/2_merge_and_convert.py
    python scripts/3_split_dataset.py
    python scripts/4_train_yolo.py
    python scripts/5_evaluate_and_export.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
STEPS = [
    ("1_download_kaggle_datasets.py", "Download Kaggle datasets"),
    ("2_merge_and_convert.py",        "Merge & convert datasets"),
    ("3_split_dataset.py",            "Split train/val/test"),
    ("4_train_yolo.py",               "Train YOLOv8m model"),
    ("5_evaluate_and_export.py",      "Evaluate & export model"),
]


def run_step(script: str, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"  FILE: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)])
    if result.returncode != 0:
        print(f"\n[FAILED] Step '{description}' exited with code {result.returncode}")
        return False
    return True


def main():
    print("\n" + "=" * 60)
    print("  Smart Civic AI — Full Training Pipeline")
    print("=" * 60)

    for i, (script, desc) in enumerate(STEPS, 1):
        ok = run_step(script, f"[{i}/{len(STEPS)}] {desc}")
        if not ok:
            print(f"\n  Pipeline stopped at step {i}. Fix the error and re-run.")
            print(f"  You can resume from: python scripts/{script}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE!")
    print("  Model is ready at: ai/models/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
