"""
Script 4: Train YOLOv8 on the final road damage dataset.

Model   : YOLOv8m (medium — best accuracy/speed trade-off)
Epochs  : 100 (early stopping enabled)
Dataset : ai/dataset/final_dataset/data.yaml
Output  : ai/runs/train/road_damage_v1/

GPU auto-detected; falls back to CPU if no CUDA device.
"""

from pathlib import Path
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
DATASET_YAML  = BASE_DIR / "dataset" / "final_dataset" / "data.yaml"
RUNS_DIR      = BASE_DIR / "runs"
RUN_NAME      = "road_damage_v1"

# ── Hyperparameters ──────────────────────────────────────────────────────────
# Tuned for road damage detection with mixed lighting / surface conditions
HP = dict(
    data          = str(DATASET_YAML),
    epochs        = 100,
    patience      = 20,          # Early stopping: stop if no improvement for 20 epochs
    imgsz         = 640,         # Standard YOLO input size
    batch         = -1,          # Auto batch size (fills 60% GPU VRAM)
    optimizer     = "AdamW",
    lr0           = 0.001,       # Initial learning rate
    lrf           = 0.01,        # Final LR as fraction of lr0
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3.0,
    warmup_momentum = 0.8,
    box           = 7.5,         # Box loss gain
    cls           = 0.5,         # Class loss gain
    dfl           = 1.5,         # Distribution focal loss gain
    # Augmentation — robust for outdoor road imagery
    hsv_h         = 0.015,       # Hue shift
    hsv_s         = 0.7,         # Saturation shift
    hsv_v         = 0.4,         # Brightness shift
    degrees       = 5.0,         # Rotation
    translate     = 0.1,         # Translation
    scale         = 0.5,         # Scale
    shear         = 2.0,
    perspective   = 0.0001,
    flipud        = 0.1,
    fliplr        = 0.5,
    mosaic        = 1.0,         # Mosaic augmentation (great for small objects)
    mixup         = 0.1,         # MixUp augmentation
    copy_paste    = 0.1,         # Copy-paste augmentation
    # Output
    project       = str(RUNS_DIR / "train"),
    name          = RUN_NAME,
    exist_ok      = True,
    pretrained    = True,
    verbose       = True,
    save          = True,
    save_period   = 10,          # Save checkpoint every 10 epochs
    plots         = True,
    val           = True,
    amp           = True,        # Automatic mixed precision (faster on GPU)
    workers       = 8,
    seed          = 42,
    cos_lr        = True,        # Cosine LR scheduler
    close_mosaic  = 10,          # Disable mosaic last 10 epochs for fine-tuning
    rect          = False,
    cache         = False,       # Set True to cache images in RAM if >32GB RAM
)


def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("  Smart Civic AI — YOLOv8 Training")
    print("=" * 60)

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("[CPU] No CUDA GPU detected — training on CPU (will be slow)")
        # Reduce batch & workers for CPU
        HP["batch"]   = 8
        HP["workers"] = 2
        HP["amp"]     = False

    # Verify dataset exists
    if not DATASET_YAML.exists():
        print(f"[ERROR] Dataset YAML not found: {DATASET_YAML}")
        print("  Run scripts 1, 2, 3 first.")
        return

    # Count training images
    train_img_dir = DATASET_YAML.parent / "train" / "images"
    n_train = len(list(train_img_dir.glob("*"))) if train_img_dir.exists() else "?"
    print(f"[DATA] Training images: {n_train}")
    print(f"[DATA] YAML: {DATASET_YAML}")

    # ── Load model ───────────────────────────────────────────────────────────
    # YOLOv8m — 25.9M params, good for detecting small road defects
    model = YOLO("yolov8m.pt")
    print(f"\n[MODEL] YOLOv8m loaded (pretrained COCO weights)")
    print(f"[RUN]   Output → {RUNS_DIR / 'train' / RUN_NAME}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    results = model.train(**HP)

    # ── Post-training summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete!")
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print(f"  Best weights : {best_weights}")
    print(f"  Last weights : {last_weights}")
    print(f"  Metrics      : {save_dir / 'results.csv'}")
    print(f"  Plots        : {save_dir}")

    # Print final metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print(f"\n  Final Metrics:")
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                    "metrics/precision(B)", "metrics/recall(B)"]:
            val = metrics.get(key, "N/A")
            label = key.replace("metrics/", "").replace("(B)", "")
            print(f"    {label:<20} {val}")

    print(f"\n[DONE] Next step → Run: python scripts/5_evaluate_and_export.py")


if __name__ == "__main__":
    main()
