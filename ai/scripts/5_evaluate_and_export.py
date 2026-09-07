"""
Script 5: Evaluate trained model on test set & export to FastAPI-ready format.

Actions:
  1. Run validation on hold-out test split → mAP, precision, recall, F1
  2. Generate confusion matrix and PR curve plots
  3. Run batch inference on 10 sample test images (visual check)
  4. Export best.pt → ONNX (for production FastAPI deployment)
  5. Copy best.pt → ai/models/best.pt (used by FastAPI server)
"""

import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
DATASET_YAML = BASE_DIR / "dataset" / "final_dataset" / "data.yaml"
RUNS_DIR     = BASE_DIR / "runs"
RUN_NAME     = "road_damage_v1"
TRAIN_DIR    = RUNS_DIR / "train" / RUN_NAME
BEST_PT      = TRAIN_DIR / "weights" / "best.pt"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["Crack", "Pothole", "Surface Erosion"]


def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("  Smart Civic AI — Model Evaluation & Export")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"[ERROR] Best weights not found: {BEST_PT}")
        print("  Run script 4 (training) first.")
        return

    model = YOLO(str(BEST_PT))
    print(f"[MODEL] Loaded: {BEST_PT}")

    # ── 1. Validate on test split ─────────────────────────────────────────
    print("\n[1/4] Running validation on TEST split...")
    metrics = model.val(
        data    = str(DATASET_YAML),
        split   = "test",
        imgsz   = 640,
        batch   = 16,
        conf    = 0.25,
        iou     = 0.5,
        plots   = True,
        save_json = True,
        project = str(RUNS_DIR / "eval"),
        name    = "test_eval",
        exist_ok = True,
        verbose  = True,
    )

    print("\n  Per-class metrics:")
    print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'mAP@50':>10} {'mAP@50-95':>12}")
    print("  " + "-" * 64)
    try:
        box = metrics.box
        for i, cls in enumerate(CLASS_NAMES):
            p  = box.p[i]  if hasattr(box, "p")  else "?"
            r  = box.r[i]  if hasattr(box, "r")  else "?"
            ap = box.ap50[i] if hasattr(box, "ap50") else "?"
            ap95 = box.ap[i] if hasattr(box, "ap")  else "?"
            print(f"  {cls:<20} {float(p):>10.4f} {float(r):>10.4f} {float(ap):>10.4f} {float(ap95):>12.4f}")
        print(f"\n  Overall mAP@50     : {metrics.box.map50:.4f}")
        print(f"  Overall mAP@50-95  : {metrics.box.map:.4f}")
    except Exception as e:
        print(f"  (Could not extract per-class metrics: {e})")
        print(f"  Results saved to: {RUNS_DIR / 'eval' / 'test_eval'}")

    # ── 2. Batch inference on sample test images ─────────────────────────
    print("\n[2/4] Running inference on sample test images...")
    test_img_dir = BASE_DIR / "dataset" / "final_dataset" / "test" / "images"
    sample_imgs  = list(test_img_dir.glob("*.jpg"))[:10] + \
                   list(test_img_dir.glob("*.png"))[:10]
    sample_imgs  = sample_imgs[:10]

    if sample_imgs:
        results = model.predict(
            source  = sample_imgs,
            conf    = 0.25,
            iou     = 0.5,
            imgsz   = 640,
            save    = True,
            save_txt = True,
            project = str(RUNS_DIR / "predict"),
            name    = "sample_test",
            exist_ok = True,
        )
        print(f"  Predictions saved to: {RUNS_DIR / 'predict' / 'sample_test'}")
        for r in results:
            dets = [(CLASS_NAMES[int(b.cls)], float(b.conf))
                    for b in r.boxes] if r.boxes else []
            label_str = ", ".join(f"{c}({conf:.2f})" for c, conf in dets) or "No detection"
            print(f"    {Path(r.path).name:<50} → {label_str}")
    else:
        print("  [WARN] No test images found for sample inference.")

    # ── 3. Export to ONNX ────────────────────────────────────────────────
    print("\n[3/4] Exporting to ONNX format...")
    try:
        export_path = model.export(
            format  = "onnx",
            imgsz   = 640,
            dynamic = True,     # Dynamic batch size
            simplify = True,    # ONNX simplifier
            opset   = 12,
        )
        print(f"  ONNX model saved: {export_path}")
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")

    # ── 4. Copy best.pt to models/ ───────────────────────────────────────
    print("\n[4/4] Copying best weights to models/...")
    dest_pt   = MODELS_DIR / "best.pt"
    dest_yaml = MODELS_DIR / "data.yaml"
    shutil.copy2(BEST_PT, dest_pt)
    shutil.copy2(DATASET_YAML, dest_yaml)
    print(f"  best.pt  → {dest_pt}")
    print(f"  data.yaml → {dest_yaml}")

    # Write model info
    info_path = MODELS_DIR / "model_info.txt"
    with open(info_path, "w") as f:
        f.write("Smart Civic AI — Road Damage Detection Model\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture : YOLOv8m\n")
        f.write(f"Classes      : {CLASS_NAMES}\n")
        f.write(f"Input size   : 640x640\n")
        f.write(f"Weights file : best.pt\n")
        f.write(f"ONNX export  : best.onnx\n")
        f.write(f"Training run : {TRAIN_DIR}\n")
    print(f"  model_info.txt → {info_path}")

    print(f"\n{'='*60}")
    print("  Evaluation & Export complete!")
    print(f"  FastAPI can now load: {dest_pt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
