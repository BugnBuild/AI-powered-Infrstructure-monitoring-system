from pathlib import Path

d = Path(r'c:\Users\Raghwendra Singh\Desktop\AI Powered infrastructure monitoring system\datasets')
classes = ['Crack', 'Pothole', 'Surface_Erosion']
splits  = ['train', 'val', 'test']
n
print()
print("=" * 55)
print("  EXACT DATASET COUNTS (read from disk right now)")
print("=" * 55)
print(f"  {'Split':<8} {'Class':<22} {'Images':>8}  {'Labels':>8}")
print("  " + "-" * 44)

split_totals = {}
grand = 0
for split in splits:
    s = 0
    for cls in classes:
        img_dir = d / split / "images" / cls
        lbl_dir = d / split / "labels" / cls
        imgs = len(list(img_dir.glob("*")))    if img_dir.exists() else 0
        lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"  {split:<8} {cls:<22} {imgs:>8,}  {lbls:>8,}")
        s     += imgs
        grand += imgs
    split_totals[split] = s
    print(f"  {'':8} {'── SUBTOTAL ──':<22} {s:>8,}")
    print()

print("=" * 55)
print(f"  {'TRAIN  images (70%)':<32} {split_totals['train']:>8,}")
print(f"  {'VAL    images (20%)':<32} {split_totals['val']:>8,}")
print(f"  {'TEST   images (10%)':<32} {split_totals['test']:>8,}")
print(f"  {'-'*42}")
print(f"  {'GRAND TOTAL':<32} {grand:>8,}")
print("=" * 55)
print()
