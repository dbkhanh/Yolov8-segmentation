import random
import shutil
from pathlib import Path

# -------- config --------
ROOT = Path("../photo")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

TRAIN_RATIO = 0.75
SEED = 42

IMG_EXTS = {".png", ".jpg", ".jpeg"}
# ------------------------

random.seed(SEED)

# create output dirs
for split in ["train", "val"]:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (LBL_DIR / split).mkdir(parents=True, exist_ok=True)

# collect image files
images = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]

assert len(images) > 0, "❌ No images found"

random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_pair(img_path, split):
    label_path = LBL_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"❌ Missing label for {img_path.name}")

    shutil.move(str(img_path), IMG_DIR / split / img_path.name)
    shutil.move(str(label_path), LBL_DIR / split / label_path.name)

for img in train_imgs:
    move_pair(img, "train")

for img in val_imgs:
    move_pair(img, "val")

print(f"✅ Done!")
print(f"   Train: {len(train_imgs)} images")
print(f"   Val:   {len(val_imgs)} images")
