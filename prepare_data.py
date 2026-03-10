import os
import random
import shutil
from pathlib import Path

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
RAW_DIR        = Path("data/raw/chest_xray")
OUT_DIR        = Path("data/chest_xray")
CLASSES        = ['NORMAL', 'PNEUMONIA']
VAL_FRACTION   = 0.15
SEED           = 42

def download():
    import kaggle
    kaggle.api.authenticate()  
    print("Downloading dataset...")
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path="data/raw",
        unzip=True
    )
    print("Done.")

def copy_split(split):
    for cls in CLASSES:
        src = RAW_DIR / split / cls
        dst = OUT_DIR / split / cls
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy(f, dst / f.name)
        n = len(list(dst.iterdir()))
        print(f"  {split}/{cls}: {n} images")
        print(f"Created {dst}")

def build_val_from_train():
    random.seed(SEED)
    
    for cls in CLASSES:
        src = OUT_DIR / "train" / cls
        dst = OUT_DIR / "val" / cls
        dst.mkdir(parents=True, exist_ok=True)
        
        files = list(src.iterdir())
        random.shuffle(files)
        n_val = int(len(files) * VAL_FRACTION)
        val_files = files[:n_val]
        for f in val_files:
            shutil.move(f, dst / f.name)
        print(f"  {dst.parent.name}/{cls}: {len(val_files)} images")

def main():
    if not RAW_DIR.exists():
        download()
    
    print("\nSetting up train...")
    copy_split("train")
    
    print("\nBuilding val...")
    build_val_from_train()
    
    print("\nSetting up test...")
    copy_split("test")

if __name__ == "__main__":
    main()