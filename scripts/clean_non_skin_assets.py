import os
import glob
from PIL import Image

RAW_DIR = os.path.join("data", "raw")

# Patterns of files that are diagrams, illustrations, charts, or non-skin
non_skin_keywords = [
    "soap", "formation", "awareness", "pathogenesis", "classification",
    "pores", "text-book", "mechanisms", "plate", "chart", "diagram",
    "illustration", "siegerehrung", "bikini", "gum"
]

removed_count = 0
for folder in os.listdir(RAW_DIR):
    folder_path = os.path.join(RAW_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        lower = fname.lower()
        if any(k in lower for k in non_skin_keywords):
            fpath = os.path.join(folder_path, fname)
            try:
                os.remove(fpath)
                print(f"Removed non-skin asset: {folder}/{fname}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {fpath}: {e}")

print(f"\nTotal non-skin assets removed: {removed_count}")
