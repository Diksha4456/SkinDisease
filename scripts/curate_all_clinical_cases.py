import os
import glob
from PIL import Image

RAW_DIR = os.path.join("data", "raw")
acne_dir = os.path.join(RAW_DIR, "acne")
mel_dir = os.path.join(RAW_DIR, "melasma")
pso_dir = os.path.join(RAW_DIR, "psoriasis")
rw_dir = os.path.join(RAW_DIR, "ringworm")
vit_dir = os.path.join(RAW_DIR, "vitiligo")

for d in [acne_dir, mel_dir, pso_dir, rw_dir, vit_dir]:
    os.makedirs(d, exist_ok=True)

sc_dir = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots"

cases = [
    # 1. Forehead Acne Vulgaris (Screenshot 2026-10-14 214534.png)
    ("acne", "Screenshot 2026-10-14 214534.png", "user_forehead_acne", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.10, 0.85, 0.90),
        (0.20, 0.05, 0.80, 0.70),
        (0.10, 0.25, 0.90, 0.85),
        (0.25, 0.15, 0.75, 0.65),
    ]),
    # 2. Cheek Melasma (Screenshot 2026-10-14 195441.png)
    ("melasma", "Screenshot 2026-10-14 195441.png", "user_cheek_melasma_patch", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.15, 0.85, 0.85),
        (0.20, 0.20, 0.80, 0.80),
        (0.10, 0.05, 0.90, 0.75),
        (0.25, 0.25, 0.75, 0.75),
    ]),
    # 3. Facial Melasma (Screenshot 2026-10-14 195532.png)
    ("melasma", "Screenshot 2026-10-14 195532.png", "user_facial_melasma_patch", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.10, 0.85, 0.90),
        (0.20, 0.20, 0.80, 0.80),
        (0.10, 0.15, 0.90, 0.75),
    ]),
    # 4. Plaque Psoriasis on Back/Body (Screenshot 2026-10-14 195622.png)
    ("psoriasis", "Screenshot 2026-10-14 195622.png", "user_back_psoriasis_plaque", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.10, 0.85, 0.90),
        (0.20, 0.15, 0.80, 0.85),
        (0.25, 0.20, 0.75, 0.75),
    ]),
    # 5. Elbow Plaque Psoriasis (Screenshot 2026-10-14 214608.png)
    ("psoriasis", "Screenshot 2026-10-14 214608.png", "user_elbow_psoriasis", [
        (0.0, 0.0, 1.0, 1.0),
        (0.10, 0.15, 0.85, 0.85),
        (0.15, 0.20, 0.75, 0.80),
        (0.20, 0.25, 0.70, 0.75),
    ]),
    # 6. Leg Ringworm (Screenshot 2026-10-14 214633.png)
    ("ringworm", "Screenshot 2026-10-14 214633.png", "user_leg_ringworm", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.15, 0.85, 0.85),
        (0.20, 0.20, 0.80, 0.80),
        (0.25, 0.25, 0.75, 0.75),
    ]),
    # 7. Dorsal Hand Vitiligo (Screenshot 2026-10-14 214738.png)
    ("vitiligo", "Screenshot 2026-10-14 214738.png", "user_hand_vitiligo", [
        (0.0, 0.0, 1.0, 1.0),
        (0.10, 0.10, 0.90, 0.90),
        (0.20, 0.15, 0.80, 0.85),
        (0.15, 0.25, 0.85, 0.80),
    ]),
    # 8. Portrait Vitiligo (Screenshot 2026-10-14 214900.png)
    ("vitiligo", "Screenshot 2026-10-14 214900.png", "user_portrait_vitiligo", [
        (0.0, 0.0, 1.0, 1.0),
        (0.10, 0.05, 0.90, 0.95),
        (0.15, 0.10, 0.85, 0.75),
        (0.20, 0.05, 0.80, 0.65),
    ]),
    # 9. Perioral Vitiligo Child (Screenshot 2026-10-14 214753.png)
    ("vitiligo", "Screenshot 2026-10-14 214753.png", "user_perioral_vitiligo", [
        (0.0, 0.0, 1.0, 1.0),
        (0.15, 0.15, 0.85, 0.85),
        (0.20, 0.20, 0.80, 0.80),
    ])
]

total_generated = 0
for disease, filename, prefix, crop_coords in cases:
    src_path = os.path.join(sc_dir, filename)
    target_dir = os.path.join(RAW_DIR, disease)
    if not os.path.exists(src_path):
        print(f"[Skip] {filename} not found.")
        continue
    
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    print(f"Processing {filename} ({w}x{h}) -> {disease}")
    
    for i, (x1, y1, x2, y2) in enumerate(crop_coords):
        cropped = im.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
        out_name = f"{prefix}_c{i}.jpg"
        cropped.save(os.path.join(target_dir, out_name), "JPEG", quality=95)
        # Horizontal flip
        flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(os.path.join(target_dir, f"{prefix}_c{i}_flip.jpg"), "JPEG", quality=95)
        total_generated += 2

print(f"\nSuccessfully generated {total_generated} high-resolution clinical multi-scale samples across all 5 classes.")
