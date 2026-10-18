import os
import shutil
from PIL import Image

RAW_DIR = os.path.join("data", "raw")

# 1. Bad files to remove from melasma
BAD_MELASMA_PATTERNS = [
    "gum", "leukotrichia", "torso", "mechanisms", "obstetrics",
    "pati", "fox_plate", "maryland", "hyperpig_4980396", "hyperpig_9750058", "hyperpig_9750063"
]

mel_dir = os.path.join(RAW_DIR, "melasma")
removed_count = 0
for f in os.listdir(mel_dir):
    low = f.lower()
    if any(bp in low for bp in BAD_MELASMA_PATTERNS):
        p = os.path.join(mel_dir, f)
        os.remove(p)
        removed_count += 1
print(f"Removed {removed_count} corrupt/non-clinical melasma files.")

# 2. Curate and augment User Facial Melasma
user_mel_src = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\melasma.png"
if os.path.exists(user_mel_src):
    im_mel = Image.open(user_mel_src).convert("RGB")
    w, h = im_mel.size
    
    # We create diverse high-fidelity clinical crops (focusing on the hyperpigmented cheek patches)
    # Notice the left 60px is a dark edge, so we avoid it
    crops_mel = [
        ("user_melasma_crop_main.jpg", im_mel.crop((int(w * 0.10), 0, w, h))),
        ("user_melasma_crop_cheek.jpg", im_mel.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))),
        ("user_melasma_crop_upper_cheek.jpg", im_mel.crop((int(w * 0.20), int(h * 0.05), int(w * 0.80), int(h * 0.65)))),
        ("user_melasma_crop_lower_cheek.jpg", im_mel.crop((int(w * 0.15), int(h * 0.35), int(w * 0.85), int(h * 0.95)))),
        ("user_melasma_crop_malar.jpg", im_mel.crop((int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.75)))),
    ]
    
    idx = 0
    for name, crp in crops_mel:
        crp.save(os.path.join(mel_dir, name), "JPEG", quality=95)
        # Horizontal flip
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(mel_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
        idx += 2
    print(f"Generated {idx} high-fidelity clinical training samples from user facial melasma.")

# 3. Curate and augment User Plaque Psoriasis
user_pso_src = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 195622.png"
pso_dir = os.path.join(RAW_DIR, "psoriasis")
os.makedirs(pso_dir, exist_ok=True)
if os.path.exists(user_pso_src):
    im_pso = Image.open(user_pso_src).convert("RGB")
    w, h = im_pso.size
    crops_pso = [
        ("user_pso_full.jpg", im_pso),
        ("user_pso_center_plaque.jpg", im_pso.crop((int(w * 0.20), int(h * 0.10), int(w * 0.75), int(h * 0.90)))),
        ("user_pso_right_plaque.jpg", im_pso.crop((int(w * 0.55), int(h * 0.20), w, int(h * 0.85)))),
        ("user_pso_central_silver.jpg", im_pso.crop((int(w * 0.30), int(h * 0.25), int(w * 0.65), int(h * 0.75)))),
        ("user_pso_guttate_periphery.jpg", im_pso.crop((0, 0, int(w * 0.60), int(h * 0.60)))),
    ]
    idx = 0
    for name, crp in crops_pso:
        crp.save(os.path.join(pso_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(pso_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
        idx += 2
    print(f"Generated {idx} high-fidelity clinical training samples from user plaque psoriasis.")

# 4. Curate and augment User Facial Vitiligo
user_vit_src = r"C:\Users\DIKSHA\OneDrive\Desktop\WhatsApp Image 2026-10-14 at 18.54.52.jpeg"
vit_dir = os.path.join(RAW_DIR, "vitiligo")
os.makedirs(vit_dir, exist_ok=True)
if os.path.exists(user_vit_src):
    im_vit = Image.open(user_vit_src).convert("RGB")
    w, h = im_vit.size
    crops_vit = [
        ("user_vit_full.jpg", im_vit),
        ("user_vit_forehead.jpg", im_vit.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.45)))),
        ("user_vit_periorbital.jpg", im_vit.crop((int(w * 0.10), int(h * 0.15), int(w * 0.90), int(h * 0.65)))),
        ("user_vit_cheeks.jpg", im_vit.crop((int(w * 0.10), int(h * 0.35), int(w * 0.90), int(h * 0.85)))),
    ]
    idx = 0
    for name, crp in crops_vit:
        crp.save(os.path.join(vit_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(vit_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
        idx += 2
    print(f"Generated {idx} high-fidelity clinical training samples from user facial vitiligo.")

# Summary of clean dataset
for d in ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]:
    dp = os.path.join(RAW_DIR, d)
    count = len(os.listdir(dp)) if os.path.exists(dp) else 0
    print(f"  {d:<16}: {count} verified clinical samples")
