import os
from PIL import Image
import requests

RAW_DIR = os.path.join("data", "raw")
acne_dir = os.path.join(RAW_DIR, "acne")
pso_dir = os.path.join(RAW_DIR, "psoriasis")
mel_dir = os.path.join(RAW_DIR, "melasma")

os.makedirs(acne_dir, exist_ok=True)
os.makedirs(pso_dir, exist_ok=True)
os.makedirs(mel_dir, exist_ok=True)

# 1. Curate User Cheek Acne (Screenshot 2026-10-14 230937.png)
src_acne = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230937.png"
if os.path.exists(src_acne):
    im = Image.open(src_acne).convert("RGB")
    w, h = im.size
    crops_acne = [
        ("user_acne_full.jpg", im),
        ("user_acne_center_papules.jpg", im.crop((int(w * 0.15), int(h * 0.10), int(w * 0.85), int(h * 0.90)))),
        ("user_acne_upper_pustules.jpg", im.crop((int(w * 0.20), int(h * 0.05), int(w * 0.80), int(h * 0.65)))),
        ("user_acne_lower_papules.jpg", im.crop((int(w * 0.15), int(h * 0.35), int(w * 0.85), int(h * 0.95)))),
        ("user_acne_macro_cluster.jpg", im.crop((int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.75)))),
    ]
    count = 0
    for name, crp in crops_acne:
        crp.save(os.path.join(acne_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(acne_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
        count += 2
    print(f"Generated {count} clinical samples from user acne screenshot.")

# 2. Curate User Psoriasis Plaque (Screenshot 2026-10-14 231135.png and 230327.png)
src_pso1 = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 231135.png"
if os.path.exists(src_pso1):
    im = Image.open(src_pso1).convert("RGB")
    w, h = im.size
    crops_pso1 = [
        ("user_pso_macro_plaque.jpg", im),
        ("user_pso_macro_silver_scale.jpg", im.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))),
        ("user_pso_macro_erythema_border.jpg", im.crop((int(w * 0.10), int(h * 0.05), int(w * 0.90), int(h * 0.90)))),
    ]
    for name, crp in crops_pso1:
        crp.save(os.path.join(pso_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(pso_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
    print("Generated 6 clinical samples from user psoriasis close-up.")

src_pso2 = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230327.png"
if os.path.exists(src_pso2):
    im = Image.open(src_pso2).convert("RGB")
    w, h = im.size
    crops_pso2 = [
        ("user_pso_arms_full.jpg", im),
        ("user_pso_upper_arm_plaque.jpg", im.crop((int(w * 0.20), 0, int(w * 0.80), int(h * 0.55)))),
        ("user_pso_lower_arm_plaque.jpg", im.crop((int(w * 0.10), int(h * 0.40), int(w * 0.70), h))),
    ]
    for name, crp in crops_pso2:
        crp.save(os.path.join(pso_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(pso_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
    print("Generated 6 clinical samples from user arm psoriasis.")

# 3. Curate User Melasma (Screenshot 2026-10-14 230513.png)
src_mel = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230513.png"
if os.path.exists(src_mel):
    im = Image.open(src_mel).convert("RGB")
    w, h = im.size
    crops_mel = [
        ("user_mel_3_full.jpg", im),
        ("user_mel_3_cheek_pigment.jpg", im.crop((int(w * 0.20), int(h * 0.10), int(w * 0.85), int(h * 0.70)))),
        ("user_mel_3_upper_cheek.jpg", im.crop((int(w * 0.30), 0, int(w * 0.80), int(h * 0.60)))),
    ]
    for name, crp in crops_mel:
        crp.save(os.path.join(mel_dir, name), "JPEG", quality=95)
        crp.transpose(Image.FLIP_LEFT_RIGHT).save(
            os.path.join(mel_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
        )
    print("Generated 6 clinical samples from user melasma screenshot 3.")

# 4. Download additional open clinical acne photos from Wikimedia Commons
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
extra_acne = [
    ("wiki_acne_pustulosa.jpg", "https://upload.wikimedia.org/wikipedia/commons/e/ee/Acne-pustulosa.jpg"),
    ("wiki_acne_papulopustulosa.jpg", "https://upload.wikimedia.org/wikipedia/commons/9/91/Acne_papulopustulosa.jpg"),
    ("wiki_acne_face_severe.jpg", "https://upload.wikimedia.org/wikipedia/commons/0/08/Severe_acne_on_face.jpg"),
    ("wiki_acne_nodulocystic.jpg", "https://upload.wikimedia.org/wikipedia/commons/3/36/Nodulocystic_acne_on_the_face.jpg"),
    ("wiki_acne_comedonal_forehead.jpg", "https://upload.wikimedia.org/wikipedia/commons/b/bc/Comedonal_acne.jpg"),
]
for fname, u in extra_acne:
    dest = os.path.join(acne_dir, fname)
    if not os.path.exists(dest):
        try:
            r = requests.get(u, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.content) > 2000:
                with open(dest, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded extra clinical acne: {fname}")
        except Exception:
            pass

print("\nCuration complete.")
for d in ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]:
    dp = os.path.join(RAW_DIR, d)
    print(f"  {d:<16}: {len(os.listdir(dp))} files")
