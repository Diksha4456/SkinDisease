import os
import urllib.request
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

RAW_DIR = os.path.join("data", "raw")
acne_dir = os.path.join(RAW_DIR, "acne")
mel_dir = os.path.join(RAW_DIR, "melasma")
os.makedirs(acne_dir, exist_ok=True)
os.makedirs(mel_dir, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 1. Download Atlas Dermatologico Acne (Disease ID 3)
def download_atlas_acne():
    url = "http://www.atlasdermatologico.com.br/disease.jsf?diseaseId=3"
    req = urllib.request.Request(url, headers=HEADERS)
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    img_ids = sorted(set(re.findall(r"imageId=(\d+)", html)), key=int)
    print(f"Found {len(img_ids)} Acne images on Atlas Dermatologico.")
    
    def fetch_img(iid):
        dest = os.path.join(acne_dir, f"atlas_acne_{iid}.jpg")
        if not os.path.exists(dest):
            i_url = f"http://atlasdermatologico.com.br/img?imageId={iid}"
            try:
                r = urllib.request.Request(i_url, headers=HEADERS)
                b = urllib.request.urlopen(r, timeout=10).read()
                if len(b) > 2000:
                    with open(dest, "wb") as f:
                        f.write(b)
                    return True
            except Exception:
                return False
        return True

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_img, img_ids))
    print(f"Downloaded/verified {sum(results)} Atlas Acne images.")

# 2. Add real-world clinical photos of Acne and Melasma from Wikimedia
def download_wikimedia_clinical():
    wiki_acne_urls = [
        ("wiki_acne_akne_jugend.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Akne-jugend.jpg"),
        ("wiki_acne_vulgaris_2.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/47/Acne_vulgaris_2.jpg"),
        ("wiki_acne_0601.jpg", "https://upload.wikimedia.org/wikipedia/commons/b/b3/0601_Acne_Vulgaris.jpg"),
        ("wiki_acne_0602.jpg", "https://upload.wikimedia.org/wikipedia/commons/e/e9/0602_Acne_Vulgaris.jpg"),
        ("wiki_acne_0603.jpg", "https://upload.wikimedia.org/wikipedia/commons/d/dd/0603_Acne_Vulgaris.jpg"),
        ("wiki_acne_comedonica.jpg", "https://upload.wikimedia.org/wikipedia/commons/c/cd/Acne_comedonica-2.jpg"),
    ]
    for name, u in wiki_acne_urls:
        dest = os.path.join(acne_dir, name)
        if not os.path.exists(dest):
            try:
                req = urllib.request.Request(u, headers=HEADERS)
                b = urllib.request.urlopen(req, timeout=10).read()
                if len(b) > 2000:
                    with open(dest, "wb") as f:
                        f.write(b)
                    print(f"Downloaded Wikimedia Acne: {name}")
            except Exception as e:
                print(f"Note: {name} ({e})")

# 3. Curate crops from the user's second melasma screenshot (Screenshot 2026-10-14 214936.png)
def curate_user_melasma_screenshot_2():
    src = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214936.png"
    if os.path.exists(src):
        im = Image.open(src).convert("RGB")
        w, h = im.size
        # The woman's face has melasma on cheek, forehead, and temple
        crops = [
            ("user_mel_2_full_face.jpg", im.crop((0, 0, int(w * 0.85), h))),
            ("user_mel_2_cheek.jpg", im.crop((int(w * 0.15), int(h * 0.20), int(w * 0.80), int(h * 0.80)))),
            ("user_mel_2_forehead_temple.jpg", im.crop((int(w * 0.10), 0, int(w * 0.80), int(h * 0.50)))),
            ("user_mel_2_malar.jpg", im.crop((int(w * 0.20), int(h * 0.25), int(w * 0.70), int(h * 0.70)))),
            ("user_mel_2_lower_cheek.jpg", im.crop((int(w * 0.15), int(h * 0.40), int(w * 0.75), int(h * 0.90)))),
        ]
        count = 0
        for name, crp in crops:
            dest = os.path.join(mel_dir, name)
            crp.save(dest, "JPEG", quality=95)
            # Flipped
            crp.transpose(Image.FLIP_LEFT_RIGHT).save(
                os.path.join(mel_dir, name.replace(".jpg", "_flip.jpg")), "JPEG", quality=95
            )
            count += 2
        print(f"Generated {count} clinical samples from user melasma screenshot 2.")

download_atlas_acne()
download_wikimedia_clinical()
curate_user_melasma_screenshot_2()

print(f"\nFinal Raw Dataset Counts:")
for d in ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]:
    dp = os.path.join(RAW_DIR, d)
    print(f"  {d:<16}: {len(os.listdir(dp))} files")
