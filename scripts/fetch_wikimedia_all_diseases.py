import os
import json
import urllib.request
from urllib.parse import urlparse

RAW_DIR = os.path.join("data", "raw")
headers = {"User-Agent": "FairSkinAIResearch/1.0 (academic dermatological research; contact: research@example.org)"}

category_map = {
    "acne": ["Category:Acne_vulgaris", "Category:Acne"],
    "melasma": ["Category:Melasma", "Category:Hyperpigmentation"],
    "psoriasis": ["Category:Psoriasis", "Category:Plaque_psoriasis"],
    "ringworm": ["Category:Tinea_corporis", "Category:Dermatophytosis", "Category:Tinea"],
    "vitiligo": ["Category:Vitiligo"]
}

total_new = 0

for disease, cats in category_map.items():
    dest_dir = os.path.join(RAW_DIR, disease)
    os.makedirs(dest_dir, exist_ok=True)
    for cat in cats:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=categorymembers&gcmtitle={cat}&gcmlimit=30&prop=imageinfo&iiprop=url|size"
        try:
            req = urllib.request.Request(url, headers=headers)
            data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                ii = p.get("imageinfo", [{}])[0]
                img_url = ii.get("url", "")
                title = p.get("title", "")
                path_part = urlparse(img_url).path.lower()
                if not any(path_part.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                    continue
                clean_name = "".join(c for c in title.replace("File:", "").replace(" ", "_") if c.isalnum() or c in "_-.")[:35]
                fname = f"wiki_{pid}_{clean_name}.jpg"
                dest = os.path.join(dest_dir, fname)
                if not os.path.exists(dest):
                    try:
                        r_img = urllib.request.Request(img_url, headers=headers)
                        content = urllib.request.urlopen(r_img, timeout=10).read()
                        if len(content) > 5000:
                            with open(dest, "wb") as f:
                                f.write(content)
                            total_new += 1
                            print(f"[{disease}] Downloaded {fname}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[{disease}] Error fetching {cat}: {e}")

print(f"\nDone. Downloaded {total_new} new verified clinical images from Wikimedia Commons.")
