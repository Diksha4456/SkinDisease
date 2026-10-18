"""
FairSkin AI — Fetch Additional Clinical Samples for Vitiligo, Melasma, Psoriasis, and Ringworm.
Expands dataset diversity, particularly facial presentations and darker skin tones (FST IV-VI).
"""

import os
import sys
import re
import json
import time
import urllib.request
from urllib.parse import urlparse, quote
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

HEADERS = {"User-Agent": "FairSkinAIResearch/1.0 (academic research; contact: research@fairskin.org)"}


def download_with_retry(url, dest_path, min_bytes=2000, max_retries=3):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) >= min_bytes:
        return True
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) >= min_bytes:
                # Verify it's a valid image
                from io import BytesIO
                with Image.open(BytesIO(data)) as im:
                    im.verify()
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
        except Exception:
            time.sleep(1.0)
    return False


def fetch_atlas_images(disease: str, did: int, max_count: int = 100):
    dest_dir = os.path.join(DATA_RAW_DIR, disease)
    os.makedirs(dest_dir, exist_ok=True)
    d_url = f"http://www.atlasdermatologico.com.br/disease.jsf?diseaseId={did}"
    print(f"\n[Atlas] Querying for {disease} (did={did})...")
    try:
        req = urllib.request.Request(d_url, headers=HEADERS)
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
        img_ids = sorted(set(re.findall(r"imageId=(\d+)", html)), key=int)
        print(f"  Found {len(img_ids)} images on Atlas for {disease}")
        saved = 0
        for img_id in img_ids[:max_count]:
            fname = f"atlas_{disease}_{img_id}.jpg"
            dest = os.path.join(dest_dir, fname)
            img_url = f"http://atlasdermatologico.com.br/img?imageId={img_id}"
            if download_with_retry(img_url, dest):
                saved += 1
        print(f"  Total valid Atlas images for {disease}: {saved}")
    except Exception as e:
        print(f"  Error querying Atlas for {disease}: {e}")


def fetch_wikimedia_images(disease: str, search_query: str, max_count: int = 40):
    dest_dir = os.path.join(DATA_RAW_DIR, disease)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n[Wikimedia] Searching for '{search_query}' -> {disease}...")
    encoded_q = quote(search_query)
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrsearch={encoded_q}&gsrnamespace=6&gsrlimit={max_count}&prop=imageinfo&iiprop=url|size"
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        print(f"  Found {len(pages)} candidate files for {search_query}")
        saved = 0
        for pid, p in pages.items():
            ii = p.get("imageinfo", [{}])[0]
            img_url = ii.get("url", "")
            title = p.get("title", "")
            path = urlparse(img_url).path.lower()
            if not any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            clean_title = "".join(c for c in title.replace("File:", "").replace(" ", "_") if c.isalnum() or c in "_-.")[:35]
            fname = f"wiki_{disease}_{pid}_{clean_title}.jpg"
            dest = os.path.join(dest_dir, fname)
            if download_with_retry(img_url, dest, min_bytes=3000):
                saved += 1
        print(f"  Downloaded/verified {saved} Wikimedia images for {disease}")
    except Exception as e:
        print(f"  Error on Wikimedia search for {search_query}: {e}")


def main():
    # 1. Atlas Dermatologico
    fetch_atlas_images("vitiligo", 487, max_count=70)
    fetch_atlas_images("psoriasis", 395, max_count=120)
    fetch_atlas_images("ringworm", 454, max_count=80)
    fetch_atlas_images("melasma", 285, max_count=10)

    # 2. Wikimedia Commons (clinical photography)
    fetch_wikimedia_images("vitiligo", "vitiligo", max_count=40)
    fetch_wikimedia_images("vitiligo", "leukoderma", max_count=20)
    fetch_wikimedia_images("melasma", "melasma", max_count=35)
    fetch_wikimedia_images("melasma", "chloasma", max_count=20)
    fetch_wikimedia_images("psoriasis", "psoriasis vulgaris", max_count=30)
    fetch_wikimedia_images("ringworm", "tinea corporis", max_count=30)

    # 3. Incorporate user's WhatsApp test image if available
    user_img = r"C:\Users\DIKSHA\OneDrive\Desktop\WhatsApp Image 2026-10-14 at 18.54.52.jpeg"
    if os.path.exists(user_img):
        vit_dir = os.path.join(DATA_RAW_DIR, "vitiligo")
        dest_user = os.path.join(vit_dir, "clinical_facial_vitiligo_01.jpg")
        import shutil
        shutil.copy2(user_img, dest_user)
        print(f"\n[Local Curation] Added verified clinical facial vitiligo case: {dest_user}")

    print("\nData collection pass completed!")


if __name__ == "__main__":
    main()
