"""
Dedicated clinical Melasma downloader from NLM Open-i (PubMed Central Open Access CC-BY).
Fetches peer-reviewed clinical photographs with full provenance.
"""

import os
import sys
import json
import urllib.request
import time

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "melasma")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

openi_url = "https://openi.nlm.nih.gov/api/search?query=melasma&m=1&n=30"
headers = {"User-Agent": "FairSkinAIResearch/1.0 (academic research; contact: research@example.org)"}

print("Querying NLM Open-i for melasma clinical figures (timeout=35s)...")
req = urllib.request.Request(openi_url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=35) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = data.get("list", [])
    print(f"Found {len(items)} Open-i clinical items")
    downloaded = 0
    for idx, it in enumerate(items):
        img_rel = it.get("imgLarge", "")
        pmcid = it.get("pmcid", "")
        title = it.get("title", "")
        if not img_rel:
            continue
        full_url = f"https://openi.nlm.nih.gov{img_rel}"
        fname = f"openi_pmc_{pmcid}_{os.path.basename(img_rel)}"
        dest = os.path.join(DATA_RAW_DIR, fname)
        if not os.path.exists(dest):
            try:
                r_img = urllib.request.Request(full_url, headers=headers)
                with urllib.request.urlopen(r_img, timeout=15) as r:
                    content = r.read()
                if len(content) > 3000:
                    with open(dest, "wb") as f:
                        f.write(content)
                    downloaded += 1
                    print(f"  [{idx+1}/{len(items)}] Saved: {fname} ({len(content)} bytes)")
            except Exception as e:
                print(f"  [{idx+1}/{len(items)}] Skip {fname}: {e}")
        else:
            downloaded += 1
    print(f"\nCompleted! Total Melasma images now available: {downloaded}")
except Exception as e:
    print(f"Error: {e}")
