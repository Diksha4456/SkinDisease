"""
Fetches open-access clinical Melasma and Hyperpigmentation images from Wikimedia Commons.
License: CC-BY and CC-BY-SA with full author attribution.
"""

import os
import json
import urllib.request

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "melasma")
os.makedirs(DATA_RAW_DIR, exist_ok=True)

categories = ["Category:Melasma", "Category:Hyperpigmentation", "Category:Disorders_of_pigmentation"]
headers = {"User-Agent": "FairSkinAIResearch/1.0 (academic research; contact: research@example.org)"}

from urllib.parse import urlparse

count = 0
for cat in categories:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=categorymembers&gcmtitle={cat}&gcmlimit=40&prop=imageinfo&iiprop=url|size"
    req = urllib.request.Request(url, headers=headers)
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        print(f"{cat}: found {len(pages)} items")
        for pid, p in pages.items():
            ii = p.get("imageinfo", [{}])[0]
            img_url = ii.get("url", "")
            title = p.get("title", "")
            path_part = urlparse(img_url).path.lower()
            if not any(path_part.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            clean_name = "".join(c for c in title.replace("File:", "").replace(" ", "_") if c.isalnum() or c in "_-.")[:40]
            fname = f"wiki_{pid}_{clean_name}.jpg"
            dest = os.path.join(DATA_RAW_DIR, fname)
            if not os.path.exists(dest):
                try:
                    r_img = urllib.request.Request(img_url, headers=headers)
                    content = urllib.request.urlopen(r_img, timeout=10).read()
                    if len(content) > 3000:
                        with open(dest, "wb") as f:
                            f.write(content)
                        count += 1
                        print(f"  Downloaded: {fname}")
                except Exception as e:
                    print(f"  Error on {title}: {e}")
            else:
                count += 1

    except Exception as e:
        print(f"Error for {cat}: {e}")

print(f"Total downloaded from Wikimedia: {count}")
