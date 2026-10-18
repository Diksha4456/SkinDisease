import urllib.request
import json
from urllib.parse import quote
import os
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SERVER_URL = "http://127.0.0.1:5000/api/screen"

def fetch_and_test(query: str, disease: str, max_items: int = 4):
    out_dir = os.path.join("scratch", "online_test", disease)
    os.makedirs(out_dir, exist_ok=True)
    
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
        f"&gsrsearch={quote(query)}&gsrnamespace=6&gsrlimit={max_items*3}&prop=imageinfo&iiprop=url"
    )
    req = urllib.request.Request(api_url, headers=HEADERS)
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
    except Exception as e:
        print(f"Failed query for {query}: {e}")
        return

    print(f"\n=======================================================")
    print(f"Testing Real Online Images for: {disease.upper()} (Query: '{query}')")
    print(f"=======================================================")

    count = 0
    for pid, p in pages.items():
        ii = p.get("imageinfo", [{}])[0]
        url = ii.get("url", "")
        title = p.get("title", "")
        if not any(url.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
            continue
        # Avoid PDFs and SVGs
        if ".pdf" in url.lower() or ".svg" in url.lower():
            continue
        
        dest = os.path.join(out_dir, f"test_{count}.jpg")
        try:
            r = urllib.request.Request(url, headers=HEADERS)
            with open(dest, "wb") as f:
                f.write(urllib.request.urlopen(r, timeout=10).read())
        except Exception:
            continue
        
        # Test on live server
        with open(dest, "rb") as f:
            resp = requests.post(SERVER_URL, files={"image": f})
        
        if resp.status_code == 200:
            res_data = resp.json()
            pred = res_data.get("predicted_disease", "N/A")
            conf = res_data.get("confidence_pct", 0)
            feats = res_data.get("observable_features", [])
            is_match = disease.lower() in pred.lower()
            sym = "[PASS]" if is_match else "[FAIL]"
            clean_title = title.replace("File:", "")[:40]
            print(f"  {sym} {clean_title}")
            print(f"         Prediction: {pred:<20} ({conf}%)")
            print(f"         Features:   {', '.join(feats[:3])}")
            count += 1
            if count >= max_items:
                break

fetch_and_test("Acne vulgaris", "acne", 4)
fetch_and_test("Melasma", "melasma", 4)
