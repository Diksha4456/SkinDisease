import urllib.request
import re
from concurrent.futures import ThreadPoolExecutor

targets = []

def check_id(did):
    d_url = f"http://www.atlasdermatologico.com.br/disease.jsf?diseaseId={did}"
    req = urllib.request.Request(d_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8", errors="ignore")
        title_m = re.search(r"<h2>([^<]+)</h2>", html)
        if title_m:
            name = title_m.group(1).strip()
            img_ids = re.findall(r"imageId=(\d+)", html)
            unique_imgs = len(set(img_ids))
            lower = name.lower()
            if any(k in lower for k in ["melasma", "cloasma", "chloasma", "hipercromia", "pigment", "lentigo", "discromia", "efel"]):
                print(f"Match: did={did} -> '{name}' ({unique_imgs} images)")
                return (did, name, unique_imgs)
    except Exception:
        pass
    return None

with ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(check_id, range(1, 520)))

matches = [r for r in results if r is not None]
print(f"\nTotal matches found: {len(matches)}")
for m in matches:
    print(m)
