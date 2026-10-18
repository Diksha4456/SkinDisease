import urllib.request
import re
from concurrent.futures import ThreadPoolExecutor

def check_id(did):
    url = f"http://www.atlasdermatologico.com.br/disease.jsf?diseaseId={did}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8", errors="ignore")
        m = re.search(r"<h2>([^<]+)</h2>", html)
        if m:
            name = m.group(1).strip()
            imgs = re.findall(r"imageId=(\d+)", html)
            if "acne" in name.lower() or "melasma" in name.lower() or "cloasma" in name.lower():
                return (did, name, len(set(imgs)))
    except Exception:
        pass
    return None

with ThreadPoolExecutor(max_workers=10) as ex:
    results = [r for r in ex.map(check_id, range(1, 100)) if r is not None]

for r in results:
    print(r)
