import urllib.request
import re
from concurrent.futures import ThreadPoolExecutor

targets = ["acne", "melasma", "cloasma", "chloasma", "psorias", "tinea", "vitiligo"]

def check_id(did):
    d_url = f"http://www.atlasdermatologico.com.br/disease.jsf?diseaseId={did}"
    req = urllib.request.Request(d_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=6).read().decode("utf-8", errors="ignore")
        title_m = re.search(r"<h2>([^<]+)</h2>", html)
        if title_m:
            name = title_m.group(1).strip()
            img_ids = set(re.findall(r"imageId=(\d+)", html))
            lower = name.lower()
            for t in targets:
                if t in lower:
                    print(f"Match [{t}]: did={did} -> '{name}' ({len(img_ids)} images)")
                    return (did, name, t, len(img_ids))
    except Exception:
        pass
    return None

if __name__ == "__main__":
    print("Scanning Atlas Dermatologico for target diseases...")
    with ThreadPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(check_id, range(1, 550)))
    
    matches = [r for r in results if r is not None]
    print(f"\nTotal matches found: {len(matches)}")
    for m in matches:
        print(m)
