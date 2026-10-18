import urllib.request
import re

url = "http://www.atlasdermatologico.com.br/browse.jsf"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    matches = re.findall(r'<a href="disease\.jsf\?diseaseId=(\d+)">([^<]+)</a>', html)
    print(f"Total diseases on Atlas: {len(matches)}")
    for did, name in matches:
        lower = name.lower()
        if any(k in lower for k in ["melasma", "chloasma", "cloasma", "pigment", "lentigo", "efelid"]):
            print(f"  did={did}: {name}")
except Exception as e:
    print("Error:", e)
