import urllib.request
import requests
import io

headers = {"User-Agent": "FairSkinApp/1.0 (contact: test@gmail.com)"}

urls = [
    ("Akne-jugend (Face)", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Akne-jugend.jpg"),
    ("Acne vulgaris 2 (Cheek)", "https://upload.wikimedia.org/wikipedia/commons/4/47/Acne_vulgaris_2.jpg"),
    ("Acne on the back (Pustules)", "https://upload.wikimedia.org/wikipedia/commons/0/07/Acne_on_the_back.jpg"),
    ("User Melasma 214936", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214936.png"),
]

for name, u in urls:
    try:
        if u.startswith("http"):
            req = urllib.request.Request(u, headers=headers)
            img_bytes = urllib.request.urlopen(req, timeout=10).read()
        else:
            with open(u, "rb") as f:
                img_bytes = f.read()
        
        resp = requests.post("http://127.0.0.1:5000/api/screen", files={"image": ("img.jpg", img_bytes, "image/jpeg")})
        data = resp.json()
        print(f"=== {name} ===")
        print(f"   Prediction: {data.get('predicted_disease')} ({data.get('confidence_pct')}%)")
        print(f"   Features:   {data.get('observable_features')}")
        print(f"   Severity:   {data.get('acne_severity')}")
    except Exception as e:
        print(f"{name}: Error {e}")
