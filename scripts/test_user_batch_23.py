import sys, os
sys.path.append(os.path.abspath("."))
import requests

shots = [
    ("230327 (Psoriasis arms)", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230327.png"),
    ("230513 (Melasma cheek)", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230513.png"),
    ("230937 (Acne cheek)", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 230937.png"),
    ("231135 (Psoriasis plaque)", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 231135.png")
]

for name, p in shots:
    with open(p, "rb") as f:
        resp = requests.post("http://127.0.0.1:5000/api/screen", files={"image": f})
    data = resp.json()
    print(f"=== {name} ===")
    print(f"  Prediction: {data.get('predicted_disease')} ({data.get('confidence_pct')}%)")
    print(f"  Features:   {data.get('observable_features')}")
