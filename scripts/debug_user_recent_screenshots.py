import os
import requests
from PIL import Image

shots = [
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214534.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214608.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214633.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214738.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214753.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214900.png",
    r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214936.png",
]

for s in shots:
    if os.path.exists(s):
        im = Image.open(s)
        with open(s, "rb") as f:
            resp = requests.post("http://127.0.0.1:5000/api/screen", files={"image": f})
        data = resp.json() if resp.status_code == 200 else {"error": resp.text}
        print(f"=== {os.path.basename(s)}: size={im.size} mode={im.mode} ===")
        print(f"   Predicted: {data.get('predicted_disease')} ({data.get('confidence_pct')}%)")
        print(f"   Features: {data.get('observable_features')}")
        print(f"   Rationale: {data.get('clinical_rationale')}")
        print()
