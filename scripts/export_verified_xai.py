import os
import requests
import base64
from PIL import Image
import io

SERVER_URL = "http://127.0.0.1:5000/api/screen"
OUT_DIR = "reports/xai_evidence"
os.makedirs(OUT_DIR, exist_ok=True)

test_images = [
    ("user_facial_vitiligo", r"C:\Users\DIKSHA\OneDrive\Desktop\WhatsApp Image 2026-10-14 at 18.54.52.jpeg"),
    ("user_facial_melasma", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\melasma.png"),
    ("user_plaque_psoriasis", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 195622.png")
]

for name, p in test_images:
    with open(p, "rb") as f:
        resp = requests.post(SERVER_URL, files={"image": f})
    data = resp.json()
    b64_heat = data.get("gradcam_b64")
    if b64_heat:
        img_bytes = base64.b64decode(b64_heat)
        img = Image.open(io.BytesIO(img_bytes))
        out_path = os.path.join(OUT_DIR, f"{name}_gradcam_verified.jpg")
        img.save(out_path)
        print(f"Saved verified Grad-CAM overlay for {name} ({data['predicted_disease']} - {data['confidence_pct']}%) -> {out_path}")
