import os
import requests

SERVER_URL = "http://127.0.0.1:5000/api/screen"

acne_dir = "data/raw/acne"
atlas_acne_files = [f for f in os.listdir(acne_dir) if "atlas" in f][:15]

print(f"Testing {len(atlas_acne_files)} Atlas Acne samples on live server:")
correct = 0
for f in atlas_acne_files:
    p = os.path.join(acne_dir, f)
    with open(p, "rb") as img_file:
        resp = requests.post(SERVER_URL, files={"image": img_file})
    data = resp.json()
    pred = data.get("predicted_disease", "")
    conf = data.get("confidence_pct", 0)
    is_acne = "acne" in pred.lower()
    if is_acne:
        correct += 1
    print(f"  {f:<22}: {pred:<18} ({conf}%) {'[OK]' if is_acne else '[WRONG]'}")

print(f"Atlas Acne Accuracy: {correct}/{len(atlas_acne_files)} ({correct/len(atlas_acne_files)*100:.1f}%)")
