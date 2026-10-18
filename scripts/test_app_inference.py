"""
Test local offline screening endpoint with genuine clinical images.
"""

import os
import requests

test_cases = [
    ("acne", "data/raw/acne"),
    ("psoriasis", "data/raw/psoriasis"),
    ("ringworm", "data/raw/ringworm"),
    ("vitiligo", "data/raw/vitiligo"),
    ("melasma", "data/raw/melasma")
]

print("=" * 70)
print("TESTING FAIRSKIN AI OFFLINE INFERENCE ENDPOINT")
print("=" * 70)

for label, folder in test_cases:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            img_path = os.path.join(folder, files[0])
            with open(img_path, "rb") as f:
                r = requests.post("http://127.0.0.1:5000/api/screen", files={"image": f}, timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    pred = d["predicted_disease"]
                    conf = d["confidence_pct"]
                    fst = d["fitzpatrick_scale"]
                    sev = d["acne_severity"]
                    tone = d["ita_tone_band"]
                    print(f"Ground Truth: {label:<10} -> Predicted: {pred:<18} | Conf: {conf:5.1f}% | Tone: {tone:<12} (FST {fst}) | Severity: {sev}")
                else:
                    print(f"Error {r.status_code} on {label}: {r.text[:80]}")
print("=" * 70)
