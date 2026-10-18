import os
import requests
import json

SERVER_URL = "http://127.0.0.1:5000/api/screen"

test_cases = [
    {
        "name": "User Facial Vitiligo",
        "path": r"C:\Users\DIKSHA\OneDrive\Desktop\WhatsApp Image 2026-10-14 at 18.54.52.jpeg",
        "expected": "Vitiligo"
    },
    {
        "name": "User Facial Melasma Screenshot",
        "path": r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\melasma.png",
        "expected": "Melasma"
    },
    {
        "name": "User Plaque Psoriasis Screenshot",
        "path": r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 195622.png",
        "expected": "Psoriasis"
    }
]

# Add representative samples from each class in data/raw
for disease in ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]:
    d_dir = os.path.join("data", "raw", disease)
    if os.path.exists(d_dir):
        files = [os.path.join(d_dir, f) for f in os.listdir(d_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            # Pick the 2nd or 3rd sample
            pick_idx = min(2, len(files) - 1)
            expected_name = "Condition Not Recognized" if disease == "non_target_ood" else disease.title()
            test_cases.append({
                "name": f"Clinical Sample: {disease}",
                "path": files[pick_idx],
                "expected": expected_name
            })

print(f"Executing self-checks across {len(test_cases)} test images on {SERVER_URL}...\n")
results = []

for tc in test_cases:
    path = tc["path"]
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        continue
    
    with open(path, "rb") as f:
        resp = requests.post(SERVER_URL, files={"image": f})
    
    if resp.status_code == 200:
        data = resp.json()
        pred = data.get("predicted_disease", "N/A")
        conf = data.get("confidence_pct", 0)
        tone = data.get("ita_tone_band", "N/A")
        features = data.get("observable_features", [])
        warning = data.get("quality_warning", None)
        
        passed = (tc["expected"].lower() in pred.lower())
        status_sym = "[PASS]" if passed else "[CHECK]"
        
        print(f"{status_sym} {tc['name']}")
        print(f"       Expected: {tc['expected']} | Predicted: {pred} ({conf}%) | Tone: {tone}")
        print(f"       Features: {', '.join(features[:3])}")
        if warning:
            print(f"       Quality Notice: {warning}")
        print()
        
        results.append({
            "name": tc["name"],
            "expected": tc["expected"],
            "predicted": pred,
            "confidence": conf,
            "tone": tone,
            "passed": passed
        })
    else:
        print(f"[FAIL] Server error ({resp.status_code}) on {tc['name']}: {resp.text}\n")

total = len(results)
num_passed = sum(1 for r in results if r["passed"])
print("=" * 60)
print(f"Self-Check Summary: {num_passed}/{total} correctly predicted ({num_passed/total*100:.1f}%)")
print("=" * 60)
