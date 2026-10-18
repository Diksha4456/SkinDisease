import os
import requests

SERVER_URL = "http://127.0.0.1:5000/api/screen"

for disease in ["acne", "melasma"]:
    d_dir = os.path.join("data", "raw", disease)
    files = [f for f in os.listdir(d_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
    print(f"=== Testing 10 samples of {disease.upper()} ===")
    correct = 0
    for f in files:
        p = os.path.join(d_dir, f)
        with open(p, "rb") as img_file:
            resp = requests.post(SERVER_URL, files={"image": img_file})
        data = resp.json()
        pred = data.get("predicted_disease", "")
        conf = data.get("confidence_pct", 0)
        is_correct = disease.lower() in pred.lower()
        if is_correct:
            correct += 1
        print(f"  {f:<35}: {pred:<20} ({conf}%) {'[OK]' if is_correct else '[WRONG]'}")
    print(f"Accuracy on {disease}: {correct}/{len(files)} ({correct/len(files)*100:.1f}%)\n")
