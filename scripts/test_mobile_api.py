import requests
import base64
import os

print("--- Testing GET /api/health ---")
r_health = requests.get("http://127.0.0.1:5000/api/health")
print("Health Status:", r_health.status_code, r_health.json())

print("\n--- Testing GET /api/dwh/analytics ---")
r_dwh = requests.get("http://127.0.0.1:5000/api/dwh/analytics")
print("DWH Status:", r_dwh.status_code)
dwh_data = r_dwh.json()
print("Total Screenings in DWH:", dwh_data.get("total_screenings"))
print("Fairness Audit Samples:", len(dwh_data.get("fairness_audit", [])))
print("Prevalence Samples:", len(dwh_data.get("disease_prevalence", [])))

print("\n--- Testing POST /api/screen_base64 ---")
img_path = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214534.png"
if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    
    r_screen = requests.post("http://127.0.0.1:5000/api/screen_base64", json={"image_base64": b64})
    print("Screen Status:", r_screen.status_code)
    res = r_screen.json()
    print("Predicted:", res.get("predicted_disease"), f"({res.get('confidence_pct')}%)")
    print("Features:", res.get("observable_features"))
    print("Tone:", res.get("ita_tone_band"), f"(FST {res.get('fitzpatrick_scale')})")
    print("Grad-CAM b64 length:", len(res.get("gradcam_b64", "")))
