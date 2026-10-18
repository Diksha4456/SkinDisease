import sys
import os
sys.path.append(os.path.abspath("."))
import torch
from PIL import Image
from src.models.backbone import FairSkinMobileNet, DISEASE_CLASSES, FEATURE_VOCABULARY
from src.data.dataset_loader import get_transforms

model = FairSkinMobileNet(pretrained=False)
ckpt = torch.load('models/best_fairskin_model.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tf = get_transforms(is_train=False, image_size=224)

cases = [
    ("Akne-jugend", "scratch/test_akne.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Akne-jugend.jpg"),
    ("Acne vulgaris 2", "scratch/test_acne2.jpg", "https://upload.wikimedia.org/wikipedia/commons/4/47/Acne_vulgaris_2.jpg"),
    ("Melasma SS 214936", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\Screenshot 2026-10-14 214936.png", None),
    ("Melasma SS melasma.png", r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots\melasma.png", None)
]

import requests
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for name, local_path, url in cases:
    if url and not os.path.exists(local_path):
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
    
    p = local_path if url else local_path
    im = Image.open(p).convert("RGB")
    tensor = tf(im).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        d_logits = out["disease_logits"][0].numpy()
        d_probs = out["disease_probs"][0].numpy()
        f_probs = out["feature_probs"][0].numpy()
    
    print(f"\n==================== {name} ====================")
    print("Disease Probabilities:")
    for c, prob, logit in zip(DISEASE_CLASSES, d_probs, d_logits):
        print(f"  {c:<16}: {prob*100:5.2f}% (logit: {logit:6.2f})")
    
    print("Observable Features (> 30%):")
    for f, prob in zip(FEATURE_VOCABULARY, f_probs):
        if prob > 0.30:
            print(f"  {f:<32}: {prob*100:5.2f}%")
