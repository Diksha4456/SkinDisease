import os
import glob
import sys
sys.path.append(".")
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as TVTransforms

from src.models.backbone import FairSkinMobileNet, DISEASE_CLASSES, FEATURE_VOCABULARY
from src.data.dataset_loader import get_transforms

model = FairSkinMobileNet(pretrained=False)
ckpt = torch.load("models/best_fairskin_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

infer_transforms = get_transforms(is_train=False, image_size=224)
full_frame_infer_tf = TVTransforms.Compose([
    TVTransforms.Resize((224, 224)),
    TVTransforms.ToTensor(),
    TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sc_dir = r"C:\Users\DIKSHA\OneDrive\Pictures\Screenshots"

test_cases = [
    # Image name, ground truth, clinical description
    ("Screenshot 2026-10-14 214534.png", "acne", "User Forehead Acne Vulgaris"),
    ("Screenshot 2026-10-14 230937.png", "acne", "User Cheek Acne Vulgaris"),
    ("Screenshot 2026-10-14 195441.png", "melasma", "User Malar Cheek Melasma Patch"),
    ("Screenshot 2026-10-14 195532.png", "melasma", "User Facial Melasma"),
    ("Screenshot 2026-10-14 214936.png", "melasma", "User Facial Melasma Cheek"),
    ("Screenshot 2026-10-14 230513.png", "melasma", "User Cheek Melasma"),
    ("Screenshot 2026-10-14 195622.png", "psoriasis", "User Plaque Psoriasis Back"),
    ("Screenshot 2026-10-14 214608.png", "psoriasis", "User Elbow Plaque Psoriasis"),
    ("Screenshot 2026-10-14 230327.png", "psoriasis", "User Arms Plaque Psoriasis"),
    ("Screenshot 2026-10-14 231135.png", "psoriasis", "User Close-Up Plaque Psoriasis"),
    ("Screenshot 2026-10-14 214633.png", "ringworm", "User Leg Ringworm Annular Plaque"),
    ("Screenshot 2026-10-14 214738.png", "vitiligo", "User Dorsal Hand Vitiligo"),
    ("Screenshot 2026-10-14 214753.png", "vitiligo", "User Perioral Vitiligo Child"),
    ("Screenshot 2026-10-14 214900.png", "vitiligo", "User Portrait Vitiligo Model"),
]

print("=" * 80)
print(f"EVALUATING UPGRADED MODEL ON {len(test_cases)} REAL CLINICAL USER SCREENSHOTS")
print("=" * 80)

correct = 0
for fname, ground_truth, desc in test_cases:
    p = os.path.join(sc_dir, fname)
    if not os.path.exists(p):
        print(f"[Missing] {fname}")
        continue
    
    pil_img = Image.open(p).convert("RGB")
    w, h = pil_img.size
    t_full = full_frame_infer_tf(pil_img).unsqueeze(0)
    sub_crops = [
        pil_img,
        pil_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92))),
        pil_img.crop((int(w * 0.15), 0, int(w * 0.85), int(h * 0.70))),
        pil_img.crop((int(w * 0.15), int(h * 0.30), int(w * 0.85), h))
    ]
    crop_tensors = torch.cat([t_full, torch.stack([infer_transforms(c) for c in sub_crops])], dim=0)
    with torch.no_grad():
        multi_outs = model(crop_tensors)
        avg_disease_probs = multi_outs["disease_probs"].mean(dim=0).cpu().numpy()
        avg_feat_probs = multi_outs["feature_probs"].mean(dim=0).cpu().numpy()

    # Differential clinical prior
    feat_evidence = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
    # 0: Acne: papules/pustules (0), comedo (1), nodules (2)
    acne_score = float(max(avg_feat_probs[0], avg_feat_probs[1]) * 0.70 + avg_feat_probs[2] * 0.30)
    # 1: Melasma: reticulated hyperpigmentation (3), irregular pigment border (4)
    mel_score = float(max(avg_feat_probs[3], avg_feat_probs[4]))
    # 2: Psoriasis: scaly plaque (7), silvery scale (8)
    pso_score = float(max(avg_feat_probs[7], avg_feat_probs[8]))
    # 3: Ringworm: annular ring (9), advancing border (10), central clearing (11)
    rw_score = float(max(avg_feat_probs[9], avg_feat_probs[10]) * 0.70 + avg_feat_probs[11] * 0.30)
    # 4: Vitiligo: chalky white depigmentation (5), well demarcated (6)
    vit_score = float(max(avg_feat_probs[5], avg_feat_probs[6]))

    # Differential rules
    if acne_score > 0.30:
        # Suppress fungal annular ring & psoriasis
        rw_score *= 0.20
        pso_score *= 0.35
    if mel_score > 0.35:
        # Suppress ringworm & vitiligo
        rw_score *= 0.20
        vit_score *= 0.10
    if vit_score > 0.40:
        mel_score *= 0.10
    if pso_score > 0.40:
        acne_score *= 0.25

    feat_evidence[0] = acne_score
    feat_evidence[1] = mel_score
    feat_evidence[2] = pso_score
    feat_evidence[3] = rw_score
    feat_evidence[4] = vit_score
    feat_evidence[5] = 0.05

    fused_probs = 0.70 * avg_disease_probs + 0.30 * feat_evidence
    probs = fused_probs / fused_probs.sum()
    pred_idx = int(np.argmax(probs))
    pred_disease = DISEASE_CLASSES[pred_idx]
    conf = float(probs[pred_idx]) * 100

    is_pass = (pred_disease == ground_truth)
    if is_pass: correct += 1
    status = "PASS" if is_pass else "FAIL"

    top_feats = [FEATURE_VOCABULARY[i] for i in np.where(avg_feat_probs > 0.30)[0]]
    print(f"[{status}] {desc:<35} | Target: {ground_truth:<10} | Pred: {pred_disease:<10} ({conf:5.1f}%) | Feats: {top_feats}")

print("=" * 80)
print(f"Accuracy across real clinical user screenshots: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
print("=" * 80)
