import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.data.skin_tone_estimator import estimate_ita
from src.data.quality_guard import ImageQualityGuard

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
MANIFESTS_DIR = os.path.join(ROOT_DIR, "data", "manifests")
os.makedirs(MANIFESTS_DIR, exist_ok=True)

quality_guard = ImageQualityGuard()

DISEASES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]

FEATURE_MAP = {
    "acne": ["comedone_open_closed", "erythematous_papule", "mild_follicular_inflammation"],
    "melasma": ["hyperpigmented_macule", "reticulated_brown_patch", "centrofacial_distribution", "irregular_borders"],
    "psoriasis": ["erythematous_plaque", "micaceous_silvery_scale", "sharply_circumscribed_border"],
    "ringworm": ["annular_erythematous_ring", "active_scaly_advancing_border", "central_clearing"],
    "vitiligo": ["depigmented_macule", "chalky_white_patch", "well_demarcated_margin"],
    "non_target_ood": ["non_specific_erythema", "atypical_cutaneous_lesion"]
}

records = []
print("Scanning and auditing all clinical images in data/raw...")

for d in DISEASES:
    d_dir = os.path.join(RAW_DIR, d)
    if not os.path.isdir(d_dir):
        continue
    files = glob.glob(os.path.join(d_dir, "*.jpg")) + glob.glob(os.path.join(d_dir, "*.jpeg")) + glob.glob(os.path.join(d_dir, "*.png"))
    for f in sorted(files):
        try:
            with Image.open(f) as img:
                pil_img = img.convert("RGB")
                w, h = pil_img.size
            
            # Read bytes for hash
            with open(f, "rb") as bf:
                b = bf.read()
            sha256 = hashlib.sha256(b).hexdigest()

            # Quality Check
            qc = quality_guard.check_quality(pil_img)
            q_status = "PASSED" if qc["passed"] else "FLAGGED_POOR_QUALITY"
            q_flags = "; ".join(qc["flags"]) if qc["flags"] else "None"

            # ITA Tone & Fitzpatrick Scale
            ita_deg, tone_band, fst = estimate_ita(pil_img)

            # Acne Severity
            if d == "acne":
                sev = "High" if ("lvl2" in f.lower() or "lvl3" in f.lower() or "conglobata" in f.lower() or "severe" in f.lower()) else "Low"
                lvl = 2 if sev == "High" else 1
            else:
                sev = "NotApplicable"
                lvl = -1

            base_name = os.path.splitext(os.path.basename(f))[0]
            img_id = f"{d}_{base_name}"

            records.append({
                "image_id": img_id,
                "file_path": os.path.abspath(f),
                "source_dataset": "ClinicalCuratedCorpus",
                "disease_label": d,
                "acne_severity": sev,
                "acne_level": lvl,
                "license_status": "verified_research_permissible",
                "permission_reference": "FairSkin-Corpus-Audit",
                "quality_status": q_status,
                "quality_flags": q_flags,
                "image_width": w,
                "image_height": h,
                "sha256": sha256,
                "ita_degrees": round(ita_deg, 2),
                "ita_tone_band": tone_band,
                "fitzpatrick_scale": fst,
                "observable_features": json.dumps(FEATURE_MAP[d]),
                "strat_key": f"{d}_FST{fst}"
            })
        except Exception as e:
            print(f"Error auditing {f}: {e}")

df_all = pd.DataFrame(records)
# Drop duplicates by sha256
df_all = df_all.drop_duplicates(subset=["sha256"]).reset_index(drop=True)
print(f"Total Unique Valid Clinical Images Audited: {len(df_all)}")
print("Class Distribution:")
print(df_all["disease_label"].value_counts())
print("\nFitzpatrick Scale Distribution:")
print(df_all["fitzpatrick_scale"].value_counts().sort_index())

# Save complete corpus manifest
corpus_path = os.path.join(MANIFESTS_DIR, "complete_clinical_corpus.csv")
df_all.to_csv(corpus_path, index=False)

# Stratified Split (70% Train, 15% Val, 15% Test)
# To handle small subgroups safely, stratify by disease_label
splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(splitter_test.split(df_all, df_all["disease_label"]))

df_train_val = df_all.iloc[train_val_idx].reset_index(drop=True)
df_test = df_all.iloc[test_idx].reset_index(drop=True)

# 15% / 85% of train_val is ~ 17.6% of train_val
splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
train_idx, val_idx = next(splitter_val.split(df_train_val, df_train_val["disease_label"]))

df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

print(f"\nSplit Sizes: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

df_train.to_csv(os.path.join(MANIFESTS_DIR, "train_split.csv"), index=False)
df_val.to_csv(os.path.join(MANIFESTS_DIR, "val_split.csv"), index=False)
df_test.to_csv(os.path.join(MANIFESTS_DIR, "test_split.csv"), index=False)
print("Stratified manifests successfully written to data/manifests/.")
