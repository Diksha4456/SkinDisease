"""
FairSkin AI — PyTorch Dataset Loader with Melanin-Preserving Augmentation.
Loads verified clinical dermatology images from train/val/test split manifests.
Provides targets for:
  - Disease classification
  - Acne severity grading (gated)
  - Observable clinical feature multi-label vector
  - Fitzpatrick skin tone scale & ITA value for subgroup auditing
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from src.models.backbone import DISEASE_CLASSES, SEVERITY_CLASSES, FEATURE_VOCABULARY


def get_transforms(is_train: bool = True, image_size: int = 224):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            # Melanin-preserving color jitter (moderate brightness & contrast, zero hue shift)
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


FEATURE_MAPPING = {
    "comedone_open_closed": ["open_closed_comedo"],
    "erythematous_papule": ["erythematous_papule_pustule"],
    "inflammatory_pustule": ["erythematous_papule_pustule"],
    "erythematous_nodule": ["inflammatory_nodule_scarring"],
    "confluent_lesions": ["inflammatory_nodule_scarring"],
    "scarring": ["inflammatory_nodule_scarring"],
    "mild_follicular_inflammation": ["erythematous_papule_pustule"],
    "hyperpigmented_macule": ["reticulated_hyperpigmentation"],
    "reticulated_brown_patch": ["reticulated_hyperpigmentation"],
    "centrofacial_distribution": ["reticulated_hyperpigmentation"],
    "irregular_borders": ["irregular_pigment_border"],
    "depigmented_macule": ["chalky_white_depigmentation"],
    "chalky_white_patch": ["chalky_white_depigmentation"],
    "well_demarcated_margin": ["well_demarcated_border"],
    "trichrome_transition": ["well_demarcated_border"],
    "erythematous_plaque": ["erythematous_scaly_plaque"],
    "micaceous_silvery_scale": ["micaceous_silvery_scale"],
    "sharply_circumscribed_border": ["erythematous_scaly_plaque"],
    "auspitz_sign_suspect": ["micaceous_silvery_scale"],
    "annular_erythematous_ring": ["annular_erythematous_ring"],
    "active_scaly_advancing_border": ["active_scaly_advancing_border"],
    "central_clearing": ["central_clearing"],
    "pruritic_margin": ["active_scaly_advancing_border"]
}


class FairSkinDataset(Dataset):
    def __init__(self, manifest_csv: str, is_train: bool = True, image_size: int = 224):
        self.df = pd.read_csv(manifest_csv)
        self.transforms = get_transforms(is_train=is_train, image_size=image_size)

        self.disease_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}
        self.severity_to_idx = {"Low": 0, "High": 1}
        self.feature_to_idx = {name: i for i, name in enumerate(FEATURE_VOCABULARY)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["file_path"]

        # Fallback to local path if file moved
        if not os.path.exists(img_path):
            base = os.path.basename(img_path)
            for folder in ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]:
                cand = os.path.join("data", "raw", folder, base)
                if os.path.exists(cand):
                    img_path = cand
                    break

        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")
        except Exception:
            # Fallback black image if corrupted
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        img_tensor = self.transforms(img)

        # 1. Disease target
        disease_name = str(row["disease_label"]).strip().lower()
        disease_idx = self.disease_to_idx.get(disease_name, 5)  # Default to non_target_ood

        # 2. Acne severity target
        sev_name = str(row["acne_severity"]).strip()
        sev_idx = self.severity_to_idx.get(sev_name, -1)  # -1 if not applicable

        # 3. Multi-label features target
        feat_target = torch.zeros(len(FEATURE_VOCABULARY), dtype=torch.float32)
        try:
            feats = json.loads(row["observable_features"]) if "observable_features" in row and pd.notna(row["observable_features"]) else []
            for f in feats:
                mapped_feats = FEATURE_MAPPING.get(f.strip(), [])
                for mf in mapped_feats:
                    if mf in self.feature_to_idx:
                        feat_target[self.feature_to_idx[mf]] = 1.0
        except Exception:
            pass

        # 4. Demographic & Subgroup metadata
        fst = int(row["fitzpatrick_scale"]) if "fitzpatrick_scale" in row and pd.notna(row["fitzpatrick_scale"]) else 3
        ita = float(row["ita_degrees"]) if "ita_degrees" in row and pd.notna(row["ita_degrees"]) else 30.0

        return {
            "image": img_tensor,
            "disease_target": torch.tensor(disease_idx, dtype=torch.long),
            "severity_target": torch.tensor(sev_idx, dtype=torch.long),
            "feature_target": feat_target,
            "fitzpatrick_scale": fst,
            "ita_degrees": ita,
            "image_id": str(row["image_id"]),
            "disease_label": disease_name
        }
