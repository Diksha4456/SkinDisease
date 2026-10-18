"""
FairSkin AI — Multi-Task Neural Network Architecture for Mobile Inference.
Backbone: MobileNetV3-Small (lightweight, high-efficiency, on-device optimized).
Heads:
  1. Disease Classification (5 target diseases + 1 non-target/OOD class)
  2. Acne Severity Grading (Low vs High severity, gated for acne cases)
  3. Observable Feature Multi-Label Predictor (Dermatological visual evidence)
Includes hooks for Grad-CAM activation mapping.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, Optional


DISEASE_CLASSES = ["acne", "melasma", "psoriasis", "ringworm", "vitiligo", "non_target_ood"]
SEVERITY_CLASSES = ["Low", "High"]

FEATURE_VOCABULARY = [
    "erythematous_papule_pustule",
    "open_closed_comedo",
    "inflammatory_nodule_scarring",
    "reticulated_hyperpigmentation",
    "irregular_pigment_border",
    "chalky_white_depigmentation",
    "well_demarcated_border",
    "erythematous_scaly_plaque",
    "micaceous_silvery_scale",
    "annular_erythematous_ring",
    "active_scaly_advancing_border",
    "central_clearing"
]


class FairSkinMobileNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)

        # Convolutional feature extractor
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_features = 576  # MobileNetV3-Small last conv channels

        # Shared dense representation
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25)
        )

        # Head 1: Primary Disease Classifier (6 classes)
        self.disease_head = nn.Linear(256, len(DISEASE_CLASSES))

        # Head 2: Acne Severity Classifier (Binary: Low vs High)
        self.severity_head = nn.Linear(256, 2)

        # Head 3: Observable Clinical Features (Multi-label Sigmoid)
        self.feature_head = nn.Linear(256, len(FEATURE_VOCABULARY))

        # Hook storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract convolutional feature maps
        conv_out = self.features(x)

        # Register hook on the last feature map for Grad-CAM
        if x.requires_grad:
            h = conv_out.register_hook(self.activations_hook)
            self.activations = conv_out

        pooled = self.avgpool(conv_out)
        flat = torch.flatten(pooled, 1)
        feat = self.shared_fc(flat)

        disease_logits = self.disease_head(feat)
        severity_logits = self.severity_head(feat)
        feature_logits = self.feature_head(feat)

        return {
            "disease_logits": disease_logits,
            "severity_logits": severity_logits,
            "feature_logits": feature_logits,
            "feature_probs": torch.sigmoid(feature_logits),
            "disease_probs": torch.softmax(disease_logits, dim=-1)
        }

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations
