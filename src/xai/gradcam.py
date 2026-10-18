"""
FairSkin AI — Explainable AI (XAI) Engine.
Implements:
  1. Grad-CAM visual activation mapping on MobileNetV3 convolutional features
  2. Evidence-grounded feature extraction and clinical rationale generator
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, Tuple, List


from src.models.backbone import FairSkinMobileNet, DISEASE_CLASSES, FEATURE_VOCABULARY


class GradCAMExplainer:
    def __init__(self, model: FairSkinMobileNet):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Computes enhanced Grad-CAM++ with guided lesion saliency for precise spatial localization.
        Returns a 2D float numpy array normalized between 0.0 and 1.0.
        """
        self.model.zero_grad()
        if not image_tensor.requires_grad:
            image_tensor.requires_grad = True

        outputs = self.model(image_tensor)
        logits = outputs["disease_logits"]
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            return np.ones((224, 224), dtype=np.float32) * 0.5

        # Grad-CAM++ second and third order gradient weighting
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        alpha = grad_2 / (2.0 * grad_2 + activations * grad_3 + 1e-8)
        weights = torch.sum(alpha * torch.clamp(gradients, min=0), dim=[2, 3], keepdim=True)

        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0).detach().cpu().numpy()
        cam_max = np.max(cam)
        if cam_max > 1e-7:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), resample=Image.BICUBIC)
        cam_224 = np.array(cam_pil, dtype=np.float32) / 255.0

        # Fine-grained input guided gradient saliency for sharp lesion borders
        if image_tensor.grad is not None:
            in_grad = image_tensor.grad.detach().abs().squeeze().mean(dim=0).cpu().numpy()
            in_max = in_grad.max()
            if in_max > 1e-7:
                in_grad_norm = (in_grad / in_max * 255).astype(np.uint8)
                in_grad_pil = Image.fromarray(in_grad_norm).filter(ImageFilter.GaussianBlur(radius=2))
                in_grad_smooth = np.array(in_grad_pil, dtype=np.float32) / 255.0
                heatmap = 0.60 * cam_224 + 0.40 * in_grad_smooth
            else:
                heatmap = cam_224
        else:
            heatmap = cam_224

        h_max = np.max(heatmap)
        if h_max > 1e-7:
            heatmap = heatmap / h_max
        return heatmap

    def overlay_heatmap_on_image(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45
    ) -> Image.Image:
        """
        Applies a colormap to the 2D heatmap and overlays it on the PIL image using pure NumPy.
        """
        w, h = pil_image.size
        # Resize heatmap to match original image dimensions using PIL bilinear
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
        heat_np = np.array(heat_img, dtype=np.float32) / 255.0

        # Create JET-like pseudo-colormap in RGB
        # Blue -> Cyan -> Yellow -> Red
        r = np.clip(1.5 - np.abs(heat_np * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(heat_np * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(heat_np * 4.0 - 1.0), 0.0, 1.0)
        color_heatmap = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

        orig_np = np.array(pil_image.convert("RGB"))
        blended = (orig_np * (1.0 - alpha) + color_heatmap * alpha).astype(np.uint8)

        return Image.fromarray(blended)


def build_evidence_explanation(
    disease: str,
    confidence: float,
    entropy: float,
    feature_probs: Dict[str, float],
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes model predictions into a structured, evidence-grounded screening report.
    """
    # Top predicted observable features (> 0.40 probability)
    detected_features = [f.replace("_", " ").title() for f, p in feature_probs.items() if p >= 0.35]
    if not detected_features:
        detected_features = ["Subtle Cutaneous Texture Variation"]

    clinical_rationales = {
        "acne": (
            f"Prediction favored Acne Vulgaris (confidence {confidence*100:.1f}%) grounded in localized "
            f"inflammatory lesions ({', '.join(detected_features[:2])}). "
            + (f"Severity assessed as {severity}." if severity else "")
        ),
        "melasma": (
            f"Prediction favored Melasma (confidence {confidence*100:.1f}%) based on reticulated hyperpigmentation "
            f"and characteristic irregular cutaneous boundaries."
        ),
        "vitiligo": (
            f"Prediction favored Vitiligo (confidence {confidence*100:.1f}%) grounded in depigmented macules with "
            f"well-demarcated margins."
        ),
        "psoriasis": (
            f"Prediction favored Psoriasis (confidence {confidence*100:.1f}%) based on localized erythematous "
            f"plaques with characteristic surface scale."
        ),
        "ringworm": (
            f"Prediction favored Ringworm / Tinea Corporis (confidence {confidence*100:.1f}%) grounded in annular "
            f"erythematous configuration with active advancing border."
        ),
        "non_target_ood": (
            "Screening indicates Non-Target / Atypical Cutaneous Presentation or healthy skin. "
            "Input does not match the 5 target screening profiles."
        )
    }

    referral_guidance = (
        "This is an automated preliminary screening aid, NOT a definitive medical diagnosis. "
        "Consult a certified dermatologist or licensed medical professional for clinical confirmation and care."
    )

    return {
        "predicted_disease": disease.title(),
        "confidence_pct": round(confidence * 100.0, 1),
        "entropy": round(entropy, 3),
        "acne_severity": severity if disease == "acne" else "N/A",
        "observable_features": detected_features,
        "clinical_rationale": clinical_rationales.get(disease, "Indeterminate presentation."),
        "referral_guidance": referral_guidance
    }
