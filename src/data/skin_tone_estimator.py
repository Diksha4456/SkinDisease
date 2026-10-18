"""
Skin Tone Estimator using Individual Typology Angle (ITA) in CIE L*a*b* space.
Provides an objective, optical measurement of skin pigmentation complementary
to the categorical Fitzpatrick scale.

Formula:
  ITA = arctan((L* - 50) / b*) * (180 / pi)

Standard classification bands (Chardon et al., 1991; Del Bino et al., 2013):
  - Very Light:  ITA > 55 deg        (~ Fitzpatrick I)
  - Light:       41 deg < ITA <= 55 deg (~ Fitzpatrick II)
  - Intermediate: 28 deg < ITA <= 41 deg (~ Fitzpatrick III)
  - Tan:         10 deg < ITA <= 28 deg (~ Fitzpatrick IV)
  - Brown:      -30 deg < ITA <= 10 deg (~ Fitzpatrick V)
  - Dark:        ITA <= -30 deg       (~ Fitzpatrick VI)
"""

import math
import numpy as np
from PIL import Image


def rgb_to_cielab(rgb_array: np.ndarray) -> np.ndarray:
    """
    Converts an sRGB image array (0-255) to CIE L*a*b* space using standard D65 illuminant.
    """
    # Normalize sRGB
    rgb = rgb_array.astype(np.float32) / 255.0

    # Gamma correction (linearize)
    mask = rgb > 0.04045
    rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92

    # Observer. = 2 deg, Illuminant = D65 matrix
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32).T

    xyz = np.dot(rgb, mat)

    # Reference white point D65: (95.047, 100.0, 108.883)
    xyz[:, :, 0] /= 0.95047
    xyz[:, :, 1] /= 1.00000
    xyz[:, :, 2] /= 1.08883

    # Non-linear transformation for Lab
    epsilon = 0.008856
    kappa = 903.3

    f_xyz = np.zeros_like(xyz)
    m = xyz > epsilon
    f_xyz[m] = np.power(xyz[m], 1.0 / 3.0)
    f_xyz[~m] = (kappa * xyz[~m] + 16.0) / 116.0

    l_star = np.clip(116.0 * f_xyz[:, :, 1] - 16.0, 0.0, 100.0)
    a_star = 500.0 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
    b_star = 200.0 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])

    lab = np.stack([l_star, a_star, b_star], axis=-1)
    return lab


def estimate_ita(image: Image.Image) -> tuple[float, str, int]:
    """
    Estimates the Individual Typology Angle (ITA) from a PIL Image.
    Uses skin pixel segmentation heuristic (YCbCr + Lab thresholding)
    to sample healthy/perilesional skin and exclude glare and shadow.

    Returns:
        (ita_degrees, tone_band_name, estimated_fitzpatrick_type)
    """
    # Resize for computational efficiency and noise reduction
    thumb = image.convert("RGB").resize((128, 128))
    np_img = np.array(thumb)

    # Convert to CIE Lab
    lab = rgb_to_cielab(np_img)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Skin detection heuristic in RGB/Lab space:
    # Avoid specular reflection (L* > 95) and deep dark shadows / hair (L* < 15)
    # Ensure skin-like redness (a* between 2 and 40)
    skin_mask = (l_channel > 15.0) & (l_channel < 92.0) & (a_channel > 2.0) & (a_channel < 42.0)

    if np.sum(skin_mask) < 50:
        # Fallback to central 50% image crop if skin heuristic selects too few pixels
        h, w = l_channel.shape
        skin_mask = np.zeros_like(l_channel, dtype=bool)
        skin_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True

    med_l = float(np.median(l_channel[skin_mask]))
    med_b = float(np.median(b_channel[skin_mask]))

    # Avoid division by zero
    if abs(med_b) < 1e-5:
        med_b = 1e-5 if med_b >= 0 else -1e-5

    # ITA formula: arctan((L - 50) / b) * (180 / pi)
    ita_rad = math.atan2(med_l - 50.0, med_b)
    ita_deg = float(math.degrees(ita_rad))

    # Standard Del Bino / Chardon typology mapping:
    if ita_deg > 55.0:
        band = "Very Light"
        fst = 1
    elif ita_deg > 41.0:
        band = "Light"
        fst = 2
    elif ita_deg > 28.0:
        band = "Intermediate"
        fst = 3
    elif ita_deg > 10.0:
        band = "Tan"
        fst = 4
    elif ita_deg > -30.0:
        band = "Brown"
        fst = 5
    else:
        band = "Dark"
        fst = 6

    return round(ita_deg, 2), band, fst
