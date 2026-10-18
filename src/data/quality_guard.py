"""
Image Quality Pre-Flight Guard for Offline Clinical Screening.
Rejects ungradable, blurry, dark, overexposed, or poorly framed images
BEFORE model inference to guarantee patient safety and prevent forced predictions.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


class ImageQualityGuard:
    def __init__(
        self,
        min_resolution: int = 160,
        blur_laplacian_threshold: float = 8.0,
        min_luminance: float = 30.0,
        max_luminance: float = 230.0,
        aspect_ratio_range: Tuple[float, float] = (0.35, 2.85),
    ):
        self.min_resolution = min_resolution
        self.blur_laplacian_threshold = blur_laplacian_threshold
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.aspect_ratio_range = aspect_ratio_range

    def check_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs comprehensive image quality checks on an input PIL Image.

        Returns:
            dict containing:
              - 'passed': bool
              - 'flags': list of identified quality warnings/reasons
              - 'metrics': dictionary of measured numerical quality scores
        """
        flags = []
        metrics = {}

        # 1. Resolution & Dimension check
        w, h = image.size
        aspect_ratio = w / float(h)
        metrics["width"] = w
        metrics["height"] = h
        metrics["aspect_ratio"] = round(aspect_ratio, 2)

        if w < self.min_resolution or h < self.min_resolution:
            flags.append(f"Low resolution: {w}x{h} (minimum {self.min_resolution}x{self.min_resolution} required)")

        if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
            flags.append(f"Extreme aspect ratio ({aspect_ratio:.2f}); lesion may be severely cropped or distorted")

        # 2. Luminance & Exposure check
        gray = image.convert("L")
        np_gray = np.array(gray, dtype=np.float32)

        mean_lum = float(np.mean(np_gray))
        std_lum = float(np.std(np_gray))
        metrics["mean_luminance"] = round(mean_lum, 2)
        metrics["std_luminance"] = round(std_lum, 2)

        if mean_lum < self.min_luminance:
            flags.append(f"Underexposed / Too dark (mean luminance {mean_lum:.1f} < {self.min_luminance})")
        elif mean_lum > self.max_luminance:
            flags.append(f"Overexposed / Specular flash washout (mean luminance {mean_lum:.1f} > {self.max_luminance})")

        if std_lum < 15.0:
            flags.append(f"Extremely low contrast (std {std_lum:.1f} < 15.0); details obscured")

        # 3. Blur & Focus check via Laplacian Variance
        # Discrete Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        padded = np.pad(np_gray, pad_width=1, mode="edge")
        laplacian = (
            padded[0:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 2:]
            - 4.0 * padded[1:-1, 1:-1]
        )
        blur_score = float(np.var(laplacian))
        metrics["blur_laplacian_var"] = round(blur_score, 2)

        if blur_score < self.blur_laplacian_threshold:
            flags.append(
                f"Image is out of focus or blurred (Laplacian variance {blur_score:.1f} < {self.blur_laplacian_threshold})"
            )

        passed = len(flags) == 0
        return {
            "passed": passed,
            "flags": flags,
            "metrics": metrics,
            "recommendation": (
                "Image accepted for screening."
                if passed
                else "Quality warning: Please retake the photo with steady hands, proper ambient lighting, and central framing."
            ),
        }
