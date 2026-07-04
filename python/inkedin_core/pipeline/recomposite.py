"""Line-art preservation (Research.MD Gap 7).

Take chroma from the model output, keep lightness anchored to the source page
in dark regions so ink lines and lettering stay sharp regardless of model drift.
Also provides LAB-moment palette anchoring for cross-page consistency (Gap 5).
"""

from __future__ import annotations

import cv2
import numpy as np


def preserve_lines(source_rgb: np.ndarray, colored_rgb: np.ndarray, ink_weight: float = 0.85) -> np.ndarray:
    """Blend L channel toward source where the source is dark (ink)."""
    if colored_rgb.shape[:2] != source_rgb.shape[:2]:
        colored_rgb = cv2.resize(
            colored_rgb, (source_rgb.shape[1], source_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )

    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    src_l = src_lab[:, :, 0]
    # Ink mask: darker source pixels pull output lightness toward the source.
    darkness = np.clip((160.0 - src_l) / 160.0, 0.0, 1.0) ** 1.5
    w = darkness * ink_weight
    out_lab[:, :, 0] = out_lab[:, :, 0] * (1.0 - w) + src_l * w

    out = cv2.cvtColor(out_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def match_palette(page_rgb: np.ndarray, anchor_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """LAB mean/std moment matching toward an anchor page. strength 0..1."""
    page = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    anchor = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        p_mean, p_std = page[:, :, c].mean(), page[:, :, c].std() + 1e-6
        a_mean, a_std = anchor[:, :, c].mean(), anchor[:, :, c].std() + 1e-6
        matched = (page[:, :, c] - p_mean) * (a_std / p_std) + a_mean
        page[:, :, c] = page[:, :, c] * (1 - strength) + matched * strength

    return cv2.cvtColor(page.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
