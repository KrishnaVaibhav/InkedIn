"""Color-page detection + nearest-reference selection (Research.MD
"Self-Reference Colorization Plan").

A page is "color" when a meaningful share of its pixels carry chroma AND the
hues are spread out. Uniformly tinted scans (yellowed paper, sepia print)
have chroma but essentially one hue — they stay classified B&W.
"""

from __future__ import annotations

import cv2
import numpy as np

CHROMA_THRESHOLD = 12.0  # LAB chroma magnitude that counts as "a real color"
COLOR_FRAC_MIN = 0.10  # >=10% colorful pixels
HUE_SPREAD_MIN_DEG = 8.0  # circular hue std-dev; uniform tint scans sit < 5 deg,
#                           real color art (even warm-dominant covers) > 10 deg
ANALYZE_EDGE = 512  # analysis runs on a downscale


def chroma_stats(rgb: np.ndarray) -> tuple[float, float]:
    """(colorful_frac, hue_circular_std_degrees) on a downscaled copy."""
    h, w = rgb.shape[:2]
    scale = ANALYZE_EDGE / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    colorful = chroma > CHROMA_THRESHOLD
    frac = float(colorful.mean())
    if not colorful.any():
        return frac, 0.0

    # circular std-dev of the hue angle over colorful pixels, in degrees.
    # R = |mean unit vector|; circ_std = sqrt(-2 ln R). A uniformly tinted
    # scan is a single tight hue cluster (R→1, std→0); color art spreads.
    ang = np.arctan2(b[colorful], a[colorful])
    r = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    r = min(max(r, 1e-9), 1.0 - 1e-9)
    circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(r))))
    return frac, circ_std_deg


def is_color_page(rgb: np.ndarray) -> tuple[bool, float]:
    """(is_color, score). score = colorful_frac, stored for diagnostics."""
    frac, spread_deg = chroma_stats(rgb)
    return frac >= COLOR_FRAC_MIN and spread_deg >= HUE_SPREAD_MIN_DEG, frac


def lab_moments(rgb: np.ndarray) -> np.ndarray:
    """Per-channel LAB mean+std (6 numbers) — cheap page signature."""
    h, w = rgb.shape[:2]
    scale = ANALYZE_EDGE / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.array([f(lab[:, :, c]) for c in range(3) for f in (np.mean, np.std)], np.float32)


def nearest_reference(page_rgb: np.ndarray, refs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Pick the reference whose L-channel statistics are closest to the page
    (tone/structure proxy; the refs are color, the page is B&W, so chroma
    channels are useless for matching). refs = [(ref_rgb, ref_moments)]."""
    if not refs:
        return None
    page_m = lab_moments(page_rgb)
    best, best_d = None, np.inf
    for ref_rgb, ref_m in refs:
        d = float(np.abs(page_m[:2] - ref_m[:2]).sum())  # L mean+std distance
        if d < best_d:
            best, best_d = ref_rgb, d
    return best
