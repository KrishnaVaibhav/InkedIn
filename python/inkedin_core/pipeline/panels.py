"""Panel detection (Research.MD: panel-wise diffusion rendering).

Comic panels are ink-bordered regions on a light gutter background. Classic
CV is enough: threshold ink, close the borders, take big external contours.
Falls back to the whole page when detection looks wrong (borderless layouts,
full-bleed art) so the caller never has to special-case failure.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_PANEL_FRAC = 0.02  # ignore blobs smaller than 2% of the page
MIN_COVERAGE = 0.30  # panels must cover 30% of the page or we bail to full-page


def detect_panels(page_rgb: np.ndarray, margin_frac: float = 0.015) -> list[tuple[int, int, int, int]]:
    """Return panel rects as (x, y, w, h), row-major order.

    Always returns at least one rect (the full page as fallback).
    """
    h, w = page_rgb.shape[:2]
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)

    ink = (gray < 200).astype(np.uint8)
    # Close small gaps in panel borders so each panel is one solid contour.
    k = max(3, (min(h, w) // 300) | 1)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(h * w)
    margin = int(min(h, w) * margin_frac)

    rects: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < page_area * MIN_PANEL_FRAC:
            continue
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + cw + margin), min(h, y + ch + margin)
        rects.append((x0, y0, x1 - x0, y1 - y0))

    covered = sum(rw * rh for _, _, rw, rh in rects)
    if not rects or covered < page_area * MIN_COVERAGE or covered > page_area * 1.5:
        return [(0, 0, w, h)]

    rects.sort(key=lambda r: (r[1] // max(1, h // 8), r[0]))  # rough rows, then x
    return rects
