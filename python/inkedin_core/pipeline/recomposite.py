"""Line-art preservation (Research.MD Gap 7).

Two strategies live here:

- `recompose` (model modes): keep the FULL source lightness channel and take
  only chroma (a/b) from the model output. Chroma is joint-upsampled with a
  guided filter against the full-resolution page so color edges snap to the
  ink instead of bleeding past it, ink pixels are desaturated so lines never
  carry a color halo, and detected speech bubbles are forced back to neutral
  so lettering stays on clean white.
- `preserve_lines` (theme grade): blend L toward the source in dark regions
  only; themes synthesize chroma from the source itself so bleed cannot occur.

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


def text_bubble_mask(source_rgb: np.ndarray) -> np.ndarray:
    """Soft (0..1 float) mask over speech bubbles / caption boxes.

    Uses the RT-DETR bubble detector when its weights are cached locally
    (models/bubbles.py), otherwise a heuristic: near-white connected
    components of plausible bubble size and fill ratio whose interior holes
    contain a moderate share of dark pixels (the lettering). Panels and page
    background fail the size or text-density checks.
    """
    try:
        from ..models import bubbles as bubbles_mod

        boxes = bubbles_mod.detect_bubble_boxes(source_rgb)
    except Exception:
        boxes = None
    if boxes is not None:
        return _mask_from_bubble_boxes(source_rgb, boxes)

    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    page_area = float(h * w)
    white = (gray >= 235).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    mask = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if not (page_area * 3e-4 <= area <= page_area * 0.15):
            continue
        if area / float(cw * ch) < 0.35:  # bubbles are roundish, panels of art are sparse
            continue
        comp = (labels[y : y + ch, x : x + cw] == i).astype(np.uint8)
        # Close the component to swallow the lettering holes.
        k = max(3, (min(cw, ch) // 20) | 1)
        filled = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        holes = (filled == 1) & (comp == 0)
        dark = (gray[y : y + ch, x : x + cw] < 110) & holes
        dark_ratio = dark.sum() / float(area)
        if 0.005 <= dark_ratio <= 0.6:
            mask[y : y + ch, x : x + cw] |= filled

    if not mask.any():
        return mask.astype(np.float32)
    # Feather so the neutralization has no hard seam.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)


def _mask_from_bubble_boxes(
    source_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Detector boxes are rectangles; bubbles are not. Inside each box, take
    the near-white pixels, drop the components connected to the box border
    (that is surrounding paper, not bubble — whitening it caused rectangular
    'missing color' halos), and close what remains: the bubble body."""
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in boxes:
        if x1 <= x0 or y1 <= y0:
            continue
        white = (gray[y0:y1, x0:x1] >= 200).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        rh, rw = white.shape
        inner = np.zeros_like(white)
        for i in range(1, num):
            bx, by, bw2, bh2, _area = stats[i]
            if bx == 0 or by == 0 or bx + bw2 >= rw or by + bh2 >= rh:
                continue  # touches box border: surrounding paper
            inner[labels == i] = 1
        if not inner.any():
            inner = white  # tight box: bubble fills it, keep the plain mask
        k = max(3, (min(x1 - x0, y1 - y0) // 20) | 1)
        inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask[y0:y1, x0:x1] |= inner
    if not mask.any():
        return mask.astype(np.float32)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)


def _smooth_chroma(channel: np.ndarray, guide: np.ndarray, radius: int = 8) -> np.ndarray:
    """Edge-aware smoothing of one chroma channel against the source page."""
    try:
        from cv2 import ximgproc

        return ximgproc.guidedFilter(guide, channel, radius=radius, eps=150.0)
    except ImportError:  # plain opencv build: fall back to joint-ish bilateral
        return cv2.bilateralFilter(channel, d=radius * 2 + 1, sigmaColor=25, sigmaSpace=radius)


def recompose(
    source_rgb: np.ndarray,
    colored_rgb: np.ndarray,
    protect_text: bool = True,
) -> np.ndarray:
    """Chroma-only recomposite: L from the source page, a/b from the model.

    Ink, lettering and screentone survive untouched because lightness never
    comes from the model; the guided filter aligns upsampled chroma to the
    full-resolution line art so color stays inside the lines.
    """
    if colored_rgb.shape[:2] != source_rgb.shape[:2]:
        colored_rgb = cv2.resize(
            colored_rgb, (source_rgb.shape[1], source_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )

    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    out_lab = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB)

    guide = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    a = _smooth_chroma(out_lab[:, :, 1], guide).astype(np.float32)
    b = _smooth_chroma(out_lab[:, :, 2], guide).astype(np.float32)

    src_l = src_lab[:, :, 0].astype(np.float32)
    # Ink never carries color: pull chroma to neutral where the source is dark.
    # Ramp starts at L=65 (was 90): dark shading/hair keeps its color, only
    # true ink goes neutral — the old ramp was a "missing color" source in
    # heavily shaded regions.
    ink = np.clip((65.0 - src_l) / 65.0, 0.0, 1.0)
    a = a * (1.0 - ink) + 128.0 * ink
    b = b * (1.0 - ink) + 128.0 * ink

    if protect_text:
        bubbles = text_bubble_mask(source_rgb)
        if bubbles.any():
            a = a * (1.0 - bubbles) + 128.0 * bubbles
            b = b * (1.0 - bubbles) + 128.0 * bubbles

    out = np.stack([src_l, a, b], axis=2)
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def neutralize_ink_and_bubbles(
    source_rgb: np.ndarray, colored_rgb: np.ndarray, protect_text: bool = True
) -> np.ndarray:
    """Cheap post-guard after palette operations: re-neutralize chroma on ink
    and inside speech bubbles WITHOUT the full recompose pass. Running the
    whole recompose twice (guided filter + L rebuild) washes the colors out —
    that was a visible quality regression."""
    lab = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[:, :, 1], lab[:, :, 2]
    src_l = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    ink = np.clip((65.0 - src_l) / 65.0, 0.0, 1.0)
    a = a * (1.0 - ink) + 128.0 * ink
    b = b * (1.0 - ink) + 128.0 * ink
    if protect_text:
        bubbles = text_bubble_mask(source_rgb)
        if bubbles.any():
            a = a * (1.0 - bubbles) + 128.0 * bubbles
            b = b * (1.0 - bubbles) + 128.0 * bubbles
    lab[:, :, 1], lab[:, :, 2] = a, b
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def fill_chroma_voids(source_rgb: np.ndarray, colored_rgb: np.ndarray, protect_text: bool = True) -> np.ndarray:
    """Fill 'missing spots': bright regions the model left gray although they
    are surrounded by color (GAN uncertainty, undetected panels). Chroma is
    inpainted from the colorful surroundings; lightness stays untouched, so
    line art is unaffected.

    Deliberately NOT filled: page margins/gutters (components touching the
    border), speech bubbles (protect mask), tiny highlights, and anything not
    actually enclosed by colored pixels.
    """
    h, w = colored_rgb.shape[:2]
    page_area = float(h * w)
    lab = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    src_l = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    void = ((chroma < 8.0) & (src_l > 140)).astype(np.uint8)
    if protect_text:
        bubbles = text_bubble_mask(source_rgb)
        if bubbles.any():
            void[bubbles > 0.2] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(void, connectivity=8)
    fill = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < page_area * 5e-4 or area > page_area * 0.12:
            continue  # specks / large regions (large bright areas are usually
            #           intentional: white clothes, backgrounds, flashbacks)
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue  # touches the border: gutter or margin, keep white
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((15, 15), np.uint8)) - comp
        sel = ring == 1
        if not sel.any() or chroma[sel].mean() < 14.0:
            continue  # not clearly enclosed by color: intentional white
        # the surrounding color must be ONE coherent hue — inpainting a mix of
        # hues into a void smears wrong colors into the art
        ang = np.arctan2(b[sel], a[sel])
        if np.hypot(np.cos(ang).mean(), np.sin(ang).mean()) < 0.75:
            continue
        fill |= comp

    if not fill.any():
        return colored_rgb
    lab_u8 = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB)
    lab_u8[:, :, 1] = cv2.inpaint(lab_u8[:, :, 1], fill, 5, cv2.INPAINT_TELEA)
    lab_u8[:, :, 2] = cv2.inpaint(lab_u8[:, :, 2], fill, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)


def match_palette(
    page_rgb: np.ndarray,
    anchor_rgb: np.ndarray,
    strength: float = 0.5,
    channels: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """LAB mean/std moment matching toward an anchor page. strength 0..1.
    channels=(1, 2) matches chroma only — right for color-reference anchoring,
    where dragging lightness would wash the page."""
    page = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    anchor = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in channels:
        p_mean, p_std = page[:, :, c].mean(), page[:, :, c].std() + 1e-6
        a_mean, a_std = anchor[:, :, c].mean(), anchor[:, :, c].std() + 1e-6
        matched = (page[:, :, c] - p_mean) * (a_std / p_std) + a_mean
        page[:, :, c] = page[:, :, c] * (1 - strength) + matched * strength

    return cv2.cvtColor(page.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
