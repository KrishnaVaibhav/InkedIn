"""Colorizer interface + implementations.

Modes (Research.MD Model Strategy):
- "fast": manga-colorization-v2 style GAN (models/gan.py) — real ML colorization.
- "theme:<name>": deterministic LAB duotone grading — no ML, instant, used both as
  a standalone stylizer and as a post-grade on top of "fast".
Reference/diffusion modes arrive in Milestone 4 behind this same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import recomposite

# name -> (shadow LAB(a,b), highlight LAB(a,b)) tint anchors
THEMES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "sepia": ((8, 18), (6, 22)),
    "noir": ((0, -4), (0, 2)),
    "sunset": ((18, 24), (8, 30)),
    "ocean": ((-6, -22), (-4, -12)),
    "forest": ((-18, 14), (-8, 18)),
    "pastel": ((10, -6), (-6, 12)),
}


@dataclass
class ColorizeRequest:
    page_rgb: np.ndarray  # HxWx3 uint8, source page
    mode: str = "fast"  # "fast" | "theme:<name>" | "fast+theme:<name>"
    ink_weight: float = 0.85
    protect_text: bool = True  # keep speech bubbles neutral white
    anchor_rgb: np.ndarray | None = None  # palette anchor page (consistency)
    anchor_strength: float = 0.4
    anchor_chroma_only: bool = False  # color-ref anchoring: don't drag lightness
    fill_voids: bool = False  # inpaint chroma into enclosed gray "missing spots"
    extra: dict = field(default_factory=dict)


class Colorizer(Protocol):
    def colorize(self, req: ColorizeRequest) -> np.ndarray: ...

    def close(self) -> None: ...


class PassthroughColorizer:
    """mode 'none': keep the original art (translation-only jobs)."""

    def colorize(self, req: ColorizeRequest) -> np.ndarray:
        return req.page_rgb.copy()

    def close(self) -> None:
        pass


class ThemeColorizer:
    """Deterministic duotone grade in LAB space. CPU, instant, zero model risk."""

    def __init__(self, theme: str = "sepia"):
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}; have {sorted(THEMES)}")
        self.theme = theme

    def colorize(self, req: ColorizeRequest) -> np.ndarray:
        import cv2

        (sa, sb), (ha, hb) = THEMES[self.theme]
        lab = cv2.cvtColor(req.page_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        t = lab[:, :, 0] / 255.0  # 0 = shadow, 1 = highlight
        lab[:, :, 1] = 128 + sa + (ha - sa) * t
        lab[:, :, 2] = 128 + sb + (hb - sb) * t
        out = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return recomposite.preserve_lines(req.page_rgb, out, req.ink_weight)

    def close(self) -> None:
        pass


def run_page(colorizer: Colorizer, req: ColorizeRequest) -> np.ndarray:
    """One page through: model -> void fill -> palette anchor -> ink guard.

    The colorizer already recomposed once; after palette matching only the
    cheap ink/bubble chroma guard runs — a second full recompose (guided
    filter + L rebuild) visibly washes the model's colors out.
    """
    out = colorizer.colorize(req)
    if req.fill_voids:
        out = recomposite.fill_chroma_voids(req.page_rgb, out, protect_text=req.protect_text)
    if req.anchor_rgb is not None:
        channels = (1, 2) if req.anchor_chroma_only else (0, 1, 2)
        out = recomposite.match_palette(out, req.anchor_rgb, req.anchor_strength, channels=channels)
        out = recomposite.neutralize_ink_and_bubbles(req.page_rgb, out, protect_text=req.protect_text)
    return out


def build_colorizer(mode: str, device_pref: str = "auto") -> tuple[Colorizer, str | None]:
    """Returns (colorizer, theme_overlay). mode grammar:
    "none", "fast", "ai", "ai:<prompt>", "theme:sepia", "fast+theme:sunset"
    """
    theme_overlay: str | None = None
    base = mode
    if "+theme:" in mode:
        base, theme_overlay = mode.split("+theme:", 1)
    elif mode.startswith("theme:"):
        return ThemeColorizer(mode.removeprefix("theme:")), None

    if base == "none":
        return PassthroughColorizer(), theme_overlay
    if base == "fast":
        from ..models.gan import V2GanColorizer

        return V2GanColorizer(device_pref=device_pref), theme_overlay
    if base == "ai" or base.startswith("ai:"):
        from ..models.diffusion import DiffusionColorizer

        prompt = base[3:] if base.startswith("ai:") else ""
        return DiffusionColorizer(device_pref=device_pref, prompt=prompt), theme_overlay
    raise ValueError(f"unknown mode: {mode!r}")
