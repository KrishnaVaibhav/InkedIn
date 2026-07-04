"""Typesetting layer: replace detected source-language text with English,
in place, without overflow.

Layout rule: when a text region sits inside a speech bubble, the ENGLISH text
is laid out in the whole bubble interior (the visual area a reader assigns
the text to), not the tight glyph box — tight boxes are why translations used
to overflow. Regions outside bubbles keep their own box. Fitting shrinks the
font, wraps on words, and breaks over-long words character-wise, so text
never spills outside its area.

Detection recovery: bubbles that the detector returned WITHOUT a text region
are OCR'd anyway (their interior becomes a synthetic region) — missed-text
insurance.

Pure image logic — OCR/MT/detector arrive as callables so tests can fake them.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import cv2
import numpy as np

PAD = 6  # px around detected boxes: glyphs must sit clear of the region border
#         so the border-connected-component filter never mistakes them for art
MAX_FONT = 44
MIN_FONT = 8
LINE_SPACING = 1.08
TEXT_DARK_THRESHOLD = 150  # source pixels darker than this inside a box = glyphs
BUBBLE_INSET = 0.08  # bubble box shrink per side -> usable interior

# scripts where ASCII text means "leave it alone" (SFX, numbers, English).
# For Latin-script sources (es/fr/de/...) ASCII IS the language.
NONLATIN_LANGS = {"ja", "ko", "zh", "ru"}
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\comicbd.ttf",  # Comic Sans MS Bold: the comic classic
    r"C:\Windows\Fonts\comic.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
]
_font_path: str | None | bool = False  # False = not probed yet


def _font(size: int):
    from PIL import ImageFont

    global _font_path
    if _font_path is False:
        _font_path = None
        for cand in _FONT_CANDIDATES:
            try:
                ImageFont.truetype(cand, 12)
                _font_path = cand
                break
            except OSError:
                continue
    if _font_path:
        return ImageFont.truetype(_font_path, size)
    return ImageFont.load_default(size=size)  # Pillow's embedded scalable font


def _text_width(font, s: str) -> int:
    box = font.getbbox(s)
    return box[2] - box[0]


def _break_word(word: str, font, max_w: int) -> list[str]:
    """Character-wise split of a word wider than the box."""
    chunks, cur = [], ""
    for ch in word:
        if cur and _text_width(font, cur + ch) > max_w:
            chunks.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        chunks.append(cur)
    return chunks


def _wrap(words: list[str], font, max_w: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for w in words:
        if _text_width(font, w) > max_w:  # over-long word: hard-break it
            for chunk in _break_word(w, font, max_w):
                if cur and _text_width(font, f"{cur} {chunk}") > max_w:
                    lines.append(cur)
                    cur = chunk
                elif cur:
                    cur = f"{cur} {chunk}"
                else:
                    cur = chunk
            continue
        trial = f"{cur} {w}".strip()
        if cur and _text_width(font, trial) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def fit_text(text: str, box_w: int, box_h: int) -> tuple[object, list[str], int]:
    """Largest font whose wrapped lines fit the box; long words are broken, so
    width never overflows. Height overflow only possible below MIN_FONT."""
    words = text.split()
    start = max(MIN_FONT, min(MAX_FONT, int(box_h * 0.8)))
    for size in range(start, MIN_FONT - 1, -1):
        font = _font(size)
        lines = _wrap(words, font, box_w)
        line_h = max(size + 1, int(size * LINE_SPACING))
        if lines and len(lines) * line_h <= box_h and all(_text_width(font, ln) <= box_w for ln in lines):
            return font, lines, line_h
    font = _font(MIN_FONT)
    return font, _wrap(words, font, box_w), MIN_FONT + 1


def should_translate(original: str, src_lang: str) -> bool:
    original = original.strip()
    if not original or not _HAS_LETTER.search(original):
        return False  # empty / digits / punctuation only
    if src_lang in NONLATIN_LANGS and original.isascii():
        return False  # ASCII in a CJK/Cyrillic book = SFX or already English
    return True


def _pad_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, x0 - PAD), max(0, y0 - PAD), min(w, x1 + PAD), min(h, y1 + PAD)


def _inset_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    dx = int((x1 - x0) * BUBBLE_INSET)
    dy = int((y1 - y0) * BUBBLE_INSET)
    return x0 + dx, y0 + dy, x1 - dx, y1 - dy


def _glyph_mask(src_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Mask of the original lettering inside a box, from the SOURCE page
    (colorization never moves glyphs, so source geometry is authoritative).

    Dark components CONNECTED TO THE BOX BORDER are excluded: those are bubble
    outlines or art crossing the region, not letters — erasing them punched
    holes in bubble outlines."""
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(src_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    dark = (gray < TEXT_DARK_THRESHOLD).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    h, w = dark.shape
    mask = np.zeros_like(dark)
    for i in range(1, num):
        bx, by, bw2, bh2, _area = stats[i]
        if bx == 0 or by == 0 or bx + bw2 >= w or by + bh2 >= h:
            continue  # touches region border: outline/art, not a letter
        mask[labels == i] = 1
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)


def _erase_bubble_text(out_rgb: np.ndarray, src_rgb: np.ndarray, box) -> None:
    """Fill lettering with the bubble's own background color (median of the
    non-glyph pixels), so themed/tinted bubbles stay seamless."""
    x0, y0, x1, y1 = box
    mask = _glyph_mask(src_rgb, box)
    region = out_rgb[y0:y1, x0:x1]
    bg_pixels = region[mask == 0]
    bg = np.median(bg_pixels, axis=0) if len(bg_pixels) else np.array([255, 255, 255])
    region[mask == 1] = bg.astype(np.uint8)


def _erase_free_text(out_rgb: np.ndarray, src_rgb: np.ndarray, box) -> None:
    """SFX sits on art: inpaint instead of flat fill."""
    x0, y0, x1, y1 = box
    mask = _glyph_mask(src_rgb, box)
    region = out_rgb[y0:y1, x0:x1]
    out_rgb[y0:y1, x0:x1] = cv2.inpaint(region, mask, 3, cv2.INPAINT_TELEA)


def _region_luminance(out_rgb: np.ndarray, box) -> float:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return 255.0
    return float(cv2.cvtColor(out_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).mean())


def _draw_text(out_rgb: np.ndarray, box, text: str, outlined: bool) -> np.ndarray:
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    font, lines, line_h = fit_text(text, bw, bh)

    # readable on any background: dark letters on light, light letters on dark
    dark_bg = _region_luminance(out_rgb, box) < 128
    if outlined or dark_bg:
        fill, stroke = (255, 255, 255), (0, 0, 0)
        stroke_w = max(2, line_h // 10)
    else:
        fill, stroke = (10, 10, 10), None
        stroke_w = 0

    img = Image.fromarray(out_rgb)
    d = ImageDraw.Draw(img)
    total_h = len(lines) * line_h
    y = y0 + max(0, (bh - total_h) // 2)
    for ln in lines:
        x = x0 + max(0, (bw - _text_width(font, ln)) // 2)
        d.text((x, y), ln, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
        y += line_h
    return np.array(img)


def _center_inside(box, outer) -> bool:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, by0, bx1, by1 = outer
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def translate_page(
    src_rgb: np.ndarray,
    out_rgb: np.ndarray,
    detect: Callable[[np.ndarray], list[dict] | None],
    ocr: Callable[[np.ndarray], str],
    translate: Callable[[str], str],
    include_sfx: bool = False,
    src_lang: str = "ja",
) -> np.ndarray:
    """Replace source-language text on out_rgb with English, using detections
    and glyph geometry from src_rgb. Returns the typeset page."""
    items = detect(src_rgb)
    if items is None:
        raise RuntimeError("text detector unavailable (weights not downloaded?)")

    h, w = src_rgb.shape[:2]
    bubbles = [it["box"] for it in items if it["label"] == "bubble"]
    texts = [it for it in items if it["label"] in ("text_bubble", "text_free")]

    # missed-text insurance: bubbles with no text region inside get one — but
    # only when the interior looks like a real bubble (light background) and
    # actually contains letter-shaped pixels; otherwise OCR hallucinates.
    for bb in bubbles:
        if any(_center_inside(t["box"], bb) for t in texts):
            continue
        inner = _inset_box(bb)
        ix0, iy0, ix1, iy1 = inner
        if ix1 - ix0 < 16 or iy1 - iy0 < 16:
            continue
        interior = cv2.cvtColor(src_rgb[iy0:iy1, ix0:ix1], cv2.COLOR_RGB2GRAY)
        if np.median(interior) < 190:
            continue  # dark/gray interior: art, not a speech bubble
        if _glyph_mask(src_rgb, inner).mean() < 0.005:
            continue  # effectively empty: nothing to read
        texts.append({"label": "text_bubble", "box": inner, "score": 0.0})

    # manga reading order: top-to-bottom, right-to-left (only affects MT cache warmth)
    texts.sort(key=lambda it: (it["box"][1] // 64, -it["box"][0]))

    out = out_rgb.copy()
    for it in texts:
        box = _pad_box(it["box"], w, h)
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        original = ocr(src_rgb[y0:y1, x0:x1])
        if not should_translate(original, src_lang):
            continue
        english = translate(original).strip()
        if not english:
            continue

        bubble = next((bb for bb in bubbles if _center_inside(box, bb)), None)
        in_bubble = it["label"] == "text_bubble" or bubble is not None
        if in_bubble:
            # lay the English out in the whole bubble interior when known:
            # same visual area, far more room than the tight glyph box
            layout = _inset_box(bubble) if bubble else box
            lx0, ly0, lx1, ly1 = layout
            if lx1 - lx0 < 8 or ly1 - ly0 < 8:
                layout = box
            # erase over the union of glyph box and interior, so glyph tails
            # just outside the detected text box don't survive
            ex0, ey0, ex1, ey1 = layout
            erase_box = (min(box[0], ex0), min(box[1], ey0), max(box[2], ex1), max(box[3], ey1))
            _erase_bubble_text(out, src_rgb, erase_box)
            out = _draw_text(out, layout, english, outlined=False)
        elif include_sfx:
            _erase_free_text(out, src_rgb, box)
            out = _draw_text(out, box, english, outlined=True)
    return out
