"""WD14 auto-tagger: turn a panel into booru tags for the diffusion prompt.

The diffusion model can only color what it recognizes; a generic prompt on a
tiny multi-panel page is why subjects got the wrong colors. Tagging each
panel ("1girl, long hair, school uniform, night sky") tells the model what
it is looking at — the same trick ComfyUI/Auto1111 colorization workflows use
(WD14 Tagger nodes).

ONNX on CPU (~0.5 s/panel), weights cached in the workspace like the rest.
"""

from __future__ import annotations

import csv

import numpy as np

TAGGER_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.75
MAX_TAGS = 24

# Tags that describe the monochrome source or comic furniture, not content —
# feeding them back would fight the colorization prompt.
BLOCKLIST = {
    "monochrome", "greyscale", "grayscale", "comic", "halftone", "screentone",
    "speech bubble", "speech_bubble", "text", "english text", "japanese text",
    "sketch", "lineart", "line art", "no humans", "signature", "watermark",
    "white background", "grey background", "traditional media", "4koma",
}


class WD14Tagger:
    def __init__(self):
        from ..models.diffusion import _hf_cache_into_workspace

        _hf_cache_into_workspace()
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(TAGGER_REPO, "model.onnx")
        tags_path = hf_hub_download(TAGGER_REPO, "selected_tags.csv")

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[1]  # NHWC square

        self.tags: list[tuple[str, int]] = []  # (name, category) per output index
        with open(tags_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.tags.append((row["name"].replace("_", " "), int(row["category"])))

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 255, np.uint8)
        canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = rgb
        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return canvas[:, :, ::-1].astype(np.float32)[None]  # RGB->BGR, NHWC

    def tag(self, rgb: np.ndarray) -> str:
        probs = self.session.run(None, {self.input_name: self._preprocess(rgb)})[0][0]
        picked: list[tuple[float, str]] = []
        for (name, category), p in zip(self.tags, probs):
            if name in BLOCKLIST:
                continue
            if category == 0 and p >= GENERAL_THRESHOLD:  # general
                picked.append((float(p), name))
            elif category == 4 and p >= CHARACTER_THRESHOLD:  # character
                picked.append((float(p), name))
        picked.sort(reverse=True)
        return ", ".join(name for _, name in picked[:MAX_TAGS])
