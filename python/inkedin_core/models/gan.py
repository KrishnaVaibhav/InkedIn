"""Fast-automatic colorizer: manga-colorization-v2 generator.

Preprocessing kept faithful to the upstream inference code (resize to width 576 /
height 864, pad to /32, grayscale + zero hint channels), then our recomposite step
restores full-resolution lines (Research.MD Gap 7).
"""

from __future__ import annotations

import numpy as np

from ..pipeline import recomposite
from ..pipeline.colorize import ColorizeRequest
from . import device as device_mod
from .registry import ensure_model

INFER_SIZE = 576  # multiple of 32, upstream default


def _resize_pad(gray: np.ndarray, size: int = INFER_SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """Upstream resize_pad, grayscale-only variant. Returns (HxWx1 float-ready, (pad_h, pad_w))."""
    import cv2

    img = gray[:, :, None] if gray.ndim == 2 else gray
    img = np.repeat(img[:, :, :1], 3, 2)

    if img.shape[0] < img.shape[1]:  # landscape spread
        ratio = img.shape[0] / (size * 1.5)
        width = int(np.ceil(img.shape[1] / ratio))
        img = cv2.resize(img, (width, int(size * 1.5)), interpolation=cv2.INTER_AREA)
        pad = (0, 32 - width % 32)  # upstream pads a full 32 when already aligned
        img = np.pad(img, ((0, 0), (0, pad[1]), (0, 0)), "maximum")
    else:
        ratio = img.shape[1] / size
        height = int(np.ceil(img.shape[0] / ratio))
        img = cv2.resize(img, (size, height), interpolation=cv2.INTER_AREA)
        pad = (32 - height % 32, 0)
        img = np.pad(img, ((0, pad[0]), (0, 0), (0, 0)), "maximum")

    return img[:, :, :1], pad


class V2GanColorizer:
    def __init__(self, device_pref: str = "auto"):
        import torch

        from .v2_nets import Generator

        self.torch = torch
        self.dev = device_mod.probe(device_pref)
        weights = ensure_model("manga-colorization-v2")

        self.net = Generator()
        state = torch.load(weights.path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval().to(self.dev.device)
        self.use_fp16 = self.dev.device == "cuda"
        if self.use_fp16:
            self.net.half()

    def colorize(self, req: ColorizeRequest) -> np.ndarray:
        import cv2

        torch = self.torch
        src = req.page_rgb
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        if req.extra.get("denoise", True):
            gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

        img, pad = _resize_pad(gray, INFER_SIZE)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        hint = torch.zeros(1, 4, x.shape[2], x.shape[3])
        inp = torch.cat([x, hint], 1).to(self.dev.device)
        if self.use_fp16:
            inp = inp.half()

        with torch.inference_mode():
            fake, _ = self.net(inp)
            out = (fake[0].float().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

        if pad[0]:
            out = out[: -pad[0]]
        if pad[1]:
            out = out[:, : -pad[1]]
        out8 = (out * 255).astype(np.uint8)
        out8 = cv2.resize(out8, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return recomposite.recompose(src, out8, protect_text=req.protect_text)

    def close(self) -> None:
        del self.net
        if self.dev.device == "cuda":
            self.torch.cuda.empty_cache()
