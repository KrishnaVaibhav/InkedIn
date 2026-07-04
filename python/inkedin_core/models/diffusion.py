"""Semantic colorization: SD 1.5 anime checkpoint + ControlNet lineart-anime.

Unlike the fast GAN (palette wash), this path understands content: skin, hair,
sky, clothing get plausible colors. Runs fp16 on 8 GB VRAM with CPU offload.

Pages are colorized PANEL-WISE (Research.MD panel-wise plan): a whole page
downscaled to <=896 px leaves each panel a few hundred pixels — too small for
the model to recognize subjects, which is exactly how objects ended up with
the wrong colors. Instead each detected panel renders at 512-896 px with its
own WD14 auto-tag prompt describing its actual content, panels are palette-
matched to the first panel for cross-panel consistency (fixed seed, Gap 5),
and the chroma-only recomposite restores full-resolution ink (Gap 7).

Weights come from Hugging Face into the workspace cache (HF_HOME), safetensors
only, revisions pinned below. This is the manifest exception noted in
Research.MD: diffusers repos are multi-file, so we pin repo+revision instead of
a single SHA-256.
"""

from __future__ import annotations

import os

import numpy as np

from ..pipeline import panels as panels_mod
from ..pipeline import recomposite
from ..pipeline.colorize import ColorizeRequest
from ..workspace import workspace_root
from . import device as device_mod

BASE_MODEL = "gsdf/Counterfeit-V2.5"  # anime SD1.5, CreativeML OpenRAIL-M
BASE_FALLBACK = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/control_v11p_sd15s2_lineart_anime"  # OpenRAIL

IP_ADAPTER_REPO = "h94/IP-Adapter"  # Apache-2.0
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.safetensors"

MAX_EDGE = 896  # /8-aligned render size cap for 8 GB VRAM
MIN_EDGE = 512  # upscale small panels to at least this so SD recognizes content
PANEL_ANCHOR_STRENGTH = 0.35  # palette pull toward the first panel
REF_SCALE = 0.65  # IP-Adapter strength for a user-supplied reference image
SELF_REF_SCALE = 0.4  # panels 2+ follow panel 1's colors when no user ref
AUTO_REF_SCALE = 0.3  # detected color page: seeds panel 1 only, gently —
#                       forcing it on every panel overrode correct semantics
DEFAULT_PROMPT = (
    "masterpiece, best quality, highly detailed anime color illustration, "
    "vivid natural colors, correct skin tones, soft shading"
)
NEGATIVE_PROMPT = (
    "monochrome, greyscale, sketch, lowres, blurry, bad anatomy, "
    "text, watermark, signature, jpeg artifacts"
)


def _hf_cache_into_workspace() -> None:
    os.environ.setdefault("HF_HOME", str(workspace_root() / "models" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class DiffusionColorizer:
    def __init__(self, device_pref: str = "auto", prompt: str = "", seed: int = 31337):
        _hf_cache_into_workspace()
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        self.torch = torch
        self.dev = device_mod.probe(device_pref)
        if self.dev.device == "cpu":
            raise RuntimeError("ai mode needs a GPU; use mode 'fast' on CPU")
        dtype = torch.float16

        # diffusers 0.39 + hub 1.x drops sub-folder config.json from its download
        # filter; snapshot_download the full repos ourselves and load locally.
        from huggingface_hub import snapshot_download

        patterns = ["*.json", "*.txt", "*.safetensors", "*.model"]
        cn_dir = snapshot_download(CONTROLNET, allow_patterns=patterns, ignore_patterns=["*.bin", "*non_ema*"])
        controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype)
        try:
            base_dir = snapshot_download(BASE_MODEL, allow_patterns=patterns, ignore_patterns=["*.bin", "*non_ema*"])
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_dir, controlnet=controlnet, torch_dtype=dtype, safety_checker=None,
            )
        except Exception:
            base_dir = snapshot_download(BASE_FALLBACK, allow_patterns=patterns, ignore_patterns=["*.bin", "*non_ema*"])
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_dir, controlnet=controlnet, torch_dtype=dtype, safety_checker=None,
            )
        # Reference conditioning (IP-Adapter): a colored reference image steers
        # the palette; also used panel-to-panel for cross-panel consistency.
        from diffusers.schedulers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHT)

        # NOTE: no enable_attention_slicing() — it swaps in SlicedAttnProcessor,
        # silently discarding the IP-Adapter attention processors (tuple crash).
        # PyTorch 2 SDPA is already memory-efficient.
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.prompt = (prompt.strip() + ", " if prompt.strip() else "") + DEFAULT_PROMPT
        self.seed = seed
        self.tagger = self._build_tagger()

    @staticmethod
    def _build_tagger():
        try:
            from .tagger import WD14Tagger

            return WD14Tagger()
        except Exception as e:  # onnxruntime missing / offline: degrade to generic prompt
            print(f"[inkedin] WD14 tagger unavailable ({e}); using generic prompts")
            return None

    def _control_image(self, page_rgb: np.ndarray):
        """lineart-anime control: white lines on black, render-size, /8 aligned."""
        import cv2
        from PIL import Image

        gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        long_edge = max(h, w)
        scale = min(MAX_EDGE, max(MIN_EDGE, long_edge)) / long_edge
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        # suppress screentone dots so the controlnet sees lines, not noise
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        lines = 255 - gray
        return Image.fromarray(cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)), (nw, nh)

    def _panel_prompt(self, panel_rgb: np.ndarray, req: ColorizeRequest) -> str:
        user = req.extra.get("prompt") or ""
        tags = self.tagger.tag(panel_rgb) if self.tagger is not None else ""
        parts = [p for p in (user.strip(), tags, DEFAULT_PROMPT) if p]
        return ", ".join(parts)

    def _render(
        self, panel_rgb: np.ndarray, prompt: str, steps: int,
        ref_rgb: np.ndarray | None, ref_scale: float,
    ) -> np.ndarray:
        import cv2
        from PIL import Image

        control, (nw, nh) = self._control_image(panel_rgb)
        # IP-Adapter always needs an image once loaded; scale 0 disables it.
        self.pipe.set_ip_adapter_scale(ref_scale if ref_rgb is not None else 0.0)
        ref = Image.fromarray(ref_rgb if ref_rgb is not None else panel_rgb)
        gen = self.torch.Generator(device="cpu").manual_seed(self.seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=control,
            ip_adapter_image=ref,
            width=nw,
            height=nh,
            num_inference_steps=steps,
            guidance_scale=7.0,
            controlnet_conditioning_scale=1.0,
            generator=gen,
        ).images[0]
        return cv2.resize(
            np.array(out), (panel_rgb.shape[1], panel_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def colorize(self, req: ColorizeRequest) -> np.ndarray:
        steps = int(req.extra.get("steps", 24))
        user_ref: np.ndarray | None = req.extra.get("ref_rgb")
        auto_ref: np.ndarray | None = req.extra.get("auto_ref_rgb")
        canvas = req.page_rgb.copy()  # gutters keep the source page
        anchor: np.ndarray | None = None

        ip_scale = float(req.extra.get("ip_scale", REF_SCALE))
        self_scale = float(req.extra.get("self_ref_scale", SELF_REF_SCALE))
        auto_scale = float(req.extra.get("auto_ref_scale", AUTO_REF_SCALE))
        for x, y, w, h in panels_mod.detect_panels(req.page_rgb):
            crop = req.page_rgb[y : y + h, x : x + w]
            if user_ref is not None:  # explicit reference: applies to every panel
                ref, scale = user_ref, ip_scale
            elif anchor is None and auto_ref is not None:  # color-page seed, panel 1 only
                ref, scale = auto_ref, auto_scale
            else:  # the liked behavior: panels follow panel 1's own colors
                ref, scale = anchor, self_scale
            out = self._render(crop, self._panel_prompt(crop, req), steps, ref, scale)
            if anchor is None:
                anchor = out
            else:
                out = recomposite.match_palette(out, anchor, PANEL_ANCHOR_STRENGTH)
            canvas[y : y + h, x : x + w] = out

        return recomposite.recompose(req.page_rgb, canvas, protect_text=req.protect_text)

    def close(self) -> None:
        del self.pipe
        if self.dev.device == "cuda":
            self.torch.cuda.empty_cache()
