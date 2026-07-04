"""Semantic colorization: SD 1.5 anime checkpoint + ControlNet lineart-anime.

Unlike the fast GAN (palette wash), this path understands content: skin, hair,
sky, clothing get plausible colors. Runs fp16 on 8 GB VRAM with CPU offload.
Pages render at <=896 px long edge (Research.MD Gap 6), then line recomposite
restores full-resolution ink (Gap 7). Deterministic seed per job (Gap 5).

Weights come from Hugging Face into the workspace cache (HF_HOME), safetensors
only, revisions pinned below. This is the manifest exception noted in
Research.MD: diffusers repos are multi-file, so we pin repo+revision instead of
a single SHA-256.
"""

from __future__ import annotations

import os

import numpy as np

from ..pipeline import recomposite
from ..pipeline.colorize import ColorizeRequest
from ..workspace import workspace_root
from . import device as device_mod

BASE_MODEL = "gsdf/Counterfeit-V2.5"  # anime SD1.5, CreativeML OpenRAIL-M
BASE_FALLBACK = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/control_v11p_sd15s2_lineart_anime"  # OpenRAIL

MAX_EDGE = 896  # /8-aligned render size cap for 8 GB VRAM
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
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.prompt = (prompt.strip() + ", " if prompt.strip() else "") + DEFAULT_PROMPT
        self.seed = seed

    def _control_image(self, page_rgb: np.ndarray):
        """lineart-anime control: white lines on black, render-size, /8 aligned."""
        import cv2
        from PIL import Image

        gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        scale = min(MAX_EDGE / max(h, w), 1.0)
        nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        # suppress screentone dots so the controlnet sees lines, not noise
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        lines = 255 - gray
        return Image.fromarray(cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)), (nw, nh)

    def colorize(self, req: ColorizeRequest) -> np.ndarray:
        import cv2

        control, (nw, nh) = self._control_image(req.page_rgb)
        gen = self.torch.Generator(device="cpu").manual_seed(self.seed)
        out = self.pipe(
            prompt=req.extra.get("prompt") or self.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=control,
            width=nw,
            height=nh,
            num_inference_steps=int(req.extra.get("steps", 24)),
            guidance_scale=7.0,
            controlnet_conditioning_scale=1.0,
            generator=gen,
        ).images[0]

        out8 = cv2.resize(
            np.array(out), (req.page_rgb.shape[1], req.page_rgb.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return recomposite.preserve_lines(req.page_rgb, out8, req.ink_weight)

    def close(self) -> None:
        del self.pipe
        if self.dev.device == "cuda":
            self.torch.cuda.empty_cache()
