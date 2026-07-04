"""Local OCR + machine translation for comic text, multi-language.

OCR engines:
- Japanese: kha-white/manga-ocr-base (Apache-2.0) — ViT encoder-decoder built
  for manga: vertical text, furigana, stylized fonts. Best-in-class for JA.
- Korean / Chinese / Russian / Spanish / other: EasyOCR (Apache-2.0),
  loaded with the language's recognizer + English. Models download once into
  the workspace (models/easyocr).

MT: facebook/m2m100_418M (MIT) — one model for every source language.

Everything runs on CPU so it never fights the diffusion model for VRAM.
"""

from __future__ import annotations

import numpy as np

from ..workspace import workspace_root

OCR_REPO = "kha-white/manga-ocr-base"  # Apache-2.0
MT_REPO = "facebook/m2m100_418M"  # MIT
MT_MAX_TOKENS = 128

# language -> (m2m100 code, easyocr code). ja routes to manga-ocr instead.
LANGS: dict[str, tuple[str, str]] = {
    "ja": ("ja", "ja"),
    "ko": ("ko", "ko"),
    "zh": ("zh", "ch_sim"),
    "ru": ("ru", "ru"),
    "es": ("es", "es"),
    "fr": ("fr", "fr"),
    "de": ("de", "de"),
    "it": ("it", "it"),
    "pt": ("pt", "pt"),
}


def _hf_cache_into_workspace() -> None:
    from .diffusion import _hf_cache_into_workspace as _f

    _f()


def is_cached() -> bool:
    """True when the JA models are already in the local cache (the common path)."""
    _hf_cache_into_workspace()
    try:
        from huggingface_hub import snapshot_download

        for repo in (OCR_REPO, MT_REPO):
            snapshot_download(
                repo,
                allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
                local_files_only=True,
            )
        return True
    except Exception:
        return False


class MangaTranslator:
    """OCR a text crop in `src_lang`, translate to English. Small in-memory
    cache because comics repeat lines (SFX, catchphrases) constantly."""

    def __init__(self, src_lang: str = "ja"):
        if src_lang not in LANGS:
            raise ValueError(f"unsupported language {src_lang!r}; have {sorted(LANGS)}")
        self.src_lang = src_lang
        _hf_cache_into_workspace()
        import torch

        self.torch = torch
        if src_lang == "ja":
            self._init_manga_ocr()
        else:
            self._init_easyocr(LANGS[src_lang][1])
        self._init_mt(LANGS[src_lang][0])
        self._cache: dict[str, str] = {}

    def _init_manga_ocr(self) -> None:
        from transformers import (
            AutoImageProcessor,
            BertJapaneseTokenizer,
            VisionEncoderDecoderModel,
        )

        print("[inkedin] loading manga-ocr (downloads once) ...")
        self.ocr_proc = AutoImageProcessor.from_pretrained(OCR_REPO, use_fast=True)
        # transformers 5.x AutoTokenizer insists on a fast tokenizer and manga-ocr
        # ships a slow BertJapanese vocab — instantiate the class directly.
        self.ocr_tok = BertJapaneseTokenizer.from_pretrained(OCR_REPO)
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(OCR_REPO)
        self.ocr_model.eval()
        self._easy = None

    def _init_easyocr(self, easy_code: str) -> None:
        import easyocr

        mdir = workspace_root() / "models" / "easyocr"
        mdir.mkdir(parents=True, exist_ok=True)
        print(f"[inkedin] loading EasyOCR [{easy_code}] (downloads once) ...")
        langs = [easy_code] if easy_code in ("ja", "ch_sim", "ko") else [easy_code, "en"]
        self._easy = easyocr.Reader(
            langs,
            gpu=False,
            model_storage_directory=str(mdir),
            user_network_directory=str(mdir),
            verbose=False,
        )

    def _init_mt(self, m2m_code: str) -> None:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        print("[inkedin] loading translation model (downloads once) ...")
        self.mt_tok = M2M100Tokenizer.from_pretrained(MT_REPO)
        self.mt_tok.src_lang = m2m_code
        self.mt_model = M2M100ForConditionalGeneration.from_pretrained(MT_REPO)
        self.mt_model.eval()
        self._en_id = self.mt_tok.get_lang_id("en")

    def ocr(self, crop_rgb: np.ndarray) -> str:
        """Text in one detected region ('' when nothing readable)."""
        if self._easy is not None:
            # upscale small crops: EasyOCR misses text below ~20 px height
            h, w = crop_rgb.shape[:2]
            if min(h, w) > 0 and min(h, w) < 64:
                import cv2

                s = 64 / min(h, w)
                crop_rgb = cv2.resize(crop_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
            pieces = self._easy.readtext(crop_rgb, detail=0, paragraph=True)
            return " ".join(p.strip() for p in pieces if p.strip())

        from PIL import Image

        img = Image.fromarray(crop_rgb).convert("RGB")
        with self.torch.inference_mode():
            pixel = self.ocr_proc(images=img, return_tensors="pt").pixel_values
            ids = self.ocr_model.generate(pixel, max_length=300)[0]
        text = self.ocr_tok.decode(ids, skip_special_tokens=True)
        return text.replace(" ", "").strip()

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text in self._cache:
            return self._cache[text]
        with self.torch.inference_mode():
            enc = self.mt_tok(text, return_tensors="pt", truncation=True, max_length=MT_MAX_TOKENS)
            ids = self.mt_model.generate(
                **enc, forced_bos_token_id=self._en_id, max_new_tokens=MT_MAX_TOKENS, num_beams=4
            )[0]
        out = self.mt_tok.decode(ids, skip_special_tokens=True).strip()
        self._cache[text] = out
        return out

    def close(self) -> None:
        if self._easy is None:
            del self.ocr_model
        self._easy = None
        del self.mt_model
