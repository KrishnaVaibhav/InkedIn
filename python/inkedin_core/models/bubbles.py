"""ML speech-bubble/text detector (Research.MD 2026-07-04 update, Gap 3).

ogkalu/comic-text-and-bubble-detector: RT-DETR-v2 r50vd, Apache-2.0, 42.9M
params, trained on ~11k manga/webtoon/manhua/western pages. Classes:
bubble, text_bubble (text inside bubbles), text_free (text outside bubbles).

Policy: the model is only *used* when its weights are already in the local
cache (`local_files_only`), so offline runs and tests never touch the network.
`ensure_downloaded()` is the single explicit download path — the UI checkbox,
`--ml-text`, and `doctor` call it. recomposite falls back to the white-blob
heuristic whenever this module reports unavailable.
"""

from __future__ import annotations

import threading

import numpy as np

REPO = "ogkalu/comic-text-and-bubble-detector"
SCORE_THRESHOLD = 0.30  # recall matters more than precision here: missed text
#                         stays untranslated, a weak false box just OCRs to ''

_lock = threading.Lock()
_detector = None  # (processor, model) once loaded
_state = "unknown"  # unknown | ready | unavailable


def _hf_cache_into_workspace() -> None:
    from .diffusion import _hf_cache_into_workspace as _f

    _f()


def ensure_downloaded() -> bool:
    """Download weights into the workspace HF cache (explicit user action)."""
    global _state
    _hf_cache_into_workspace()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        with _lock:
            if _state == "unavailable":
                _state = "unknown"  # retry the load now that files exist
        return True
    except Exception as e:
        print(f"[inkedin] bubble detector download failed: {e}")
        return False


def is_cached() -> bool:
    _hf_cache_into_workspace()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"], local_files_only=True)
        return True
    except Exception:
        return False


def _load():
    """Load once; never downloads (local_files_only)."""
    global _detector, _state
    with _lock:
        if _state == "ready":
            return _detector
        if _state == "unavailable":
            return None
        try:
            _hf_cache_into_workspace()
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            proc = AutoImageProcessor.from_pretrained(REPO, local_files_only=True, use_fast=True)
            model = AutoModelForObjectDetection.from_pretrained(REPO, local_files_only=True)
            model.eval()
            _detector = (proc, model)
            _state = "ready"
            return _detector
        except Exception:
            _state = "unavailable"
            return None


def detect_all(page_rgb: np.ndarray) -> list[dict] | None:
    """All detections as {label, box, score} where label is one of
    bubble / text_bubble / text_free and box is (x0, y0, x1, y1).
    None when the ML detector is not available."""
    det = _load()
    if det is None:
        return None
    proc, model = det
    import torch
    from PIL import Image

    h, w = page_rgb.shape[:2]
    with torch.inference_mode():
        inputs = proc(images=Image.fromarray(page_rgb), return_tensors="pt")
        outputs = model(**inputs)
        res = proc.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=SCORE_THRESHOLD
        )[0]

    id2label = model.config.id2label
    items: list[dict] = []
    for label_id, box, score in zip(
        res["labels"].tolist(), res["boxes"].tolist(), res["scores"].tolist()
    ):
        x0, y0, x1, y1 = (int(v) for v in box)
        items.append(
            {
                "label": id2label.get(int(label_id), "?"),
                "box": (max(0, x0), max(0, y0), min(w, x1), min(h, y1)),
                "score": float(score),
            }
        )
    return items


def detect_bubble_boxes(page_rgb: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    """Bubble bounding boxes (x0, y0, x1, y1), or None when the ML detector is
    not available (caller falls back to the heuristic)."""
    items = detect_all(page_rgb)
    if items is None:
        return None
    return [it["box"] for it in items if it["label"] == "bubble"]
