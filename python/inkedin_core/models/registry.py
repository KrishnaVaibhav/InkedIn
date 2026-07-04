"""Model manifest + verified download into the workspace model cache.

Rules: HTTPS only, SHA-256 pinned in the manifest, atomic move after verify,
weights loaded with torch.load(weights_only=True), never execute repo code.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..workspace import models_dir

MANIFESTS = {
    "manga-colorization-v2": {
        "file": "generator.zip",  # torch zip-serialized state_dict
        "url": "https://huggingface.co/vergil1000/manga-colorization-v2/resolve/main/generator.zip",
        # Pinned after first verified download; see models/manifests/manga-colorization-v2.json
        "sha256": None,
        "license": "UNSPECIFIED (research repo, no license file) - local use only, do not redistribute",
        "source": "https://github.com/qweasdd/manga-colorization-v2",
    },
}

_PIN_DIR = Path(__file__).resolve().parents[3] / "models" / "manifests"


@dataclass
class ModelFile:
    model_id: str
    path: Path
    sha256: str


def _pinned_sha(model_id: str) -> str | None:
    pin = _PIN_DIR / f"{model_id}.json"
    if pin.exists():
        return json.loads(pin.read_text())["sha256"]
    return MANIFESTS[model_id].get("sha256")


def _write_pin(model_id: str, sha256: str, size: int) -> None:
    _PIN_DIR.mkdir(parents=True, exist_ok=True)
    m = MANIFESTS[model_id]
    (_PIN_DIR / f"{model_id}.json").write_text(
        json.dumps(
            {"model_id": model_id, "url": m["url"], "sha256": sha256, "size": size,
             "license": m["license"], "source": m["source"]},
            indent=2,
        )
    )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_id: str, progress=None) -> ModelFile:
    """Return verified local weights, downloading once if missing."""
    m = MANIFESTS[model_id]
    dest = models_dir() / model_id / m["file"]
    pinned = _pinned_sha(model_id)

    if dest.exists():
        digest = _sha256_of(dest)
        if pinned and digest != pinned:
            raise RuntimeError(f"model {model_id} hash mismatch: {digest} != pinned {pinned}")
        if not pinned:
            _write_pin(model_id, digest, dest.stat().st_size)
        return ModelFile(model_id, dest, digest)

    if not m["url"].startswith("https://"):
        raise RuntimeError("model URLs must be https")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(m["url"], headers={"User-Agent": "InkedIn/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)

    digest = _sha256_of(tmp)
    if pinned and digest != pinned:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {model_id} hash mismatch: {digest} != pinned {pinned}")
    os.replace(tmp, dest)
    if not pinned:
        # Trust-on-first-download: pin now so every later load is verified.
        _write_pin(model_id, digest, dest.stat().st_size)
    return ModelFile(model_id, dest, digest)
