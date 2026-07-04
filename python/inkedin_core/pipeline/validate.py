"""Input validation: size, magic bytes, image geometry. Extension is never trusted."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..security.limits import (
    MAGIC,
    MAX_IMAGE_DIM,
    MAX_IMAGE_MEGAPIXELS,
    MAX_INPUT_FILE_BYTES,
    LimitExceeded,
    ValidationFailed,
)

# Pillow's own decompression-bomb guard, aligned with our limit.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_MEGAPIXELS * 1_000_000


def sniff_type(path: Path) -> str:
    """Return detected type from magic bytes:
    png/jpeg/webp/bmp/tiff/gif/pdf/zip/7z/rar/tar."""
    size = path.stat().st_size
    if size == 0:
        raise ValidationFailed(f"empty file: {path.name}")
    if size > MAX_INPUT_FILE_BYTES:
        raise LimitExceeded(f"file exceeds {MAX_INPUT_FILE_BYTES} bytes")

    with open(path, "rb") as f:
        head = f.read(16)
        # tar (CBT): "ustar" signature sits at offset 257, not 0.
        f.seek(257)
        tar_sig = f.read(8)

    for magic, kind in MAGIC.items():
        if head.startswith(magic):
            if kind == "webp" and head[8:12] != b"WEBP":
                continue
            return kind
    if tar_sig[:5] == b"ustar":
        return "tar"
    raise ValidationFailed(f"unrecognized file signature: {path.name}")


def check_image_geometry(path: Path) -> tuple[int, int]:
    """Header-only geometry check before full decode."""
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:  # Pillow raises many types on hostile input
        raise ValidationFailed(f"undecodable image: {path.name}: {e}") from e
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise LimitExceeded(f"image dimension {w}x{h} exceeds {MAX_IMAGE_DIM}")
    if w * h > MAX_IMAGE_MEGAPIXELS * 1_000_000:
        raise LimitExceeded(f"image {w}x{h} exceeds {MAX_IMAGE_MEGAPIXELS} MP")
    return w, h


def decode_image(path: Path) -> Image.Image:
    """Full decode to RGB after validation. EXIF orientation applied, metadata dropped."""
    check_image_geometry(path)
    from PIL import ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
