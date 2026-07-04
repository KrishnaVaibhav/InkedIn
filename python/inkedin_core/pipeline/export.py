"""Export colorized pages to folder / PDF / CBZ.

Rules: write to a temp path first, atomic rename on success, metadata stripped,
filenames app-generated, never overwrite input.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from PIL import Image


def _atomic(dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    return tmp


def export_folder(page_files: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(page_files):
        dest = out_dir / f"page_{i:05d}.png"
        tmp = _atomic(dest)
        with Image.open(p) as im:
            im.convert("RGB").save(tmp, format="PNG")
        os.replace(tmp, dest)
    return out_dir


def export_cbz(page_files: list[Path], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic(dest)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(page_files):
            zf.write(p, arcname=f"page_{i:05d}.png")
    os.replace(tmp, dest)
    return dest


def export_pdf(page_files: list[Path], dest: Path, jpeg_quality: int = 92) -> Path:
    import pymupdf

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic(dest)
    with pymupdf.open() as doc:
        for p in page_files:
            with Image.open(p) as im:
                w, h = im.size
            page = doc.new_page(width=w * 72 / 200, height=h * 72 / 200)
            page.insert_image(page.rect, filename=str(p))
        doc.save(tmp, garbage=4, deflate=True)
    os.replace(tmp, dest)
    return dest


def export(page_files: list[Path], target: Path, fmt: str) -> Path:
    if fmt == "folder":
        return export_folder(page_files, target)
    if fmt == "cbz":
        return export_cbz(page_files, target)
    if fmt == "pdf":
        return export_pdf(page_files, target)
    raise ValueError(f"unknown export format {fmt!r}")
