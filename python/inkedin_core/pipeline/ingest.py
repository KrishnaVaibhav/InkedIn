"""Ingest: turn any supported input (image, folder, PDF, CBZ) into per-page files
inside a job-scoped temp workspace. Pages stream one at a time; the full book is
never held in RAM. All page filenames are app-generated.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..security import archives
from ..security.limits import (
    ALLOWED_IMAGE_EXTS,
    MAX_PAGE_COUNT,
    LimitExceeded,
    ValidationFailed,
)
from . import validate

PDF_RENDER_DPI = 200  # ~1650x2350 for a typical tankobon page


@dataclass
class PageRef:
    index: int  # 0-based page order
    source_path: Path  # normalized page image inside job workspace
    width: int
    height: int


def _page_name(index: int) -> str:
    return f"page_{index:05d}.png"


def ingest(input_path: Path, job_dir: Path) -> list[PageRef]:
    """Dispatch on sniffed type, never on extension."""
    input_path = input_path.resolve()
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        return _ingest_folder(input_path, pages_dir)

    kind = validate.sniff_type(input_path)
    if kind == "pdf":
        return _ingest_pdf(input_path, pages_dir)
    if kind == "zip":
        return _ingest_cbz(input_path, pages_dir)
    if kind in {"png", "jpeg", "webp", "bmp", "tiff", "gif"}:
        return [_ingest_single_image(input_path, pages_dir, 0)]
    raise ValidationFailed(f"unsupported input type: {kind}")


def _normalize_to(src_img, dest: Path) -> tuple[int, int]:
    """Save decoded RGB image as PNG; original file is never reused downstream."""
    src_img.save(dest, format="PNG")
    return src_img.size


def _ingest_single_image(path: Path, pages_dir: Path, index: int) -> PageRef:
    img = validate.decode_image(path)
    dest = pages_dir / _page_name(index)
    w, h = _normalize_to(img, dest)
    img.close()
    return PageRef(index=index, source_path=dest, width=w, height=h)


def _ingest_folder(folder: Path, pages_dir: Path) -> list[PageRef]:
    candidates = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS),
        key=lambda p: archives._natural_key(p.name),
    )
    if not candidates:
        raise ValidationFailed("folder contains no supported images")
    if len(candidates) > MAX_PAGE_COUNT:
        raise LimitExceeded(f"folder has {len(candidates)} images (max {MAX_PAGE_COUNT})")
    pages = []
    for i, p in enumerate(candidates):
        validate.sniff_type(p)
        pages.append(_ingest_single_image(p, pages_dir, i))
    return pages


def _ingest_pdf(pdf_path: Path, pages_dir: Path) -> list[PageRef]:
    import pymupdf

    pages: list[PageRef] = []
    with pymupdf.open(pdf_path) as doc:
        if doc.page_count > MAX_PAGE_COUNT:
            raise LimitExceeded(f"PDF has {doc.page_count} pages (max {MAX_PAGE_COUNT})")
        zoom = PDF_RENDER_DPI / 72
        mat = pymupdf.Matrix(zoom, zoom)
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            dest = pages_dir / _page_name(i)
            pix.save(dest)
            pages.append(PageRef(index=i, source_path=dest, width=pix.width, height=pix.height))
            del pix
    if not pages:
        raise ValidationFailed("PDF has no pages")
    return pages


def _ingest_cbz(zip_path: Path, pages_dir: Path) -> list[PageRef]:
    members = archives.list_image_members(zip_path)
    if len(members) > MAX_PAGE_COUNT:
        raise LimitExceeded(f"archive has {len(members)} pages (max {MAX_PAGE_COUNT})")
    pages = []
    raw_dir = pages_dir / "_raw"
    for i, member in enumerate(members):
        raw = archives.extract_member(zip_path, member, raw_dir / f"raw_{i:05d}")
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages
