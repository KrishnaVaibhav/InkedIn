"""Ingest: turn any supported input (image, folder, PDF, CBZ/CB7/CBT/CBR, EPUB)
into per-page files inside a job-scoped temp workspace. Pages stream one at a
time; the full book is never held in RAM. All page filenames are app-generated.
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
SPREAD_RATIO = 1.15  # wider than this x height = double-page spread


@dataclass
class PageRef:
    index: int  # 0-based page order
    source_path: Path  # normalized page image inside job workspace
    width: int
    height: int


def _page_name(index: int) -> str:
    return f"page_{index:05d}.png"


def ingest(
    input_path: Path,
    job_dir: Path,
    split_spreads: bool = False,
    rtl: bool = False,
) -> list[PageRef]:
    """Dispatch on sniffed type, never on extension.

    split_spreads: cut landscape pages (double-page spreads) into two pages.
    rtl: right-to-left reading order — affects which spread half comes first.
    """
    input_path = input_path.resolve()
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        pages = _ingest_folder(input_path, pages_dir)
    else:
        kind = validate.sniff_type(input_path)
        if kind == "pdf":
            pages = _ingest_pdf(input_path, pages_dir)
        elif kind == "zip":
            if archives.is_epub(input_path):
                pages = _ingest_epub(input_path, pages_dir)
            else:
                pages = _ingest_cbz(input_path, pages_dir)
        elif kind == "7z":
            pages = _ingest_cb7(input_path, pages_dir)
        elif kind == "tar":
            pages = _ingest_cbt(input_path, pages_dir)
        elif kind == "rar":
            pages = _ingest_cbr(input_path, pages_dir)
        elif kind in {"png", "jpeg", "webp", "bmp", "tiff", "gif"}:
            pages = [_ingest_single_image(input_path, pages_dir, 0)]
        else:
            raise ValidationFailed(f"unsupported input type: {kind}")

    if split_spreads:
        pages = _split_spreads(pages, pages_dir, rtl)
    return pages


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


def _check_page_count(n: int) -> None:
    if n > MAX_PAGE_COUNT:
        raise LimitExceeded(f"archive has {n} pages (max {MAX_PAGE_COUNT})")


def _ingest_cbz(zip_path: Path, pages_dir: Path) -> list[PageRef]:
    members = archives.list_image_members(zip_path)
    _check_page_count(len(members))
    pages = []
    raw_dir = pages_dir / "_raw"
    for i, member in enumerate(members):
        raw = archives.extract_member(zip_path, member, raw_dir / f"raw_{i:05d}")
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages


def _ingest_epub(epub_path: Path, pages_dir: Path) -> list[PageRef]:
    """EPUB = ZIP + OPF spine. Container rules are identical to CBZ; page order
    comes from the spine when parsable, natural filename sort otherwise."""
    infos = archives.list_image_members(epub_path)
    _check_page_count(len(infos))
    by_name = {i.filename: i for i in infos}
    order = archives.epub_spine_order(epub_path, list(by_name))
    names = order if order else sorted(by_name, key=archives._natural_key)

    pages = []
    raw_dir = pages_dir / "_raw"
    for i, name in enumerate(names):
        raw = archives.extract_member(epub_path, by_name[name], raw_dir / f"raw_{i:05d}")
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages


def _ingest_cb7(path: Path, pages_dir: Path) -> list[PageRef]:
    members = archives.list_7z_image_members(path)
    _check_page_count(len(members))
    raw_dir = pages_dir / "_raw"
    extracted = archives.extract_7z_images(path, members, raw_dir)
    pages = []
    for i, member in enumerate(members):
        raw = extracted[member]
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages


def _ingest_cbt(path: Path, pages_dir: Path) -> list[PageRef]:
    members = archives.list_tar_image_members(path)
    _check_page_count(len(members))
    pages = []
    raw_dir = pages_dir / "_raw"
    for i, member in enumerate(members):
        raw = archives.extract_tar_member(path, member, raw_dir / f"raw_{i:05d}")
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages


def _ingest_cbr(path: Path, pages_dir: Path) -> list[PageRef]:
    members = archives.list_rar_image_members(path)
    _check_page_count(len(members))
    pages = []
    raw_dir = pages_dir / "_raw"
    for i, member in enumerate(members):
        raw = archives.extract_rar_member(path, member, raw_dir / f"raw_{i:05d}")
        validate.sniff_type(raw)
        pages.append(_ingest_single_image(raw, pages_dir, i))
        raw.unlink(missing_ok=True)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return pages


def _split_spreads(pages: list[PageRef], pages_dir: Path, rtl: bool) -> list[PageRef]:
    """Cut landscape pages in half. RTL books read the right half first."""
    from PIL import Image

    tmp_dir = pages_dir / "_split"
    tmp_dir.mkdir(exist_ok=True)
    halves: list[Path] = []
    for p in pages:
        with Image.open(p.source_path) as im:
            w, h = im.size
            if w > h * SPREAD_RATIO:
                mid = w // 2
                left = im.crop((0, 0, mid, h))
                right = im.crop((mid, 0, w, h))
                first, second = (right, left) if rtl else (left, right)
                for half in (first, second):
                    dest = tmp_dir / f"s_{len(halves):05d}.png"
                    half.save(dest, format="PNG")
                    halves.append(dest)
            else:
                dest = tmp_dir / f"s_{len(halves):05d}.png"
                im.save(dest, format="PNG")
                halves.append(dest)
        p.source_path.unlink(missing_ok=True)

    if len(halves) > MAX_PAGE_COUNT:
        raise LimitExceeded(f"spread split yields {len(halves)} pages (max {MAX_PAGE_COUNT})")

    out: list[PageRef] = []
    for i, tmp in enumerate(halves):
        dest = pages_dir / _page_name(i)
        tmp.replace(dest)
        with Image.open(dest) as im:
            w, h = im.size
        out.append(PageRef(index=i, source_path=dest, width=w, height=h))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out
