"""Hostile-archive (CBZ/ZIP) handling.

Rules (see Research.MD):
- never trust member names: no absolute paths, no '..', no symlinks
- app generates all output filenames
- enforce entry count, expanded size, expansion ratio
- decode from stream into generated temp paths
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

from .limits import (
    ALLOWED_IMAGE_EXTS,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_EXPANDED_BYTES,
    MAX_EXPANSION_RATIO,
    LimitExceeded,
    ValidationFailed,
)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    # Upper 16 bits of external_attr hold Unix mode; 0xA000 = symlink.
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _member_is_unsafe(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute():
        return True
    if any(part == ".." for part in p.parts):
        return True
    # Windows drive letters or reserved characters smuggled in names
    if ":" in name:
        return True
    return False


def list_image_members(zip_path: Path) -> list[zipfile.ZipInfo]:
    """Validate archive and return image members in natural page order."""
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise LimitExceeded(f"archive has {len(infos)} entries (max {MAX_ARCHIVE_ENTRIES})")

        total_expanded = 0
        members: list[zipfile.ZipInfo] = []
        for info in infos:
            if info.is_dir():
                continue
            if _is_symlink(info):
                raise ValidationFailed("archive contains a symlink entry")
            if _member_is_unsafe(info.filename):
                raise ValidationFailed(f"unsafe member path: {info.filename!r}")
            total_expanded += info.file_size
            if total_expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                raise LimitExceeded("archive expanded size exceeds limit")
            if info.compress_size > 0 and info.file_size / info.compress_size > MAX_EXPANSION_RATIO:
                raise LimitExceeded(f"suspicious expansion ratio for {info.filename!r}")
            ext = PurePosixPath(info.filename).suffix.lower()
            if ext in ALLOWED_IMAGE_EXTS:
                members.append(info)

        if not members:
            raise ValidationFailed("archive contains no supported image entries")

        members.sort(key=lambda i: _natural_key(i.filename))
        return members


def extract_member(zip_path: Path, member: zipfile.ZipInfo, dest: Path) -> Path:
    """Stream one validated member into an app-generated path (never the member's own name)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, open(dest, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    return dest


def _natural_key(name: str) -> list:
    """'page10' sorts after 'page2'."""
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]
