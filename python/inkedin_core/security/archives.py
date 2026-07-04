"""Hostile-archive handling: CBZ/ZIP, CB7/7z, CBT/tar, CBR/RAR, EPUB.

Rules (see Research.MD):
- never trust member names: no absolute paths, no '..', no symlinks
- app generates all output filenames
- enforce entry count, expanded size, expansion ratio
- decode from stream into generated temp paths
- EPUB metadata is parsed with size-capped regex, never an XML parser
  (no entity expansion engine = no XML bombs)
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path, PurePosixPath

from .limits import (
    ALLOWED_IMAGE_EXTS,
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_EXPANDED_BYTES,
    MAX_EPUB_META_BYTES,
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
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# -- CB7 / 7z ---------------------------------------------------------------


def list_7z_image_members(path: Path) -> list[str]:
    """Validate a 7z/CB7 archive and return image member names in page order.

    7z solid archives don't expose reliable per-member compressed sizes, so the
    expansion-ratio guard compares total uncompressed size against the archive
    file size.
    """
    import py7zr

    archive_bytes = path.stat().st_size
    try:
        with py7zr.SevenZipFile(path, mode="r") as z:
            infos = z.list()
    except py7zr.exceptions.ArchiveError as e:
        raise ValidationFailed(f"corrupt 7z archive: {e}") from e

    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise LimitExceeded(f"archive has {len(infos)} entries (max {MAX_ARCHIVE_ENTRIES})")

    total = 0
    members: list[str] = []
    for info in infos:
        if info.is_directory:
            continue
        if _member_is_unsafe(info.filename):
            raise ValidationFailed(f"unsafe member path: {info.filename!r}")
        total += info.uncompressed or 0
        if total > MAX_ARCHIVE_EXPANDED_BYTES:
            raise LimitExceeded("archive expanded size exceeds limit")
        if PurePosixPath(info.filename).suffix.lower() in ALLOWED_IMAGE_EXTS:
            members.append(info.filename)

    if archive_bytes > 0 and total / archive_bytes > MAX_EXPANSION_RATIO:
        raise LimitExceeded("suspicious 7z expansion ratio")
    if not members:
        raise ValidationFailed("archive contains no supported image entries")
    members.sort(key=_natural_key)
    return members


def extract_7z_images(path: Path, members: list[str], raw_dir: Path) -> dict[str, Path]:
    """Extract validated image members into raw_dir in one sequential pass
    (solid 7z decompresses front-to-back; per-member opens would be O(n^2)).
    Member names were validated relative-safe, and the result is immediately
    re-sniffed and renamed to app-generated paths by the caller."""
    import py7zr

    raw_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(path, mode="r") as z:
        z.extract(path=raw_dir, targets=members)
    out = {}
    for m in members:
        p = raw_dir / PurePosixPath(m)
        if not p.is_file():
            raise ValidationFailed(f"7z member missing after extract: {m!r}")
        out[m] = p
    return out


# -- CBT / tar ---------------------------------------------------------------


def list_tar_image_members(path: Path) -> list[str]:
    """Validate a tar/CBT archive and return image member names in page order."""
    import tarfile

    try:
        with tarfile.open(path, mode="r:*") as tf:
            infos = tf.getmembers()
    except tarfile.TarError as e:
        raise ValidationFailed(f"corrupt tar archive: {e}") from e

    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise LimitExceeded(f"archive has {len(infos)} entries (max {MAX_ARCHIVE_ENTRIES})")

    total = 0
    members: list[str] = []
    for info in infos:
        if info.isdir():
            continue
        if not info.isreg():  # symlinks, devices, fifos: hostile in a comic
            raise ValidationFailed(f"archive contains a non-regular entry: {info.name!r}")
        if _member_is_unsafe(info.name):
            raise ValidationFailed(f"unsafe member path: {info.name!r}")
        total += info.size
        if total > MAX_ARCHIVE_EXPANDED_BYTES:
            raise LimitExceeded("archive expanded size exceeds limit")
        if PurePosixPath(info.name).suffix.lower() in ALLOWED_IMAGE_EXTS:
            members.append(info.name)

    if not members:
        raise ValidationFailed("archive contains no supported image entries")
    members.sort(key=_natural_key)
    return members


def extract_tar_member(path: Path, member: str, dest: Path) -> Path:
    import tarfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="r:*") as tf:
        src = tf.extractfile(member)
        if src is None:
            raise ValidationFailed(f"tar member not extractable: {member!r}")
        with src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    return dest


# -- CBR / RAR ---------------------------------------------------------------


def _rarfile():
    try:
        import rarfile
    except ImportError as e:
        raise ValidationFailed("CBR support needs the 'rarfile' package") from e
    return rarfile


def rar_backend_available() -> bool:
    try:
        _rarfile().tool_setup()
        return True
    except Exception:
        return False


def list_rar_image_members(path: Path) -> list[str]:
    """Validate a RAR/CBR archive and return image member names in page order.

    Entries are streamed through rarfile's read API — the external tool never
    extracts to disk, so unrar path-traversal bugs (CVE-2022-30333 class) have
    no write primitive here.
    """
    rarfile = _rarfile()
    try:
        rarfile.tool_setup()
    except rarfile.RarCannotExec as e:
        raise ValidationFailed(
            "no RAR extractor found (install unar, bsdtar or unrar >= 6.12, "
            "or convert the file to CBZ)"
        ) from e

    try:
        with rarfile.RarFile(path) as rf:
            infos = rf.infolist()
    except rarfile.Error as e:
        raise ValidationFailed(f"corrupt RAR archive: {e}") from e

    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise LimitExceeded(f"archive has {len(infos)} entries (max {MAX_ARCHIVE_ENTRIES})")

    total = 0
    members: list[str] = []
    for info in infos:
        if info.is_dir():
            continue
        if getattr(info, "is_symlink", lambda: False)():
            raise ValidationFailed("archive contains a symlink entry")
        if _member_is_unsafe(info.filename):
            raise ValidationFailed(f"unsafe member path: {info.filename!r}")
        total += info.file_size or 0
        if total > MAX_ARCHIVE_EXPANDED_BYTES:
            raise LimitExceeded("archive expanded size exceeds limit")
        if (info.compress_size or 0) > 0 and info.file_size / info.compress_size > MAX_EXPANSION_RATIO:
            raise LimitExceeded(f"suspicious expansion ratio for {info.filename!r}")
        if PurePosixPath(info.filename).suffix.lower() in ALLOWED_IMAGE_EXTS:
            members.append(info.filename)

    if not members:
        raise ValidationFailed("archive contains no supported image entries")
    members.sort(key=_natural_key)
    return members


def extract_rar_member(path: Path, member: str, dest: Path) -> Path:
    rarfile = _rarfile()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(path) as rf, rf.open(member) as src, open(dest, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)
    return dest


# -- EPUB --------------------------------------------------------------------

_RE_ROOTFILE = re.compile(rb'full-path\s*=\s*["\']([^"\']+)["\']')
_RE_ITEM = re.compile(r"<item\b[^>]*>", re.I)
_RE_ITEMREF = re.compile(r'<itemref\b[^>]*\bidref\s*=\s*["\']([^"\']+)["\']', re.I)
_RE_ATTR_ID = re.compile(r'\bid\s*=\s*["\']([^"\']+)["\']', re.I)
_RE_ATTR_HREF = re.compile(r'\bhref\s*=\s*["\']([^"\']+)["\']', re.I)
_RE_IMG_SRC = re.compile(r'<img\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']', re.I)
_RE_SVG_IMG = re.compile(r'<image\b[^>]*\b(?:xlink:)?href\s*=\s*["\']([^"\']+)["\']', re.I)


def is_epub(zip_path: Path) -> bool:
    """A ZIP is an EPUB when its 'mimetype' entry says so."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "mimetype" not in zf.namelist():
                return False
            return zf.read("mimetype")[:64].strip().startswith(b"application/epub+zip")
    except Exception:
        return False


def _read_meta(zf: zipfile.ZipFile, name: str) -> str | None:
    """Size-capped read of a metadata member; None when absent or oversized."""
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_EPUB_META_BYTES:
        return None
    return zf.read(name).decode("utf-8", errors="replace")


def epub_spine_order(zip_path: Path, image_members: list[str]) -> list[str] | None:
    """Reading order of image members from the OPF spine, or None to fall back
    to natural sort. Regex-scanned with per-file size caps; malformed books
    simply return None."""
    image_set = set(image_members)
    with zipfile.ZipFile(zip_path) as zf:
        try:
            container = zf.read("META-INF/container.xml")[:MAX_EPUB_META_BYTES]
        except KeyError:
            return None
        m = _RE_ROOTFILE.search(container)
        if not m:
            return None
        opf_name = m.group(1).decode("utf-8", errors="replace")
        if _member_is_unsafe(opf_name):
            return None
        opf = _read_meta(zf, opf_name)
        if opf is None:
            return None
        opf_dir = posixpath.dirname(opf_name)

        # manifest: id -> href (resolved against the OPF directory)
        hrefs: dict[str, str] = {}
        for tag in _RE_ITEM.findall(opf):
            mid, mhref = _RE_ATTR_ID.search(tag), _RE_ATTR_HREF.search(tag)
            if mid and mhref:
                href = posixpath.normpath(posixpath.join(opf_dir, mhref.group(1)))
                if not _member_is_unsafe(href):
                    hrefs[mid.group(1)] = href

        ordered: list[str] = []
        seen: set[str] = set()
        for idref in _RE_ITEMREF.findall(opf):
            href = hrefs.get(idref)
            if not href:
                continue
            if href in image_set:  # spine references the image directly
                candidates = [href]
            else:  # spine references an (X)HTML page wrapping one image
                doc = _read_meta(zf, href)
                if doc is None:
                    continue
                doc_dir = posixpath.dirname(href)
                candidates = [
                    posixpath.normpath(posixpath.join(doc_dir, u))
                    for u in (_RE_IMG_SRC.findall(doc) + _RE_SVG_IMG.findall(doc))
                ]
            for c in candidates:
                if c in image_set and c not in seen:
                    ordered.append(c)
                    seen.add(c)

    # Trust the spine only when it accounts for most of the book.
    if len(ordered) >= max(1, int(0.6 * len(image_members))):
        return ordered + [m for m in sorted(image_set - seen, key=_natural_key)]
    return None
