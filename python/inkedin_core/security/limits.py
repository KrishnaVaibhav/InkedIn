"""Hard resource limits. Inputs are hostile until validated."""

from __future__ import annotations

MAX_INPUT_FILE_BYTES = 1_500_000_000  # 1.5 GB book
MAX_PAGE_COUNT = 2_000
MAX_IMAGE_MEGAPIXELS = 80  # per page, pre-decode check
MAX_IMAGE_DIM = 20_000  # px, either axis

# Archive (CBZ) limits
MAX_ARCHIVE_ENTRIES = MAX_PAGE_COUNT + 64
MAX_ARCHIVE_EXPANDED_BYTES = 4_000_000_000
MAX_EXPANSION_RATIO = 120  # zip-bomb guard; screentone PNGs compress hard

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
ALLOWED_BOOK_EXTS = {".pdf", ".cbz", ".zip", ".cb7", ".7z", ".cbt", ".tar", ".cbr", ".rar", ".epub"}

# EPUB metadata files (container.xml / OPF / XHTML) are parsed with size-capped
# regex scans, never an XML parser, so entity-expansion bombs have no engine.
MAX_EPUB_META_BYTES = 1_000_000  # per metadata file

# Magic-byte prefixes for sniffing (extension alone is never trusted).
# tar has no signature at offset 0 ("ustar" lives at offset 257) and is
# handled separately in validate.sniff_type.
MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"RIFF": "webp",  # + WEBP at offset 8, checked in validate
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"GIF8": "gif",
    b"%PDF": "pdf",
    b"PK\x03\x04": "zip",  # also CBZ and EPUB; ingest disambiguates
    b"7z\xbc\xaf\x27\x1c": "7z",
    b"Rar!\x1a\x07": "rar",  # covers RAR4 (\x00) and RAR5 (\x01\x00)
}


class LimitExceeded(ValueError):
    """Raised when an input violates a hard limit."""


class ValidationFailed(ValueError):
    """Raised when an input fails structural validation."""
