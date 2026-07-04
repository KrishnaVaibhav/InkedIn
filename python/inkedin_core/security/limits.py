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
ALLOWED_BOOK_EXTS = {".pdf", ".cbz", ".zip"}

# Magic-byte prefixes for sniffing (extension alone is never trusted).
MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"RIFF": "webp",  # + WEBP at offset 8, checked in validate
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"GIF8": "gif",
    b"%PDF": "pdf",
    b"PK\x03\x04": "zip",
}


class LimitExceeded(ValueError):
    """Raised when an input violates a hard limit."""


class ValidationFailed(ValueError):
    """Raised when an input fails structural validation."""
