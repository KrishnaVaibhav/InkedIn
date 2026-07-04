import io
import zipfile

import pytest
from PIL import Image

from inkedin_core.security import archives
from inkedin_core.security.limits import LimitExceeded, ValidationFailed
from inkedin_core.pipeline import validate


def _zip_with(tmp_path, entries):
    p = tmp_path / "evil.cbz"
    with zipfile.ZipFile(p, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return p


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_zip_path_traversal_rejected(tmp_path):
    p = _zip_with(tmp_path, [("../../evil.png", _png_bytes())])
    with pytest.raises(ValidationFailed):
        archives.list_image_members(p)


def test_zip_absolute_path_rejected(tmp_path):
    p = _zip_with(tmp_path, [("/etc/evil.png", _png_bytes())])
    with pytest.raises(ValidationFailed):
        archives.list_image_members(p)


def test_zip_drive_letter_rejected(tmp_path):
    p = _zip_with(tmp_path, [("C:evil.png", _png_bytes())])
    with pytest.raises(ValidationFailed):
        archives.list_image_members(p)


def test_zip_symlink_rejected(tmp_path):
    p = tmp_path / "sym.cbz"
    with zipfile.ZipFile(p, "w") as zf:
        info = zipfile.ZipInfo("link.png")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "target")
    with pytest.raises(ValidationFailed):
        archives.list_image_members(p)


def test_zip_bomb_ratio_rejected(tmp_path):
    p = tmp_path / "bomb.cbz"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.png", b"\0" * 50_000_000)  # zeros deflate far beyond 120x
    with pytest.raises(LimitExceeded):
        archives.list_image_members(p)


def test_zip_no_images_rejected(tmp_path):
    p = _zip_with(tmp_path, [("readme.txt", b"hi")])
    with pytest.raises(ValidationFailed):
        archives.list_image_members(p)


def test_natural_page_order(tmp_path):
    p = _zip_with(tmp_path, [(f"page{i}.png", _png_bytes()) for i in (10, 2, 1)])
    names = [m.filename for m in archives.list_image_members(p)]
    assert names == ["page1.png", "page2.png", "page10.png"]


def test_fake_extension_rejected(tmp_path):
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"MZ\x90\x00 this is not an image")
    with pytest.raises(ValidationFailed):
        validate.sniff_type(fake)


def test_empty_file_rejected(tmp_path):
    f = tmp_path / "empty.png"
    f.touch()
    with pytest.raises(ValidationFailed):
        validate.sniff_type(f)


def test_magic_sniff_correct(manga_page):
    assert validate.sniff_type(manga_page) == "png"


def test_oversize_dimensions_rejected(tmp_path):
    # Hand-build a PNG header claiming 30000x30000 (header check, no full decode)
    big = tmp_path / "big.png"
    im = Image.new("RGB", (10, 10))
    im.save(big)
    import struct

    data = bytearray(big.read_bytes())
    data[16:24] = struct.pack(">II", 30000, 30000)
    big.write_bytes(bytes(data))
    with pytest.raises((LimitExceeded, ValidationFailed)):
        validate.check_image_geometry(big)
