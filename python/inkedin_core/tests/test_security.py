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


def test_tar_path_traversal_rejected(tmp_path, manga_page):
    import tarfile

    p = tmp_path / "evil.cbt"
    with tarfile.open(p, "w") as tf:
        tf.add(manga_page, arcname="../../evil.png")
    with pytest.raises(ValidationFailed):
        archives.list_tar_image_members(p)


def test_tar_symlink_rejected(tmp_path):
    import tarfile

    p = tmp_path / "sym.cbt"
    with tarfile.open(p, "w") as tf:
        info = tarfile.TarInfo("link.png")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(ValidationFailed):
        archives.list_tar_image_members(p)


def test_member_name_validator_blocks_hostile_paths():
    # shared by zip/7z/tar/rar listing (py7zr's writer refuses to *create*
    # traversal names, so the validator is exercised directly)
    for bad in ("../../evil.png", "/abs/evil.png", "C:evil.png", "a/../../b.png", "\\\\srv\\share.png"):
        assert archives._member_is_unsafe(bad), bad
    for ok in ("page1.png", "vol1/page1.png", "a.b/c-d_e.png"):
        assert not archives._member_is_unsafe(ok), ok


def test_7z_bomb_ratio_rejected(tmp_path):
    import py7zr

    p = tmp_path / "bomb.cb7"
    with py7zr.SevenZipFile(p, "w") as z:
        z.writestr(b"\0" * 60_000_000, "a.png")  # zeros compress far beyond 120x
    with pytest.raises(LimitExceeded):
        archives.list_7z_image_members(p)


def test_7z_no_images_rejected(tmp_path):
    import py7zr

    p = tmp_path / "empty.cb7"
    with py7zr.SevenZipFile(p, "w") as z:
        z.writestr(b"hi", "readme.txt")
    with pytest.raises(ValidationFailed):
        archives.list_7z_image_members(p)


def test_epub_detection(tmp_path):
    epub = tmp_path / "b.epub"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("x.png", _png_bytes())
    assert archives.is_epub(epub)

    cbz = tmp_path / "b.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        zf.writestr("x.png", _png_bytes())
    assert not archives.is_epub(cbz)


def test_sniff_new_formats(tmp_path):
    import tarfile

    import py7zr

    t = tmp_path / "x.bin"
    with tarfile.open(t, "w") as tf:
        info = tarfile.TarInfo("a.png")
        info.size = 0
        tf.addfile(info, io.BytesIO(b""))
    assert validate.sniff_type(t) == "tar"

    s = tmp_path / "y.bin"
    with py7zr.SevenZipFile(s, "w") as z:
        z.writestr(b"d", "a.txt")
    assert validate.sniff_type(s) == "7z"

    r = tmp_path / "z.bin"
    r.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"\0" * 64)
    assert validate.sniff_type(r) == "rar"


def test_epub_without_container_falls_back_to_natural_order(tmp_path):
    """Broken/absent OPF must degrade to filename sort, never crash."""
    from inkedin_core.pipeline import ingest

    epub = tmp_path / "b.epub"
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        for i in (10, 2, 1):
            zf.writestr(f"img/p{i}.png", _png_bytes())
    assert archives.epub_spine_order(epub, [f"img/p{i}.png" for i in (10, 2, 1)]) is None
    pages = ingest.ingest(epub, tmp_path / "job")
    assert len(pages) == 3  # natural order ingest still works


def test_epub_hostile_rootfile_path_ignored(tmp_path):
    p = tmp_path / "evil.epub"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="../../evil.opf"/></rootfiles></container>',
        )
        zf.writestr("x.png", _png_bytes())
    assert archives.epub_spine_order(p, ["x.png"]) is None  # unsafe path -> fallback


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
