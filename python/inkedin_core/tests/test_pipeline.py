import zipfile

import numpy as np
from PIL import Image, ImageDraw

from inkedin_core.jobs import JobManager
from inkedin_core.pipeline import export as export_mod
from inkedin_core.pipeline import ingest
from inkedin_core.pipeline.colorize import THEMES, ColorizeRequest, ThemeColorizer
from inkedin_core.pipeline.recomposite import match_palette, preserve_lines, recompose, text_bubble_mask


def test_ingest_single_image(manga_page, tmp_path):
    pages = ingest.ingest(manga_page, tmp_path / "job")
    assert len(pages) == 1
    assert pages[0].source_path.exists()
    assert (pages[0].width, pages[0].height) == (620, 900)


def test_ingest_cbz_roundtrip(manga_page, tmp_path):
    cbz = tmp_path / "book.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        for i in (3, 1, 2):
            zf.write(manga_page, arcname=f"p{i}.png")
    pages = ingest.ingest(cbz, tmp_path / "job")
    assert [p.index for p in pages] == [0, 1, 2]


def test_ingest_cb7_roundtrip(manga_page, tmp_path):
    import py7zr

    cb7 = tmp_path / "book.cb7"
    with py7zr.SevenZipFile(cb7, "w") as z:
        for i in (2, 1):
            z.write(manga_page, arcname=f"p{i}.png")
    pages = ingest.ingest(cb7, tmp_path / "job")
    assert [p.index for p in pages] == [0, 1]
    assert all(p.source_path.exists() for p in pages)


def test_ingest_cbt_roundtrip(manga_page, tmp_path):
    import tarfile

    cbt = tmp_path / "book.cbt"
    with tarfile.open(cbt, "w") as tf:
        for i in (10, 2):  # natural order check: p2 before p10
            tf.add(manga_page, arcname=f"p{i}.png")
    pages = ingest.ingest(cbt, tmp_path / "job")
    assert len(pages) == 2


def _color_png_bytes(rgb):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (60, 90), rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_ingest_epub_spine_order(tmp_path):
    """Spine order must beat natural filename sort."""
    epub = tmp_path / "book.epub"
    opf = """<package><manifest>
      <item id="g1" href="pages/one.xhtml" media-type="application/xhtml+xml"/>
      <item id="g2" href="pages/two.xhtml" media-type="application/xhtml+xml"/>
      <item id="i1" href="images/zz.png" media-type="image/png"/>
      <item id="i2" href="images/aa.png" media-type="image/png"/>
    </manifest><spine><itemref idref="g1"/><itemref idref="g2"/></spine></package>"""
    with zipfile.ZipFile(epub, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/pages/one.xhtml", '<html><body><img src="../images/zz.png"/></body></html>')
        zf.writestr("OEBPS/pages/two.xhtml", '<html><body><img src="../images/aa.png"/></body></html>')
        zf.writestr("OEBPS/images/zz.png", _color_png_bytes((255, 0, 0)))  # spine page 1: red
        zf.writestr("OEBPS/images/aa.png", _color_png_bytes((0, 0, 255)))  # spine page 2: blue
    pages = ingest.ingest(epub, tmp_path / "job")
    assert len(pages) == 2
    first = np.array(Image.open(pages[0].source_path))
    assert first[5, 5, 0] > 200 and first[5, 5, 2] < 50  # red first (spine), not aa.png


def test_split_spreads_rtl_order(tmp_path):
    """A landscape spread splits into two pages; RTL reads the right half first."""
    im = Image.new("RGB", (800, 400), (255, 255, 255))
    px = im.load()
    for y in range(400):  # left half dark, right half light
        for x in range(400):
            px[x, y] = (10, 10, 10)
    src = tmp_path / "spread.png"
    im.save(src)

    pages = ingest.ingest(src, tmp_path / "j1", split_spreads=True, rtl=False)
    assert len(pages) == 2
    p0 = np.array(Image.open(pages[0].source_path))
    assert p0.mean() < 100  # LTR: dark left half first

    pages = ingest.ingest(src, tmp_path / "j2", split_spreads=True, rtl=True)
    p0 = np.array(Image.open(pages[0].source_path))
    assert p0.mean() > 150  # RTL: light right half first


def test_bubble_mask_from_detector_boxes():
    from inkedin_core.pipeline.recomposite import _mask_from_bubble_boxes

    page = _bubble_page()
    mask = _mask_from_bubble_boxes(page, [(90, 90, 310, 260)])
    assert mask[175, 200] > 0.5  # inside bubble body
    assert mask[450, 200] == 0.0  # outside every box
    assert mask[100, 20] == 0.0  # dark art inside no box


def test_mode_none_passthrough(page_rgb):
    from inkedin_core.pipeline.colorize import build_colorizer

    col, overlay = build_colorizer("none")
    assert overlay is None
    out = col.colorize(ColorizeRequest(page_rgb=page_rgb))
    assert np.array_equal(out, page_rgb)


def test_run_mode_none_without_translate_rejected(manga_page):
    import pytest

    jm = JobManager()
    job = jm.create(manga_page)
    with pytest.raises(ValueError):
        jm.run(job.id, mode="none")
    jm.delete(job.id)


def test_fit_text_wraps_and_fits():
    from inkedin_core.pipeline.textlayer import _text_width, fit_text

    font, lines, line_h = fit_text("THIS IS A FAIRLY LONG SPEECH BUBBLE SENTENCE", 160, 120)
    assert len(lines) >= 2  # wrapped
    assert len(lines) * line_h <= 120
    assert all(_text_width(font, ln) <= 160 for ln in lines)


def test_translate_page_replaces_bubble_text():
    from inkedin_core.pipeline import textlayer

    page = _bubble_page()  # bubble ellipse (100,100)-(300,250) with dark bars inside
    text_box = (130, 135, 270, 220)
    items = [
        {"label": "bubble", "box": (100, 100, 300, 250), "score": 0.9},
        {"label": "text_bubble", "box": text_box, "score": 0.9},
    ]
    calls = {}

    def detect(src):
        return items

    def ocr(crop):
        return "こんにちは世界"

    def translate(t):
        calls["mt"] = t
        return "HELLO WORLD"

    erased = page.copy()
    layout = textlayer._inset_box(items[0]["box"])
    padded_text = textlayer._pad_box(text_box, page.shape[1], page.shape[0])
    erase_box = (
        min(padded_text[0], layout[0]),
        min(padded_text[1], layout[1]),
        max(padded_text[2], layout[2]),
        max(padded_text[3], layout[3]),
    )
    textlayer._erase_bubble_text(erased, page, erase_box)
    # original lettering bar at y=142,x=250 was black; erasing fills it with bubble bg
    assert erased[142, 250, 0] > 200

    out = textlayer.translate_page(page, page.copy(), detect, ocr, translate)
    assert calls["mt"] == "こんにちは世界"
    # new dark text pixels exist inside the text box
    x0, y0, x1, y1 = text_box
    region = out[y0:y1, x0:x1]
    assert (region < 100).any()
    # art outside every box untouched
    assert np.array_equal(out[400:, :, :], page[400:, :, :])


def test_translate_page_skips_ascii_text():
    from inkedin_core.pipeline import textlayer

    page = _bubble_page()
    items = [{"label": "text_bubble", "box": (130, 135, 270, 220), "score": 0.9}]
    mt_called = []

    out = textlayer.translate_page(
        page, page.copy(), lambda s: items, lambda c: "OK!!", lambda t: mt_called.append(t) or "X"
    )
    assert not mt_called  # already-English text left alone
    assert np.array_equal(out, page)


def test_should_translate_language_rules():
    from inkedin_core.pipeline.textlayer import should_translate

    assert should_translate("こんにちは", "ja")
    assert not should_translate("WHAM!!", "ja")  # ascii in a CJK book = SFX
    assert not should_translate("123 !?", "es")  # no letters anywhere
    assert should_translate("Hola amigo", "es")  # ascii IS Spanish
    assert should_translate("Привет", "ru")
    assert not should_translate("OK", "ru")


def test_wrap_breaks_overlong_words():
    from inkedin_core.pipeline.textlayer import _text_width, fit_text

    font, lines, _ = fit_text("Supercalifragilisticexpialidocious", 80, 200)
    assert len(lines) >= 2  # word wider than 80px got hard-broken
    assert all(_text_width(font, ln) <= 80 for ln in lines)


def test_translate_page_recovers_bubble_without_text_box():
    """Detector missed the text region: the bubble interior is OCR'd anyway."""
    from inkedin_core.pipeline import textlayer

    page = _bubble_page()
    items = [{"label": "bubble", "box": (100, 100, 300, 250), "score": 0.9}]  # no text_bubble!
    ocr_boxes = []

    def ocr(crop):
        ocr_boxes.append(crop.shape)
        return "こんにちは"

    out = textlayer.translate_page(page, page.copy(), lambda s: items, ocr, lambda t: "HI THERE")
    assert ocr_boxes  # OCR ran on the synthetic interior region
    region = out[112:238, 116:284]
    assert (region < 100).any()  # English drawn inside the bubble


def test_draw_text_readable_on_dark_background():
    from inkedin_core.pipeline.textlayer import _draw_text

    dark = np.full((200, 300, 3), 30, np.uint8)
    out = _draw_text(dark, (20, 20, 280, 180), "NIGHT SCENE", outlined=False)
    assert (out > 200).any()  # switched to light lettering


def test_translate_page_detector_missing_raises():
    import pytest

    from inkedin_core.pipeline import textlayer

    page = _bubble_page()
    with pytest.raises(RuntimeError):
        textlayer.translate_page(page, page.copy(), lambda s: None, lambda c: "", lambda t: "")


def _rainbow_page(w=400, h=600):
    """Genuine color page: several distinct hues."""
    im = np.full((h, w, 3), 255, np.uint8)
    im[50:200, 40:360] = (220, 60, 50)
    im[220:370, 40:360] = (60, 180, 70)
    im[390:560, 40:360] = (70, 90, 210)
    return im


def test_color_page_detection(page_rgb):
    from inkedin_core.pipeline.colorstats import is_color_page

    assert is_color_page(_rainbow_page())[0] is True
    assert is_color_page(page_rgb)[0] is False  # B&W manga fixture

    sepia = np.full((600, 400, 3), 255, np.uint8)  # uniformly tinted paper
    sepia[:, :] = (222, 205, 170)
    assert is_color_page(sepia)[0] is False  # chroma yes, hue spread no

    # warm-dominant cover (clustered hues, like real volume openers)
    warm = np.full((900, 620, 3), (250, 240, 225), np.uint8)
    warm[100:420, 150:470] = (230, 150, 110)
    warm[430:820, 180:440] = (180, 60, 50)
    warm[180:240, 230:290] = (90, 140, 190)
    assert is_color_page(warm)[0] is True


def test_fill_chroma_voids_fills_enclosed_gray_only():
    from inkedin_core.pipeline.recomposite import fill_chroma_voids

    src = np.full((400, 400, 3), 255, np.uint8)  # source page (bright everywhere)
    out = np.full((400, 400, 3), 255, np.uint8)
    out[20:380, 20:380] = (200, 120, 60)  # colored art with white page margin
    out[150:250, 150:250] = (255, 255, 255)  # missing spot inside the color

    fixed = fill_chroma_voids(src, out, protect_text=False)
    center = fixed[200, 200].astype(int)
    assert center[0] - center[2] > 30  # got the surrounding orange chroma
    assert tuple(fixed[5, 5]) == (255, 255, 255)  # border margin untouched
    # lightness preserved: the filled spot stays bright (chroma-only inpaint)
    assert fixed[200, 200].max() > 180


def test_auto_ref_skips_color_pages_and_steers_palette(tmp_path):

    from PIL import Image as PILImage

    def make_book(folder):
        folder.mkdir()
        PILImage.fromarray(_rainbow_page()).save(folder / "p0_color.png")
        gray = np.full((600, 400, 3), 255, np.uint8)
        gray[100:500, 60:340] = 128
        PILImage.fromarray(gray).save(folder / "p1_bw.png")
        PILImage.fromarray(gray).save(folder / "p2_bw.png")

    jm = JobManager()

    b1 = tmp_path / "book1"
    make_book(b1)
    job = jm.create(b1)
    assert jm.color_pages(job.id) == [0]
    res = jm.run(job.id, mode="theme:noir", auto_ref=True)
    states = {p["page"]: p["status"] for p in jm.list_pages(job.id)}
    assert states[0] == "pending"  # color page skipped, kept as reference
    assert states[1] == states[2] == "done"
    with PILImage.open(jm.job_dir(job.id) / "colored" / "page_00001.png") as im:
        ref_run = np.array(im).astype(np.float32)

    b2 = tmp_path / "book2"
    make_book(b2)
    job2 = jm.create(b2)
    jm.run(job2.id, mode="theme:noir", selected=[1, 2], auto_ref=False)
    with PILImage.open(jm.job_dir(job2.id) / "colored" / "page_00001.png") as im:
        plain_run = np.array(im).astype(np.float32)

    def mean_chroma(rgb):
        import cv2

        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        return np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128).mean()

    # the anchored run must carry more color than the plain noir duotone —
    # a LIGHT bias (strength 0.15): visible shift, not a repaint
    assert mean_chroma(ref_run) > mean_chroma(plain_run) + 0.2
    assert res["done"] == 2


def test_ingest_pdf(manga_page, tmp_path):
    import pymupdf

    pdf = tmp_path / "book.pdf"
    with pymupdf.open() as doc:
        for _ in range(2):
            page = doc.new_page(width=310, height=450)
            page.insert_image(page.rect, filename=str(manga_page))
        doc.save(pdf)
    pages = ingest.ingest(pdf, tmp_path / "job")
    assert len(pages) == 2
    assert pages[0].width > 300  # rendered at 200 dpi, not 72


def test_theme_colorizer_adds_chroma(page_rgb):
    out = ThemeColorizer("sepia").colorize(ColorizeRequest(page_rgb=page_rgb))
    assert out.shape == page_rgb.shape
    # source is grayscale (R==G==B); themed output must not be
    assert not np.array_equal(out[:, :, 0], out[:, :, 2])


def test_all_themes_run(page_rgb):
    for t in THEMES:
        out = ThemeColorizer(t).colorize(ColorizeRequest(page_rgb=page_rgb))
        assert out.dtype == np.uint8


def test_preserve_lines_keeps_ink_dark(page_rgb):
    flat = np.full_like(page_rgb, 200)  # model output that lost all lines
    out = preserve_lines(page_rgb, flat)
    ink = page_rgb[:, :, 0] < 40
    assert out[ink].mean() < 90  # ink pulled back toward dark
    assert out[~ink].mean() > 150  # paper stays light


def test_recompose_keeps_lightness_no_ink_halo(page_rgb):
    import cv2

    # crude "model output": half-res orange wash painted over everything, lines included
    colored = cv2.resize(page_rgb, (310, 450))
    colored[:] = (220, 150, 80)
    out = recompose(page_rgb, colored, protect_text=False)

    ink = page_rgb[:, :, 0] < 40
    gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    assert gray_out[ink].mean() < 60  # ink stays dark
    assert gray_out[~ink].mean() > 150  # paper stays light
    assert not np.array_equal(out[:, :, 0], out[:, :, 2])  # paper got chroma
    # ink pixels carry (near-)neutral chroma: no color halo on lines/text
    rb = out[ink][:, 0].astype(int) - out[ink][:, 2].astype(int)
    assert abs(rb.mean()) < 12


def _bubble_page():
    im = Image.new("RGB", (400, 600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([10, 10, 390, 590], fill=(90, 90, 90))  # panel "art"
    d.ellipse([100, 100, 300, 250], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    for i, y in enumerate(range(140, 210, 14)):  # fake lettering
        d.rectangle([135, y, 265 - i * 10, y + 6], fill=(0, 0, 0))
    return np.array(im)


def test_text_bubble_mask_and_protection():
    page = _bubble_page()
    mask = text_bubble_mask(page)
    assert mask[175, 120] > 0.5  # inside bubble
    assert mask[450, 200] == 0.0  # panel art untouched

    colored = page.copy()
    colored[:] = (220, 150, 80)
    out = recompose(page, colored, protect_text=True)
    rb = out[:, :, 0].astype(int) - out[:, :, 2].astype(int)
    assert abs(rb[175, 120]) < 10  # bubble interior stays neutral
    assert rb[450, 200] > 25  # panel keeps its color


def test_detect_panels_finds_bordered_panels(page_rgb):
    from inkedin_core.pipeline.panels import detect_panels

    rects = detect_panels(page_rgb)
    assert len(rects) == 3  # fixture page: one wide panel, two below
    assert rects[0][1] < rects[1][1]  # row-major order
    for x, y, w, h in rects:
        assert w > 100 and h > 100


def test_detect_panels_falls_back_to_full_page():
    from inkedin_core.pipeline.panels import detect_panels

    blank = np.full((800, 600, 3), 255, np.uint8)
    assert detect_panels(blank) == [(0, 0, 600, 800)]


def test_match_palette_moves_stats(page_rgb):
    warm = ThemeColorizer("sunset").colorize(ColorizeRequest(page_rgb=page_rgb, ink_weight=0.0))
    cold = ThemeColorizer("ocean").colorize(ColorizeRequest(page_rgb=page_rgb, ink_weight=0.0))
    moved = match_palette(cold, warm, strength=1.0)
    # red-blue balance should approach the warm anchor
    warm_rb = warm[:, :, 0].astype(int).mean() - warm[:, :, 2].astype(int).mean()
    cold_rb = cold[:, :, 0].astype(int).mean() - cold[:, :, 2].astype(int).mean()
    moved_rb = moved[:, :, 0].astype(int).mean() - moved[:, :, 2].astype(int).mean()
    assert abs(moved_rb - warm_rb) < abs(cold_rb - warm_rb)


def test_export_all_formats(page_rgb, tmp_path):
    pages = []
    for i in range(2):
        p = tmp_path / f"c{i}.png"
        Image.fromarray(page_rgb).save(p)
        pages.append(p)

    folder = export_mod.export(pages, tmp_path / "outdir", "folder")
    assert len(list(folder.glob("*.png"))) == 2

    cbz = export_mod.export(pages, tmp_path / "out.cbz", "cbz")
    with zipfile.ZipFile(cbz) as zf:
        assert len(zf.namelist()) == 2

    pdf = export_mod.export(pages, tmp_path / "out.pdf", "pdf")
    assert pdf.stat().st_size > 1000


def test_job_manager_end_to_end_theme(manga_page, tmp_path):
    jm = JobManager()
    job = jm.create(manga_page)
    res = jm.run(job.id, mode="theme:sepia")
    assert res == {"status": "done", "done": 1, "errors": 0}
    out = jm.export(job.id, "cbz", tmp_path / "final.cbz")
    assert out.exists()
    jm.delete(job.id)
    assert not (job.dir.exists())


def test_job_page_selection(manga_page, tmp_path):
    import shutil

    folder = tmp_path / "book"
    folder.mkdir()
    for i in range(4):
        shutil.copy(manga_page, folder / f"p{i}.png")
    jm = JobManager()
    job = jm.create(folder)
    res = jm.run(job.id, selected=[1, 3], mode="theme:noir")
    assert res["done"] == 2
    states = {p["page"]: p["status"] for p in jm.list_pages(job.id)}
    assert states == {0: "pending", 1: "done", 2: "pending", 3: "done"}


def test_ref_strength_zero_means_no_bias(tmp_path):
    from PIL import Image as PILImage

    def make_book(folder):
        folder.mkdir()
        PILImage.fromarray(_rainbow_page()).save(folder / "p0_color.png")
        gray = np.full((600, 400, 3), 255, np.uint8)
        gray[100:500, 60:340] = 128
        PILImage.fromarray(gray).save(folder / "p1_bw.png")

    jm = JobManager()
    b1 = tmp_path / "b1"
    make_book(b1)
    j1 = jm.create(b1)
    jm.run(j1.id, mode="theme:noir", ref_strength=0.0)  # bias off, page still skipped
    states = {p["page"]: p["status"] for p in jm.list_pages(j1.id)}
    assert states[0] == "pending" and states[1] == "done"
    with PILImage.open(jm.job_dir(j1.id) / "colored" / "page_00001.png") as im:
        zero_bias = np.array(im)

    b2 = tmp_path / "b2"
    make_book(b2)
    j2 = jm.create(b2)
    jm.run(j2.id, mode="theme:noir", selected=[1], auto_ref=False)
    with PILImage.open(jm.job_dir(j2.id) / "colored" / "page_00001.png") as im:
        no_ref = np.array(im)

    assert np.array_equal(zero_bias, no_ref)  # strength 0 == feature fully out of the loop


def test_server_download_export(manga_page):
    import importlib
    import time

    from fastapi.testclient import TestClient

    import inkedin_core.server as srv

    srv = importlib.reload(srv)  # rebind the module-level JobManager to this test's workspace
    c = TestClient(srv.app)
    H = {"x-inkedin-token": srv.TOKEN}

    job = c.post("/api/jobs", headers=H, json={"path": str(manga_page)}).json()["job"]
    assert c.get(f"/api/jobs/{job}/download/cbz", headers=H).status_code == 400  # nothing colorized
    assert c.get(f"/api/jobs/{job}/download/exe", headers=H).status_code == 400  # bad format

    c.post(f"/api/jobs/{job}/run", headers=H, json={"mode": "theme:sepia"})
    for _ in range(200):
        p = c.get(f"/api/jobs/{job}/pages", headers=H).json()
        if p["progress"] and p["progress"]["done"] >= p["progress"]["total"]:
            break
        time.sleep(0.05)

    r = c.get(f"/api/jobs/{job}/download/cbz", headers=H)
    assert r.status_code == 200
    assert r.content[:2] == b"PK"  # a real ZIP came back
    assert "attachment" in r.headers.get("content-disposition", "")


def test_cli_parse_pages():
    from inkedin_core.cli import _parse_pages

    assert _parse_pages(None) is None
    assert _parse_pages("1,3,5-9") == [0, 2, 4, 5, 6, 7, 8]
    assert _parse_pages("2-2") == [1]
    assert _parse_pages("3,1,1") == [0, 2]


def test_build_colorizer_grammar_errors():
    import pytest

    from inkedin_core.pipeline.colorize import ThemeColorizer, build_colorizer

    with pytest.raises(ValueError):
        build_colorizer("watercolor")  # unknown mode
    with pytest.raises(ValueError):
        ThemeColorizer("bogus")  # unknown theme
    col, overlay = build_colorizer("none+theme:sepia")
    assert overlay == "sepia"  # grammar parses overlay for any base


def test_neutralize_ink_and_bubbles_guard(page_rgb):
    from inkedin_core.pipeline.recomposite import neutralize_ink_and_bubbles

    tinted = page_rgb.copy()
    tinted[:, :] = (200, 120, 90)  # palette op painted EVERYTHING, ink included
    out = neutralize_ink_and_bubbles(page_rgb, tinted, protect_text=False)
    ink = page_rgb[:, :, 0] < 40
    rb_ink = out[ink][:, 0].astype(int) - out[ink][:, 2].astype(int)
    rb_paper = out[~ink][:, 0].astype(int) - out[~ink][:, 2].astype(int)
    assert abs(rb_ink.mean()) < 12  # ink chroma pulled back to neutral
    assert rb_paper.mean() > 40  # paper keeps the color


def test_match_palette_chroma_only_preserves_lightness(page_rgb):
    import cv2

    warm = ThemeColorizer("sunset").colorize(ColorizeRequest(page_rgb=page_rgb, ink_weight=0.0))
    dark_anchor = (warm * 0.3).astype(np.uint8)  # much darker reference

    moved = match_palette(warm, dark_anchor, strength=1.0, channels=(1, 2))
    l_before = cv2.cvtColor(warm, cv2.COLOR_RGB2LAB)[:, :, 0].astype(float).mean()
    l_after = cv2.cvtColor(moved, cv2.COLOR_RGB2LAB)[:, :, 0].astype(float).mean()
    assert abs(l_before - l_after) < 2.0  # chroma-only: lightness untouched


def test_fill_voids_skips_mixed_hue_surroundings():
    from inkedin_core.pipeline.recomposite import fill_chroma_voids

    src = np.full((400, 400, 3), 255, np.uint8)
    out = np.full((400, 400, 3), 255, np.uint8)
    # void surrounded by two OPPOSING hues: inpainting would smear a mess
    out[100:300, 100:200] = (220, 60, 50)  # red left of the hole
    out[100:300, 200:300] = (60, 90, 210)  # blue right of the hole
    out[170:230, 170:230] = (255, 255, 255)  # the hole
    fixed = fill_chroma_voids(src, out, protect_text=False)
    assert tuple(fixed[200, 200]) == (255, 255, 255)  # left alone


def test_server_upload_batch_order_and_name(manga_page):
    import importlib
    import io

    from fastapi.testclient import TestClient
    from PIL import Image as PILImage

    import inkedin_core.server as srv

    srv = importlib.reload(srv)
    c = TestClient(srv.app)
    H = {"x-inkedin-token": srv.TOKEN}

    def png(color):
        b = io.BytesIO()
        PILImage.new("RGB", (60, 90), color).save(b, format="PNG")
        b.seek(0)
        return b

    files = [
        ("files", ("p1.png", png((255, 0, 0)), "image/png")),
        ("files", ("p2.png", png((0, 255, 0)), "image/png")),
        ("files", ("p3.png", png((0, 0, 255)), "image/png")),
    ]
    r = c.post("/api/jobs/upload-batch?name=MyBook", headers=H, files=files)
    assert r.status_code == 200
    assert r.json() == {"job": r.json()["job"], "pages": 3, "name": "MyBook"}
    job = r.json()["job"]

    img = c.get(f"/api/jobs/{job}/image/src/0", headers=H).content
    first = PILImage.open(io.BytesIO(img)).getpixel((5, 5))
    assert first[0] > 200 and first[2] < 60  # page 0 = first file sent (red)

    r = c.post("/api/jobs/upload-batch", headers=H,
               files=[("files", ("a.txt", io.BytesIO(b"hello"), "text/plain"))])
    assert r.status_code == 400  # non-image batch rejected
    c.delete(f"/api/jobs/{job}", headers=H)


def test_export_refuses_job_temp_target(manga_page):
    jm = JobManager()
    job = jm.create(manga_page)
    jm.run(job.id, mode="theme:sepia")
    import pytest

    with pytest.raises(ValueError):
        jm.export(job.id, "cbz", job.dir / "inside.cbz")
