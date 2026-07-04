import zipfile

import numpy as np
from PIL import Image

from inkedin_core.jobs import JobManager
from inkedin_core.pipeline import export as export_mod
from inkedin_core.pipeline import ingest
from inkedin_core.pipeline.colorize import THEMES, ColorizeRequest, ThemeColorizer, run_page
from inkedin_core.pipeline.recomposite import match_palette, preserve_lines


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


def test_export_refuses_job_temp_target(manga_page):
    jm = JobManager()
    job = jm.create(manga_page)
    jm.run(job.id, mode="theme:sepia")
    import pytest

    with pytest.raises(ValueError):
        jm.export(job.id, "cbz", job.dir / "inside.cbz")
