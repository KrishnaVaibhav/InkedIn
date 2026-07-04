import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Every test gets its own workspace root; nothing touches the real data dirs."""
    monkeypatch.setenv("INKEDIN_ROOT", str(tmp_path / "ws"))
    yield tmp_path


@pytest.fixture
def manga_page(tmp_path):
    """Synthetic B/W manga-ish page: panels, lines, text block, screentone dots."""
    im = Image.new("L", (620, 900), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 600, 430], outline=0, width=4)
    d.rectangle([20, 460, 300, 880], outline=0, width=4)
    d.rectangle([320, 460, 600, 880], outline=0, width=4)
    d.ellipse([200, 100, 420, 320], outline=0, width=3)
    d.text((60, 60), "WHAM!", fill=0)
    for y in range(480, 860, 8):  # screentone
        for x in range(340, 580, 8):
            d.ellipse([x, y, x + 3, y + 3], fill=0)
    p = tmp_path / "page.png"
    im.convert("RGB").save(p)
    return p


@pytest.fixture
def page_rgb(manga_page):
    return np.array(Image.open(manga_page).convert("RGB"))
