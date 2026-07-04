# InkedIn
[![CI](https://github.com/KrishnaVaibhav/InkedIn/actions/workflows/ci.yml/badge.svg)](https://github.com/KrishnaVaibhav/InkedIn/actions/workflows/ci.yml)
Local manga/comic colorizer. Everything runs on this machine — no cloud, no telemetry, no open ports beyond a token-guarded localhost UI.

## Quick start

```powershell
# one-time (already done in this workspace)
uv venv .venv --python 3.13
uv pip install -p .venv -e ".[dev]"
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv -e ".[ai]"   # optional: diffusion "ai" mode + WD14 tagger + translation
uv pip install -p .venv --no-deps easyocr   # multi-language OCR (ko/zh/ru/es/…); --no-deps keeps our opencv-contrib build

# launch the UI (opens browser, URL carries a per-session token)
.venv\Scripts\inkedin.exe ui

# or one-shot CLI
.venv\Scripts\inkedin.exe color chapter1.cbz -o chapter1_color.cbz --mode fast
.venv\Scripts\inkedin.exe color book.pdf -o out.pdf --pages 1,3,5-9 --mode fast+theme:sunset
.venv\Scripts\inkedin.exe doctor
```

## Features

- Input: PNG/JPG/JPEG/WEBP/BMP/TIFF/GIF images, folders, PDF, CBZ, CB7, CBT, CBR (needs unar/bsdtar/unrar ≥ 6.12; Windows 11 ships bsdtar), EPUB (spine-ordered) — type sniffed by magic bytes, never extension
- `--split-spreads` cuts double-page spreads in two; `--rtl` for right-to-left books
- Output: CBZ, PDF, image folder
- Bulk processing with per-page status; select only the pages you want
- Modes: `fast` (GAN model, auto-downloads verified weights), `ai` / `ai:<prompt>` (SD1.5 + ControlNet lineart — panel-wise, WD14 auto-tagged prompts, IP-Adapter reference via `--ref colored.png`, panel-to-panel color consistency, GPU only), `theme:<name>` (instant duotone: sepia, noir, sunset, ocean, forest, pastel), `fast+theme:<name>` (model + grade)
- **Self-reference**: already-colored pages in a book (covers, color inserts) are auto-detected (LAB chroma + hue-spread — uniform-tinted scans don't false-positive), skipped from colorizing, and used as a **light color bias** for the rest — the model's own semantic colors always win (gentle chroma anchor in `fast`/`theme`; in `ai` the color page only seeds panel 1, then panels follow each other as usual). 🎨 badge + toggle in UI, `--no-auto-ref` to disable
- Missing-spot repair: enclosed gray regions the model left uncolored get chroma inpainted from their colorful surroundings (gutters, margins and bubbles are excluded); dark-shading color survives (ink neutralization starts at L 65); detector bubble masks keep only the bubble's own white component, so no rectangular white halos
- Line/text preservation via chroma-only LAB recomposite (guided-filter edge snap, speech-bubble whitening); optional `--ml-text` RT-DETR bubble detector (Apache-2.0, downloads once); palette anchoring for cross-page consistency
- **Translation** (`--translate`, all local): Japanese via manga-ocr (Apache-2.0, best for manga); Korean/Chinese/Russian/Spanish/French/German/Italian/Portuguese via EasyOCR (`--lang ko|zh|ru|es|…`); M2M100 (MIT) translates everything to English. Original lettering is erased outline-safely (border-connected art/outlines are never touched) and the English is laid out in the **whole bubble interior** — word-wrapped, font-fitted down to 8 px with character-level breaking of long words, auto light-on-dark lettering — so it stays in the same visual area without overflowing. Bubbles whose text region the detector missed are OCR'd anyway (white-interior + glyph checks stop hallucinations). Colorize only, translate only (`--mode none --translate`), or both; `--translate-sfx` for free-floating SFX. Models download once into the workspace
- Web UI **reader**: 📖 button or double-click a page — full-screen reading view with ←/→ navigation; shows colorized pages as they finish (live-swaps in while the job is still running), original for pages not done yet
- Export: destination defaults to the folder the book was opened from (uploads default to `data/exports/`); **⬇ Download** button exports and saves through the browser
- Web UI: drag-drop files **or a whole folder** (a folder of images imports as one book, pages in natural order); dropping/picking multiple loose images asks "one book or separate?"; folder picker button; job list, theme swatches, reference-image upload, shift-click range selection, live progress, before/after compare slider, export presets
- **Tunable weights** (UI "⚙ Advanced" panel / CLI flags): reference influence `--ref-strength` (0 = model colors untouched), ai reference strength `--ip-scale`, panel consistency `--self-consistency`, diffusion `--steps`, plus toggles for missing-spot repair (`--no-fill-voids`) and bubble/text protection (`--no-protect-text`)
- GPU: CUDA auto-detected (fp16), CPU fallback; `--device` to force

## Storage

Everything stays inside this workspace: `.venv/`, `.uv-cache/`, `models/weights/`, `data/`.

## Layout

- `python/inkedin_core/` — pipeline, models, security controls, tests, CLI, UI server
- `models/manifests/` — pinned SHA-256 manifests for downloaded weights
- `Research.MD` — architecture, threat model, roadmap (Tauri desktop shell is Milestone 3+)

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

44 tests cover zip/tar/7z-slip, symlink and zip-bomb rejection, magic-byte sniffing (incl. tar@257, 7z, rar), dimension limits, ingest round-trips (image/CBZ/CB7/CBT/EPUB/PDF), EPUB spine ordering, spread splitting (LTR/RTL), colorize modes (incl. `none` passthrough), line preservation, bubble masks (heuristic + detector-box path), translation typesetting (erase/fit/wrap, ASCII skip, detector-missing error), palette matching, export formats, page selection, and job lifecycle.

## License & models

Code: **Apache-2.0** (see `LICENSE`).

No model weights live in this repository — everything downloads on demand
into the workspace (`models/`, gitignored) and keeps its upstream license.
Full table in [`models/manifests/NOTICE.md`](models/manifests/NOTICE.md).
The important one: `fast` mode's manga-colorization-v2 weights have **no
published license** — local use only, never redistributed by us.

User data (`data/` — imported books, jobs, exports, the job DB) and all
caches (`.venv/`, `.uv-cache/`, `models/weights|hf|easyocr/`) are gitignored:
the repo is code + pinned manifests only (~260 KB).

## Setting up from a clean clone

```powershell
git clone <repo> InkedIn && cd InkedIn
uv venv .venv --python 3.13
uv pip install -p .venv -e ".[dev]"                # core: theme modes + UI + all import formats
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or cpu wheels
uv pip install -p .venv -e ".[ai]"                 # ai mode, tagging, translation
uv pip install -p .venv --no-deps easyocr          # multi-language OCR
.venv\Scripts\python.exe -m pytest -q              # should be all green
.venv\Scripts\inkedin.exe ui
```
