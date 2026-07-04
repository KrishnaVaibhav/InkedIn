# InkedIn

Local manga/comic colorizer. Everything runs on this machine — no cloud, no telemetry, no open ports beyond a token-guarded localhost UI.

## Quick start

```powershell
# one-time (already done in this workspace)
uv venv .venv --python 3.13
uv pip install -p .venv -e ".[dev]"
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128

# launch the UI (opens browser, URL carries a per-session token)
.venv\Scripts\inkedin.exe ui

# or one-shot CLI
.venv\Scripts\inkedin.exe color chapter1.cbz -o chapter1_color.cbz --mode fast
.venv\Scripts\inkedin.exe color book.pdf -o out.pdf --pages 1,3,5-9 --mode fast+theme:sunset
.venv\Scripts\inkedin.exe doctor
```

## Features

- Input: PNG/JPG/JPEG/WEBP/BMP/TIFF/GIF images, folders, PDF, CBZ (type sniffed by magic bytes, never extension)
- Output: CBZ, PDF, image folder
- Bulk processing with per-page status; select only the pages you want
- Modes: `fast` (GAN model, auto-downloads verified weights), `theme:<name>` (instant duotone: sepia, noir, sunset, ocean, forest, pastel), `fast+theme:<name>` (model + grade)
- Line/text preservation via LAB recomposite; palette anchoring for cross-page consistency
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

Covers zip-slip/symlink/zip-bomb rejection, magic-byte sniffing, dimension limits, ingest round-trips (image/CBZ/PDF), colorize modes, line preservation, palette matching, export formats, page selection, and job lifecycle.

## Model licensing note

`fast` mode uses manga-colorization-v2 weights (research repo, no license file). Local use only; do not redistribute the weights. See `models/manifests/`.
