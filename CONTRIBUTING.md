# Contributing to InkedIn

Thanks for helping! InkedIn is a local-first manga/comic colorizer & translator:
**no cloud, no telemetry, everything stays on the user's machine.** PRs that
violate that principle won't be merged, no matter how cool.

## Dev setup

```bash
git clone <repo> && cd InkedIn
uv venv .venv --python 3.13
uv pip install -p .venv -e ".[dev]"          # enough to run the full test suite
# optional, for ai/translate features:
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -e ".[ai]"
uv pip install -p .venv --no-deps easyocr    # --no-deps: keeps opencv-contrib intact
```

## Before you open a PR

```bash
.venv/bin/python -m pytest -q      # Windows: .venv\Scripts\python.exe -m pytest -q
uvx ruff check python/
```

CI runs the suite on Linux + Windows + macOS × Python 3.12/3.13, lints, does a
full AI-stack import check, reviews new dependencies, and **builds + smoke-tests
the Windows executable** (PyInstaller one-file of the torch-free core — if your
change breaks packaging, the PR goes red). All jobs must be green to merge.

A scheduled weekly workflow additionally runs `pip-audit` (CVEs), a
deprecation-strict test pass, and an outdated-packages report.

Build the exe locally:

```bash
uv pip install -p .venv -e ".[build]"
.venv/Scripts/pyinstaller.exe --noconfirm --onefile --name inkedin \
  --collect-submodules uvicorn --workpath build/_work --specpath build \
  --distpath dist build/pyinstaller_entry.py
dist/inkedin.exe doctor
```

## Ground rules

1. **Inputs are hostile.** Anything parsed from user files (archives, PDFs,
   images, EPUB metadata) goes through `security/` limits and validators.
   New parsers need adversarial tests (traversal, bombs, fake magic bytes).
2. **The test suite must run without torch or network.** Model-dependent code
   is imported lazily and degrades gracefully; tests fake OCR/MT/detector
   callables instead of loading models.
3. **No weights in git.** Models download on demand, pinned by manifest
   (SHA-256 or repo+revision), each listed in `models/manifests/NOTICE.md`
   with its license. No `trust_remote_code`, prefer safetensors/ONNX.
4. **Workspace-local storage only.** Everything the app writes goes under the
   repo workspace (`data/`, `models/`), never to system/user directories.
5. **Line art is sacred.** Color operations work on chroma; lightness comes
   from the source page (see `pipeline/recomposite.py`). Don't break this.

Architecture, threat model, and roadmap live in `Research.MD`.
