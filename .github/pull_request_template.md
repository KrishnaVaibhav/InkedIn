## What & why

<!-- one or two sentences -->

## Checklist

- [ ] `pytest -q` passes locally (CI runs it on Linux/Windows/macOS, py3.12 + 3.13)
- [ ] `uvx ruff check python/` is clean
- [ ] New behavior has tests (security-relevant code — ingest, archives, validation — **must** have tests)
- [ ] No model weights, user data, books, or caches committed (`git status` shows code/docs only)
- [ ] New models or dependencies are listed in `models/manifests/NOTICE.md` with their license
- [ ] Everything the app writes stays inside the workspace (no `%APPDATA%`, no `/tmp`, no home-dir caches)
- [ ] No telemetry, no open ports beyond the token-guarded localhost UI, no cloud calls except explicit model downloads
