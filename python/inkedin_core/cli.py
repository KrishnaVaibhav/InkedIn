"""CLI entrypoints.

  inkedin ui                          launch the local web UI (127.0.0.1 + token)
  inkedin color INPUT -o OUT [...]    one-shot bulk colorize
  inkedin doctor                      device / model / env report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inkedin", description="InkedIn manga/comic colorizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ui = sub.add_parser("ui", help="launch local web UI")
    ui.add_argument("--port", type=int, default=8317)

    col = sub.add_parser("color", help="colorize a file/folder/PDF/CBZ")
    col.add_argument("input", type=Path)
    col.add_argument("-o", "--output", type=Path, required=True)
    col.add_argument("--format", choices=["folder", "pdf", "cbz"], default=None)
    col.add_argument("--mode", default="fast", help="fast | theme:<name> | fast+theme:<name>")
    col.add_argument("--pages", default=None, help="e.g. 1,3,5-9 (1-based); default all")
    col.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    col.add_argument("--ink", type=float, default=0.85, help="line preservation weight 0..1")
    col.add_argument("--anchor", type=int, default=None, help="1-based palette anchor page")

    sub.add_parser("doctor", help="environment report")

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "ui":
        from .server import serve

        return serve(port=args.port)
    if args.cmd == "color":
        return _color(args)
    return 2


def _parse_pages(spec: str | None) -> list[int] | None:
    """1-based '1,3,5-9' -> 0-based indices."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a) - 1, int(b)))
        else:
            out.add(int(part) - 1)
    return sorted(out)


def _color(args) -> int:
    from .jobs import JobManager

    fmt = args.format
    if fmt is None:
        suffix = args.output.suffix.lower()
        fmt = {"pdf": "pdf", "cbz": "cbz"}.get(suffix.lstrip("."), "folder")

    jm = JobManager()
    print(f"[inkedin] ingesting {args.input} ...")
    job = jm.create(args.input)
    print(f"[inkedin] {len(job.pages)} page(s), job {job.id}")

    anchor = args.anchor - 1 if args.anchor else None
    res = jm.run(
        job.id,
        selected=_parse_pages(args.pages),
        mode=args.mode,
        device=args.device,
        ink_weight=args.ink,
        anchor_page=anchor,
    )
    print(f"[inkedin] colorize: {res}")
    if res["status"].startswith("done"):
        out = jm.export(job.id, fmt, args.output)
        print(f"[inkedin] exported -> {out}")
    jm.delete(job.id)
    return 0 if res.get("errors", 0) == 0 else 1


def _doctor() -> int:
    from . import __version__
    from .models.device import probe
    from .workspace import models_dir, workspace_root

    print(f"InkedIn {__version__}")
    print(f"workspace : {workspace_root()}")
    print(f"python    : {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch     : {torch.__version__} (cuda available: {torch.cuda.is_available()})")
    except ImportError:
        print("torch     : NOT INSTALLED (theme modes only)")
    d = probe()
    vram = f", {d.vram_total_mb} MB VRAM" if d.vram_total_mb else ""
    print(f"device    : {d.device} ({d.name}{vram})")
    w = models_dir() / "manga-colorization-v2" / "generator.zip"
    print(f"weights   : {'present' if w.exists() else 'not downloaded (auto-downloads on first fast run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
