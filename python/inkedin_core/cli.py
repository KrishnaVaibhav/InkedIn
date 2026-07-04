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

    col = sub.add_parser("color", help="colorize a file/folder/PDF/CBZ/CB7/CBT/CBR/EPUB")
    col.add_argument("input", type=Path)
    col.add_argument("-o", "--output", type=Path, required=True)
    col.add_argument("--format", choices=["folder", "pdf", "cbz"], default=None)
    col.add_argument("--mode", default="fast", help="fast | ai | ai:<prompt> | theme:<name> | fast+theme:<name> | none (with --translate)")
    col.add_argument("--pages", default=None, help="e.g. 1,3,5-9 (1-based); default all")
    col.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    col.add_argument("--ink", type=float, default=0.85, help="line preservation weight 0..1")
    col.add_argument("--anchor", type=int, default=None, help="1-based palette anchor page")
    col.add_argument("--ref", type=Path, default=None, help="colored reference image (ai mode: steers palette via IP-Adapter)")
    col.add_argument("--split-spreads", action="store_true", help="cut double-page spreads into two pages")
    col.add_argument("--rtl", action="store_true", help="right-to-left reading order (manga spread split)")
    col.add_argument("--ml-text", action="store_true", help="AI bubble/text protection (downloads RT-DETR model on first use)")
    col.add_argument("--translate", action="store_true",
                     help="OCR Japanese text and typeset English in place (local models, ~2.3 GB one-time download); combine with --mode none for translate-only")
    col.add_argument("--translate-sfx", action="store_true", help="also translate free-floating SFX text (inpainted over art)")
    col.add_argument("--lang", default="ja", choices=["ja", "ko", "zh", "ru", "es", "fr", "de", "it", "pt"],
                     help="source language for --translate (ja uses manga-ocr; others use EasyOCR)")
    col.add_argument("--no-auto-ref", action="store_true",
                     help="disable self-reference: by default detected color pages are skipped and used as the color reference for the rest")
    tune = col.add_argument_group("tuning weights")
    tune.add_argument("--ref-strength", type=float, default=0.15,
                      help="how strongly detected color pages bias the palette, 0..1 (default 0.15; 0 disables the bias but still skips color pages)")
    tune.add_argument("--ip-scale", type=float, default=0.65,
                      help="ai: IP-Adapter strength of an explicit --ref image, 0..1 (default 0.65)")
    tune.add_argument("--self-consistency", type=float, default=0.4,
                      help="ai: how strongly panels follow panel 1's colors, 0..1 (default 0.4)")
    tune.add_argument("--steps", type=int, default=24, help="ai: diffusion steps (default 24; more = slower, finer)")
    tune.add_argument("--no-fill-voids", action="store_true", help="disable missing-spot chroma repair")
    tune.add_argument("--no-protect-text", action="store_true", help="disable bubble/lettering protection")

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
    job = jm.create(args.input, split_spreads=args.split_spreads, rtl=args.rtl)
    print(f"[inkedin] {len(job.pages)} page(s), job {job.id}")

    anchor = args.anchor - 1 if args.anchor else None
    res = jm.run(
        job.id,
        selected=_parse_pages(args.pages),
        mode=args.mode,
        device=args.device,
        ink_weight=args.ink,
        anchor_page=anchor,
        ref_image=args.ref,
        ml_text=args.ml_text,
        translate=args.translate,
        translate_sfx=args.translate_sfx,
        translate_lang=args.lang,
        auto_ref=not args.no_auto_ref,
        ref_strength=max(0.0, min(1.0, args.ref_strength)),
        ip_scale=max(0.0, min(1.0, args.ip_scale)),
        self_ref_scale=max(0.0, min(1.0, args.self_consistency)),
        steps=max(4, min(60, args.steps)),
        fill_voids=not args.no_fill_voids,
        protect_text=not args.no_protect_text,
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

    from .security.archives import rar_backend_available

    rar = "available" if rar_backend_available() else "MISSING (install unar/bsdtar/unrar >= 6.12 for CBR)"
    print(f"formats   : image/folder/PDF/CBZ/CB7/CBT/EPUB built-in; CBR backend {rar}")
    try:
        from .models import bubbles

        print(f"ml-text   : {'cached' if bubbles.is_cached() else 'not downloaded (run with --ml-text to fetch)'}")
    except Exception:
        print("ml-text   : unavailable (needs [ai] extra: transformers)")
    try:
        from .models import translator

        print(f"translate : {'models cached' if translator.is_cached() else 'not downloaded (first --translate run fetches ~2.3 GB)'}")
    except Exception:
        print("translate : unavailable (needs [ai] extra: transformers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
