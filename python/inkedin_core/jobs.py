"""Job orchestration: SQLite state + per-job temp workspace + page loop.

One page in memory at a time. Cancellation checked between pages. Failed pages
mark error state and the job continues (partial export allowed).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import colorize as colorize_mod
from .pipeline import export as export_mod
from .pipeline import ingest
from .workspace import db_path, ensure_dirs, jobs_dir

THUMB_LONG_EDGE = 320

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created_at REAL, status TEXT, input TEXT,
  mode TEXT, device TEXT, selected TEXT, export_fmt TEXT, export_path TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS pages (
  job_id TEXT, page INTEGER, status TEXT, out_path TEXT, error TEXT,
  PRIMARY KEY (job_id, page)
);
"""


def _db() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.executescript(_SCHEMA)
    for ddl in (  # additive migrations for pre-existing databases
        "ALTER TABLE pages ADD COLUMN is_color INTEGER DEFAULT 0",
        "ALTER TABLE pages ADD COLUMN color_score REAL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


@dataclass
class Job:
    id: str
    dir: Path
    pages: list[ingest.PageRef]


class JobManager:
    def __init__(self):
        self.conn = _db()
        self.lock = threading.Lock()
        self.cancel_flags: dict[str, threading.Event] = {}
        self.progress: dict[str, dict] = {}
        # sequential run queue: one book on the GPU at a time; further Run
        # clicks line up instead of competing for VRAM
        self._queue: list[tuple[str, dict]] = []
        self._queue_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._running_job: str | None = None

    # -- queued execution ----------------------------------------------------

    def enqueue_run(self, job_id: str, **kwargs) -> dict:
        """Queue a run; a single worker drains jobs FIFO. Returns queue info."""
        with self._queue_lock:
            if self._running_job == job_id or any(j == job_id for j, _ in self._queue):
                raise ValueError("job already running or queued")
            position = len(self._queue) + (1 if self._running_job else 0)
            self._queue.append((job_id, kwargs))
            self.progress[job_id] = {"done": 0, "total": 0, "current": None, "stage": "queued"}
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain_queue, daemon=True)
                self._worker.start()
        self._set_job(job_id, status="queued")
        return {"queued": True, "position": position}

    def _drain_queue(self) -> None:
        while True:
            with self._queue_lock:
                if not self._queue:
                    self._worker = None
                    return
                job_id, kwargs = self._queue.pop(0)
                self._running_job = job_id
            try:
                self.run(job_id, **kwargs)
            except Exception as e:  # keep draining; surface the error on the job
                self._set_job(job_id, status="error", error=str(e)[:300])
                self.progress[job_id] = {
                    "done": 1, "total": 1, "current": None,
                    "stage": "error", "message": str(e)[:200],
                }
            finally:
                with self._queue_lock:
                    self._running_job = None

    # -- lifecycle ---------------------------------------------------------

    def create(
        self,
        input_path: Path,
        split_spreads: bool = False,
        rtl: bool = False,
        display_name: str | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        jdir = jobs_dir() / job_id
        jdir.mkdir(parents=True)
        pages = ingest.ingest(input_path, jdir, split_spreads=split_spreads, rtl=rtl)
        color_flags = self._make_thumbs(jdir, pages)
        with self.lock:
            self.conn.execute(
                "INSERT INTO jobs (id, created_at, status, input, mode, device, selected, export_fmt) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, time.time(), "ready", display_name or str(input_path), "", "", "[]", ""),
            )
            self.conn.executemany(
                "INSERT INTO pages (job_id, page, status, is_color, color_score) VALUES (?,?,?,?,?)",
                [
                    (job_id, p.index, "pending", int(color_flags[p.index][0]), color_flags[p.index][1])
                    for p in pages
                ],
            )
            self.conn.commit()
        return Job(job_id, jdir, pages)

    def _make_thumbs(self, jdir: Path, pages: list[ingest.PageRef]) -> dict[int, tuple[bool, float]]:
        """Thumbnail pass; the decode is reused to detect already-colored pages."""
        from .pipeline import colorstats

        tdir = jdir / "thumbs"
        tdir.mkdir(exist_ok=True)
        flags: dict[int, tuple[bool, float]] = {}
        for p in pages:
            with Image.open(p.source_path) as im:
                flags[p.index] = colorstats.is_color_page(np.array(im.convert("RGB")))
                im.thumbnail((THUMB_LONG_EDGE, THUMB_LONG_EDGE))
                im.save(tdir / f"t_{p.index:05d}.jpg", quality=80)
        return flags

    def job_dir(self, job_id: str) -> Path:
        d = jobs_dir() / job_id
        if not d.is_dir():
            raise KeyError(f"unknown job {job_id}")
        return d

    def list_jobs(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT j.id, j.created_at, j.status, j.input, COUNT(p.page) "
            "FROM jobs j LEFT JOIN pages p ON p.job_id = j.id "
            "GROUP BY j.id ORDER BY j.created_at DESC"
        )
        out = []
        for jid, created, status, inp, n in cur.fetchall():
            if not (jobs_dir() / jid).is_dir():
                continue  # row without workspace (cleaned externally)
            name = Path(inp).name if inp else jid
            parent = str(Path(inp).parent) if inp and Path(inp).is_absolute() else ""
            out.append(
                {"job": jid, "created": created, "status": status, "name": name,
                 "pages": n, "dir": parent}
            )
        return out

    def list_pages(self, job_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT page, status, out_path, error, is_color FROM pages WHERE job_id=? ORDER BY page",
            (job_id,),
        )
        return [
            {"page": r[0], "status": r[1], "out": r[2], "error": r[3], "is_color": bool(r[4])}
            for r in cur.fetchall()
        ]

    def color_pages(self, job_id: str) -> list[int]:
        cur = self.conn.execute(
            "SELECT page FROM pages WHERE job_id=? AND is_color=1 ORDER BY page", (job_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def cancel(self, job_id: str) -> None:
        with self._queue_lock:  # still waiting in line: just drop it
            before = len(self._queue)
            self._queue = [(j, k) for j, k in self._queue if j != job_id]
            dequeued = len(self._queue) != before
        if dequeued:
            self.progress.pop(job_id, None)
            self._set_job(job_id, status="ready")
            return
        if job_id in self.cancel_flags:
            self.cancel_flags[job_id].set()

    def delete(self, job_id: str) -> None:
        self.cancel(job_id)
        shutil.rmtree(jobs_dir() / job_id, ignore_errors=True)
        with self.lock:
            self.conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self.conn.execute("DELETE FROM pages WHERE job_id=?", (job_id,))
            self.conn.commit()

    def cleanup_stale(self) -> int:
        """Startup cleanup: remove job dirs with no DB row and reset stuck jobs."""
        known = {r[0] for r in self.conn.execute("SELECT id FROM jobs")}
        removed = 0
        for d in jobs_dir().iterdir():
            if d.is_dir() and d.name not in known:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        with self.lock:
            self.conn.execute("UPDATE jobs SET status='ready' WHERE status IN ('running','queued')")
            self.conn.commit()
        return removed

    # -- processing --------------------------------------------------------

    def run(
        self,
        job_id: str,
        selected: list[int] | None = None,
        mode: str = "fast",
        device: str = "auto",
        ink_weight: float = 0.85,
        anchor_page: int | None = None,
        ref_image: Path | None = None,
        ml_text: bool = False,
        translate: bool = False,
        translate_sfx: bool = False,
        translate_lang: str = "ja",
        auto_ref: bool = True,
        ref_strength: float = 0.15,  # auto color-page bias (0 = off)
        ip_scale: float = 0.65,  # ai: user reference IP-Adapter strength
        self_ref_scale: float = 0.4,  # ai: panels follow panel 1
        steps: int = 24,  # ai: diffusion steps
        fill_voids: bool = True,  # inpaint enclosed gray "missing spots"
        protect_text: bool = True,  # keep bubbles/lettering neutral
        page_consistency: float = 0.25,  # cross-page: later pages follow the
        #   first colorized page so characters keep hair/outfit colors (0 = off)
    ) -> dict:
        """Colorize and/or translate selected pages (all if None). Blocking;
        call from a worker thread. mode 'none' + translate = translation only."""
        if mode == "none" and not translate:
            raise ValueError("mode 'none' does nothing unless translate is enabled")
        if ml_text or translate:  # detector needed: fetch RT-DETR weights
            from .models import bubbles

            bubbles.ensure_downloaded()
        jdir = self.job_dir(job_id)
        pages_dir = jdir / "pages"
        out_dir = jdir / "colored"
        out_dir.mkdir(exist_ok=True)

        all_pages = sorted(int(p.stem.split("_")[1]) for p in pages_dir.glob("page_*.png"))
        color_set = set(self.color_pages(job_id))
        targets = [p for p in all_pages if selected is None or p in set(selected)]
        if selected is None and auto_ref and color_set and mode != "none":
            # already-colored pages are references, not work
            targets = [p for p in targets if p not in color_set]
            if not targets:
                raise ValueError("every page is already colored — nothing to colorize")
        if not targets:
            raise ValueError("no pages selected")

        # Self-reference: detected color pages steer the palette of the rest.
        refs: list[tuple[np.ndarray, np.ndarray]] = []
        if auto_ref and color_set and mode != "none":
            from .pipeline import colorstats

            for cp in sorted(color_set):
                f = pages_dir / f"page_{cp:05d}.png"
                if f.exists():
                    rgb = np.array(Image.open(f).convert("RGB"))
                    refs.append((rgb, colorstats.lab_moments(rgb)))

        cancel = threading.Event()
        self.cancel_flags[job_id] = cancel
        self._set_job(job_id, status="running", mode=mode, selected=json.dumps(targets))
        # stage lets the UI distinguish "downloading/loading models" (first run
        # can pull gigabytes) from actual page processing
        self.progress[job_id] = {"done": 0, "total": len(targets), "current": None, "stage": "loading"}

        colorizer, theme_overlay = colorize_mod.build_colorizer(mode, device)
        translator = None
        if translate:
            from .models.translator import MangaTranslator

            translator = MangaTranslator(src_lang=translate_lang)
        self.progress[job_id]["stage"] = "processing"
        ref_rgb = np.array(Image.open(ref_image).convert("RGB")) if ref_image else None
        anchor_rgb = None
        rolling_anchor = None  # first successfully colorized page of THIS run:
        #   the stable color identity every later page is nudged toward
        model_mode = not mode.startswith("theme:") and mode != "none"
        errors = 0
        try:
            for page in targets:
                if cancel.is_set():
                    self._set_job(job_id, status="cancelled")
                    return {"status": "cancelled", "done": self.progress[job_id]["done"]}
                self.progress[job_id]["current"] = page
                try:
                    src = np.array(Image.open(pages_dir / f"page_{page:05d}.png").convert("RGB"))
                    auto_anchor = None
                    auto_ipref = None
                    if refs:
                        from .pipeline import colorstats

                        nearest = colorstats.nearest_reference(src, refs)
                        if ref_rgb is None:
                            auto_ipref = nearest  # ai mode: gentle panel-1 seed
                        if anchor_rgb is None:
                            auto_anchor = nearest  # fast/theme: chroma anchor
                    extra = {"steps": steps, "ip_scale": ip_scale, "self_ref_scale": self_ref_scale}
                    if ref_rgb is not None:
                        extra["ref_rgb"] = ref_rgb  # explicit user ref: ip_scale strength
                    elif auto_ipref is not None and ref_strength > 0:
                        # NOT ref_rgb: a color insert must nudge, never override the
                        # model's own semantics on every panel (that regressed quality)
                        extra["auto_ref_rgb"] = auto_ipref
                        extra["auto_ref_scale"] = min(1.0, ref_strength * 2)
                    elif rolling_anchor is not None and page_consistency > 0:
                        # cross-page character consistency: later pages seed from
                        # the first colorized page instead of drifting freely
                        extra["auto_ref_rgb"] = rolling_anchor
                        extra["auto_ref_scale"] = min(1.0, page_consistency * 1.4)

                    # anchor priority: user's anchor page > detected color page >
                    # this run's first colorized page (cross-page consistency)
                    eff_anchor = anchor_rgb
                    eff_strength, chroma_only = 0.4, False
                    if eff_anchor is None and auto_anchor is not None and ref_strength > 0:
                        eff_anchor, eff_strength, chroma_only = auto_anchor, ref_strength, True
                    elif eff_anchor is None and rolling_anchor is not None and page_consistency > 0:
                        eff_anchor, eff_strength, chroma_only = rolling_anchor, page_consistency, True
                    req = colorize_mod.ColorizeRequest(
                        page_rgb=src, mode=mode, ink_weight=ink_weight,
                        protect_text=protect_text,
                        anchor_rgb=eff_anchor,
                        anchor_strength=eff_strength,
                        anchor_chroma_only=chroma_only,
                        fill_voids=fill_voids and model_mode,
                        extra=extra,
                    )
                    out = colorize_mod.run_page(colorizer, req)
                    if theme_overlay:
                        themed = colorize_mod.ThemeColorizer(theme_overlay)
                        out = themed.colorize(colorize_mod.ColorizeRequest(page_rgb=out, ink_weight=0.0))
                    if translator is not None:
                        from .models import bubbles
                        from .pipeline import textlayer

                        out = textlayer.translate_page(
                            src, out, bubbles.detect_all,
                            translator.ocr, translator.translate,
                            include_sfx=translate_sfx, src_lang=translate_lang,
                        )
                    if anchor_page is not None and page == anchor_page:
                        anchor_rgb = out.copy()
                    if rolling_anchor is None and model_mode:
                        rolling_anchor = out.copy()  # first page done = book's color identity
                    dest = out_dir / f"page_{page:05d}.png"
                    Image.fromarray(out).save(dest)
                    self._set_page(job_id, page, "done", str(dest))
                except Exception as e:  # keep going; page marked failed
                    errors += 1
                    self._set_page(job_id, page, "error", None, str(e)[:500])
                self.progress[job_id]["done"] += 1
        finally:
            colorizer.close()
            if translator is not None:
                translator.close()
            self.cancel_flags.pop(job_id, None)

        status = "done" if errors == 0 else "done_with_errors"
        self._set_job(job_id, status=status)
        return {"status": status, "done": len(targets) - errors, "errors": errors}

    def export(self, job_id: str, fmt: str, dest: Path) -> Path:
        jdir = self.job_dir(job_id)
        colored = sorted((jdir / "colored").glob("page_*.png"))
        if not colored:
            raise ValueError("no colorized pages to export")
        dest = dest.resolve()
        if jobs_dir() in dest.parents:
            raise ValueError("export target must be outside job temp")
        result = export_mod.export(colored, dest, fmt)
        self._set_job(job_id, export_fmt=fmt, export_path=str(result))
        return result

    # -- helpers -----------------------------------------------------------

    def _set_job(self, job_id: str, **cols) -> None:
        sets = ", ".join(f"{k}=?" for k in cols)
        with self.lock:
            self.conn.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*cols.values(), job_id))
            self.conn.commit()

    def _set_page(self, job_id: str, page: int, status: str, out_path: str | None = None, error: str | None = None) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE pages SET status=?, out_path=?, error=? WHERE job_id=? AND page=?",
                (status, out_path, error, job_id, page),
            )
            self.conn.commit()
