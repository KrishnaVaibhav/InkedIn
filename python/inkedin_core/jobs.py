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

    # -- lifecycle ---------------------------------------------------------

    def create(self, input_path: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        jdir = jobs_dir() / job_id
        jdir.mkdir(parents=True)
        pages = ingest.ingest(input_path, jdir)
        with self.lock:
            self.conn.execute(
                "INSERT INTO jobs (id, created_at, status, input, mode, device, selected, export_fmt) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job_id, time.time(), "ready", str(input_path), "", "", "[]", ""),
            )
            self.conn.executemany(
                "INSERT INTO pages (job_id, page, status) VALUES (?,?,?)",
                [(job_id, p.index, "pending") for p in pages],
            )
            self.conn.commit()
        self._make_thumbs(jdir, pages)
        return Job(job_id, jdir, pages)

    def _make_thumbs(self, jdir: Path, pages: list[ingest.PageRef]) -> None:
        tdir = jdir / "thumbs"
        tdir.mkdir(exist_ok=True)
        for p in pages:
            with Image.open(p.source_path) as im:
                im.thumbnail((THUMB_LONG_EDGE, THUMB_LONG_EDGE))
                im.save(tdir / f"t_{p.index:05d}.jpg", quality=80)

    def job_dir(self, job_id: str) -> Path:
        d = jobs_dir() / job_id
        if not d.is_dir():
            raise KeyError(f"unknown job {job_id}")
        return d

    def list_pages(self, job_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT page, status, out_path, error FROM pages WHERE job_id=? ORDER BY page", (job_id,)
        )
        return [{"page": r[0], "status": r[1], "out": r[2], "error": r[3]} for r in cur.fetchall()]

    def cancel(self, job_id: str) -> None:
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
            self.conn.execute("UPDATE jobs SET status='ready' WHERE status='running'")
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
    ) -> dict:
        """Colorize selected pages (all if None). Blocking; call from a worker thread."""
        jdir = self.job_dir(job_id)
        pages_dir = jdir / "pages"
        out_dir = jdir / "colored"
        out_dir.mkdir(exist_ok=True)

        all_pages = sorted(int(p.stem.split("_")[1]) for p in pages_dir.glob("page_*.png"))
        targets = [p for p in all_pages if selected is None or p in set(selected)]
        if not targets:
            raise ValueError("no pages selected")

        cancel = threading.Event()
        self.cancel_flags[job_id] = cancel
        self._set_job(job_id, status="running", mode=mode, selected=json.dumps(targets))
        self.progress[job_id] = {"done": 0, "total": len(targets), "current": None}

        colorizer, theme_overlay = colorize_mod.build_colorizer(mode, device)
        anchor_rgb = None
        errors = 0
        try:
            for page in targets:
                if cancel.is_set():
                    self._set_job(job_id, status="cancelled")
                    return {"status": "cancelled", "done": self.progress[job_id]["done"]}
                self.progress[job_id]["current"] = page
                try:
                    src = np.array(Image.open(pages_dir / f"page_{page:05d}.png").convert("RGB"))
                    req = colorize_mod.ColorizeRequest(
                        page_rgb=src, mode=mode, ink_weight=ink_weight, anchor_rgb=anchor_rgb
                    )
                    out = colorize_mod.run_page(colorizer, req)
                    if theme_overlay:
                        themed = colorize_mod.ThemeColorizer(theme_overlay)
                        out = themed.colorize(colorize_mod.ColorizeRequest(page_rgb=out, ink_weight=0.0))
                    if anchor_page is not None and page == anchor_page:
                        anchor_rgb = out.copy()
                    dest = out_dir / f"page_{page:05d}.png"
                    Image.fromarray(out).save(dest)
                    self._set_page(job_id, page, "done", str(dest))
                except Exception as e:  # keep going; page marked failed
                    errors += 1
                    self._set_page(job_id, page, "error", None, str(e)[:500])
                self.progress[job_id]["done"] += 1
        finally:
            colorizer.close()
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
