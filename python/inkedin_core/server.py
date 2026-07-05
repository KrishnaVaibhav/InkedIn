"""Local web UI. Security posture per Research.MD Gap 10:
- binds 127.0.0.1 only
- random per-session token required on every API call
- strict CSP, no external origins
- page/job identifiers validated; files served only from job dirs by integer index
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .jobs import JobManager
from .pipeline import validate
from .pipeline.colorize import THEMES
from .workspace import ensure_dirs, exports_dir, jobs_dir

TOKEN = secrets.token_urlsafe(24)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
jm = JobManager()


def _auth(request: Request) -> None:
    tok = request.headers.get("x-inkedin-token") or request.query_params.get("token")
    if not secrets.compare_digest(tok or "", TOKEN):
        raise HTTPException(401, "bad token")


@app.middleware("http")
async def _csp(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
        "script-src 'unsafe-inline'; connect-src 'self'"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


class CreateJob(BaseModel):
    path: str
    split_spreads: bool = False
    rtl: bool = False


class RunJob(BaseModel):
    pages: list[int] | None = None
    mode: str = "fast"
    device: str = "auto"
    ink_weight: float = 0.85
    anchor_page: int | None = None
    use_ref: bool = False
    ml_text: bool = False
    translate: bool = False
    translate_sfx: bool = False
    translate_lang: str = "ja"
    auto_ref: bool = True
    ref_strength: float = 0.15
    ip_scale: float = 0.65
    self_ref_scale: float = 0.4
    page_consistency: float = 0.25
    steps: int = 24
    fill_voids: bool = True
    protect_text: bool = True


class ExportJob(BaseModel):
    fmt: str
    dest: str


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/api/meta", dependencies=[Depends(_auth)])
def meta():
    from . import __version__
    from .models.device import probe
    from .security.archives import rar_backend_available

    d = probe()
    try:
        from .models import bubbles

        ml_text = "cached" if bubbles.is_cached() else "downloadable"
    except Exception:
        ml_text = "unavailable"
    try:
        from .models import translator

        translate = "cached" if translator.is_cached() else "downloadable"
    except Exception:
        translate = "unavailable"
    return {
        "version": __version__,
        "themes": sorted(THEMES),
        "device": d.__dict__,
        "rar_ok": rar_backend_available(),
        "ml_text": ml_text,
        "translate": translate,
        "export_dir": str(exports_dir()),
    }


@app.get("/api/jobs", dependencies=[Depends(_auth)])
def list_jobs():
    return {"jobs": jm.list_jobs()}


@app.post("/api/jobs", dependencies=[Depends(_auth)])
def create_job(body: CreateJob):
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(400, "path does not exist")
    try:
        job = jm.create(p, split_spreads=body.split_spreads, rtl=body.rtl)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"job": job.id, "pages": len(job.pages)}


@app.post("/api/jobs/upload", dependencies=[Depends(_auth)])
async def upload_job(file: UploadFile, split_spreads: bool = False, rtl: bool = False):
    ensure_dirs()
    staging = jobs_dir() / "_uploads"
    staging.mkdir(parents=True, exist_ok=True)
    # app-generated name; original filename only trusted for its extension hint
    ext = Path(file.filename or "x").suffix.lower()[:8]
    dest = staging / f"up_{secrets.token_hex(8)}{ext}"
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    display = Path(file.filename or "upload").name  # basename only, display use
    try:
        job = jm.create(dest, split_spreads=split_spreads, rtl=rtl, display_name=display)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        dest.unlink(missing_ok=True)
    return {"job": job.id, "pages": len(job.pages), "name": display}


@app.post("/api/jobs/upload-batch", dependencies=[Depends(_auth)])
async def upload_batch(
    files: list[UploadFile],
    split_spreads: bool = False,
    rtl: bool = False,
    name: str = "",
):
    """Many loose images become ONE book. The client sends files already in
    reading order; index-prefixed staging names preserve that order through
    the folder ingest's natural sort."""
    import shutil

    if not files:
        raise HTTPException(400, "no files")
    ensure_dirs()
    staging = jobs_dir() / "_uploads" / f"batch_{secrets.token_hex(8)}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for i, f in enumerate(files):
            ext = Path(f.filename or "x").suffix.lower()[:8]
            with open(staging / f"{i:05d}{ext}", "wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
        display = Path(name).name[:80] if name.strip() else f"book ({len(files)} pages)"
        try:
            job = jm.create(staging, split_spreads=split_spreads, rtl=rtl, display_name=display)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"job": job.id, "pages": len(job.pages), "name": display}


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) > 16:
        raise HTTPException(400, "bad job id")
    try:
        return jm.job_dir(job_id)
    except KeyError:
        raise HTTPException(404, "unknown job") from None


@app.post("/api/jobs/{job_id}/ref", dependencies=[Depends(_auth)])
async def upload_ref(job_id: str, file: UploadFile):
    """Colored reference image for ai mode (IP-Adapter palette steering)."""
    jdir = _job_dir(job_id)
    tmp = jdir / "ref_upload.tmp"
    with open(tmp, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    try:
        kind = validate.sniff_type(tmp)
        if kind in {"pdf", "zip", "7z", "rar", "tar"}:
            raise ValueError("reference must be a single image")
        img = validate.decode_image(tmp)
        img.save(jdir / "ref.png", format="PNG")
        img.close()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/ref", dependencies=[Depends(_auth)])
def get_ref(job_id: str):
    f = _job_dir(job_id) / "ref.png"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


@app.delete("/api/jobs/{job_id}/ref", dependencies=[Depends(_auth)])
def delete_ref(job_id: str):
    (_job_dir(job_id) / "ref.png").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/pages", dependencies=[Depends(_auth)])
def pages(job_id: str):
    _job_dir(job_id)
    return {"pages": jm.list_pages(job_id), "progress": jm.progress.get(job_id)}


@app.get("/api/jobs/{job_id}/thumb/{page}", dependencies=[Depends(_auth)])
def thumb(job_id: str, page: int):
    f = _job_dir(job_id) / "thumbs" / f"t_{page:05d}.jpg"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/image/{kind}/{page}", dependencies=[Depends(_auth)])
def image(job_id: str, kind: str, page: int):
    sub = {"src": "pages", "out": "colored"}.get(kind)
    if not sub:
        raise HTTPException(400)
    f = _job_dir(job_id) / sub / f"page_{page:05d}.png"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


@app.post("/api/jobs/{job_id}/run", dependencies=[Depends(_auth)])
def run(job_id: str, body: RunJob):
    """Enqueue a run. One job processes at a time (single GPU); pressing Run
    on another book lines it up behind the current one."""
    jdir = _job_dir(job_id)
    if body.mode == "none" and not body.translate:
        raise HTTPException(400, "mode 'none' does nothing unless translate is enabled")
    ref = jdir / "ref.png"
    try:
        info = jm.enqueue_run(
            job_id,
            selected=body.pages, mode=body.mode, device=body.device,
            ink_weight=body.ink_weight, anchor_page=body.anchor_page,
            ref_image=ref if (body.use_ref and ref.exists()) else None,
            ml_text=body.ml_text,
            translate=body.translate, translate_sfx=body.translate_sfx,
            translate_lang=body.translate_lang,
            auto_ref=body.auto_ref,
            ref_strength=max(0.0, min(1.0, body.ref_strength)),
            ip_scale=max(0.0, min(1.0, body.ip_scale)),
            self_ref_scale=max(0.0, min(1.0, body.self_ref_scale)),
            page_consistency=max(0.0, min(1.0, body.page_consistency)),
            steps=max(4, min(60, body.steps)),
            fill_voids=body.fill_voids,
            protect_text=body.protect_text,
        )
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {"started": True, **info}


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(_auth)])
def cancel(job_id: str):
    _job_dir(job_id)
    jm.cancel(job_id)
    return {"cancelled": True}


@app.post("/api/jobs/{job_id}/export", dependencies=[Depends(_auth)])
def export(job_id: str, body: ExportJob):
    _job_dir(job_id)
    if body.fmt not in ("folder", "pdf", "cbz"):
        raise HTTPException(400, "fmt must be folder|pdf|cbz")
    try:
        out = jm.export(job_id, body.fmt, Path(body.dest))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"exported": str(out)}


@app.get("/api/jobs/{job_id}/download/{fmt}", dependencies=[Depends(_auth)])
def download(job_id: str, fmt: str):
    """Export to the workspace exports dir and hand the file to the browser —
    the natural path for books that were uploaded rather than opened from a
    local path. All formats: cbz, pdf, and folder (delivered as a .zip of the
    page images, since a browser can't receive a directory)."""
    import shutil

    _job_dir(job_id)
    if fmt not in ("pdf", "cbz", "folder"):
        raise HTTPException(400, "downloadable formats: cbz, pdf, folder")
    name = next((j["name"] for j in jm.list_jobs() if j["job"] == job_id), job_id)
    base = Path(name).stem or job_id
    try:
        if fmt == "folder":
            outdir = jm.export(job_id, "folder", exports_dir() / f"{base}_color_pages")
            zip_path = shutil.make_archive(str(outdir), "zip", root_dir=outdir)
            return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)
        out = jm.export(job_id, fmt, exports_dir() / f"{base}_color.{fmt}")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    media = "application/pdf" if fmt == "pdf" else "application/vnd.comicbook+zip"
    return FileResponse(out, media_type=media, filename=out.name)


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(_auth)])
def delete(job_id: str):
    _job_dir(job_id)
    jm.delete(job_id)
    return JSONResponse({"deleted": True})


def serve(port: int = 8317) -> int:
    import uvicorn

    ensure_dirs()
    jm.cleanup_stale()
    url = f"http://127.0.0.1:{port}/?token={TOKEN}"
    print(f"\n  InkedIn UI:  {url}\n  (local only; token dies with this process)\n", flush=True)
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


_INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>InkedIn</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d0d12;--panel:#16161d;--card:#1c1c26;--line:#2b2b38;--ink:#eceaf2;
  --mut:#8f8da1;--acc:#e0564f;--acc2:#ff7a6e;--ok:#4fb06a;--warn:#d9a13d;--err:#d05555}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:18px}header h1 b{color:var(--acc)}
.pill{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:20px;padding:2px 10px}
.pill.gpu{color:var(--ok);border-color:#2c4a35}
#layout{flex:1;display:flex;min-height:0}
#side{width:270px;min-width:270px;background:var(--panel);border-right:1px solid var(--line);padding:14px;overflow-y:auto}
#main{flex:1;padding:16px 20px;overflow-y:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:10px}
input[type=text],select{background:#0c0c11;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:8px;width:100%;font:inherit}
input[type=range]{width:100%;accent-color:var(--acc)}
label{display:block;color:var(--mut);font-size:12px;margin:8px 0 3px}
.chk{display:flex;align-items:center;gap:7px;color:var(--mut);font-size:12.5px;margin:6px 0;cursor:pointer}
.chk input{accent-color:var(--acc)}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:9px 15px;font-weight:600;cursor:pointer;font:inherit}
button:hover{background:var(--acc2)}
button.alt{background:#2c2c38}button.alt:hover{background:#3a3a4a}
button:disabled{opacity:.4;cursor:default}
.btnrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
#drop{border:2px dashed var(--line);border-radius:10px;padding:20px 12px;text-align:center;color:var(--mut);font-size:12.5px;transition:.15s}
#drop.hot{border-color:var(--acc);color:var(--ink);background:#e0564f14}
#drop b{color:var(--ink)}
.fmt{font-size:10.5px;color:#6a6878;margin-top:6px}
.job{display:flex;align-items:center;gap:8px;padding:8px 9px;border:1px solid transparent;border-radius:8px;cursor:pointer;margin-bottom:4px}
.job:hover{background:#ffffff0a}
.job.on{background:#e0564f1a;border-color:#e0564f55}
.job .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.job .np{font-size:11px;color:var(--mut)}
.job .del{background:none;border:0;color:var(--mut);padding:0 4px;font-size:15px;cursor:pointer}
.job .del:hover{color:var(--err);background:none}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:170px}
.hint{font-size:12px;color:var(--mut);margin-top:4px;min-height:18px}
#swatches{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.sw{width:26px;height:26px;border-radius:50%;cursor:pointer;border:2px solid transparent;position:relative}
.sw.on{border-color:#fff}
.sw:hover::after{content:attr(data-n);position:absolute;top:-24px;left:50%;transform:translateX(-50%);
  background:#000c;color:#fff;font-size:10px;padding:1px 7px;border-radius:6px;white-space:nowrap}
#bar{height:7px;background:#2c2c38;border-radius:4px;overflow:hidden;margin-top:12px;display:none}
#bar>div{height:100%;background:linear-gradient(90deg,var(--acc),var(--ok));width:0%;transition:width .4s}
#log{color:var(--mut);font-size:12px;margin-top:8px;white-space:pre-wrap}
#gridbar{display:flex;align-items:center;gap:10px;margin:14px 0 8px}
#gridbar .sp{flex:1}
.mini{font-size:12px;color:var(--mut)}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.pg{position:relative;border:2px solid var(--line);border-radius:9px;overflow:hidden;cursor:pointer;background:#0a0a0e}
.pg img{width:100%;display:block;aspect-ratio:3/4.3;object-fit:contain}
.pg.sel{border-color:var(--acc)}
.pg.done{border-color:var(--ok)}
.pg.err{border-color:var(--err)}
.pg .n{position:absolute;top:5px;left:6px;background:#000b;padding:0 7px;border-radius:9px;font-size:11px}
.pg .st{position:absolute;bottom:5px;right:6px;font-size:11px;background:#000b;padding:0 7px;border-radius:9px}
.pg .cmp{position:absolute;bottom:5px;left:6px;font-size:10px;background:#000b;padding:1px 7px;border-radius:9px;
  color:var(--ok);display:none}
.pg.done .cmp{display:block}
.pg .spin{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:#0008}
.pg.busy .spin{display:flex}
.spin::after{content:"";width:22px;height:22px;border:3px solid #fff3;border-top-color:var(--acc);border-radius:50%;
  animation:r 1s linear infinite}
@keyframes r{to{transform:rotate(360deg)}}
.btnspin{display:inline-block;width:12px;height:12px;border:2px solid #fff5;border-top-color:#fff;
  border-radius:50%;animation:r .8s linear infinite;vertical-align:-2px;margin-right:7px}
.upbar{height:5px;background:#2c2c38;border-radius:3px;overflow:hidden;margin-top:8px}
.upbar>div{height:100%;background:var(--acc);width:0%;transition:width .2s}
#refbox{display:flex;align-items:center;gap:10px;margin-top:6px}
#refimg{width:44px;height:44px;object-fit:cover;border-radius:6px;border:1px solid var(--line);display:none}
#modal{position:fixed;inset:0;background:#000d;display:none;flex-direction:column;align-items:center;
  justify-content:center;z-index:10;padding:24px}
#modal.on{display:flex}
#cwrap{position:relative;max-width:92vw;max-height:82vh;user-select:none;touch-action:none}
#cwrap img{max-width:92vw;max-height:82vh;display:block}
#cover{position:absolute;inset:0;overflow:hidden}
#cover img{filter:grayscale(0)}
#cbarline{position:absolute;top:0;bottom:0;width:2px;background:var(--acc);cursor:ew-resize}
#cbarline::after{content:"◂ ▸";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  background:var(--acc);border-radius:14px;padding:3px 8px;font-size:11px;white-space:nowrap}
#mtools{margin-top:12px;display:flex;gap:10px;align-items:center;color:var(--mut);font-size:12px}
.tag{position:absolute;top:8px;background:#000b;padding:2px 10px;border-radius:8px;font-size:11px}
kbd{background:#2c2c38;border-radius:4px;padding:0 5px;font-size:11px}
#empty{color:var(--mut);text-align:center;padding:60px 0;font-size:13px}
#reader{position:fixed;inset:0;background:#08080b;display:none;z-index:20}
#reader.on{display:block}
#rimg{position:absolute;inset:0;margin:auto;max-width:100%;max-height:100%}
.rnav{position:absolute;top:0;bottom:0;width:26%;display:flex;align-items:center;color:#fff0;font-size:54px;cursor:pointer;user-select:none;z-index:2}
.rnav:hover{color:#fff8}
#rprev{left:0;justify-content:flex-start;padding-left:18px}
#rnext{right:0;justify-content:flex-end;padding-right:18px}
#rbar{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:#000c;border-radius:20px;padding:5px 16px;font-size:12.5px;z-index:3;white-space:nowrap}
#rstat{color:var(--mut);margin-left:10px}
#rclose{position:absolute;top:10px;right:18px;font-size:22px;color:#fff9;cursor:pointer;z-index:3}
#rclose:hover{color:#fff}
</style></head><body>

<header>
  <h1>Inked<b>In</b></h1>
  <span class="pill" id="devpill">device: …</span>
  <span class="pill">local only — nothing leaves this machine</span>
  <span style="flex:1"></span>
  <span class="pill" id="verpill"></span>
</header>

<div id="layout">
<div id="side">
  <div class="card">
    <h2>Import</h2>
    <div id="drop"></div>
    <input type="file" id="fpick" multiple style="display:none">
    <input type="file" id="dpick" webkitdirectory style="display:none">
    <div id="chooser" style="display:none;margin-top:10px;border:1px solid var(--line);border-radius:8px;padding:10px">
      <div class="hint" id="chq" style="margin:0 0 8px"></div>
      <div class="btnrow" style="margin-top:0">
        <button id="chone">One book</button>
        <button class="alt" id="chsep">Separate</button>
        <button class="alt" id="chx">✕</button>
      </div>
    </div>
    <label>…or a local path (file / folder)</label>
    <input type="text" id="path" placeholder="D:\manga\chapter1.cbz">
    <label class="chk"><input type="checkbox" id="split"> split double-page spreads</label>
    <label class="chk"><input type="checkbox" id="rtl"> right-to-left book (manga)</label>
    <div class="btnrow"><button id="load" style="width:100%">Import</button></div>
    <div class="hint" id="imperr"></div>
  </div>
  <div class="card">
    <h2>Books</h2>
    <div id="joblist"><div class="mini">nothing imported yet</div></div>
  </div>
</div>

<div id="main">
  <div id="empty">Import a book or image on the left to begin.</div>

  <div id="work" style="display:none">
  <div class="card">
    <h2>Colorize</h2>
    <div class="row">
      <div>
        <label>Mode</label><select id="mode"></select>
        <div class="hint" id="modehint"></div>
        <div id="swatches"></div>
      </div>
      <div>
        <label>Prompt <span class="mini">(ai mode — describe colors)</span></label>
        <input type="text" id="prompt" placeholder='e.g. "red jacket, blonde hair, sunset"'>
        <label>Reference image <span class="mini">(ai mode — colors copied from it)</span></label>
        <div id="refbox">
          <img id="refimg"><button class="alt" id="refbtn">Choose…</button>
          <button class="alt" id="refclr" style="display:none">✕</button>
          <input type="file" id="refpick" accept="image/*" style="display:none">
        </div>
      </div>
      <div>
        <label>Line strength <span class="mini" id="inkv">0.85</span></label>
        <input type="range" id="ink" min="0" max="1" step="0.05" value="0.85">
        <label>Device</label>
        <select id="dev"><option>auto</option><option>cuda</option><option>cpu</option></select>
        <label class="chk" id="mlrow"><input type="checkbox" id="mltext"> AI text protection
          <span class="mini" id="mlhint"></span></label>
        <label class="chk" id="trrow"><input type="checkbox" id="trans"> <b>Translate to English</b>
          <span class="mini" id="trhint"></span></label>
        <div id="langrow" style="margin-left:22px;display:flex;gap:8px;align-items:center">
          <span class="mini">from</span>
          <select id="tlang" style="width:auto;padding:4px 8px">
            <option value="ja">Japanese</option><option value="ko">Korean</option>
            <option value="zh">Chinese</option><option value="ru">Russian</option>
            <option value="es">Spanish</option><option value="fr">French</option>
            <option value="de">German</option><option value="it">Italian</option>
            <option value="pt">Portuguese</option>
          </select>
          <label class="chk" id="sfxrow" style="margin:0"><input type="checkbox" id="sfx"> also SFX</label>
        </div>
      </div>
    </div>
    <div class="hint" id="refbanner" style="display:none;margin-top:8px">🎨 <span id="refmsg"></span>
      <label class="chk" style="display:inline-flex;margin:0 0 0 10px"><input type="checkbox" id="autoref" checked>
        use as color reference</label></div>
    <details id="adv" style="margin-top:10px">
      <summary style="cursor:pointer;color:var(--mut);font-size:12px;user-select:none">⚙ Advanced — weights &amp; toggles</summary>
      <div class="row" style="margin-top:10px">
        <div><label>Reference influence <span class="mini" id="refsV">0.15</span></label>
          <input type="range" id="refs" min="0" max="1" step="0.05" value="0.15">
          <div class="hint">bias from detected color pages — 0 keeps the model's own colors untouched</div></div>
        <div><label>Ref image strength (ai) <span class="mini" id="ipsV">0.65</span></label>
          <input type="range" id="ips" min="0" max="1" step="0.05" value="0.65">
          <div class="hint">pull of the uploaded reference image on every panel</div></div>
        <div><label>Panel consistency (ai) <span class="mini" id="scV">0.4</span></label>
          <input type="range" id="sc" min="0" max="1" step="0.05" value="0.4">
          <div class="hint">how strongly panels follow panel 1's colors</div></div>
        <div><label>Cross-page consistency <span class="mini" id="pcV">0.25</span></label>
          <input type="range" id="pc" min="0" max="1" step="0.05" value="0.25">
          <div class="hint">later pages follow the first colorized page — keeps character hair/outfit colors stable</div></div>
        <div><label>AI steps <span class="mini" id="stV">24</span></label>
          <input type="range" id="st" min="8" max="50" step="1" value="24">
          <div class="hint">more = slower, finer detail</div></div>
      </div>
      <label class="chk"><input type="checkbox" id="fillv" checked> repair missing color spots (chroma inpaint)</label>
      <label class="chk"><input type="checkbox" id="ptext" checked> protect bubbles &amp; lettering (keep them clean)</label>
    </details>
    <div class="btnrow">
      <button id="run">▶ Colorize selected</button>
      <button class="alt" id="cancel">Cancel</button>
    </div>
    <div id="bar"><div></div></div>
    <div id="log"></div>
  </div>

  <div class="card">
    <h2>Export</h2>
    <div class="row">
      <div style="flex:0;min-width:110px"><label>Format</label>
        <select id="fmt"><option>cbz</option><option>pdf</option><option>folder</option></select></div>
      <div><label>Destination <span class="mini">(defaults to the book's own folder)</span></label>
        <input type="text" id="dest" placeholder="D:\manga\book_color.cbz"></div>
      <div style="flex:0;align-self:end"><button id="exp">Export</button></div>
      <div style="flex:0;align-self:end"><button class="alt" id="dl" title="export and save through the browser">⬇ Download</button></div>
    </div>
    <div class="hint" id="exphint"></div>
  </div>

  <div id="gridbar">
    <button class="alt" id="all">All</button>
    <button class="alt" id="none">None</button>
    <button class="alt" id="inv">Invert</button>
    <button class="alt" id="read">📖 Read</button>
    <span class="mini" id="selinfo"></span>
    <span class="sp"></span>
    <span class="mini">click = select · <kbd>shift</kbd>+click = range · double-click = read · ⇆ = compare</span>
  </div>
  <div id="grid"></div>
  </div>
</div>
</div>

<div id="reader">
  <img id="rimg">
  <div class="rnav" id="rprev">‹</div>
  <div class="rnav" id="rnext">›</div>
  <div id="rclose">✕</div>
  <div id="rbar"><span id="rinfo"></span><span id="rstat"></span></div>
</div>

<div id="modal">
  <div id="cwrap">
    <img id="mout">
    <div id="cover"><img id="msrc"></div>
    <div id="cbarline"></div>
    <span class="tag" style="left:10px">original</span>
    <span class="tag" style="right:10px">colorized</span>
  </div>
  <div id="mtools">drag the divider · <kbd>esc</kbd> or click outside to close</div>
</div>

<script>
const TOKEN=new URLSearchParams(location.search).get('token');
const H={'x-inkedin-token':TOKEN,'content-type':'application/json'};
const UH={'x-inkedin-token':TOKEN};
const $=id=>document.getElementById(id);
let JOB=null,NPAGES=0,SEL=new Set(),POLL=null,LASTCLICK=null,JOBNAMES={},META=null,PAGESTATE={};
const log=m=>$('log').textContent=m;
async function api(p,opt){const r=await fetch(p,opt);
  if(!r.ok){let d;try{d=(await r.json()).detail}catch(_){d=r.status}throw new Error(d)}return r.json()}

const THEME_COLORS={sepia:'#c8a06a',noir:'#8a8a95',sunset:'#e58a5a',ocean:'#5a9ad0',forest:'#6aa06a',pastel:'#d0a0c8'};
const MODE_HINTS={
 none:'No colorization — keep the original art. Use with "Translate to English" for a translate-only pass.',
 fast:'GAN model — quick, works on CPU, soft palette wash.',
 ai:'SD1.5 + ControlNet — understands content (skin/sky/hair), panel-wise, needs GPU. Prompt + reference apply.',
 theme:'Instant duotone grade — no ML, deterministic.',
 'fast+theme':'GAN colorization, then the theme grade on top.'};

(async()=>{try{
  const m=await api('/api/meta',{headers:H});META=m;
  const modes=['ai','fast',...m.themes.map(t=>'theme:'+t),...m.themes.map(t=>'fast+theme:'+t),'none'];
  $('mode').innerHTML=modes.map(x=>`<option value="${x}">${x==='none'?'none — translate only':x}</option>`).join('');
  $('devpill').textContent=`device: ${m.device.device} (${m.device.name||'?'})`;
  if(m.device.device==='cuda')$('devpill').classList.add('gpu');
  $('verpill').textContent='v'+m.version;
  $('mlhint').textContent=m.ml_text==='cached'?'(model cached)':m.ml_text==='downloadable'?'(~170 MB download on first run)':'';
  if(m.ml_text==='unavailable')$('mlrow').style.display='none';
  $('trhint').textContent=m.translate==='cached'?'(models cached)':m.translate==='downloadable'?'(Japanese OCR + MT, ~2.3 GB download on first run)':'';
  if(m.translate==='unavailable'){$('trrow').style.display='none';$('langrow').style.display='none'}
  $('swatches').innerHTML=m.themes.map(t=>
    `<div class="sw" data-n="${t}" data-t="${t}" style="background:${THEME_COLORS[t]||'#888'}"></div>`).join('');
  document.querySelectorAll('.sw').forEach(s=>s.onclick=()=>{
    $('mode').value='theme:'+s.dataset.t;modeHint();
    document.querySelectorAll('.sw').forEach(x=>x.classList.toggle('on',x===s))});
  modeHint();refreshJobs();
}catch(e){log('init failed: '+e.message)}})();

function modeHint(){const v=$('mode').value;
  const k=v.startsWith('fast+theme')?'fast+theme':v.startsWith('theme')?'theme':v.split(':')[0];
  $('modehint').textContent=MODE_HINTS[k]||'';
  if(!v.startsWith('theme:'))document.querySelectorAll('.sw').forEach(x=>x.classList.remove('on'))}
$('mode').onchange=modeHint;
$('ink').oninput=()=>$('inkv').textContent=$('ink').value;
[['refs','refsV'],['ips','ipsV'],['sc','scV'],['pc','pcV'],['st','stV']].forEach(([s,v])=>
  $(s).oninput=()=>$(v).textContent=$(s).value);

/* ---- import ---- */
const drop=$('drop');
const DROP_HTML='<b>Drop files or a folder here</b><br>click: pick files · <u id="pickdir" style="cursor:pointer">pick a folder…</u>'+
  '<div class="fmt">PDF · CBZ · CB7 · CBT · CBR · EPUB · ZIP · images<br>a folder of images = one book</div>';
drop.innerHTML=DROP_HTML;
const IMG_RE=/\.(png|jpe?g|jpe|jfif|webp|bmp|tiff?|gif)$/i;
const natkey=s=>s.toLowerCase().split(/(\d+)/).map(t=>/^\d+$/.test(t)?t.padStart(12,'0'):t).join('');
const relname=f=>f._rel||f.webkitRelativePath||f.name;
const natsort=fs=>[...fs].sort((a,b)=>{const x=natkey(relname(a)),y=natkey(relname(b));return x<y?-1:x>y?1:0});

drop.addEventListener('click',e=>{
  if(e.target.id==='pickdir'){e.stopPropagation();$('dpick').click()}else $('fpick').click()});
$('fpick').onchange=()=>{handleFiles([...$('fpick').files]);$('fpick').value=''};
$('dpick').onchange=()=>{
  const fs=[...$('dpick').files].filter(f=>IMG_RE.test(f.name));
  if(!fs.length){$('imperr').textContent='✗ that folder has no images'}
  else{const top=(fs[0].webkitRelativePath||'').split('/')[0]||'folder';uploadBatch(natsort(fs),top)}
  $('dpick').value=''};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hot');
  const items=[...(e.dataTransfer.items||[])],dirs=[],loose=[];
  for(const it of items){
    const en=it.webkitGetAsEntry?it.webkitGetAsEntry():null;
    if(en&&en.isDirectory)dirs.push(en);
    else{const f=it.getAsFile();if(f)loose.push(f)}
  }
  if(!items.length&&e.dataTransfer.files.length)loose.push(...e.dataTransfer.files);
  (async()=>{
    for(const d of dirs){
      const fs=(await readDirAll(d)).filter(f=>IMG_RE.test(f.name));
      if(fs.length)await uploadBatch(natsort(fs),d.name);
      else $('imperr').textContent='✗ '+d.name+': no images inside';
    }
    if(loose.length)handleFiles(loose);
  })()};

function readDirAll(dir){return new Promise(res=>{const out=[];
  const walk=(d,done)=>{const rd=d.createReader();
    const pump=()=>rd.readEntries(ents=>{
      if(!ents.length){done();return}
      let waiting=ents.length;
      const next=()=>{if(--waiting===0)pump()};
      ents.forEach(en=>{
        if(en.isFile)en.file(f=>{try{f._rel=d.fullPath+'/'+f.name}catch(_){}out.push(f);next()},next);
        else if(en.isDirectory)walk(en,next);
        else next()})});
    pump()};
  walk(dir,()=>res(out))})}

/* multiple loose images: user decides — one book or separate jobs */
let PENDING=null;
function handleFiles(files){
  if(!files.length)return;
  if(files.length>1&&files.every(f=>IMG_RE.test(f.name))){
    PENDING=files;
    $('chq').textContent=files.length+' images selected — import as one book (pages in order) or separately?';
    $('chooser').style.display='block';
  }else uploadFiles(files)}
$('chone').onclick=()=>{$('chooser').style.display='none';if(PENDING)uploadBatch(natsort(PENDING),'');PENDING=null};
$('chsep').onclick=()=>{$('chooser').style.display='none';if(PENDING)uploadFiles(PENDING);PENDING=null};
$('chx').onclick=()=>{$('chooser').style.display='none';PENDING=null};

/* upload with real progress (fetch can't report upload %, XHR can). The
   server ingests synchronously after the upload lands, so the bar fills to
   100% and then shows an "importing" spinner phase. */
function xhrUpload(url,fd,label){
  return new Promise((resolve,reject)=>{
    drop.innerHTML=`<b>${label}</b><div class="upbar"><div></div></div><div class="mini" id="upst">uploading… 0%</div>`;
    const bar=drop.querySelector('.upbar>div'),st=drop.querySelector('#upst');
    const x=new XMLHttpRequest();
    x.open('POST',url);
    x.setRequestHeader('x-inkedin-token',TOKEN);
    x.upload.onprogress=e=>{if(e.lengthComputable){
      const pc=Math.round(100*e.loaded/e.total);
      bar.style.width=pc+'%';
      st.textContent=pc<100?('uploading… '+pc+'%'):'processing pages…';
      if(pc>=100)st.innerHTML='<span class="btnspin"></span>importing pages… (big books take a moment)'}};
    x.onload=()=>{try{const j=JSON.parse(x.responseText);
      x.status<400?resolve(j):reject(new Error(j.detail||x.status))}
      catch(e){reject(new Error('bad response'))}};
    x.onerror=()=>reject(new Error('network error'));
    x.send(fd)});
}

async function uploadFiles(files){
  if(!files.length)return;
  const q=`?split_spreads=${$('split').checked}&rtl=${$('rtl').checked}`;
  let i=0;
  for(const f of files){
    i++;
    const fd=new FormData();fd.append('file',f);
    try{
      const r=await xhrUpload('/api/jobs/upload'+q,fd,
        files.length>1?`file ${i}/${files.length}: ${f.name}`:f.name);
      JOBNAMES[r.job]=r.name;$('imperr').textContent='';await refreshJobs();openJob(r.job,r.pages);
    }catch(err){$('imperr').textContent='✗ '+f.name+': '+err.message}}
  drop.innerHTML=DROP_HTML;
}

async function uploadBatch(files,name){
  const q=`?split_spreads=${$('split').checked}&rtl=${$('rtl').checked}&name=${encodeURIComponent(name)}`;
  const fd=new FormData();
  for(const f of files)fd.append('files',f);
  try{
    const r=await xhrUpload('/api/jobs/upload-batch'+q,fd,`${files.length} page(s) → one book`);
    JOBNAMES[r.job]=r.name;$('imperr').textContent='';await refreshJobs();openJob(r.job,r.pages);
  }catch(err){$('imperr').textContent='✗ batch import: '+err.message}
  drop.innerHTML=DROP_HTML;
}

$('load').onclick=async()=>{try{
  $('load').disabled=true;$('load').innerHTML='<span class="btnspin"></span>Importing… (large books take a while)';
  const r=await api('/api/jobs',{method:'POST',headers:H,body:JSON.stringify(
    {path:$('path').value,split_spreads:$('split').checked,rtl:$('rtl').checked})});
  $('imperr').textContent='';await refreshJobs();openJob(r.job,r.pages);
}catch(e){$('imperr').textContent='✗ import failed: '+e.message}
finally{$('load').disabled=false;$('load').textContent='Import'}};

/* ---- job list ---- */
let JOBDIRS={};
async function refreshJobs(){
  const r=await api('/api/jobs',{headers:H});
  const el=$('joblist');
  r.jobs.forEach(j=>{if(!JOBNAMES[j.job])JOBNAMES[j.job]=j.name;JOBDIRS[j.job]=j.dir||''});
  if(!r.jobs.length){el.innerHTML='<div class="mini">nothing imported yet</div>';return}
  el.innerHTML='';
  r.jobs.forEach(j=>{
    const d=document.createElement('div');d.className='job'+(j.job===JOB?' on':'');
    const icon=j.status==='running'?'⏳':j.status==='queued'?'🕐':j.status==='error'?'⚠️'
      :j.status?.startsWith('done')?'✅':'📖';
    d.innerHTML=`<span>${icon}</span><span class="nm">${JOBNAMES[j.job]||j.name}</span>
      <span class="np">${j.pages}p</span><button class="del" title="delete">✕</button>`;
    d.onclick=()=>openJob(j.job,j.pages);
    d.querySelector('.del').onclick=async e=>{e.stopPropagation();
      if(!confirm('Delete this book and its results?'))return;
      await api('/api/jobs/'+j.job,{method:'DELETE',headers:H});
      if(JOB===j.job){JOB=null;$('work').style.display='none';$('empty').style.display='block'}
      refreshJobs()};
    el.appendChild(d)});
}

/* ---- open job / grid ---- */
let COLORSET=new Set();
async function openJob(id,n){
  JOB=id;NPAGES=n;LASTCLICK=null;
  const pr=await api(`/api/jobs/${id}/pages`,{headers:H});
  COLORSET=new Set(pr.pages.filter(p=>p.is_color).map(p=>p.page));
  SEL=new Set([...Array(n).keys()].filter(i=>!COLORSET.has(i)));
  $('refbanner').style.display=COLORSET.size?'block':'none';
  $('refmsg').textContent=COLORSET.size+' already-colored page(s) detected — skipped from colorizing';
  $('empty').style.display='none';$('work').style.display='block';
  $('bar').style.display='none';log('');
  $('run').disabled=false;  // each book has its own Run; the queue serializes them
  POLL&&clearInterval(POLL);POLL=setInterval(refresh,1200);
  const nm=JOBNAMES[id]||'book';
  // default export location: the folder the book came from; uploaded books
  // (no source folder known to the server) fall back to the workspace exports dir
  const srcdir=JOBDIRS[id]||(META?META.export_dir:'');
  const dir=srcdir?srcdir.replace(/[\\/]$/,'')+'/':'';  // '/' works on Windows too
  const base=dir+nm.replace(/\.[^.]+$/,'')+'_color';
  $('dest').value=$('fmt').value==='folder'?base:base+'.'+$('fmt').value;
  PAGESTATE={};pr.pages.forEach(p=>PAGESTATE[p.page]=p.status);
  renderGrid(n);await refreshRef();await refresh();refreshJobs();
}

function renderGrid(n){const g=$('grid');g.innerHTML='';
  for(let i=0;i<n;i++){const d=document.createElement('div');
    d.className='pg'+(SEL.has(i)?' sel':'');d.id='pg'+i;
    const badge=COLORSET.has(i)?`<span class="n" style="left:auto;right:6px" title="already colored — used as reference">🎨</span>`:'';
    d.innerHTML=`<img loading="lazy" src="/api/jobs/${JOB}/thumb/${i}?token=${TOKEN}">
      <span class="n">${i+1}</span>${badge}<span class="st"></span><span class="cmp">⇆ compare</span><div class="spin"></div>`;
    d.onclick=e=>clickPage(i,e);
    d.ondblclick=()=>openReader(i);
    d.querySelector('.cmp').onclick=e=>{e.stopPropagation();openCompare(i)};
    g.appendChild(d)}
  selInfo()}

function clickPage(i,e){
  if(e.shiftKey&&LASTCLICK!==null){
    const [a,b]=[Math.min(i,LASTCLICK),Math.max(i,LASTCLICK)];
    const on=!SEL.has(i);
    for(let k=a;k<=b;k++){on?SEL.add(k):SEL.delete(k);$('pg'+k).classList.toggle('sel',on)}
  }else{
    SEL.has(i)?SEL.delete(i):SEL.add(i);
    $('pg'+i).classList.toggle('sel',SEL.has(i));
  }
  LASTCLICK=i;selInfo()}

const selInfo=()=>$('selinfo').textContent=`${SEL.size}/${NPAGES} selected`;
$('all').onclick=()=>{SEL=new Set([...Array(NPAGES).keys()]);document.querySelectorAll('.pg').forEach(d=>d.classList.add('sel'));selInfo()};
$('none').onclick=()=>{SEL.clear();document.querySelectorAll('.pg').forEach(d=>d.classList.remove('sel'));selInfo()};
$('inv').onclick=()=>{for(let i=0;i<NPAGES;i++){SEL.has(i)?SEL.delete(i):SEL.add(i);$('pg'+i).classList.toggle('sel')}selInfo()};

/* ---- reference image ---- */
$('refbtn').onclick=()=>$('refpick').click();
$('refpick').onchange=async()=>{const f=$('refpick').files[0];if(!f||!JOB)return;
  const fd=new FormData();fd.append('file',f);
  try{await api(`/api/jobs/${JOB}/ref`,{method:'POST',headers:UH,body:fd});await refreshRef()}
  catch(e){log('reference upload failed: '+e.message)}};
$('refclr').onclick=async()=>{await api(`/api/jobs/${JOB}/ref`,{method:'DELETE',headers:H});refreshRef()};
async function refreshRef(){
  const r=await fetch(`/api/jobs/${JOB}/ref?token=${TOKEN}&_=${Date.now()}`,{headers:UH});
  const has=r.ok;
  $('refimg').style.display=has?'block':'none';
  $('refclr').style.display=has?'inline-block':'none';
  if(has)$('refimg').src=URL.createObjectURL(await r.blob())}

/* ---- run ---- */
$('run').onclick=async()=>{if(!JOB||!SEL.size)return;
  let mode=$('mode').value;
  if(mode==='ai'&&$('prompt').value.trim())mode='ai:'+$('prompt').value.trim();
  try{
    document.querySelectorAll('.pg').forEach(d=>{d.classList.remove('done','err');d.querySelector('.st').textContent=''});
    const rr=await api(`/api/jobs/${JOB}/run`,{method:'POST',headers:H,body:JSON.stringify(
      {pages:[...SEL],mode,device:$('dev').value,ink_weight:parseFloat($('ink').value)||0.85,
       use_ref:$('refimg').style.display!=='none',ml_text:$('mltext').checked,
       translate:$('trans').checked,translate_sfx:$('sfx').checked,
       translate_lang:$('tlang').value,auto_ref:$('autoref').checked,
       ref_strength:parseFloat($('refs').value),ip_scale:parseFloat($('ips').value),
       self_ref_scale:parseFloat($('sc').value),page_consistency:parseFloat($('pc').value),
       steps:parseInt($('st').value),
       fill_voids:$('fillv').checked,protect_text:$('ptext').checked})});
    $('bar').style.display='block';$('bar').firstElementChild.style.width='0%';
    $('run').disabled=true;
    log(rr.position>0?`🕐 queued (position ${rr.position}) — another book is on the GPU right now`:'⏳ starting…');
    POLL&&clearInterval(POLL);POLL=setInterval(refresh,1200);refreshJobs();
  }catch(e){log('run failed: '+e.message);$('run').disabled=false}};

$('cancel').onclick=()=>JOB&&api(`/api/jobs/${JOB}/cancel`,{method:'POST',headers:H});

async function refresh(){if(!JOB)return;
  const r=await api(`/api/jobs/${JOB}/pages`,{headers:H});
  let done=0,err=0;
  const cur=r.progress?r.progress.current:null;
  r.pages.forEach(p=>{PAGESTATE[p.page]=p.status;const d=$('pg'+p.page);if(!d)return;const st=d.querySelector('.st');
    d.classList.toggle('busy',POLL!==null&&p.page===cur&&p.status==='pending');
    if(p.status==='done'){d.classList.add('done');d.classList.remove('err');st.textContent='✓';done++;
      const im=d.querySelector('img');
      const want=`/api/jobs/${JOB}/image/out/${p.page}?token=${TOKEN}`;
      if(!im.dataset.out){im.src=want;im.dataset.out='1'}}
    else if(p.status==='error'){d.classList.add('err');st.textContent='✗';st.title=p.error||'';err++}
    else st.textContent=''});
  if(r.progress){const st=r.progress.stage;
    $('bar').style.display='block';
    if(st==='queued'){
      $('bar').firstElementChild.style.width='0%';
      log('🕐 queued — will start automatically when the current book finishes');
    }else if(st==='error'){
      log('✗ run failed: '+(r.progress.message||'see server log'));
      if(POLL){clearInterval(POLL);POLL=null}
      $('run').disabled=false;refreshJobs();
    }else{
      const pc=r.progress.total?100*r.progress.done/r.progress.total:0;
      $('bar').firstElementChild.style.width=pc+'%';
      if(st==='loading'&&r.progress.done===0)
        log('⏳ loading / downloading models — the first run can take a few minutes…');
      else
        log(`progress: ${r.progress.done}/${r.progress.total} page(s)`+(err?` · ${err} failed`:''));
      if(r.progress.total>0&&r.progress.done>=r.progress.total&&POLL){
        clearInterval(POLL);POLL=null;$('run').disabled=false;
        document.querySelectorAll('.pg.busy').forEach(d=>d.classList.remove('busy'));
        log(`finished: ${done} page(s) colorized`+(err?`, ${err} failed`:''));refreshJobs()}}}
  if($('reader').classList.contains('on'))updateReader(false);
}

/* ---- reader: full-page view, live-updates while the job runs ---- */
let RD=-1;
function openReader(i){RD=Math.max(0,Math.min(NPAGES-1,i));$('reader').classList.add('on');updateReader(true)}
function closeReader(){$('reader').classList.remove('on');RD=-1}
function updateReader(force){
  if(RD<0||!JOB)return;
  const st=PAGESTATE[RD];
  const kind=st==='done'?'out':'src';
  const key=kind+':'+RD;
  if(force||$('rimg').dataset.cur!==key){
    $('rimg').src=`/api/jobs/${JOB}/image/${kind}/${RD}?token=${TOKEN}`;
    $('rimg').dataset.cur=key}
  $('rinfo').textContent=`page ${RD+1} / ${NPAGES}`;
  $('rstat').textContent=st==='done'?'✓ colorized':st==='error'?'✗ failed — showing original'
    :POLL?'⏳ in progress — will switch when ready':'original';
}
$('read').onclick=()=>{if(NPAGES)openReader([...SEL].sort((a,b)=>a-b)[0]??0)};
$('rprev').onclick=()=>{if(RD>0){RD--;updateReader(true)}};
$('rnext').onclick=()=>{if(RD<NPAGES-1){RD++;updateReader(true)}};
$('rclose').onclick=closeReader;

/* ---- compare modal ---- */
function openCompare(i){
  $('msrc').src=`/api/jobs/${JOB}/image/src/${i}?token=${TOKEN}`;
  $('mout').src=`/api/jobs/${JOB}/image/out/${i}?token=${TOKEN}`;
  $('modal').classList.add('on');setSplit(0.5)}
function setSplit(f){f=Math.max(0,Math.min(1,f));
  $('cover').style.width=(f*100)+'%';$('cbarline').style.left=`calc(${f*100}% - 1px)`}
{let dragging=false;
 const move=e=>{if(!dragging)return;const r=$('cwrap').getBoundingClientRect();
   setSplit(((e.touches?e.touches[0].clientX:e.clientX)-r.left)/r.width)};
 $('cbarline').onpointerdown=e=>{dragging=true;e.preventDefault()};
 $('cwrap').onpointerdown=e=>{dragging=true;move(e)};
 window.addEventListener('pointermove',move);
 window.addEventListener('pointerup',()=>dragging=false);}
$('modal').onclick=e=>{if(e.target.id==='modal')$('modal').classList.remove('on')};
window.addEventListener('keydown',e=>{
  if($('reader').classList.contains('on')){
    if(e.key==='Escape')closeReader();
    else if(e.key==='ArrowLeft')$('rprev').onclick();
    else if(e.key==='ArrowRight')$('rnext').onclick();
    return}
  if(e.key==='Escape')$('modal').classList.remove('on')});

/* ---- export ---- */
$('fmt').onchange=()=>{const base=$('dest').value.replace(/\.(cbz|pdf)$/i,'');
  $('dest').value=$('fmt').value==='folder'?base:base+'.'+$('fmt').value};
$('dl').onclick=async()=>{if(!JOB)return;
  const fmt=$('fmt').value;  // cbz | pdf | folder (arrives as a .zip of pages)
  $('exphint').textContent='preparing '+fmt+' download…';
  try{
    const r=await fetch(`/api/jobs/${JOB}/download/${fmt}`,{headers:UH});
    if(!r.ok)throw new Error((await r.json()).detail||r.status);
    // stream with a live percentage when the size is known
    let blob;
    const total=+(r.headers.get('content-length')||0);
    if(total&&r.body){
      const reader=r.body.getReader();const chunks=[];let got=0;
      for(;;){const {done,value}=await reader.read();if(done)break;
        chunks.push(value);got+=value.length;
        $('exphint').textContent=`downloading… ${Math.round(100*got/total)}%`}
      blob=new Blob(chunks,{type:r.headers.get('content-type')||''});
    }else blob=await r.blob();
    const m=(r.headers.get('content-disposition')||'').match(/filename="?([^";]+)/);
    const name=m?m[1]:('book_color.'+(fmt==='folder'?'zip':fmt));
    // Chrome/Edge: real "where do you want to save this?" dialog
    if(window.showSaveFilePicker){
      try{
        const ext='.'+name.split('.').pop();
        const h=await showSaveFilePicker({suggestedName:name,
          types:[{description:'InkedIn export',accept:{[blob.type||'application/octet-stream']:[ext]}}]});
        const w=await h.createWritable();await w.write(blob);await w.close();
        $('exphint').textContent='✓ saved as '+h.name;return;
      }catch(e){
        if(e.name==='AbortError'){$('exphint').textContent='save cancelled';return}
        // picker unavailable/denied: fall through to plain download
      }
    }
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);a.download=name;
    a.click();URL.revokeObjectURL(a.href);
    $('exphint').textContent='✓ downloaded '+name;
  }catch(e){$('exphint').textContent='download failed: '+e.message}};
$('exp').onclick=async()=>{if(!JOB)return;
  const old=$('exp').textContent;
  try{$('exp').disabled=true;$('exp').innerHTML='<span class="btnspin"></span>Exporting…';
    $('exphint').textContent='writing '+$('fmt').value+'…';
    const r=await api(`/api/jobs/${JOB}/export`,{method:'POST',headers:H,body:JSON.stringify(
      {fmt:$('fmt').value,dest:$('dest').value})});
    $('exphint').textContent='✓ exported → '+r.exported}
  catch(e){$('exphint').textContent='export failed: '+e.message}
  finally{$('exp').disabled=false;$('exp').textContent=old}};
</script></body></html>
"""
