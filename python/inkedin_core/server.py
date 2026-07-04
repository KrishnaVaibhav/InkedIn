"""Local web UI. Security posture per Research.MD Gap 10:
- binds 127.0.0.1 only
- random per-session token required on every API call
- strict CSP, no external origins
- page/job identifiers validated; files served only from job dirs by integer index
"""

from __future__ import annotations

import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .jobs import JobManager
from .pipeline.colorize import THEMES
from .workspace import ensure_dirs, jobs_dir

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


class RunJob(BaseModel):
    pages: list[int] | None = None
    mode: str = "fast"
    device: str = "auto"
    ink_weight: float = 0.85
    anchor_page: int | None = None


class ExportJob(BaseModel):
    fmt: str
    dest: str


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/api/meta", dependencies=[Depends(_auth)])
def meta():
    from .models.device import probe

    d = probe()
    return {"themes": sorted(THEMES), "device": d.__dict__}


@app.post("/api/jobs", dependencies=[Depends(_auth)])
def create_job(body: CreateJob):
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(400, "path does not exist")
    try:
        job = jm.create(p)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"job": job.id, "pages": len(job.pages)}


@app.post("/api/jobs/upload", dependencies=[Depends(_auth)])
async def upload_job(file: UploadFile):
    ensure_dirs()
    staging = jobs_dir() / "_uploads"
    staging.mkdir(parents=True, exist_ok=True)
    # app-generated name; original filename only trusted for its extension hint
    ext = Path(file.filename or "x").suffix.lower()[:8]
    dest = staging / f"up_{secrets.token_hex(8)}{ext}"
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    try:
        job = jm.create(dest)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        dest.unlink(missing_ok=True)
    return {"job": job.id, "pages": len(job.pages)}


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) > 16:
        raise HTTPException(400, "bad job id")
    try:
        return jm.job_dir(job_id)
    except KeyError:
        raise HTTPException(404, "unknown job") from None


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
    _job_dir(job_id)
    if jm.progress.get(job_id, {}).get("current") is not None and job_id in jm.cancel_flags:
        raise HTTPException(409, "job already running")
    t = threading.Thread(
        target=jm.run,
        args=(job_id,),
        kwargs=dict(
            selected=body.pages, mode=body.mode, device=body.device,
            ink_weight=body.ink_weight, anchor_page=body.anchor_page,
        ),
        daemon=True,
    )
    t.start()
    return {"started": True}


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
<style>
:root{--bg:#101014;--card:#1a1a22;--ink:#e8e8f0;--mut:#8a8a99;--acc:#e0564f;--ok:#4fb06a}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;padding:20px;max-width:1200px;margin:auto}
h1{font-size:20px;margin-bottom:4px}h1 b{color:var(--acc)}
.sub{color:var(--mut);margin-bottom:16px}
.card{background:var(--card);border-radius:10px;padding:16px;margin-bottom:14px}
input[type=text],select{background:#0c0c10;color:var(--ink);border:1px solid #333;border-radius:6px;padding:8px;width:100%}
label{display:block;color:var(--mut);font-size:12px;margin:8px 0 3px}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:9px 16px;font-weight:600;cursor:pointer;margin-right:8px}
button.alt{background:#2c2c38}button:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;margin-top:12px}
.pg{position:relative;border:2px solid #333;border-radius:8px;overflow:hidden;cursor:pointer}
.pg img{width:100%;display:block}
.pg.sel{border-color:var(--acc)}.pg.done{border-color:var(--ok)}
.pg .n{position:absolute;top:4px;left:6px;background:#000a;padding:1px 7px;border-radius:9px;font-size:11px}
.pg .st{position:absolute;bottom:4px;right:6px;font-size:11px;background:#000a;padding:1px 7px;border-radius:9px}
#bar{height:6px;background:#2c2c38;border-radius:3px;overflow:hidden;margin-top:8px}
#bar>div{height:100%;background:var(--ok);width:0%}
#log{color:var(--mut);font-size:12px;margin-top:8px;white-space:pre-wrap}
#drop{border:2px dashed #444;border-radius:10px;padding:22px;text-align:center;color:var(--mut)}
#drop.hot{border-color:var(--acc);color:var(--ink)}
</style></head><body>
<h1>Inked<b>In</b></h1><div class="sub">local manga &amp; comic colorizer — nothing leaves this machine</div>

<div class="card">
  <div id="drop">drop a PDF / CBZ / image here, or</div>
  <label>…or type a local path (file or folder)</label>
  <div class="row"><div><input type="text" id="path" placeholder="D:\manga\chapter1.cbz"></div>
  <div style="flex:0"><button id="load">Load</button></div></div>
</div>

<div class="card" id="ctl" style="display:none">
  <div class="row">
    <div><label>Mode</label><select id="mode"></select></div>
    <div><label>Line strength (ink)</label><input type="text" id="ink" value="0.85"></div>
    <div><label>Device</label><select id="dev"><option>auto</option><option>cuda</option><option>cpu</option></select></div>
  </div>
  <div style="margin-top:12px">
    <button id="run">Colorize selected</button>
    <button class="alt" id="all">Select all</button>
    <button class="alt" id="none">Select none</button>
    <button class="alt" id="cancel">Cancel</button>
  </div>
  <div id="bar"><div></div></div><div id="log"></div>
  <div class="row" style="margin-top:12px">
    <div><label>Export as</label><select id="fmt"><option>cbz</option><option>pdf</option><option>folder</option></select></div>
    <div><label>Export path</label><input type="text" id="dest" placeholder="D:\manga\chapter1_color.cbz"></div>
    <div style="flex:0;align-self:end"><button id="exp">Export</button></div>
  </div>
  <div id="grid"></div>
</div>

<script>
const TOKEN=new URLSearchParams(location.search).get('token');
const H={'x-inkedin-token':TOKEN,'content-type':'application/json'};
let JOB=null,SEL=new Set(),POLL=null;
const $=id=>document.getElementById(id);
const log=m=>$('log').textContent=m;
async function api(p,opt){const r=await fetch(p,opt);if(!r.ok){throw new Error((await r.json()).detail||r.status)}return r.json()}

(async()=>{try{const m=await api('/api/meta',{headers:H});
  const modes=['fast',...m.themes.map(t=>'theme:'+t),...m.themes.map(t=>'fast+theme:'+t)];
  $('mode').innerHTML=modes.map(x=>`<option>${x}</option>`).join('');
  log(`device: ${m.device.device} (${m.device.name})`);}catch(e){log('meta failed: '+e.message)}})();

$('load').onclick=async()=>{try{log('ingesting…');
  const r=await api('/api/jobs',{method:'POST',headers:H,body:JSON.stringify({path:$('path').value})});
  openJob(r.job,r.pages)}catch(e){log('load failed: '+e.message)}};

const drop=$('drop');
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hot')};
drop.ondragleave=()=>drop.classList.remove('hot');
drop.ondrop=async e=>{e.preventDefault();drop.classList.remove('hot');
  const f=e.dataTransfer.files[0];if(!f)return;log('uploading '+f.name+'…');
  const fd=new FormData();fd.append('file',f);
  try{const r=await api('/api/jobs/upload',{method:'POST',headers:{'x-inkedin-token':TOKEN},body:fd});
  openJob(r.job,r.pages)}catch(err){log('upload failed: '+err.message)}};

function openJob(id,n){JOB=id;SEL=new Set([...Array(n).keys()]);$('ctl').style.display='block';
  log(`job ${id}: ${n} page(s)`);renderGrid(n);refresh()}

function renderGrid(n){$('grid').innerHTML='';
  for(let i=0;i<n;i++){const d=document.createElement('div');d.className='pg sel';d.id='pg'+i;
    d.innerHTML=`<img src="/api/jobs/${JOB}/thumb/${i}?token=${TOKEN}"><span class="n">${i+1}</span><span class="st"></span>`;
    d.onclick=()=>{if(SEL.has(i)){SEL.delete(i);d.classList.remove('sel')}else{SEL.add(i);d.classList.add('sel')}};
    $('grid').appendChild(d)}}

$('all').onclick=()=>{document.querySelectorAll('.pg').forEach((d,i)=>{SEL.add(i);d.classList.add('sel')})};
$('none').onclick=()=>{SEL.clear();document.querySelectorAll('.pg').forEach(d=>d.classList.remove('sel'))};

$('run').onclick=async()=>{if(!JOB)return;
  try{await api(`/api/jobs/${JOB}/run`,{method:'POST',headers:H,body:JSON.stringify(
    {pages:[...SEL],mode:$('mode').value,device:$('dev').value,ink_weight:parseFloat($('ink').value)||0.85})});
  log('running…');POLL=setInterval(refresh,1200)}catch(e){log('run failed: '+e.message)}};

$('cancel').onclick=()=>JOB&&api(`/api/jobs/${JOB}/cancel`,{method:'POST',headers:H});

async function refresh(){if(!JOB)return;
  const r=await api(`/api/jobs/${JOB}/pages`,{headers:H});
  let done=0,err=0;
  r.pages.forEach(p=>{const d=$('pg'+p.page);if(!d)return;const st=d.querySelector('.st');
    if(p.status==='done'){d.classList.add('done');st.textContent='✓';done++;
      d.querySelector('img').src=`/api/jobs/${JOB}/image/out/${p.page}?token=${TOKEN}`}
    else if(p.status==='error'){st.textContent='✗';err++}else st.textContent=''});
  if(r.progress){const pc=100*r.progress.done/r.progress.total;$('bar').firstElementChild.style.width=pc+'%';
    log(`progress: ${r.progress.done}/${r.progress.total}`+(err?` (${err} failed)`:''));
    if(r.progress.done>=r.progress.total&&POLL){clearInterval(POLL);POLL=null;log(`finished: ${done} ok`+(err?`, ${err} failed`:''))}}}

$('exp').onclick=async()=>{if(!JOB)return;
  try{const r=await api(`/api/jobs/${JOB}/export`,{method:'POST',headers:H,body:JSON.stringify(
    {fmt:$('fmt').value,dest:$('dest').value})});log('exported → '+r.exported)}
  catch(e){log('export failed: '+e.message)}};
</script></body></html>
"""
