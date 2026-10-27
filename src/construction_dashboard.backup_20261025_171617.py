
"""
Under-construction — Step 109
Spatial Studio: Dark Edition

Run backend:
    .\\.venv\\Scripts\\python.exe src\\cad_progress_api.py

Run dashboard:
    .\\.venv\\Scripts\\python.exe src\\construction_dashboard.py

Open:
    http://127.0.0.1:5052

Notes:
- Sample 01 uses the real Step 107 CAD and walkthrough API.
- Sample 02 is a placeholder until a second dataset exists.
- Local uploads are previewed in the browser, not processed.
- No CAD/video registration or construction verification is claimed.
"""

import os
import urllib.error
import urllib.request

from flask import Flask, Response, request

app = Flask(__name__)

PORT = int(os.environ.get("DASHBOARD_PORT", "5052"))
BACKEND = os.environ.get(
    "CAD_PROGRESS_API",
    "http://127.0.0.1:5051",
).rstrip("/")


@app.get("/api/<path:path>")
def proxy_api(path):
    """Forward read-only requests to the Step 107 backend."""
    if not (
        path == "project"
        or path == "cad/elements"
        or path.startswith("walkthrough/")
    ):
        return {"error": "Endpoint not permitted"}, 404

    target = BACKEND + "/api/" + path

    if request.query_string:
        target += "?" + request.query_string.decode("ascii")

    try:
        with urllib.request.urlopen(target, timeout=25) as upstream:
            content = upstream.read()
            content_type = upstream.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content, content_type=content_type)

    except urllib.error.HTTPError as exc:
        return Response(
            exc.read(),
            status=exc.code,
            content_type=exc.headers.get(
                "Content-Type", "application/json"
            ),
        )

    except Exception as exc:
        return {
            "error": "Backend unavailable",
            "detail": str(exc),
        }, 502


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>under-construction — Spatial Studio</title>
<style>
:root{
  --bg:#101117;
  --surface:#191a22;
  --surface2:#20212b;
  --surface3:#282936;
  --line:#343542;
  --ink:#f0eef7;
  --muted:#9898aa;
  --violet:#b89aff;
  --orange:#f0784b;
  --green:#80cbb0;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:Inter,"Segoe UI",Arial,sans-serif
}
button,input{font:inherit}
button{cursor:pointer}
button:focus-visible,input:focus-visible{
  outline:2px solid var(--violet);outline-offset:3px
}
button:disabled{opacity:.4;cursor:not-allowed}
a{color:inherit;text-decoration:none}
.hidden{display:none!important}
.mono{
  font-family:"Cascadia Code",Consolas,monospace;
  letter-spacing:.05em
}
.eyebrow{
  font-size:11px;letter-spacing:.17em;
  text-transform:uppercase;color:var(--violet)
}
.topbar{
  height:72px;display:flex;align-items:center;
  justify-content:space-between;padding:0 5%;
  border-bottom:1px solid #292a34
}
.brand{display:flex;align-items:center;gap:12px;font-weight:750}
.brand-mark{
  width:36px;height:36px;border:2px dashed var(--violet);
  border-radius:50%;display:grid;place-items:center;
  color:var(--violet);font-size:20px
}
.nav-right{display:flex;gap:20px;align-items:center}
.nav-right a{font-size:13px;color:#c8c5d2}
.primary{
  background:var(--violet);color:#17131f;
  border:0;border-radius:9px;padding:12px 17px;
  font-weight:700
}
.secondary{
  background:transparent;color:var(--ink);
  border:1px solid #646272;border-radius:9px;
  padding:12px 17px
}
.primary:hover{background:#cab4ff}
.secondary:hover{background:#272733}
.hero{
  min-height:580px;
  display:grid;grid-template-columns:40% 60%;
  align-items:center;padding:65px 5% 70px;
  overflow:hidden;
  background:
    radial-gradient(ellipse at 82% 58%,#35232e 0%,transparent 43%),
    radial-gradient(ellipse at 53% 38%,#25213a 0%,transparent 42%);
}
.hero-copy{position:relative;z-index:1}
.hero h1{
  font-size:clamp(49px,5.6vw,89px);
  line-height:1.01;letter-spacing:-.065em;
  font-weight:680;margin:23px 0
}
.hero h1 em{font-style:normal;color:var(--violet)}
.hero p{
  max-width:410px;color:#b9b6c5;
  font-size:16px;line-height:1.75
}
.actions{display:flex;gap:12px;margin-top:28px;flex-wrap:wrap}
.hero-art{
  height:450px;position:relative;
  display:grid;place-items:center;
  perspective:1000px
}
.hero-grid{
  position:absolute;inset:7%;
  transform:rotateX(60deg) rotateZ(-30deg);
  background-image:
    linear-gradient(#6d618733 1px,transparent 1px),
    linear-gradient(90deg,#6d618733 1px,transparent 1px);
  background-size:32px 32px;
  mask-image:radial-gradient(ellipse,#000 35%,transparent 75%)
}
.model{
  position:relative;width:360px;height:320px;
  transform-style:preserve-3d;
  transform:rotateX(54deg) rotateZ(-38deg);
  transition:transform .3s ease-out
}
.floor{
  position:absolute;width:270px;height:195px;
  left:45px;top:50px;
  border:2px solid #c2adff;
  background:linear-gradient(130deg,#9a76e42b,#f0784b18);
  box-shadow:0 18px 42px #0005;
  transform-style:preserve-3d
}
.floor:nth-child(1){transform:translateZ(0px)}
.floor:nth-child(2){transform:translateZ(65px)}
.floor:nth-child(3){transform:translateZ(130px)}
.floor:nth-child(4){transform:translateZ(195px)}
.floor:before{
  content:"";position:absolute;
  left:35%;top:0;height:100%;width:2px;
  background:#b89affad
}
.floor:after{
  content:"";position:absolute;
  top:45%;left:0;width:100%;height:2px;
  background:#f0784b99
}
.wall{
  position:absolute;width:120px;height:65px;
  border:2px solid #bda5ff;
  background:#b89aff25;
  transform-origin:bottom;
  transform:rotateX(-90deg)
}
.wall.one{left:0;bottom:0}
.wall.two{right:0;top:0;border-color:var(--orange)}
.model-label{
  position:absolute;right:3%;top:18%;
  color:var(--violet);font-size:11px
}
.model-caption{
  position:absolute;bottom:0;right:5%;
  color:var(--muted);font-size:11px;
  text-align:right;line-height:1.8
}
.hero-divider{
  border-top:1px solid #34303c;
  padding:14px 5%;display:flex;
  justify-content:space-between;color:var(--muted);
  font-size:11px
}
.workspace{
  min-height:780px;
  display:grid;grid-template-columns:235px minmax(0,1fr);
  border-top:1px solid var(--line)
}
.sidebar{
  background:#15161e;border-right:1px solid var(--line);
  padding:24px 14px;display:flex;flex-direction:column;gap:22px
}
.sidebar-heading{
  color:var(--muted);font-size:10px;
  letter-spacing:.18em;margin:0 10px 12px
}
.project{
  display:flex;align-items:center;gap:11px;
  width:100%;padding:12px;border:1px solid transparent;
  border-radius:9px;background:transparent;
  color:var(--ink);text-align:left;margin-bottom:7px
}
.project.active{
  background:#2a2638;border-color:#514469
}
.project:hover{background:#242530}
.project-thumb{
  width:43px;height:43px;flex:none;
  border:1px solid #7e718e;
  background:
    linear-gradient(90deg,transparent 48%,#b89aff80 49%,#b89aff80 51%,transparent 52%),
    linear-gradient(transparent 47%,#b89aff80 48%,#b89aff80 51%,transparent 52%);
  border-radius:4px
}
.project-name{font-size:12px;font-weight:650}
.project-note{font-size:10px;color:var(--muted);margin-top:5px}
.new-project{
  width:100%;padding:12px;border:1px dashed #706b80;
  background:transparent;color:var(--ink);
  border-radius:8px;text-align:left
}
.sidebar-bottom{
  margin-top:auto;padding:12px;
  border-top:1px solid var(--line);
  font-size:10px;line-height:1.8;color:var(--muted)
}
.main{min-width:0;padding:26px}
.workspace-top{
  display:flex;justify-content:space-between;
  align-items:center;gap:14px;margin-bottom:20px
}
.workspace-title{
  font-size:25px;font-weight:680;letter-spacing:-.04em
}
.status{
  color:#f4a47f;font-size:10px;
  border:1px solid #7b4c40;
  border-radius:5px;padding:7px 9px;
  white-space:nowrap
}
.intro{
  margin:12px 0 24px;
  font-size:14px;line-height:1.7;color:var(--muted)
}
.upload-grid{
  display:grid;grid-template-columns:1fr 1fr;
  gap:16px
}
.dropzone{
  min-height:290px;
  border:1px dashed #665c85;
  border-radius:12px;
  background:
    radial-gradient(circle at 50% 30%,#55407930,transparent 50%),
    #1b1b25;
  display:flex;flex-direction:column;
  justify-content:center;align-items:center;
  padding:28px;text-align:center;
  transition:.2s;cursor:pointer
}
.dropzone.video{border-color:#885744}
.dropzone.dragging,.dropzone:hover{
  border-color:var(--violet);
  background-color:#292437;
  transform:translateY(-2px)
}
.dropzone.video.dragging,.dropzone.video:hover{
  border-color:var(--orange)
}
.drop-icon{font-size:39px;color:var(--violet)}
.dropzone.video .drop-icon{color:var(--orange)}
.drop-title{font-size:18px;font-weight:650;margin-top:20px}
.drop-note{font-size:12px;color:var(--muted);margin-top:10px}
.file-name{
  font-size:11px;color:var(--green);
  margin-top:15px;overflow-wrap:anywhere
}
.upload-actions{margin-top:20px;display:flex;gap:12px;align-items:center}
.note{
  margin-top:22px;color:var(--muted);
  font-size:12px;line-height:1.7
}
.viewer-grid{
  display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);
  gap:12px
}
.viewer{
  border:1px solid var(--line);
  background:var(--surface);border-radius:11px;
  overflow:hidden;min-width:0
}
.viewer-head{
  min-height:54px;padding:12px 15px;
  display:flex;align-items:center;justify-content:space-between;
  gap:10px;border-bottom:1px solid var(--line)
}
.viewer-head strong{font-size:12px}
.viewer-tools{display:flex;gap:6px}
.tool{
  background:#292a35;color:var(--ink);
  border:1px solid #42414e;border-radius:6px;
  padding:7px 10px;font-size:11px
}
.cad-stage{
  position:relative;height:410px;overflow:hidden;
  background-color:#14151d;
  background-image:radial-gradient(#5e5c752e 1px,transparent 1px);
  background-size:18px 18px;
  touch-action:none;cursor:grab
}
.cad-stage:active{cursor:grabbing}
#cad-svg{width:100%;height:100%;display:block}
.cad-line{
  fill:none;stroke:#d5d2df;stroke-width:1.35;
  vector-effect:non-scaling-stroke;
  cursor:pointer;pointer-events:stroke
}
.cad-line:hover{stroke:var(--violet);stroke-width:3}
.cad-line.selected{stroke:var(--orange);stroke-width:3.5}
.viewer-foot{
  min-height:48px;padding:13px 15px;
  border-top:1px solid var(--line);
  font-size:11px;color:var(--muted)
}
.video-stage{
  height:410px;background:#111116;
  display:grid;place-items:center;position:relative
}
.video-stage img,.video-stage video{
  width:100%;height:100%;object-fit:contain
}
.video-empty{
  color:var(--muted);font-size:12px;
  text-align:center;padding:20px;line-height:1.8
}
.video-controls{
  display:flex;align-items:center;gap:10px;
  padding:12px 15px;border-top:1px solid var(--line)
}
.video-controls input{width:100%;accent-color:var(--violet)}
.inspector{
  margin-top:14px;background:var(--surface);
  border:1px solid var(--line);border-radius:10px;
  padding:18px;display:flex;justify-content:space-between;
  align-items:flex-start;gap:18px;flex-wrap:wrap
}
.inspector h3{font-size:14px;margin:0 0 9px}
.inspector p{
  font-size:12px;line-height:1.7;
  color:var(--muted);margin:0
}
.inspector-data{
  display:grid;grid-template-columns:repeat(3,minmax(100px,1fr));
  gap:18px;font-size:12px
}
.inspector-data small{
  display:block;color:var(--muted);
  margin-bottom:7px;font-size:10px
}
.notice{
  border-left:2px solid var(--orange);
  background:#f0784b0b;
  padding:12px 15px;margin-top:16px;
  color:#c3b5b2;font-size:11px;line-height:1.7
}
@media(max-width:1100px){
  .hero{grid-template-columns:1fr}
  .hero-art{height:360px}
  .workspace{grid-template-columns:190px minmax(0,1fr)}
  .viewer-grid{grid-template-columns:1fr}
}
@media(max-width:700px){
  .topbar{padding:0 18px}
  .nav-right a{display:none}
  .hero{padding:45px 20px}
  .hero h1{font-size:52px}
  .model{transform:scale(.8) rotateX(54deg) rotateZ(-38deg)}
  .workspace{grid-template-columns:1fr}
  .sidebar{border-right:0;border-bottom:1px solid var(--line)}
  .main{padding:18px}
  .upload-grid{grid-template-columns:1fr}
  .workspace-top{align-items:flex-start;flex-direction:column}
  .inspector-data{grid-template-columns:1fr 1fr}
}
@media(prefers-reduced-motion:reduce){
  *,*:before,*:after{
    scroll-behavior:auto!important;
    transition:none!important;animation:none!important
  }
}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <div class="brand-mark">◒</div>
    <span>under-construction</span>
  </div>
  <nav class="nav-right">
    <a href="#studio">Workspace</a>
    <a href="#studio" id="nav-samples">Samples</a>
    <a href="#studio" class="primary">Get started →</a>
  </nav>
</header>

<section class="hero">
  <div class="hero-copy">
  <div class="eyebrow mono">
    UNDER-CONSTRUCTION / FOLLOW THE BUILD
  </div>

  <h1>
    Be there<br>
    as it <em>takes shape.</em>
  </h1>

  <p>
    Bring your floor plans and site walkthroughs together.
    Explore the building, follow its progress, and see
    how the plans take shape on site.
  </p>

  <div class="actions">
    <button class="primary" id="hero-sample">
      Explore a sample →
    </button>

    <button class="secondary" id="hero-own">
      Start your project
    </button>
  </div>
</div>

  <div class="hero-art" id="hero-art"
       aria-label="Illustrative layered architectural model">
    <div class="hero-grid"></div>
    <div class="model" id="hero-model">
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
    </div>
    <div class="model-label mono">ILLUSTRATIVE / SPATIAL STUDY</div>
    <div class="model-caption mono">
      MOVE TO EXPLORE<br>
      A BUILDING, NOT A CLAIM
    </div>
  </div>
</section>

<div class="hero-divider mono">
  <span>01 / THE IDEA</span>
  <span>↓ OPEN THE WORKSPACE</span>
</div>

<section class="workspace" id="studio">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">◒</div>
      <span style="font-size:12px">under-construction</span>
    </div>

    <div>
      <div class="sidebar-heading mono">SAMPLE PROJECTS</div>

      <button class="project" id="sample-one">
        <span class="project-thumb"></span>
        <span>
          <span class="project-name">Floor 2 Walkthrough</span>
          <span class="project-note" style="display:block">
            Available local dataset
          </span>
        </span>
      </button>

      <button class="project" id="sample-two">
        <span class="project-thumb"></span>
        <span>
          <span class="project-name">Sample 02</span>
          <span class="project-note" style="display:block">
            Dataset not installed
          </span>
        </span>
      </button>
    </div>

    <div>
      <div class="sidebar-heading mono">YOUR PROJECTS</div>
      <button class="new-project" id="new-project">
        + New project
      </button>
    </div>

    <div class="sidebar-bottom mono">
      PLANNED ≠ BUILT<br>
      VERIFICATION IS EARNED.
    </div>
  </aside>

  <main class="main">
    <div class="workspace-top">
      <div>
        <div class="eyebrow mono" id="workspace-eyebrow">
          PROJECT / NEW
        </div>
        <div class="workspace-title" id="workspace-title">
          Start with your files.
        </div>
      </div>
      <span class="status mono" id="workspace-status">
        NOT ANALYZED
      </span>
    </div>

    <div id="upload-view">
      <p class="intro">
        Two views of the same place. Drop in your plan and
        walkthrough to begin. We'll keep the difference
        between uploaded, processed, and verified clear.
      </p>

      <div class="upload-grid">
        <div class="dropzone" id="plan-drop"
             role="button" tabindex="0">
          <div class="drop-icon">▱</div>
          <div class="drop-title">Your floor plan.</div>
          <div class="drop-note">
            Drop a DXF file or click to browse.
          </div>
          <div class="file-name" id="plan-file"></div>
        </div>

        <div class="dropzone video" id="video-drop"
             role="button" tabindex="0">
          <div class="drop-icon">▷</div>
          <div class="drop-title">Your walkthrough.</div>
          <div class="drop-note">
            Drop an MP4, MOV, or WebM video.
          </div>
          <div class="file-name" id="video-file"></div>
        </div>
      </div>

      <input type="file" id="plan-input"
             accept=".dxf" class="hidden">
      <input type="file" id="video-input"
             accept=".mp4,.mov,.webm,video/*" class="hidden">

      <div class="upload-actions">
        <button class="primary" id="open-uploads" disabled>
          Preview selected files →
        </button>
        <span class="mono" id="upload-status"
              style="color:var(--muted);font-size:11px">
          WAITING FOR FILES
        </span>
      </div>

      <p class="note">
        Local file selection does not submit your files to
        the analysis backend. DXF parsing and progress
        verification are not performed in this step.
      </p>
    </div>

    <div id="viewer-view" class="hidden">
      <div class="viewer-grid">
        <div class="viewer">
          <div class="viewer-head">
            <strong>01 / Floor plan</strong>
            <div class="viewer-tools">
              <button class="tool" id="zoom-in">+</button>
              <button class="tool" id="zoom-out">−</button>
              <button class="tool" id="reset-cad">Fit</button>
            </div>
          </div>
          <div class="cad-stage" id="cad-stage">
            <svg id="cad-svg" xmlns="http://www.w3.org/2000/svg">
              <g id="cad-lines"></g>
            </svg>
            <div class="video-empty" id="cad-empty">
              Loading floor plan...
            </div>
          </div>
          <div class="viewer-foot mono" id="cad-footer">
            CAD / LOADING
          </div>
        </div>

        <div class="viewer">
          <div class="viewer-head">
            <strong>02 / Walkthrough</strong>
            <div class="viewer-tools">
              <button class="tool" id="replace-project">
                New project
              </button>
            </div>
          </div>

          <div class="video-stage">
            <img id="frame-image" class="hidden"
                 alt="Walkthrough frame">
            <video id="local-video" class="hidden"
                   controls playsinline></video>
            <div class="video-empty" id="video-empty">
              No walkthrough loaded.
            </div>
          </div>

          <div class="video-controls" id="frame-controls">
            <button class="tool" id="prev-frame">◀</button>
            <input type="range" id="frame-slider"
                   min="0" max="0" value="0">
            <button class="tool" id="next-frame">▶</button>
            <span class="mono" id="frame-counter"
                  style="font-size:10px;white-space:nowrap">
              0 / 0
            </span>
          </div>
        </div>
      </div>

      <div class="inspector">
        <div>
          <h3 id="inspector-title">Element inspector</h3>
          <p id="inspector-description">
            Select a CAD line to inspect its source data.
          </p>
        </div>
        <div class="inspector-data">
          <div>
            <small>CAD layer</small>
            <span id="detail-layer">—</span>
          </div>
          <div>
            <small>Entity</small>
            <span id="detail-type">—</span>
          </div>
          <div>
            <small>Verification</small>
            <span style="color:var(--orange)">
              UNVERIFIED
            </span>
          </div>
        </div>
      </div>

      <div class="notice">
        This workspace shows planned geometry and recorded
        footage. No physical correspondence, CAD registration,
        or construction completion has been verified.
      </div>
    </div>
  </main>
</section>

<script>
"use strict";

const $ = id => document.getElementById(id);

const state = {
  elements: [],
  frames: [],
  frameIndex: 0,
  box: null,
  initialBox: null,
  selected: null,
  planFile: null,
  videoFile: null,
  videoURL: null,
  dragging: null,
  moved: false
};

function setText(id,value){
  $(id).textContent = String(value);
}

function scrollToStudio(){
  $("studio").scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion:reduce)")
      .matches ? "instant" : "smooth"
  });
}

function showUpload(){
  $("upload-view").classList.remove("hidden");
  $("viewer-view").classList.add("hidden");
  $("sample-one").classList.remove("active");
  $("sample-two").classList.remove("active");
  setText("workspace-eyebrow","PROJECT / NEW");
  setText("workspace-title","Start with your files.");
  setText("workspace-status","NOT ANALYZED");
  scrollToStudio();
}

async function getJSON(path){
  const response=await fetch(path);
  if(!response.ok){
    throw new Error(
      `${path}: HTTP ${response.status}`
    );
  }
  return response.json();
}

function getGeometry(element){
  const g=element.geometry;
  if(!g || g.type!=="polyline" ||
     !Array.isArray(g.coordinates)) return null;

  const points=g.coordinates
    .filter(p=>Array.isArray(p)&&p.length>=2)
    .map(p=>[Number(p[0]),Number(p[1])])
    .filter(p=>p.every(Number.isFinite));

  return points.length>=2 ? points : null;
}

function setBox(b){
  state.box={...b};
  $("cad-svg").setAttribute(
    "viewBox",
    [b.x,b.y,b.w,b.h].join(" ")
  );
}

function renderCAD(){
  const group=$("cad-lines");
  group.replaceChildren();

  let minX=Infinity,minY=Infinity;
  let maxX=-Infinity,maxY=-Infinity;
  let count=0;

  const fragment=document.createDocumentFragment();

  for(const element of state.elements){
    const points=getGeometry(element);
    if(!points)continue;

    const path=document.createElementNS(
      "http://www.w3.org/2000/svg","path"
    );

    const d=points.map((p,i)=>{
      const x=p[0],y=-p[1];
      minX=Math.min(minX,x);
      maxX=Math.max(maxX,x);
      minY=Math.min(minY,y);
      maxY=Math.max(maxY,y);
      return `${i===0?"M":"L"} ${x} ${y}`;
    }).join(" ");

    path.setAttribute(
      "d",d+(element.geometry.closed?" Z":"")
    );
    path.setAttribute("class","cad-line");

    path.addEventListener("click",event=>{
      event.stopPropagation();
      if(state.moved)return;
      group.querySelectorAll(".selected")
        .forEach(n=>n.classList.remove("selected"));
      path.classList.add("selected");
      state.selected=element;

      setText(
        "inspector-title",
        element.element_id||"CAD element"
      );
      setText(
        "inspector-description",
        "Planned CAD geometry. No physical match established."
      );
      setText("detail-layer",element.layer||"—");
      setText(
        "detail-type",
        element.source_entity_type||"—"
      );
    });

    fragment.appendChild(path);
    count++;
  }

  group.appendChild(fragment);
  $("cad-empty").classList.toggle("hidden",count>0);

  if(count){
    const w=Math.max(1,maxX-minX);
    const h=Math.max(1,maxY-minY);
    const pad=Math.max(w,h)*.07;
    const b={
      x:minX-pad,y:minY-pad,
      w:w+2*pad,h:h+2*pad
    };
    state.initialBox=b;
    setBox(b);
  }

  setText(
    "cad-footer",
    `${count} drawable elements / ${state.elements.length} loaded`
  );
}

function zoom(factor,cx,cy){
  if(!state.box)return;
  const b=state.box;
  cx??=b.x+b.w/2;
  cy??=b.y+b.h/2;
  setBox({
    x:cx-(cx-b.x)*factor,
    y:cy-(cy-b.y)*factor,
    w:b.w*factor,h:b.h*factor
  });
}

function svgPoint(event){
  const svg=$("cad-svg");
  const matrix=svg.getScreenCTM();
  if(!matrix)return null;
  const p=svg.createSVGPoint();
  p.x=event.clientX;
  p.y=event.clientY;
  return p.matrixTransform(matrix.inverse());
}

function setupCAD(){
  $("zoom-in").onclick=()=>zoom(.8);
  $("zoom-out").onclick=()=>zoom(1.25);
  $("reset-cad").onclick=()=>{
    if(state.initialBox)setBox(state.initialBox);
  };

  const stage=$("cad-stage");

  stage.addEventListener("wheel",event=>{
    if(!state.box)return;
    event.preventDefault();
    const p=svgPoint(event);
    if(p)zoom(
      event.deltaY>0?1.12:.89,p.x,p.y
    );
  },{passive:false});

  stage.addEventListener("pointerdown",event=>{
    if(!state.box)return;
    state.moved=false;
    state.dragging={
      x:event.clientX,y:event.clientY,
      box:{...state.box}
    };
    stage.setPointerCapture(event.pointerId);
  });

  stage.addEventListener("pointermove",event=>{
    if(!state.dragging)return;
    const dx=event.clientX-state.dragging.x;
    const dy=event.clientY-state.dragging.y;
    if(Math.abs(dx)+Math.abs(dy)>5){
      state.moved=true;
    }
    const rect=stage.getBoundingClientRect();
    const b=state.dragging.box;
    setBox({
      x:b.x-dx*b.w/rect.width,
      y:b.y-dy*b.h/rect.height,
      w:b.w,h:b.h
    });
  });

  stage.addEventListener("pointerup",()=>{
    state.dragging=null;
  });
  stage.addEventListener("pointercancel",()=>{
    state.dragging=null;
  });
}

function showFrame(index){
  if(!state.frames.length)return;
  state.frameIndex=Math.max(
    0,Math.min(index,state.frames.length-1)
  );
  const frame=state.frames[state.frameIndex];
  const img=$("frame-image");
  img.src=frame.image_url;
  img.classList.remove("hidden");
  $("video-empty").classList.add("hidden");
  $("frame-slider").value=state.frameIndex;
  setText(
    "frame-counter",
    `${state.frameIndex+1} / ${state.frames.length}`
  );
}

async function loadFrames(){
  const result=[];
  let offset=0;

  for(let page=0;page<30;page++){
    const response=await getJSON(
      `/api/walkthrough/frames?offset=${offset}&limit=250`
    );
    const batch=response.frames||[];
    result.push(...batch);
    offset+=batch.length;
    if(!batch.length || offset>=response.total)break;
  }

  return result;
}

async function openSampleOne(){
  scrollToStudio();
  $("upload-view").classList.add("hidden");
  $("viewer-view").classList.remove("hidden");
  $("sample-one").classList.add("active");
  $("sample-two").classList.remove("active");

  setText("workspace-eyebrow","SAMPLE / LOCAL DATASET");
  setText("workspace-title","Floor 2 Walkthrough");
  setText("workspace-status","UNVERIFIED");
  setText("cad-footer","Loading real CAD geometry...");

  $("frame-image").classList.add("hidden");
  $("local-video").classList.add("hidden");
  $("video-empty").classList.remove("hidden");
  $("frame-controls").classList.remove("hidden");

  try{
    const [cad,frames]=await Promise.all([
      getJSON("/api/cad/elements"),
      loadFrames()
    ]);

    state.elements=cad.elements||[];
    state.frames=frames;
    renderCAD();

    $("frame-slider").max=Math.max(
      0,frames.length-1
    );

    if(frames.length){
      showFrame(0);
    }else{
      setText("video-empty","No recorded frames found.");
    }

  }catch(error){
    console.error(error);
    state.elements=[];
    renderCAD();
    setText(
      "cad-footer",
      "Backend unavailable. Start Step 107 on port 5051."
    );
    setText(
      "video-empty",
      "Could not load walkthrough frames."
    );
  }
}

function openSampleTwo(){
  showUpload();
  $("sample-two").classList.add("active");
  setText("workspace-eyebrow","SAMPLE / NOT INSTALLED");
  setText("workspace-title","Second sample coming soon.");
  setText("workspace-status","NO DATASET");
  setText(
    "upload-status",
    "A SECOND REAL DATASET IS REQUIRED"
  );
}

function bindDrop(zoneId,inputId,kind){
  const zone=$(zoneId);
  const input=$(inputId);

  zone.onclick=()=>input.click();
  zone.onkeydown=event=>{
    if(event.key==="Enter"||event.key===" "){
      event.preventDefault();
      input.click();
    }
  };

  function receive(file){
    if(!file)return;

    const extension=file.name.split(".").pop().toLowerCase();
    const valid=kind==="plan"
      ? ["dxf"].includes(extension)
      : ["mp4","mov","webm"].includes(extension);

    if(!valid){
      alert(`Unsupported ${kind} file type.`);
      return;
    }

    state[kind+"File"]=file;
    setText(
      kind==="plan"?"plan-file":"video-file",
      `SELECTED / ${file.name}`
    );

    $("open-uploads").disabled=!(
      state.planFile&&state.videoFile
    );

    setText(
      "upload-status",
      state.planFile&&state.videoFile
        ?"FILES READY FOR LOCAL PREVIEW"
        :"ONE MORE FILE NEEDED"
    );
  }

  input.onchange=()=>receive(input.files[0]);

  zone.addEventListener("dragover",event=>{
    event.preventDefault();
    zone.classList.add("dragging");
  });

  zone.addEventListener("dragleave",()=>{
    zone.classList.remove("dragging");
  });

  zone.addEventListener("drop",event=>{
    event.preventDefault();
    zone.classList.remove("dragging");
    receive(event.dataTransfer.files[0]);
  });
}

function openLocalPreview(){
  if(!state.planFile||!state.videoFile)return;

  $("upload-view").classList.add("hidden");
  $("viewer-view").classList.remove("hidden");

  setText("workspace-eyebrow","PROJECT / LOCAL PREVIEW");
  setText("workspace-title","Your project");
  setText("workspace-status","NOT PROCESSED");

  state.elements=[];
  renderCAD();
  setText(
    "cad-empty",
    `DXF selected: ${state.planFile.name}. `+
    "CAD parsing is not connected to local uploads yet."
  );
  $("cad-empty").classList.remove("hidden");

  $("frame-image").classList.add("hidden");
  $("frame-controls").classList.add("hidden");
  $("video-empty").classList.add("hidden");

  if(state.videoURL){
    URL.revokeObjectURL(state.videoURL);
  }

  state.videoURL=URL.createObjectURL(
    state.videoFile
  );

  const video=$("local-video");
  video.src=state.videoURL;
  video.classList.remove("hidden");
  video.load();

  setText(
    "inspector-title",
    "Local files selected"
  );
  setText(
    "inspector-description",
    "The video can be previewed. The DXF has not been parsed or analyzed."
  );
}

function setupHero(){
  const art=$("hero-art");
  const model=$("hero-model");
  if(matchMedia("(prefers-reduced-motion:reduce)").matches){
    return;
  }

  art.addEventListener("pointermove",event=>{
    const r=art.getBoundingClientRect();
    const x=(event.clientX-r.left)/r.width-.5;
    const y=(event.clientY-r.top)/r.height-.5;
    model.style.transform=
      `rotateX(${54-y*13}deg) rotateZ(${-38+x*16}deg)`;
  });

  art.addEventListener("pointerleave",()=>{
    model.style.transform=
      "rotateX(54deg) rotateZ(-38deg)";
  });
}

function initialize(){
  setupHero();
  setupCAD();

  bindDrop("plan-drop","plan-input","plan");
  bindDrop("video-drop","video-input","video");

  $("hero-sample").onclick=openSampleOne;
  $("hero-own").onclick=showUpload;
  $("sample-one").onclick=openSampleOne;
  $("sample-two").onclick=openSampleTwo;
  $("new-project").onclick=showUpload;
  $("replace-project").onclick=showUpload;
  $("open-uploads").onclick=openLocalPreview;

  $("prev-frame").onclick=()=>showFrame(
    state.frameIndex-1
  );
  $("next-frame").onclick=()=>showFrame(
    state.frameIndex+1
  );
  $("frame-slider").oninput=event=>showFrame(
    Number(event.target.value)
  );

  $("nav-samples").onclick=event=>{
    event.preventDefault();
    openSampleOne();
  };
}

initialize();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    print("\nUnder-construction — Step 109")
    print("Backend:", BACKEND)
    print(f"Dashboard: http://127.0.0.1:{PORT}")
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        use_reloader=False,
    )
