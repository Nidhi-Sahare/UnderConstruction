
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, abort, request


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237" / "data"
SEQUENCE = "floor_2_2025-10-28_run_2"

ELEMENTS_PATH = ROOT / "data" / "digital_twin" / "building_elements.json"
CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)
FRAMES_DIR = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frames"
MANIFEST_PATH = (
    ARCHIVE / "hilti_frames" / SEQUENCE
    / "cam0" / "frame_manifest.csv"
)

app = Flask(__name__)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_elements():
    data = read_json(ELEMENTS_PATH)
    result = []

    for raw in data.get("elements", []):
        a = raw.get("start")
        b = raw.get("end")

        if not isinstance(a, list) or not isinstance(b, list):
            continue

        if len(a) < 2 or len(b) < 2:
            continue

        coordinates = [finite(v) for v in a[:2] + b[:2]]

        if any(v is None for v in coordinates):
            continue

        x1, y1, x2, y2 = coordinates
        geometry = raw.get("geometry") or {}
        progress = raw.get("progress") or {}

        result.append({
            "id": str(raw["element_id"]),
            "category": str(raw.get("category", "unknown")),
            "start": [x1, y1],
            "end": [x2, y2],
            "length_m": finite(raw.get("length_m")),
            "height_m": finite(geometry.get("height_m")),
            "thickness_m": finite(geometry.get("thickness_m")),
            "height_source": geometry.get("height_source"),
            "thickness_source": geometry.get("thickness_source"),
            "representation": geometry.get("representation"),
            "progress_status": progress.get("status", "unobserved"),
            "progress_confidence": progress.get("confidence"),
            "progress_evidence": progress.get("evidence", []),
        })

    return result


def load_candidates():
    if not CANDIDATES_PATH.exists():
        return []

    result = []

    for item in read_json(CANDIDATES_PATH).get("candidates", []):
        try:
            candidate = {
                "rank": int(item["rank"]),
                "angle_deg": float(item["angle_deg"]),
                "tx": float(item["tx"]),
                "ty": float(item["ty"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if all(math.isfinite(v) for v in candidate.values()):
            result.append(candidate)

    return sorted(result, key=lambda c: c["rank"])


def load_positions(frame_files):
    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        records = {
            row["filename"]: row
            for row in csv.DictReader(file)
        }

    output = []

    for index, path in enumerate(frame_files):
        row = records.get(path.name, {})
        x = finite(row.get("tx"))
        y = finite(row.get("ty"))
        z = finite(row.get("tz"))

        matched = (
            str(row.get("pose_matched", "")).lower() == "true"
            and all(v is not None for v in (x, y, z))
        )

        output.append({
            "index": index,
            "filename": path.name,
            "matched": matched,
            "x": x if matched else None,
            "y": y if matched else None,
            "z": z if matched else None,
        })

    return output


elements = load_elements()
lookup = {e["id"]: e for e in elements}
candidates = load_candidates()
frames = sorted(FRAMES_DIR.glob("*.jpg"))
positions = load_positions(frames)


def transform(x, y, candidate):
    angle = math.radians(candidate["angle_deg"])
    c = math.cos(angle)
    s = math.sin(angle)

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy

    if denom < 1e-12:
        return math.dist(point, a)

    t = max(
        0.0,
        min(
            1.0,
            ((px - ax) * dx + (py - ay) * dy) / denom,
        ),
    )

    return math.hypot(
        px - (ax + t * dx),
        py - (ay + t * dy),
    )


def nearby_frames(element, candidate, limit=12):
    matches = []

    for position in positions:
        if not position["matched"]:
            continue

        x, y = transform(
            position["x"],
            position["y"],
            candidate,
        )

        distance = segment_distance(
            (x, y),
            element["start"],
            element["end"],
        )

        matches.append({
            "index": position["index"],
            "distance_m": round(distance, 3),
            "status": "PROXIMITY_ONLY",
        })

    matches.sort(
        key=lambda item: (
            item["distance_m"],
            item["index"],
        )
    )

    return matches[:limit]


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    coordinates = [
        point
        for element in elements
        for point in (
            element["start"],
            element["end"],
        )
    ]

    if not coordinates:
        return jsonify({"error": "No elements"}), 503

    categories = {}

    for element in elements:
        category = element["category"]
        categories[category] = categories.get(category, 0) + 1

    return jsonify({
        "elements": elements,
        "categories": categories,
        "candidates": candidates,
        "frame_count": len(frames),
        "matched_frames": sum(p["matched"] for p in positions),
        "bounds": {
            "min_x": min(p[0] for p in coordinates),
            "max_x": max(p[0] for p in coordinates),
            "min_y": min(p[1] for p in coordinates),
            "max_y": max(p[1] for p in coordinates),
        },
        "registration": "UNVERIFIED",
    })


@app.route("/api/element/<element_id>")
def api_element(element_id):
    element = lookup.get(element_id)

    if element is None:
        abort(404)

    rank = request.args.get("candidate", type=int)
    candidate = next(
        (c for c in candidates if c["rank"] == rank),
        None,
    )

    return jsonify({
        "element": element,
        "candidate": candidate,
        "nearby_frames": (
            nearby_frames(element, candidate)
            if candidate else []
        ),
        "registration": "UNVERIFIED",
        "visibility_verified": False,
    })


@app.route("/api/frame/<int:index>")
def api_frame(index):
    if not 0 <= index < len(frames):
        abort(404)

    image = cv2.imread(str(frames[index]))

    if image is None:
        abort(500)

    h, w = image.shape[:2]

    if w > 960:
        scale = 960 / w
        image = cv2.resize(
            image,
            (960, max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 83],
    )

    if not ok:
        abort(500)

    return Response(
        encoded.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Under-construction | Digital Twin</title>
<style>
:root {
  --bg:#101317;
  --panel:#191e24;
  --panel2:#252c34;
  --line:#353e48;
  --muted:#aab3bc;
  --accent:#ff8a43;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:var(--bg);
  color:#eef0f2;
  font-family:Inter,Arial,sans-serif;
}
header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:20px 28px;
  border-bottom:1px solid var(--line);
  gap:12px;
}
.brand {
  font-size:23px;
  font-weight:750;
  letter-spacing:-1px;
}
.brand span { color:var(--accent); }
.muted { color:var(--muted); font-size:12px; }
.metrics {
  display:flex;
  gap:35px;
  flex-wrap:wrap;
  padding:20px 28px;
}
.metric { color:var(--muted); font-size:12px; }
.metric strong {
  display:block;
  color:white;
  font-size:23px;
  margin-top:5px;
}
main {
  display:grid;
  grid-template-columns:minmax(0,1.4fr) minmax(350px,1fr);
  gap:20px;
  padding:18px 24px 30px;
}
.panel {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  overflow:hidden;
}
.title {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
  padding:17px 20px;
  border-bottom:1px solid var(--line);
  font-weight:650;
  font-size:14px;
}
.toolbar {
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  padding:14px 18px;
  border-bottom:1px solid var(--line);
}
select {
  color:white;
  background:var(--panel2);
  border:1px solid #4b5662;
  border-radius:8px;
  padding:10px;
  max-width:100%;
}
#categorySelect { flex:1; }
.canvas-wrap {
  height:650px;
  background:#eff1f2;
}
canvas {
  width:100%;
  height:100%;
  display:block;
  cursor:crosshair;
}
.side {
  display:flex;
  flex-direction:column;
  gap:18px;
  min-width:0;
}
.content { padding:18px 20px; }
.kicker {
  color:var(--accent);
  font-size:11px;
  letter-spacing:1px;
  font-weight:750;
  text-transform:uppercase;
}
h2 {
  font-size:21px;
  margin:10px 0 14px;
  overflow-wrap:anywhere;
}
.detail {
  display:grid;
  grid-template-columns:130px 1fr;
  gap:12px;
  padding:9px 0;
  border-bottom:1px solid var(--line);
  font-size:13px;
}
.detail span:first-child { color:var(--muted); }
.note {
  color:var(--muted);
  font-size:12px;
  line-height:1.6;
  padding:14px 20px;
  border-top:1px solid var(--line);
}
.warning { color:#ffb47f; }
.photo {
  width:100%;
  max-height:310px;
  object-fit:contain;
  background:#090b0e;
  display:block;
}
.photo-info {
  padding:12px 18px;
  color:var(--muted);
  font-size:12px;
}
.gallery {
  padding:12px;
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:8px;
  max-height:245px;
  overflow-y:auto;
}
.thumb {
  border:1px solid var(--line);
  background:var(--panel2);
  color:var(--muted);
  border-radius:8px;
  overflow:hidden;
  cursor:pointer;
  padding:0;
  text-align:left;
}
.thumb.active { border:2px solid var(--accent); }
.thumb img {
  display:block;
  width:100%;
  aspect-ratio:1.45;
  object-fit:cover;
}
.thumb span {
  display:block;
  font-size:11px;
  padding:7px;
}
.empty {
  color:var(--muted);
  font-size:13px;
  line-height:1.6;
  padding:24px;
}
@media(max-width:950px) {
  main { grid-template-columns:1fr; }
  .canvas-wrap { height:470px; }
}
</style>
</head>
<body>

<header>
  <div class="brand">Under<span>-construction</span></div>
  <div class="muted">DIGITAL TWIN · ELEMENT INSPECTION</div>
</header>

<section class="metrics">
  <div class="metric">
    Building elements<strong id="totalElements">—</strong>
  </div>
  <div class="metric">
    Camera frames<strong id="totalFrames">—</strong>
  </div>
  <div class="metric">
    Synchronized frames<strong id="matchedFrames">—</strong>
  </div>
  <div class="metric">
    CAD registration<strong>Unverified</strong>
  </div>
</section>

<main>
  <section class="panel">
    <div class="title">
      <span>Building elements · Floor 2</span>
      <span class="muted">Click an element</span>
    </div>
    <div class="toolbar">
      <select id="categorySelect">
        <option value="all">All element categories</option>
      </select>
      <select id="candidateSelect">
        <option value="none">Hide alignment</option>
      </select>
    </div>
    <div class="canvas-wrap">
      <canvas id="elementCanvas"></canvas>
    </div>
    <div class="note">
      Geometry is extracted from CAD centerlines.
      Height and thickness may be assumed values.
      Registration candidates are not verified.
    </div>
  </section>

  <div class="side">
    <section class="panel">
      <div class="title">
        <span>Element information</span>
        <span class="muted">Digital twin</span>
      </div>
      <div id="details" class="empty">
        Select a wall on the floor plan.
      </div>
    </section>

    <section class="panel">
      <div class="title">
        <span>Nearby site images</span>
        <span class="muted warning">PROXIMITY ONLY</span>
      </div>
      <div id="photoContainer" class="empty">
        Select an element and an alignment candidate.
      </div>
      <div id="gallery" class="gallery"></div>
      <div class="note">
        Nearby camera positions do not prove visibility
        or completion. These images are candidates for
        further inspection.
      </div>
    </section>
  </div>
</main>

<script>
let data=null;
let selected=null;
let nearby=[];
let requestVersion=0;
let projector=null;

const canvas=document.getElementById("elementCanvas");
const ctx=canvas.getContext("2d");
const categorySelect=document.getElementById("categorySelect");
const candidateSelect=document.getElementById("candidateSelect");

function makeProjector() {
  const b=data.bounds;
  const w=canvas.clientWidth;
  const h=canvas.clientHeight;
  const pad=30;

  const dx=Math.max(.001,b.max_x-b.min_x);
  const dy=Math.max(.001,b.max_y-b.min_y);
  const scale=Math.min((w-2*pad)/dx,(h-2*pad)/dy);

  const ox=(w-dx*scale)/2;
  const oy=(h-dy*scale)/2;

  return {
    project(x,y) {
      return {
        x:ox+(x-b.min_x)*scale,
        y:h-oy-(y-b.min_y)*scale
      };
    }
  };
}

function resize() {
  const rect=canvas.getBoundingClientRect();
  const ratio=window.devicePixelRatio||1;

  canvas.width=Math.round(rect.width*ratio);
  canvas.height=Math.round(rect.height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);

  draw();
}

function draw() {
  if (!data) return;

  const w=canvas.clientWidth;
  const h=canvas.clientHeight;

  ctx.fillStyle="#eff1f2";
  ctx.fillRect(0,0,w,h);

  projector=makeProjector();

  for (const element of data.elements) {
    if (
      categorySelect.value!=="all" &&
      element.category!==categorySelect.value
    ) continue;

    const a=projector.project(...element.start);
    const b=projector.project(...element.end);

    const active=selected && selected.id===element.id;

    ctx.beginPath();
    ctx.moveTo(a.x,a.y);
    ctx.lineTo(b.x,b.y);

    ctx.strokeStyle=active?"#ff6b25":"#85909a";
    ctx.lineWidth=active?4:1.3;
    ctx.stroke();
  }
}

function pointSegmentDistance(p,a,b) {
  const dx=b.x-a.x;
  const dy=b.y-a.y;
  const denominator=dx*dx+dy*dy;

  if (denominator<1e-10)
    return Math.hypot(p.x-a.x,p.y-a.y);

  const t=Math.max(0,Math.min(1,
    ((p.x-a.x)*dx+(p.y-a.y)*dy)/denominator
  ));

  return Math.hypot(
    p.x-(a.x+t*dx),
    p.y-(a.y+t*dy)
  );
}

function selectAt(x,y) {
  if (!data || !projector) return;

  let best=null;
  let bestDistance=Infinity;

  for (const element of data.elements) {
    if (
      categorySelect.value!=="all" &&
      element.category!==categorySelect.value
    ) continue;

    const a=projector.project(...element.start);
    const b=projector.project(...element.end);

    const distance=pointSegmentDistance({x,y},a,b);

    if (distance<bestDistance) {
      bestDistance=distance;
      best=element;
    }
  }

  if (best && bestDistance<=12) {
    selected=best;
    draw();
    loadSelected();
  }
}

function detail(label,value) {
  const row=document.createElement("div");
  row.className="detail";

  const left=document.createElement("span");
  left.textContent=label;

  const right=document.createElement("span");
  right.textContent=String(value);

  row.append(left,right);
  return row;
}

async function loadSelected() {
  if (!selected) return;

  const version=++requestVersion;
  const container=document.getElementById("details");

  container.className="content";
  container.replaceChildren();

  const kicker=document.createElement("div");
  kicker.className="kicker";
  kicker.textContent="BUILDING ELEMENT";

  const title=document.createElement("h2");
  title.textContent=selected.category+" wall";

  container.append(kicker,title);

  container.append(
    detail("Element ID",selected.id),
    detail("Length",selected.length_m?.toFixed(2)+" m"),
    detail("Height",selected.height_m+" m"),
    detail("Thickness",selected.thickness_m+" m"),
    detail("Height source",selected.height_source||"Unknown"),
    detail("Thickness source",selected.thickness_source||"Unknown"),
    detail("Progress",selected.progress_status),
    detail("Confidence",selected.progress_confidence??"Not available")
  );

  const photos=document.getElementById("photoContainer");
  photos.className="empty";
  photos.textContent="Finding nearby synchronized frames...";

  document.getElementById("gallery").replaceChildren();

  try {
    const rank=candidateSelect.value;

    const response=await fetch(
      "/api/element/"+encodeURIComponent(selected.id)+
      "?candidate="+encodeURIComponent(rank)
    );

    if (!response.ok) throw new Error("HTTP "+response.status);

    const result=await response.json();

    if (version!==requestVersion) return;

    nearby=result.nearby_frames||[];
    renderGallery();

  } catch(error) {
    if (version!==requestVersion) return;

    photos.textContent="Unable to retrieve nearby frames.";
    console.error(error);
  }
}

function showPhoto(item) {
  const container=document.getElementById("photoContainer");

  container.className="";
  container.replaceChildren();

  const image=document.createElement("img");
  image.className="photo";
  image.src="/api/frame/"+item.index;
  image.alt="Construction frame "+item.index;

  const info=document.createElement("div");
  info.className="photo-info";
  info.textContent=
    "Frame "+(item.index+1)+
    " · candidate distance "+item.distance_m.toFixed(2)+" m"+
    " · visibility unverified";

  container.append(image,info);

  document.querySelectorAll(".thumb").forEach(button => {
    button.classList.toggle(
      "active",
      Number(button.dataset.frame)===item.index
    );
  });
}

function renderGallery() {
  const gallery=document.getElementById("gallery");
  gallery.replaceChildren();

  if (!nearby.length) {
    const container=document.getElementById("photoContainer");
    container.className="empty";
    container.textContent=
      "No spatially matched photographs available. "+
      "Select an experimental registration candidate.";
    return;
  }

  for (const item of nearby) {
    const button=document.createElement("button");
    button.className="thumb";
    button.dataset.frame=item.index;

    const image=document.createElement("img");
    image.src="/api/frame/"+item.index;
    image.loading="lazy";
    image.alt="Frame "+item.index;

    const caption=document.createElement("span");
    caption.textContent=
      "#"+(item.index+1)+" · "+item.distance_m.toFixed(1)+" m";

    button.append(image,caption);
    button.addEventListener("click",()=>showPhoto(item));
    gallery.appendChild(button);
  }

  showPhoto(nearby[0]);
}

canvas.addEventListener("click",event => {
  const rect=canvas.getBoundingClientRect();

  selectAt(
    event.clientX-rect.left,
    event.clientY-rect.top
  );
});

categorySelect.addEventListener("change",() => {
  selected=null;
  ++requestVersion;
  draw();

  const details=document.getElementById("details");
  details.className="empty";
  details.textContent="Select a building element.";

  const photos=document.getElementById("photoContainer");
  photos.className="empty";
  photos.textContent="Select an element and alignment candidate.";

  document.getElementById("gallery").replaceChildren();
});

candidateSelect.addEventListener("change",() => {
  if (selected) loadSelected();
});

window.addEventListener("resize",resize);

async function initialize() {
  try {
    const response=await fetch("/api/data");

    if (!response.ok) throw new Error("HTTP "+response.status);

    data=await response.json();

    document.getElementById("totalElements").textContent=
      data.elements.length;

    document.getElementById("totalFrames").textContent=
      data.frame_count;

    document.getElementById("matchedFrames").textContent=
      data.matched_frames;

    for (const [category,count] of
      Object.entries(data.categories).sort()
    ) {
      const option=document.createElement("option");
      option.value=category;
      option.textContent=category+" ("+count+")";
      categorySelect.appendChild(option);
    }

    for (const candidate of data.candidates) {
      const option=document.createElement("option");
      option.value=candidate.rank;
      option.textContent=
        "Candidate #"+candidate.rank+" · unverified";
      candidateSelect.appendChild(option);
    }

    if (data.candidates.length) {
      candidateSelect.value=String(data.candidates[0].rank);
    }

    resize();

  } catch(error) {
    console.error(error);
    document.getElementById("totalElements").textContent="Error";
  }
}

initialize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not frames:
        raise RuntimeError(f"No camera frames at {FRAMES_DIR}")

    print("\nUNDER-CONSTRUCTION · DIGITAL TWIN EXPLORER")
    print("Elements:", len(elements))
    print("Frames:", len(frames))
    print("Pose-matched frames:", sum(p["matched"] for p in positions))
    print("Registration candidates:", len(candidates))
    print("Progress status: read from digital twin JSON")
    print("Open http://127.0.0.1:5002")

    app.run(
        host="127.0.0.1",
        port=5002,
        debug=False,
        threaded=True,
    )
