
from __future__ import annotations

import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import cv2
import ezdxf
from flask import Flask, abort, jsonify, render_template_string, request, Response


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237" / "data"
SEQUENCE = "floor_2_2025-10-28_run_2"

FRAMES_DIR = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frames"
MANIFEST = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frame_manifest.csv"
DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"
CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)

app = Flask(__name__)
frames = sorted(FRAMES_DIR.glob("*.jpg"))


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_positions():
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        records = {
            row["filename"]: row
            for row in csv.DictReader(file)
        }

    output = []

    for index, path in enumerate(frames):
        record = records.get(path.name, {})
        x = finite(record.get("tx"))
        y = finite(record.get("ty"))
        z = finite(record.get("tz"))

        matched = (
            str(record.get("pose_matched", "")).lower() == "true"
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


def load_candidates():
    if not CANDIDATES_PATH.exists():
        return []

    data = read_json(CANDIDATES_PATH)
    result = []

    for candidate in data.get("candidates", []):
        try:
            item = {
                "rank": int(candidate["rank"]),
                "angle_deg": float(candidate["angle_deg"]),
                "tx": float(candidate["tx"]),
                "ty": float(candidate["ty"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if all(math.isfinite(v) for v in item.values()):
            result.append(item)

    return sorted(result, key=lambda c: c["rank"])


def load_elements():
    """
    Build inspectable CAD LINE and LWPOLYLINE elements.

    These are CAD drawing primitives, not guaranteed
    to correspond one-to-one with physical building
    elements.
    """
    if not DXF_PATH.exists():
        raise FileNotFoundError(DXF_PATH)

    doc = ezdxf.readfile(DXF_PATH)
    elements = []

    for entity in doc.modelspace():
        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            points = [
                [float(a.x), float(a.y)],
                [float(b.x), float(b.y)],
            ]

        elif kind == "LWPOLYLINE":
            points = [
                [float(p[0]), float(p[1])]
                for p in entity.get_points("xy")
            ]
        else:
            continue

        if len(points) < 2:
            continue

        if not all(
            math.isfinite(value)
            for point in points
            for value in point
        ):
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        length = sum(
            math.dist(a, b)
            for a, b in zip(points[:-1], points[1:])
        )

        if kind == "LWPOLYLINE" and entity.closed:
            length += math.dist(points[-1], points[0])

        if length < 0.02:
            continue

        layer = str(entity.dxf.layer)

        elements.append({
            "id": f"cad-{len(elements):05d}",
            "handle": str(entity.dxf.handle),
            "type": kind,
            "layer": layer,
            "points": points,
            "closed": bool(
                kind == "LWPOLYLINE" and entity.closed
            ),
            "center": [cx, cy],
            "length_m": round(length, 3),
            "status": "CAD_REFERENCE_ONLY",
        })

    return elements


positions = load_positions()
candidates = load_candidates()
elements = load_elements()
element_lookup = {e["id"]: e for e in elements}


def transform(x, y, candidate):
    angle = math.radians(candidate["angle_deg"])
    c = math.cos(angle)
    s = math.sin(angle)

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def point_segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy

    if denominator < 1e-12:
        return math.dist(point, a)

    t = max(
        0.0,
        min(
            1.0,
            ((px - ax) * dx + (py - ay) * dy)
            / denominator,
        ),
    )

    closest = (ax + t * dx, ay + t * dy)
    return math.dist(point, closest)


def distance_to_element(point, element):
    points = element["points"]

    segments = list(zip(points[:-1], points[1:]))

    if element["closed"] and len(points) > 2:
        segments.append((points[-1], points[0]))

    return min(
        point_segment_distance(point, a, b)
        for a, b in segments
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

        distance = distance_to_element(
            (x, y),
            element,
        )

        matches.append({
            "index": position["index"],
            "filename": position["filename"],
            "distance_m": round(distance, 3),
            "candidate_x": round(x, 3),
            "candidate_y": round(y, 3),
            "evidence_status": "PROXIMITY_ONLY",
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
    if not elements:
        return jsonify({
            "error": "No CAD primitives available."
        }), 503

    xs = [
        p[0]
        for e in elements
        for p in e["points"]
    ]
    ys = [
        p[1]
        for e in elements
        for p in e["points"]
    ]

    layers = {}

    for element in elements:
        layer = element["layer"]
        layers[layer] = layers.get(layer, 0) + 1

    return jsonify({
        "elements": elements,
        "layers": layers,
        "candidates": candidates,
        "frames": len(frames),
        "matched_frames": sum(
            p["matched"] for p in positions
        ),
        "bounds": {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
        },
        "registration": "UNVERIFIED",
        "element_model": "CAD_DRAWING_PRIMITIVES",
    })


@app.route("/api/element/<element_id>")
def api_element(element_id):
    element = element_lookup.get(element_id)

    if element is None:
        abort(404)

    rank = request.args.get("candidate", type=int)

    candidate = next(
        (c for c in candidates if c["rank"] == rank),
        None,
    )

    result = {
        "element": element,
        "registration": "UNVERIFIED",
        "construction_status": "UNKNOWN",
        "visibility_verified": False,
        "nearby_frames": [],
    }

    if candidate is not None:
        result["candidate"] = candidate
        result["nearby_frames"] = nearby_frames(
            element,
            candidate,
        )

    return jsonify(result)


@app.route("/api/frame/<int:index>")
def api_frame(index):
    if not 0 <= index < len(frames):
        abort(404)

    image = cv2.imread(str(frames[index]))

    if image is None:
        abort(500)

    height, width = image.shape[:2]

    max_width = 1000

    if width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (
                max_width,
                max(1, round(height * scale)),
            ),
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Under-construction | Element Inspector</title>
<style>
:root {
  --bg:#101317;
  --panel:#191e24;
  --panel2:#232a32;
  --line:#343d47;
  --text:#eef0f2;
  --muted:#aab3bc;
  --accent:#ff8a43;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:Inter,Arial,sans-serif;
}
header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:16px;
  padding:20px 28px;
  border-bottom:1px solid var(--line);
}
.brand {
  font-size:23px;
  font-weight:750;
  letter-spacing:-1px;
}
.brand span { color:var(--accent); }
.sub { color:var(--muted); font-size:12px; }
main {
  display:grid;
  grid-template-columns:minmax(0,1.4fr) minmax(330px,1fr);
  gap:18px;
  padding:22px;
}
.panel {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  overflow:hidden;
}
.panel-title {
  padding:17px 20px;
  border-bottom:1px solid var(--line);
  font-weight:650;
  font-size:14px;
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:center;
}
.toolbar {
  display:flex;
  gap:10px;
  padding:14px 18px;
  flex-wrap:wrap;
  border-bottom:1px solid var(--line);
}
select,input {
  color:var(--text);
  background:var(--panel2);
  border:1px solid #4b5662;
  border-radius:8px;
  padding:10px;
  min-width:0;
}
select { max-width:100%; }
#layerSelect { flex:1; }
.canvas-wrap {
  position:relative;
  height:650px;
  background:#eff1f2;
}
canvas {
  display:block;
  width:100%;
  height:100%;
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
  font-weight:750;
  letter-spacing:1.2px;
  text-transform:uppercase;
}
h2 {
  margin:10px 0 12px;
  font-size:22px;
  overflow-wrap:anywhere;
}
.detail {
  display:grid;
  grid-template-columns:110px 1fr;
  gap:12px;
  padding:9px 0;
  border-bottom:1px solid var(--line);
  font-size:13px;
}
.detail span:first-child { color:var(--muted); }
.note {
  padding:14px 20px;
  font-size:12px;
  line-height:1.65;
  color:var(--muted);
  border-top:1px solid var(--line);
}
.warning { color:#ffb47f; }
.photo {
  width:100%;
  max-height:320px;
  object-fit:contain;
  background:#080a0d;
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
  max-height:250px;
  overflow-y:auto;
}
.thumb {
  cursor:pointer;
  background:var(--panel2);
  border:1px solid var(--line);
  border-radius:8px;
  overflow:hidden;
  color:var(--muted);
  text-align:left;
  padding:0;
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
  padding:7px;
  font-size:11px;
}
.empty {
  padding:24px;
  color:var(--muted);
  font-size:13px;
  line-height:1.6;
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
  <div class="sub">ELEMENT INSPECTOR · EXPERIMENTAL ALIGNMENT</div>
</header>

<main>
  <section class="panel">
    <div class="panel-title">
      <span>Building floor plan · Level 2</span>
      <span class="sub" id="elementCount">Loading...</span>
    </div>
    <div class="toolbar">
      <select id="layerSelect">
        <option value="all">All CAD layers</option>
      </select>
      <select id="candidateSelect">
        <option value="none">No alignment</option>
      </select>
    </div>
    <div class="canvas-wrap">
      <canvas id="cadCanvas"></canvas>
    </div>
    <div class="note">
      Click a CAD drawing primitive to inspect its geometry
      and find spatially nearby walkthrough frames.
      Camera-to-CAD registration is unverified.
    </div>
  </section>

  <div class="side">
    <section class="panel">
      <div class="panel-title">
        <span>Selected CAD element</span>
        <span class="sub">Reference geometry</span>
      </div>
      <div id="elementDetails" class="empty">
        Select a line or polyline on the floor plan.
      </div>
    </section>

    <section class="panel">
      <div class="panel-title">
        <span>Nearby site photographs</span>
        <span class="sub warning">PROXIMITY ONLY</span>
      </div>
      <div id="photoContainer" class="empty">
        Select an element and an alignment candidate.
      </div>
      <div id="gallery" class="gallery"></div>
      <div class="note">
        Nearby does not mean visible. These photographs
        are not verified evidence of construction progress.
        Element completion remains unknown.
      </div>
    </section>
  </div>
</main>

<script>
let data = null;
let selected = null;
let nearby = [];
let requestId = 0;

const canvas = document.getElementById("cadCanvas");
const ctx = canvas.getContext("2d");
const layerSelect = document.getElementById("layerSelect");
const candidateSelect = document.getElementById("candidateSelect");

let projector = null;

function makeProjector() {
  const b = data.bounds;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = 30;

  const dx = Math.max(0.001,b.max_x-b.min_x);
  const dy = Math.max(0.001,b.max_y-b.min_y);
  const scale = Math.min(
    (w-2*pad)/dx,
    (h-2*pad)/dy
  );

  const ox = (w-dx*scale)/2;
  const oy = (h-dy*scale)/2;

  return {
    project(x,y) {
      return {
        x:ox+(x-b.min_x)*scale,
        y:h-oy-(y-b.min_y)*scale
      };
    },
    scale
  };
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;

  canvas.width = Math.round(rect.width*ratio);
  canvas.height = Math.round(rect.height*ratio);

  ctx.setTransform(ratio,0,0,ratio,0,0);
  draw();
}

function draw() {
  if (!data) return;

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  ctx.fillStyle = "#eff1f2";
  ctx.fillRect(0,0,w,h);

  projector = makeProjector();

  const activeLayer = layerSelect.value;

  for (const element of data.elements) {
    if (
      activeLayer !== "all" &&
      element.layer !== activeLayer
    ) continue;

    const highlighted = (
      selected && selected.id === element.id
    );

    ctx.beginPath();

    element.points.forEach((p,i) => {
      const q = projector.project(p[0],p[1]);

      if (i===0) ctx.moveTo(q.x,q.y);
      else ctx.lineTo(q.x,q.y);
    });

    if (element.closed) ctx.closePath();

    ctx.strokeStyle = highlighted
      ? "#ff6b25"
      : "#929ba3";

    ctx.lineWidth = highlighted ? 3.5 : 0.75;
    ctx.stroke();
  }
}

function distancePointSegment(p,a,b) {
  const dx=b.x-a.x;
  const dy=b.y-a.y;
  const denom=dx*dx+dy*dy;

  if (denom<1e-9) return Math.hypot(p.x-a.x,p.y-a.y);

  const t=Math.max(0,Math.min(1,
    ((p.x-a.x)*dx+(p.y-a.y)*dy)/denom
  ));

  return Math.hypot(
    p.x-(a.x+t*dx),
    p.y-(a.y+t*dy)
  );
}

function selectAt(x,y) {
  if (!data || !projector) return;

  const layer=layerSelect.value;
  let best=null;
  let bestDistance=Infinity;

  for (const element of data.elements) {
    if (layer!=="all" && element.layer!==layer) continue;

    const points=element.points.map(p =>
      projector.project(p[0],p[1])
    );

    for (let i=1;i<points.length;i++) {
      const d=distancePointSegment(
        {x,y},points[i-1],points[i]
      );

      if (d<bestDistance) {
        bestDistance=d;
        best=element;
      }
    }

    if (element.closed && points.length>2) {
      const d=distancePointSegment(
        {x,y},points[points.length-1],points[0]
      );

      if (d<bestDistance) {
        bestDistance=d;
        best=element;
      }
    }
  }

  if (best && bestDistance<=12) {
    selected=best;
    draw();
    loadElement();
  }
}

function detail(label,value) {
  const row=document.createElement("div");
  row.className="detail";

  const a=document.createElement("span");
  const b=document.createElement("span");

  a.textContent=label;
  b.textContent=String(value);

  row.append(a,b);
  return row;
}

async function loadElement() {
  if (!selected) return;

  const version=++requestId;
  const rank=candidateSelect.value;

  const container=document.getElementById("elementDetails");
  container.replaceChildren();

  const kicker=document.createElement("div");
  kicker.className="kicker";
  kicker.textContent="CAD DRAWING REFERENCE";

  const heading=document.createElement("h2");
  heading.textContent=selected.layer;

  container.className="content";
  container.append(kicker,heading);

  container.append(
    detail("Reference",selected.id),
    detail("Entity type",selected.type),
    detail("Length",selected.length_m+" m"),
    detail("Construction","Unknown"),
    detail("Verification","Not inspected")
  );

  const photo=document.getElementById("photoContainer");
  photo.className="empty";
  photo.textContent="Finding nearby camera positions...";

  document.getElementById("gallery").replaceChildren();

  try {
    const response=await fetch(
      `/api/element/${encodeURIComponent(selected.id)}`+
      `?candidate=${encodeURIComponent(rank)}`
    );

    if (!response.ok) throw new Error("HTTP "+response.status);

    const result=await response.json();

    if (version!==requestId) return;

    nearby=result.nearby_frames||[];
    renderGallery();

  } catch(error) {
    if (version!==requestId) return;

    photo.textContent="Unable to load nearby frames.";
    console.error(error);
  }
}

function showPhoto(item) {
  const container=document.getElementById("photoContainer");
  container.className="";

  container.replaceChildren();

  const image=document.createElement("img");
  image.className="photo";
  image.alt="Construction walkthrough frame "+item.index;
  image.src="/api/frame/"+item.index;

  const info=document.createElement("div");
  info.className="photo-info";
  info.textContent=
    `Frame ${item.index+1} · `+
    `candidate distance ${item.distance_m.toFixed(2)} m · `+
    `visibility unverified`;

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
      "No nearby photographs available under the "+
      "selected registration candidate.";
    return;
  }

  for (const item of nearby) {
    const button=document.createElement("button");
    button.className="thumb";
    button.dataset.frame=item.index;

    const img=document.createElement("img");
    img.src="/api/frame/"+item.index;
    img.loading="lazy";
    img.alt="Frame "+item.index;

    const caption=document.createElement("span");
    caption.textContent=
      `#${item.index+1} · ${item.distance_m.toFixed(1)} m`;

    button.append(img,caption);
    button.addEventListener("click",() => showPhoto(item));

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

layerSelect.addEventListener("change",() => {
  selected=null;
  ++requestId;
  draw();

  const details=document.getElementById("elementDetails");
  details.className="empty";
  details.textContent="Select a CAD drawing primitive.";

  const photos=document.getElementById("photoContainer");
  photos.className="empty";
  photos.textContent="Select an element and alignment candidate.";

  document.getElementById("gallery").replaceChildren();
});

candidateSelect.addEventListener("change",() => {
  if (selected) loadElement();
});

window.addEventListener("resize",resize);

async function initialize() {
  try {
    const response=await fetch("/api/data");

    if (!response.ok) throw new Error("HTTP "+response.status);

    data=await response.json();

    document.getElementById("elementCount").textContent=
      data.elements.length+" CAD primitives";

    const layers=Object.entries(data.layers)
      .sort((a,b)=>b[1]-a[1]);

    for (const [name,count] of layers) {
      const option=document.createElement("option");
      option.value=name;
      option.textContent=name+" ("+count+")";
      layerSelect.appendChild(option);
    }

    for (const candidate of data.candidates) {
      const option=document.createElement("option");
      option.value=candidate.rank;
      option.textContent=
        "Experimental candidate #"+candidate.rank;
      candidateSelect.appendChild(option);
    }

    if (data.candidates.length) {
      candidateSelect.value=String(data.candidates[0].rank);
    }

    resize();

  } catch(error) {
    document.getElementById("elementCount").textContent=
      "Data unavailable";
    console.error(error);
  }
}

initialize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not frames:
        raise RuntimeError(
            f"No camera images found: {FRAMES_DIR}"
        )

    if not elements:
        raise RuntimeError(
            "No inspectable CAD primitives found."
        )

    print("\nUNDER-CONSTRUCTION")
    print("Building Element Inspector")
    print("-" * 55)
    print("Frames:", len(frames))
    print("Synchronized frames:", sum(p["matched"] for p in positions))
    print("CAD drawing primitives:", len(elements))
    print("Registration candidates:", len(candidates))
    print("Construction progress: UNKNOWN")
    print("Registration: UNVERIFIED")
    print("\nOpen http://127.0.0.1:5001")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True,
    )
