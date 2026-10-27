
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import ezdxf
from flask import (
    Blueprint,
    abort,
    jsonify,
    render_template_string,
    request,
    send_file,
)


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "floor_2_2025-10-28_run_2"

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"

CANDIDATES_PATH = (
    ROOT / "outputs/automatic_registration_v2"
    / "registration_candidates.json"
)

OUTPUT_DIR = ROOT / "outputs/registration_validation"
REVIEW_PATH = OUTPUT_DIR / "reviewer_hypotheses.json"

ARCHIVE = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

review_bp = Blueprint(
    "registration_review",
    __name__,
)

MAX_REVIEW_POINTS = 1000


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def find_manifest():
    preferred = (
        ARCHIVE / "hilti_frames" / SEQUENCE
        / "cam0/frame_manifest.csv"
    )

    if preferred.is_file():
        return preferred

    matches = [
        path
        for path in ROOT.rglob("frame_manifest.csv")
        if SEQUENCE in str(path)
        and "cam0" in path.parts
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one walkthrough manifest; "
            f"found {len(matches)}."
        )

    return matches[0]


def load_frames():
    manifest = find_manifest()
    frame_dir = manifest.parent / "frames"

    records = []

    with manifest.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for source_index, row in enumerate(
            csv.DictReader(handle)
        ):
            if str(
                row.get("pose_matched", "")
            ).strip().lower() != "true":
                continue

            x = finite(row.get("tx"))
            y = finite(row.get("ty"))

            if x is None or y is None:
                continue

            filename = Path(
                str(row.get("filename", ""))
            ).name

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            path = frame_dir / filename

            if not path.is_file():
                continue

            records.append({
                "frame_index": source_index,
                "filename": filename,
                "timestamp": finite(
                    row.get("header_timestamp_s")
                ),
                "x": x,
                "y": y,
                "path": path,
            })

    if not records:
        raise RuntimeError(
            "No matched walkthrough photographs found."
        )

    return records


def load_candidates():
    data = json.loads(
        CANDIDATES_PATH.read_text(
            encoding="utf-8"
        )
    )

    candidates = []

    for item in data.get("candidates", []):
        angle = finite(item.get("angle_deg"))
        tx = finite(item.get("tx"))
        ty = finite(item.get("ty"))

        if None in (angle, tx, ty):
            continue

        candidates.append({
            "rank": int(item["rank"]),
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
            "score": finite(item.get("score")),
        })

    if not candidates:
        raise RuntimeError(
            "No usable registration candidates."
        )

    return sorted(
        candidates,
        key=lambda item: item["rank"],
    )


def load_cad_segments():
    if not DXF_PATH.is_file():
        raise FileNotFoundError(DXF_PATH)

    doc = ezdxf.readfile(DXF_PATH)
    segments = []

    # Restrict the review map to architectural
    # wall geometry to avoid excessive DXF detail.
    preferred_layer = "010 Wände"

    entities = list(doc.modelspace())

    selected = [
        entity for entity in entities
        if entity.dxf.layer == preferred_layer
    ]

    if not selected:
        # Fallback to all drawable geometry.
        selected = entities

    for entity in selected:
        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end

            segments.append([
                [float(a.x), float(a.y)],
                [float(b.x), float(b.y)],
            ])

        elif kind == "LWPOLYLINE":
            points = [
                [float(p[0]), float(p[1])]
                for p in entity.get_points("xy")
            ]

            for a, b in zip(
                points[:-1], points[1:]
            ):
                segments.append([a, b])

            if entity.closed and len(points) > 2:
                segments.append([
                    points[-1], points[0]
                ])

    segments = [
        segment
        for segment in segments
        if all(
            math.isfinite(value)
            for point in segment
            for value in point
        )
    ]

    if not segments:
        raise RuntimeError(
            "No drawable CAD segments found."
        )

    return segments


def transform(x, y, candidate):
    theta = math.radians(
        candidate["angle_deg"]
    )

    c = math.cos(theta)
    s = math.sin(theta)

    return [
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    ]


def cad_bounds(segments):
    points = [
        point
        for segment in segments
        for point in segment
    ]

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    margin = 1.5

    return {
        "min_x": min(xs) - margin,
        "max_x": max(xs) + margin,
        "min_y": min(ys) - margin,
        "max_y": max(ys) + margin,
    }


def load_reviews():
    if not REVIEW_PATH.exists():
        return []

    data = json.loads(
        REVIEW_PATH.read_text(encoding="utf-8")
    )

    return data.get("points", [])


def save_reviews(points):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": "1.0",
        "status": "REVIEWER_HYPOTHESES_ONLY",
        "registration_verified": False,
        "points": points,
    }

    # Atomic replacement prevents partial JSON writes.
    fd, temporary = tempfile.mkstemp(
        dir=str(OUTPUT_DIR),
        prefix=".review_",
        suffix=".json",
    )

    try:
        with os.fdopen(
            fd, "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2)

        os.replace(temporary, REVIEW_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@review_bp.get("/api/registration-review/data")
def registration_review_data():
    try:
        frames = load_frames()
        candidates = load_candidates()
        segments = load_cad_segments()

        frame_data = []

        for record in frames:
            frame_data.append({
                "frame_index": record["frame_index"],
                "filename": record["filename"],
                "timestamp": record["timestamp"],
                "x": record["x"],
                "y": record["y"],
            })

        return jsonify({
            "status": "UNVERIFIED",
            "registration_verified": False,
            "frames": frame_data,
            "candidates": candidates,
            "cad_segments": segments,
            "bounds": cad_bounds(segments),
            "reviews": load_reviews(),
        })

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        return jsonify({
            "error": str(error),
        }), 500


@review_bp.get(
    "/api/registration-review/frame/<int:frame_index>"
)
def registration_review_frame(frame_index):
    frames = load_frames()

    record = next(
        (
            item for item in frames
            if item["frame_index"] == frame_index
        ),
        None,
    )

    if record is None:
        abort(404)

    return send_file(
        record["path"],
        mimetype="image/jpeg"
        if record["path"].suffix.lower()
        in (".jpg", ".jpeg")
        else "image/png",
        conditional=True,
    )


@review_bp.post("/api/registration-review/points")
def registration_review_save():
    if not request.is_json:
        return jsonify({
            "error": "JSON_REQUIRED",
        }), 415

    body = request.get_json(silent=True) or {}

    try:
        frame_index = int(body["frame_index"])
        candidate_rank = int(body["candidate_rank"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "INVALID_FRAME_OR_CANDIDATE",
        }), 400

    cad_x = finite(body.get("cad_x"))
    cad_y = finite(body.get("cad_y"))

    if cad_x is None or cad_y is None:
        return jsonify({
            "error": "INVALID_CAD_COORDINATES",
        }), 400

    frames = load_frames()
    candidates = load_candidates()

    frame = next(
        (
            item for item in frames
            if item["frame_index"] == frame_index
        ),
        None,
    )

    candidate = next(
        (
            item for item in candidates
            if item["rank"] == candidate_rank
        ),
        None,
    )

    if frame is None or candidate is None:
        return jsonify({
            "error": "UNKNOWN_FRAME_OR_CANDIDATE",
        }), 400

    bounds = cad_bounds(load_cad_segments())

    if not (
        bounds["min_x"] <= cad_x <= bounds["max_x"]
        and bounds["min_y"] <= cad_y <= bounds["max_y"]
    ):
        return jsonify({
            "error": "POINT_OUTSIDE_CAD_MAP",
        }), 400

    note = str(body.get("note", "")).strip()[:500]

    predicted = transform(
        frame["x"],
        frame["y"],
        candidate,
    )

    residual = math.hypot(
        predicted[0] - cad_x,
        predicted[1] - cad_y,
    )

    points = load_reviews()

    if len(points) >= MAX_REVIEW_POINTS:
        return jsonify({
            "error": "REVIEW_POINT_LIMIT_REACHED",
        }), 409

    point = {
        "id": max(
            (
                int(item.get("id", 0))
                for item in points
            ),
            default=0,
        ) + 1,
        "frame_index": frame_index,
        "filename": frame["filename"],
        "candidate_rank": candidate_rank,
        "cad_x": cad_x,
        "cad_y": cad_y,
        "predicted_cad_x": predicted[0],
        "predicted_cad_y": predicted[1],
        "residual_m": round(residual, 4),
        "note": note,
        "status": "REVIEWER_HYPOTHESIS",
        "independently_verified": False,
    }

    points.append(point)
    save_reviews(points)

    return jsonify({
        "saved": True,
        "point": point,
        "registration_verified": False,
    })


@review_bp.get("/registration-review")
def registration_review_page():
    return render_template_string(PAGE_HTML)


PAGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>Registration Review | Under-construction</title>
<style>
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #0c1118;
  color: #e6edf5;
  font: 14px system-ui, sans-serif;
}
header {
  padding: 18px 24px;
  border-bottom: 1px solid #293746;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
header strong { font-size: 20px; }
header a { color: #91bce0; text-decoration: none; }
main {
  display: grid;
  grid-template-columns: minmax(400px,1fr)
                         minmax(350px,1fr);
  gap: 18px;
  padding: 18px;
}
section {
  background: #151e29;
  border: 1px solid #293746;
  border-radius: 12px;
  padding: 16px;
  min-width: 0;
}
h2 { margin: 0 0 12px; font-size: 17px; }
small, .muted { color: #9fb1c4; }
select, input, button {
  background: #202e3d;
  color: #e6edf5;
  border: 1px solid #42556b;
  border-radius: 7px;
  padding: 9px;
}
button { cursor: pointer; }
button:hover { background: #30445a; }
.controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
#map {
  width: 100%;
  aspect-ratio: 1;
  display: block;
  background: #0d141d;
  border-radius: 8px;
  cursor: crosshair;
}
#photo {
  display: block;
  max-width: 100%;
  max-height: 540px;
  margin: 10px auto;
  border-radius: 8px;
}
#frameSlider { width: 100%; }
#status { color: #f2bc79; margin: 12px 0; }
#reviewList {
  max-height: 220px;
  overflow: auto;
  line-height: 1.8;
}
.note { width: 100%; margin: 10px 0; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; }
.dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-right: 5px;
}
@media(max-width:900px) {
  main { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <strong>Under-<span style="color:#ff704f">
    construction</span> / Registration Review</strong>
  <a href="/twin">Back to 3D model</a>
</header>
<main>
<section>
  <h2>CAD floor plan</h2>
  <div class="controls">
    <label>Candidate
      <select id="candidate"></select>
    </label>
    <span id="candidateInfo" class="muted"></span>
  </div>
  <svg id="map" xmlns="http://www.w3.org/2000/svg"
       viewBox="0 0 700 700"></svg>
  <p class="legend">
    <span><i class="dot"
      style="background:#8a9aaa"></i>CAD walls</span>
    <span><i class="dot"
      style="background:#50baff"></i>Candidate route</span>
    <span><i class="dot"
      style="background:#ffb45a"></i>Selected frame</span>
    <span><i class="dot"
      style="background:#62d9a0"></i>Proposed point</span>
  </p>
  <small>
    Click the map only when you can independently identify
    the camera's approximate location from the photograph.
    The candidate trajectory is a hypothesis, not ground truth.
  </small>
</section>
<section>
  <h2>Walkthrough photograph</h2>
  <div class="controls">
    <button id="previous">Previous</button>
    <button id="next">Next</button>
    <span id="frameLabel"></span>
  </div>
  <input type="range" id="frameSlider"
         min="0" max="0" value="0">
  <img id="photo" alt="Walkthrough frame">
  <div id="status">Loading registration data...</div>
  <div id="selection" class="muted">
    No reviewer point selected.
  </div>
  <input id="note" class="note"
         placeholder="Why is this camera location identifiable?">
  <button id="save" disabled>
    Save reviewer hypothesis
  </button>
  <h2 style="margin-top:22px">Saved hypotheses</h2>
  <div id="reviewList" class="muted">
    None yet.
  </div>
  <p class="muted">
    Saved points are not automatically accepted as verified
    correspondences. Registration and construction status
    remain unknown.
  </p>
</section>
</main>
<script>
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

let data = null;
let framePosition = 0;
let proposed = null;

const $ = id => document.getElementById(id);

function svgElement(tag, attributes) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function candidate() {
  return data.candidates.find(
    c => c.rank === Number($("candidate").value)
  );
}

function frame() {
  return data.frames[framePosition];
}

function transform(point, c) {
  const a = c.angle_deg * Math.PI / 180;
  return [
    Math.cos(a) * point[0]
      - Math.sin(a) * point[1] + c.tx,
    Math.sin(a) * point[0]
      + Math.cos(a) * point[1] + c.ty
  ];
}

function project(x, y) {
  const b = data.bounds;
  const spanX = b.max_x - b.min_x;
  const spanY = b.max_y - b.min_y;
  const scale = 640 / Math.max(spanX, spanY);
  return [
    350 + (x - (b.min_x + b.max_x)/2) * scale,
    350 - (y - (b.min_y + b.max_y)/2) * scale
  ];
}

function unproject(px, py) {
  const b = data.bounds;
  const spanX = b.max_x - b.min_x;
  const spanY = b.max_y - b.min_y;
  const scale = 640 / Math.max(spanX, spanY);
  return [
    (px - 350) / scale + (b.min_x + b.max_x)/2,
    (350 - py) / scale + (b.min_y + b.max_y)/2
  ];
}

function drawMap() {
  if (!data) return;

  const svg = $("map");
  svg.replaceChildren();

  for (const segment of data.cad_segments) {
    const a = project(...segment[0]);
    const b = project(...segment[1]);
    svg.appendChild(svgElement("line", {
      x1:a[0], y1:a[1], x2:b[0], y2:b[1],
      stroke:"#758393", "stroke-width":1
    }));
  }

  const c = candidate();

  const route = data.frames.map(f =>
    project(...transform([f.x, f.y], c))
  );

  svg.appendChild(svgElement("polyline", {
    points:route.map(p => p.join(",")).join(" "),
    fill:"none", stroke:"#50baff",
    "stroke-width":2, opacity:0.9
  }));

  const f = frame();
  const predicted = project(
    ...transform([f.x, f.y], c)
  );

  svg.appendChild(svgElement("circle", {
    cx:predicted[0], cy:predicted[1],
    r:7, fill:"#ffb45a",
    stroke:"#ffffff", "stroke-width":1.5
  }));

  for (const point of data.reviews) {
    if (point.candidate_rank !== c.rank) continue;
    const p = project(point.cad_x, point.cad_y);
    svg.appendChild(svgElement("circle", {
      cx:p[0], cy:p[1], r:5,
      fill:"#62d9a0"
    }));
  }

  if (proposed) {
    const p = project(...proposed);
    svg.appendChild(svgElement("circle", {
      cx:p[0], cy:p[1], r:8,
      fill:"none", stroke:"#62d9a0",
      "stroke-width":3
    }));
  }
}

function updateFrame() {
  if (!data) return;

  const f = frame();
  $("frameSlider").value = framePosition;
  $("frameLabel").textContent =
    `${framePosition + 1}/${data.frames.length} · ${f.filename}`;

  $("photo").src =
    `/api/registration-review/frame/${f.frame_index}`;

  proposed = null;
  $("save").disabled = true;
  $("selection").textContent =
    "No reviewer point selected.";

  drawMap();
}

function renderReviews() {
  const list = $("reviewList");
  list.replaceChildren();

  if (!data.reviews.length) {
    list.textContent = "None yet.";
    return;
  }

  for (const point of data.reviews) {
    const div = document.createElement("div");
    div.textContent =
      `Frame ${point.frame_index} · ` +
      `Candidate #${point.candidate_rank} · ` +
      `(${point.cad_x.toFixed(2)}, ` +
      `${point.cad_y.toFixed(2)}) · ` +
      `Residual ${point.residual_m.toFixed(2)} m · ` +
      `UNVERIFIED`;
    list.appendChild(div);
  }
}

async function savePoint() {
  if (!proposed) return;

  const body = {
    frame_index:frame().frame_index,
    candidate_rank:candidate().rank,
    cad_x:proposed[0],
    cad_y:proposed[1],
    note:$("note").value
  };

  $("save").disabled = true;

  try {
    const response = await fetch(
      "/api/registration-review/points",
      {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body)
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.error || "Unable to save point"
      );
    }

    data.reviews.push(result.point);
    $("status").textContent =
      "Reviewer hypothesis saved. Registration remains unverified.";
    $("note").value = "";
    proposed = null;
    renderReviews();
    drawMap();
  } catch (error) {
    $("status").textContent = error.message;
    $("save").disabled = false;
  }
}

async function initialize() {
  try {
    const response = await fetch(
      "/api/registration-review/data"
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Data load failed");
    }

    data = result;

    for (const c of data.candidates) {
      const option = document.createElement("option");
      option.value = c.rank;
      option.textContent = `Candidate #${c.rank}`;
      $("candidate").appendChild(option);
    }

    $("frameSlider").max = data.frames.length - 1;

    $("candidate").addEventListener("change", () => {
      proposed = null;
      $("save").disabled = true;
      const c = candidate();
      $("candidateInfo").textContent =
        `${c.angle_deg}° · score ` +
        `${c.score === null ? "N/A" : c.score.toFixed(3)}`;
      drawMap();
    });

    $("previous").onclick = () => {
      framePosition = Math.max(0, framePosition - 1);
      updateFrame();
    };

    $("next").onclick = () => {
      framePosition = Math.min(
        data.frames.length - 1,
        framePosition + 1
      );
      updateFrame();
    };

    $("frameSlider").oninput = event => {
      framePosition = Number(event.target.value);
      updateFrame();
    };

    $("map").addEventListener("click", event => {
      const rect = $("map").getBoundingClientRect();

      const px = (event.clientX - rect.left)
        * 700 / rect.width;
      const py = (event.clientY - rect.top)
        * 700 / rect.height;

      proposed = unproject(px, py);

      $("selection").textContent =
        `Proposed CAD camera position: ` +
        `(${proposed[0].toFixed(3)}, ` +
        `${proposed[1].toFixed(3)}) metres`;

      $("save").disabled = false;
      drawMap();
    });

    $("save").onclick = savePoint;

    $("candidate").dispatchEvent(
      new Event("change")
    );

    updateFrame();
    renderReviews();

    $("status").textContent =
      "Registration unverified. Review photographs before saving points.";
  } catch (error) {
    $("status").textContent =
      "Error: " + error.message;
  }
}

initialize();
</script>
</body>
</html>
"""
