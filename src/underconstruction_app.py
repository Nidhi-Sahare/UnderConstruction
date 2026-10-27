
from __future__ import annotations

import csv
import json
import math
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from progress.yolo_onnx import OnnxSegModel, render_overlay


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237" / "data"
SEQUENCE = "floor_2_2025-10-28_run_2"

FRAMES_DIR = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frames"
MANIFEST = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frame_manifest.csv"
CALIBRATION = ARCHIVE / "hilti_calibration" / "camera_calibration_report.json"
TRAJECTORY = ARCHIVE / "floorplan_registration" / "aligned_trajectory.csv"

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"
CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)

MODEL_PATH = ROOT / "models" / "underconstruction_seg_v0.onnx"
MODEL_CARD = ROOT / "models" / "model_card.json"

app = Flask(__name__)
inference_lock = threading.Lock()
model = None

frames = sorted(FRAMES_DIR.glob("*.jpg"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_calibration():
    data = json.loads(
        CALIBRATION.read_text(encoding="utf-8")
    )["calibration"]

    fx, fy, cx, cy = data["intrinsics"]

    K = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=np.float64,
    )
    D = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64,
    )

    return K, D


K, D = load_calibration()


def load_frame_positions():
    rows = read_csv(MANIFEST)
    by_name = {row["filename"]: row for row in rows}
    positions = []

    for index, path in enumerate(frames):
        row = by_name.get(path.name, {})

        matched = (
            str(row.get("pose_matched", "")).lower() == "true"
        )

        x = number(row.get("tx"))
        y = number(row.get("ty"))
        z = number(row.get("tz"))

        valid = matched and all(
            value is not None for value in (x, y, z)
        )

        positions.append({
            "frame": index,
            "filename": path.name,
            "timestamp": number(
                row.get("header_timestamp_s")
            ),
            "matched": bool(valid),
            "x": x if valid else None,
            "y": y if valid else None,
            "z": z if valid else None,
        })

    return positions


def load_trajectory():
    records = []

    for row in read_csv(TRAJECTORY):
        values = {
            key: number(row.get(key))
            for key in ("x", "y", "z", "timestamp")
        }

        if any(value is None for value in values.values()):
            continue

        records.append(values)

    return sorted(
        records,
        key=lambda record: record["timestamp"],
    )


def load_candidates():
    if not CANDIDATES_PATH.exists():
        app.logger.warning(
            "Registration candidates unavailable: %s",
            CANDIDATES_PATH,
        )
        return []

    data = json.loads(
        CANDIDATES_PATH.read_text(encoding="utf-8")
    )

    if data.get("verified") is True:
        app.logger.warning(
            "Candidate file claims verification; "
            "this application still treats it as unverified."
        )

    result = []

    for item in data.get("candidates", []):
        try:
            candidate = {
                "rank": int(item["rank"]),
                "angle_deg": float(item["angle_deg"]),
                "tx": float(item["tx"]),
                "ty": float(item["ty"]),
                "score": float(item["score"]),
                "inside_fraction": float(
                    item["inside_fraction"]
                ),
                "wall_occupancy_fraction": float(
                    item["wall_occupancy_fraction"]
                ),
                "sampled_crossing_fraction": float(
                    item["sampled_crossing_fraction"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if all(
            math.isfinite(value)
            for value in candidate.values()
        ):
            result.append(candidate)

    return sorted(result, key=lambda item: item["rank"])


def load_cad_geometry():
    if not DXF_PATH.exists():
        raise FileNotFoundError(DXF_PATH)

    doc = ezdxf.readfile(DXF_PATH)
    segments = []

    def add_segment(a, b):
        start = [float(a[0]), float(a[1])]
        end = [float(b[0]), float(b[1])]

        if (
            math.isfinite(start[0])
            and math.isfinite(start[1])
            and math.isfinite(end[0])
            and math.isfinite(end[1])
            and math.dist(start, end) > 1e-5
        ):
            segments.append([start, end])

    for entity in doc.modelspace():
        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            add_segment((a.x, a.y), (b.x, b.y))

        elif kind == "LWPOLYLINE":
            points = [
                (point[0], point[1])
                for point in entity.get_points("xy")
            ]

            for a, b in zip(points[:-1], points[1:]):
                add_segment(a, b)

            if entity.closed and len(points) > 2:
                add_segment(points[-1], points[0])

    if not segments:
        raise RuntimeError("No drawable CAD segments found.")

    points = np.asarray(
        segments,
        dtype=np.float64,
    ).reshape(-1, 2)

    return {
        "coordinate_system": "DXF_XY_METERS",
        "segments": segments,
        "bounds": {
            "min_x": float(points[:, 0].min()),
            "max_x": float(points[:, 0].max()),
            "min_y": float(points[:, 1].min()),
            "max_y": float(points[:, 1].max()),
        },
        "limitations": [
            "Map contains LINE and LWPOLYLINE geometry.",
            "Other DXF entity types are not rendered.",
            "Camera registration remains unverified.",
        ],
    }


frame_positions = load_frame_positions()
trajectory = load_trajectory()
candidates = load_candidates()
cad_geometry = load_cad_geometry()


def get_model():
    global model

    with inference_lock:
        if model is None:
            model = OnnxSegModel(
                model_path=MODEL_PATH,
                card_path=MODEL_CARD,
            )

    return model


@lru_cache(maxsize=4)
def rectification_maps(width, height):
    focal = 0.75 * min(width, height)

    new_K = np.array(
        [
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        new_K,
        (width, height),
        cv2.CV_16SC2,
    )


@lru_cache(maxsize=8)
def load_frame(index, mode):
    if not 0 <= index < len(frames):
        raise IndexError(index)

    image = cv2.imread(str(frames[index]))

    if image is None:
        raise RuntimeError(
            f"Could not read {frames[index]}"
        )

    if mode == "rectified":
        height, width = image.shape[:2]
        map1, map2 = rectification_maps(width, height)

        image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    elif mode != "raw":
        raise ValueError(f"Unknown mode: {mode}")

    return image


def jpeg_response(image):
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not ok:
        raise RuntimeError("JPEG encoding failed.")

    return Response(
        encoded.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


def confidence():
    value = number(request.args.get("conf"))
    return (
        0.45 if value is None
        else max(0.05, min(0.95, value))
    )


def infer(index, mode, conf):
    image = load_frame(index, mode)
    segmenter = get_model()

    with inference_lock:
        detections = segmenter.infer(
            image.copy(),
            conf=conf,
            iou=0.7,
        )

    return image, detections


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/info")
def api_info():
    return jsonify({
        "frames": len(frames),
        "matched_frames": sum(
            position["matched"]
            for position in frame_positions
        ),
        "trajectory_records": len(trajectory),
        "registration": "UNVERIFIED",
        "candidate_count": len(candidates),
    })


@app.route("/api/spatial")
def api_spatial():
    return jsonify({
        "coordinate_system": "HILTI_TRAJECTORY_XY",
        "floorplan_registration": "UNVERIFIED",
        "trajectory": trajectory,
        "frames": frame_positions,
    })


@app.route("/api/registration")
def api_registration():
    return jsonify({
        "status": "CANDIDATE_ONLY",
        "verified": False,
        "source": str(CANDIDATES_PATH.relative_to(ROOT)),
        "candidates": candidates,
        "warning": (
            "Candidate positions are hypothetical. "
            "They are not verified camera locations."
        ),
    })


@app.route("/api/cad")
def api_cad():
    return jsonify(cad_geometry)


@app.route("/api/frame/<int:index>")
def api_frame(index):
    if not 0 <= index < len(frames):
        return jsonify({"error": "Frame not found"}), 404

    mode = request.args.get("mode", "rectified")

    if mode not in ("raw", "rectified"):
        return jsonify({"error": "Invalid mode"}), 400

    try:
        if request.args.get("overlay") == "1":
            image, detections = infer(
                index, mode, confidence()
            )
            image = render_overlay(
                image.copy(), detections
            )
        else:
            image = load_frame(index, mode)

        return jpeg_response(image)

    except Exception:
        app.logger.exception("Frame processing failed")
        return jsonify({
            "error": "Frame processing failed"
        }), 500


@app.route("/api/detections/<int:index>")
def api_detections(index):
    if not 0 <= index < len(frames):
        return jsonify({"error": "Frame not found"}), 404

    mode = request.args.get("mode", "rectified")

    if mode not in ("raw", "rectified"):
        return jsonify({"error": "Invalid mode"}), 400

    try:
        _, detections = infer(
            index, mode, confidence()
        )

        counts = Counter(
            detection.class_name
            for detection in detections
        )

        return jsonify({
            "status": "MODEL_CANDIDATES_ONLY",
            "counts": dict(counts),
            "detections": [
                {
                    "class_name": detection.class_name,
                    "score": float(detection.score),
                    "bbox": list(detection.bbox),
                }
                for detection in detections
            ],
        })

    except Exception:
        app.logger.exception("Detection failed")
        return jsonify({
            "error": "Detection failed"
        }), 500


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Under-construction</title>

<style>
:root {
    --bg:#101317;
    --panel:#191e24;
    --line:#30363d;
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
}
.brand {
    font-size:24px;
    font-weight:750;
    letter-spacing:-1px;
}
.brand span { color:var(--accent); }
.muted { color:var(--muted); font-size:12px; }
.metrics {
    display:flex;
    gap:32px;
    flex-wrap:wrap;
    padding:20px 28px;
}
.metric { color:var(--muted); font-size:12px; }
.metric strong {
    display:block;
    color:white;
    font-size:22px;
    margin-top:5px;
}
main,.lower {
    display:grid;
    grid-template-columns:minmax(0,1.45fr) minmax(340px,1fr);
    gap:20px;
    padding:16px 24px;
}
.lower { padding-top:0; padding-bottom:28px; }
.panel {
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:14px;
    overflow:hidden;
}
.panel-title {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:10px;
    padding:16px 20px;
    font-size:13px;
    font-weight:650;
}
.viewer {
    height:490px;
    background:#090b0e;
    display:flex;
    align-items:center;
    justify-content:center;
}
.viewer img {
    max-width:100%;
    max-height:100%;
    object-fit:contain;
}
.controls {
    padding:16px 20px;
    border-top:1px solid var(--line);
    display:flex;
    flex-wrap:wrap;
    gap:10px;
    align-items:center;
}
button,select {
    background:#303842;
    color:white;
    border:1px solid #46505b;
    padding:10px 14px;
    border-radius:8px;
    cursor:pointer;
}
button.active {
    background:var(--accent);
    color:#171717;
}
button:hover { filter:brightness(1.12); }
input[type=range] { flex:1; min-width:100px; }
.note {
    color:var(--muted);
    font-size:12px;
    line-height:1.6;
    padding:14px 20px;
    border-top:1px solid var(--line);
}
.map-wrap {
    height:490px;
    position:relative;
    background:#f0f0f0;
}
.trajectory-wrap {
    height:420px;
    background:#14191f;
}
canvas {
    display:block;
    width:100%;
    height:100%;
}
#trajectoryCanvas { cursor:crosshair; }
#cadCanvas { cursor:crosshair; }
.legend {
    display:flex;
    flex-wrap:wrap;
    gap:16px;
    padding:12px 20px;
    font-size:12px;
    color:var(--muted);
}
.dot {
    display:inline-block;
    width:9px;
    height:9px;
    border-radius:50%;
    margin-right:6px;
}
.registration-controls {
    display:flex;
    align-items:center;
    flex-wrap:wrap;
    gap:10px;
    padding:12px 20px;
    border-top:1px solid var(--line);
}
.registration-metrics {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:10px;
    padding:0 20px 14px;
}
.registration-metric {
    background:#242b33;
    padding:10px;
    border-radius:8px;
}
.registration-metric span {
    display:block;
    color:var(--muted);
    font-size:11px;
}
.registration-metric strong {
    display:block;
    margin-top:5px;
    font-size:17px;
}
.warning {
    color:#ffb47f;
    font-weight:650;
}
.analysis-status {
    padding:0 20px 16px;
    color:var(--muted);
    font-size:12px;
}
.material-grid {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:12px;
    padding:0 20px 18px;
}
.material {
    padding:14px;
    background:#242b33;
    border:1px solid #36404a;
    border-radius:10px;
}
.material span {
    display:block;
    color:var(--muted);
    font-size:12px;
    text-transform:capitalize;
}
.material strong {
    display:block;
    font-size:25px;
    margin-top:5px;
}
.detections {
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    padding:0 20px 20px;
}
.detection {
    border:1px solid #424c56;
    border-radius:7px;
    padding:8px 10px;
    font-size:12px;
}
.conf-control {
    display:flex;
    gap:8px;
    align-items:center;
    color:var(--muted);
    font-size:12px;
}
.conf-control input { width:110px; }
@media(max-width:950px) {
    main,.lower { grid-template-columns:1fr; }
}
</style>
</head>

<body>

<header>
    <div class="brand">Under<span>-construction</span></div>
    <div class="muted">HILTI · OPENCV 5 · SPATIAL EXPLORER</div>
</header>

<section class="metrics">
    <div class="metric">Camera frames<strong id="totalFrames">—</strong></div>
    <div class="metric">Synchronized frames<strong id="matchedFrames">—</strong></div>
    <div class="metric">Trajectory points<strong id="trajectoryCount">—</strong></div>
    <div class="metric">CAD candidates<strong id="candidateCount">—</strong></div>
    <div class="metric">CAD registration<strong>Unverified</strong></div>
</section>

<main>
    <section class="panel">
        <div class="panel-title">
            <span>Construction camera</span>
            <span id="frameLabel">Loading...</span>
        </div>

        <div class="viewer">
            <img id="cameraImage" alt="Hilti construction footage">
        </div>

        <div class="controls">
            <button onclick="step(-1)">←</button>
            <button onclick="step(1)">→</button>

            <input id="frameSlider" type="range"
                   min="0" max="0" value="0">

            <button id="rawButton"
                    onclick="setMode('raw')">Fisheye</button>

            <button id="rectButton" class="active"
                    onclick="setMode('rectified')">Corrected</button>

            <button id="overlayButton"
                    onclick="toggleOverlay()">Overlay OFF</button>

            <label class="conf-control">
                Confidence
                <input id="confidenceSlider" type="range"
                       min="25" max="90" step="5" value="45">
                <span id="confidenceLabel">45%</span>
            </label>
        </div>

        <div class="note" id="positionStatus">
            Loading camera position...
        </div>
    </section>

    <section class="panel">
        <div class="panel-title">
            <span>Building floor plan · Level 2</span>
            <span class="warning">UNVERIFIED ALIGNMENT</span>
        </div>

        <div class="map-wrap">
            <canvas id="cadCanvas"></canvas>
        </div>

        <div class="registration-controls">
            <label class="muted" for="candidateSelect">
                Registration
            </label>
            <select id="candidateSelect">
                <option value="none">Hide alignment</option>
            </select>
        </div>

        <div class="registration-metrics">
            <div class="registration-metric">
                <span>Inside building</span>
                <strong id="insideMetric">—</strong>
            </div>
            <div class="registration-metric">
                <span>Wall occupancy</span>
                <strong id="wallMetric">—</strong>
            </div>
            <div class="registration-metric">
                <span>Wall crossings</span>
                <strong id="crossingMetric">—</strong>
            </div>
        </div>

        <div class="note" id="cadStatus">
            CAD geometry is loading.
        </div>

        <div class="legend">
            <span><i class="dot" style="background:#398bd3"></i>Candidate route</span>
            <span><i class="dot" style="background:#ff8a43"></i>Candidate camera position</span>
            <span><i class="dot" style="background:#9ba5ae"></i>CAD geometry</span>
        </div>
    </section>
</main>

<section class="lower">
    <section class="panel">
        <div class="panel-title">
            <span>Interactive camera trajectory</span>
            <span class="muted">Click the path to navigate</span>
        </div>

        <div class="trajectory-wrap">
            <canvas id="trajectoryCanvas"></canvas>
        </div>

        <div class="legend">
            <span><i class="dot" style="background:#66d48e"></i>Start</span>
            <span><i class="dot" style="background:#ee765f"></i>End</span>
            <span><i class="dot" style="background:#ff8a43"></i>Selected frame</span>
        </div>

        <div class="note">
            Original Hilti trajectory coordinates.
            This panel does not apply CAD registration.
            Clicking the path selects the nearest synchronized photograph.
        </div>
    </section>

    <section class="panel">
        <div class="panel-title">
            <span>Material observations</span>
            <span class="muted">Model candidates only</span>
        </div>

        <div class="analysis-status" id="analysisStatus">
            Enable the overlay to analyze the current frame.
        </div>

        <div class="material-grid">
            <div class="material">
                <span>Concrete</span><strong id="countConcrete">—</strong>
            </div>
            <div class="material">
                <span>Formwork</span><strong id="countFormwork">—</strong>
            </div>
            <div class="material">
                <span>Rebar</span><strong id="countRebar">—</strong>
            </div>
            <div class="material">
                <span>Tiles</span><strong id="countTiles">—</strong>
            </div>
        </div>

        <div class="detections" id="detectionList"></div>

        <div class="note">
            Predictions are not independently verified.
            Detection counts are not construction quantities
            or completion percentages.
        </div>
    </section>
</section>

<script>
let frameIndex = 0;
let frameCount = 0;
let mode = "rectified";
let overlay = false;
let confidence = 0.45;
let requestVersion = 0;

let trajectory = [];
let framePositions = [];
let candidates = [];
let cad = null;
let selectedCandidate = null;

let projectedFrames = [];

const trajectoryCanvas = document.getElementById("trajectoryCanvas");
const trajectoryCtx = trajectoryCanvas.getContext("2d");

const cadCanvas = document.getElementById("cadCanvas");
const cadCtx = cadCanvas.getContext("2d");

function resizeCanvas(canvas, ctx) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;

    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);

    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function resizeAll() {
    resizeCanvas(trajectoryCanvas, trajectoryCtx);
    resizeCanvas(cadCanvas, cadCtx);
    drawTrajectory();
    drawCAD();
}

function transformPoint(x, y, candidate) {
    const theta = candidate.angle_deg * Math.PI / 180;
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    return {
        x: c * x - s * y + candidate.tx,
        y: s * x + c * y + candidate.ty
    };
}

function makeProjector(bounds, width, height, padding=30) {
    const dx = Math.max(bounds.max_x - bounds.min_x, 0.001);
    const dy = Math.max(bounds.max_y - bounds.min_y, 0.001);

    const scale = Math.min(
        (width - 2 * padding) / dx,
        (height - 2 * padding) / dy
    );

    const usedW = dx * scale;
    const usedH = dy * scale;

    const offsetX = (width - usedW) / 2;
    const offsetY = (height - usedH) / 2;

    return (x, y) => ({
        x: offsetX + (x - bounds.min_x) * scale,
        y: height - offsetY - (y - bounds.min_y) * scale
    });
}

function drawCircle(ctx, point, radius, color) {
    if (!point) return;

    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawPath(ctx, points, color, width) {
    if (points.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.stroke();
}

function trajectoryBounds() {
    if (!trajectory.length) return null;

    const xs = trajectory.map(p => p.x);
    const ys = trajectory.map(p => p.y);

    return {
        min_x: Math.min(...xs),
        max_x: Math.max(...xs),
        min_y: Math.min(...ys),
        max_y: Math.max(...ys)
    };
}

function drawTrajectory() {
    const width = trajectoryCanvas.clientWidth;
    const height = trajectoryCanvas.clientHeight;

    if (!width || !height) return;

    trajectoryCtx.clearRect(0, 0, width, height);
    trajectoryCtx.fillStyle = "#14191f";
    trajectoryCtx.fillRect(0, 0, width, height);

    if (!trajectory.length) return;

    const project = makeProjector(
        trajectoryBounds(), width, height, 42
    );

    const path = trajectory.map(p => project(p.x, p.y));

    projectedFrames = framePositions.map(p =>
        p.matched ? project(p.x, p.y) : null
    );

    drawPath(trajectoryCtx, path, "#4b9bc9", 2.5);

    drawCircle(trajectoryCtx, path[0], 6, "#66d48e");
    drawCircle(
        trajectoryCtx,
        path[path.length - 1],
        6,
        "#ee765f"
    );

    const selected = projectedFrames[frameIndex];

    if (selected) {
        trajectoryCtx.beginPath();
        trajectoryCtx.arc(
            selected.x, selected.y, 13, 0, 2 * Math.PI
        );
        trajectoryCtx.strokeStyle = "#ff8a43";
        trajectoryCtx.lineWidth = 2;
        trajectoryCtx.stroke();

        drawCircle(
            trajectoryCtx, selected, 7, "#ff8a43"
        );
    }

    trajectoryCtx.fillStyle = "#aab3bc";
    trajectoryCtx.font = "12px Arial";
    trajectoryCtx.fillText(
        "Original Hilti XY · no CAD transformation",
        16, height - 15
    );
}

function drawCAD() {
    const width = cadCanvas.clientWidth;
    const height = cadCanvas.clientHeight;

    if (!width || !height) return;

    cadCtx.clearRect(0, 0, width, height);
    cadCtx.fillStyle = "#f0f0f0";
    cadCtx.fillRect(0, 0, width, height);

    if (!cad) return;

    const project = makeProjector(
        cad.bounds, width, height, 25
    );

    cadCtx.strokeStyle = "#929ba3";
    cadCtx.lineWidth = 0.8;
    cadCtx.beginPath();

    for (const segment of cad.segments) {
        const a = project(segment[0][0], segment[0][1]);
        const b = project(segment[1][0], segment[1][1]);

        cadCtx.moveTo(a.x, a.y);
        cadCtx.lineTo(b.x, b.y);
    }

    cadCtx.stroke();

    if (!selectedCandidate) {
        cadCtx.fillStyle = "#48525b";
        cadCtx.font = "13px Arial";
        cadCtx.fillText(
            "CAD reference · alignment hidden",
            16, height - 15
        );
        return;
    }

    const candidate = selectedCandidate;

    const proposedRoute = trajectory.map(p => {
        const transformed = transformPoint(
            p.x, p.y, candidate
        );
        return project(transformed.x, transformed.y);
    });

    drawPath(
        cadCtx, proposedRoute, "#398bd3", 2.2
    );

    const position = framePositions[frameIndex];

    if (position && position.matched) {
        const predicted = transformPoint(
            position.x, position.y, candidate
        );

        const marker = project(
            predicted.x, predicted.y
        );

        cadCtx.beginPath();
        cadCtx.arc(
            marker.x, marker.y, 13, 0, 2 * Math.PI
        );
        cadCtx.strokeStyle = "#ff8a43";
        cadCtx.lineWidth = 2.5;
        cadCtx.stroke();

        drawCircle(
            cadCtx, marker, 6, "#ff8a43"
        );
    }

    cadCtx.fillStyle = "#a34b18";
    cadCtx.font = "bold 12px Arial";
    cadCtx.fillText(
        "CANDIDATE ONLY · NOT VERIFIED",
        14, height - 15
    );
}

function updateRegistration() {
    const value = document.getElementById(
        "candidateSelect"
    ).value;

    selectedCandidate = value === "none"
        ? null
        : candidates.find(
            c => c.rank === Number(value)
        ) || null;

    const inside = document.getElementById("insideMetric");
    const wall = document.getElementById("wallMetric");
    const crossing = document.getElementById("crossingMetric");
    const status = document.getElementById("cadStatus");

    if (!selectedCandidate) {
        inside.textContent = "—";
        wall.textContent = "—";
        crossing.textContent = "—";
        status.textContent =
            "Showing the CAD floor plan without an alignment. " +
            "The camera trajectory remains in its original coordinates.";
    } else {
        const c = selectedCandidate;

        inside.textContent =
            (100 * c.inside_fraction).toFixed(1) + "%";

        wall.textContent =
            (100 * c.wall_occupancy_fraction).toFixed(1) + "%";

        crossing.textContent =
            (100 * c.sampled_crossing_fraction).toFixed(1) + "%";

        status.textContent =
            `Candidate #${c.rank} · ` +
            `rotation ${c.angle_deg.toFixed(1)}° · ` +
            `score ${c.score.toFixed(3)}. ` +
            "These are heuristic geometry metrics, " +
            "not registration accuracy or confidence.";
    }

    drawCAD();
}

function updatePositionStatus() {
    const p = framePositions[frameIndex];
    const element = document.getElementById(
        "positionStatus"
    );

    if (!p || !p.matched) {
        element.textContent =
            "Position unavailable — this frame has no synchronized pose.";
        return;
    }

    let message =
        `Synchronized Hilti pose · ` +
        `X ${p.x.toFixed(2)} m · ` +
        `Y ${p.y.toFixed(2)} m · ` +
        `Z ${p.z.toFixed(2)} m`;

    if (selectedCandidate) {
        const q = transformPoint(
            p.x, p.y, selectedCandidate
        );

        message +=
            ` · Candidate CAD position ` +
            `(${q.x.toFixed(2)}, ${q.y.toFixed(2)}) m` +
            ` — UNVERIFIED`;
    } else {
        message += " · CAD alignment hidden";
    }

    element.textContent = message;
}

function resetCounts() {
    for (const name of [
        "Concrete", "Formwork", "Rebar", "Tiles"
    ]) {
        document.getElementById(
            "count" + name
        ).textContent = "—";
    }

    document.getElementById(
        "detectionList"
    ).replaceChildren();
}

async function updateFrame() {
    if (!frameCount) return;

    const version = ++requestVersion;
    const params = `mode=${mode}&conf=${confidence}`;

    document.getElementById(
        "frameSlider"
    ).value = frameIndex;

    document.getElementById(
        "frameLabel"
    ).textContent =
        `Frame ${frameIndex + 1} / ${frameCount}`;

    document.getElementById(
        "cameraImage"
    ).src =
        `/api/frame/${frameIndex}?${params}` +
        `&overlay=${overlay ? 1 : 0}&v=${version}`;

    updatePositionStatus();
    drawTrajectory();
    drawCAD();

    if (!overlay) {
        resetCounts();
        document.getElementById(
            "analysisStatus"
        ).textContent =
            "Enable the overlay to analyze the current frame.";
        return;
    }

    resetCounts();

    document.getElementById(
        "analysisStatus"
    ).textContent =
        "Running OpenCV 5 segmentation...";

    try {
        const response = await fetch(
            `/api/detections/${frameIndex}?${params}`
        );

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (version !== requestVersion) return;

        const counts = data.counts || {};

        for (const name of [
            "concrete", "formwork", "rebar", "tiles"
        ]) {
            const id =
                "count" +
                name[0].toUpperCase() +
                name.slice(1);

            document.getElementById(
                id
            ).textContent = counts[name] || 0;
        }

        const list = document.getElementById(
            "detectionList"
        );

        list.replaceChildren();

        for (const detection of data.detections) {
            const item = document.createElement("div");
            item.className = "detection";

            item.textContent =
                `${detection.class_name} · ` +
                `${(detection.score * 100).toFixed(1)}%`;

            list.appendChild(item);
        }

        document.getElementById(
            "analysisStatus"
        ).textContent =
            `${data.detections.length} candidate detections`;

    } catch (error) {
        if (version !== requestVersion) return;

        document.getElementById(
            "analysisStatus"
        ).textContent =
            "Segmentation failed. Check the Python terminal.";

        console.error(error);
    }
}

function step(delta) {
    frameIndex = Math.max(
        0,
        Math.min(frameCount - 1, frameIndex + delta)
    );
    updateFrame();
}

function setMode(value) {
    mode = value;

    document.getElementById(
        "rawButton"
    ).classList.toggle(
        "active", mode === "raw"
    );

    document.getElementById(
        "rectButton"
    ).classList.toggle(
        "active", mode === "rectified"
    );

    updateFrame();
}

function toggleOverlay() {
    overlay = !overlay;

    const button = document.getElementById(
        "overlayButton"
    );

    button.classList.toggle("active", overlay);
    button.textContent =
        overlay ? "Overlay ON" : "Overlay OFF";

    updateFrame();
}

trajectoryCanvas.addEventListener(
    "click",
    event => {
        if (!projectedFrames.length) return;

        const rect =
            trajectoryCanvas.getBoundingClientRect();

        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        let bestFrame = null;
        let bestDistance = Infinity;

        projectedFrames.forEach((point, index) => {
            if (!point) return;

            const distance =
                (point.x - x) ** 2 +
                (point.y - y) ** 2;

            if (distance < bestDistance) {
                bestDistance = distance;
                bestFrame = index;
            }
        });

        if (bestFrame !== null) {
            frameIndex = bestFrame;
            updateFrame();
        }
    }
);

document.getElementById(
    "candidateSelect"
).addEventListener(
    "change",
    () => {
        updateRegistration();
        updatePositionStatus();
    }
);

document.getElementById(
    "frameSlider"
).addEventListener(
    "input",
    event => {
        frameIndex = Number(event.target.value);
        updateFrame();
    }
);

document.getElementById(
    "confidenceSlider"
).addEventListener(
    "input",
    event => {
        confidence = Number(event.target.value) / 100;

        document.getElementById(
            "confidenceLabel"
        ).textContent =
            `${Math.round(confidence * 100)}%`;
    }
);

document.getElementById(
    "confidenceSlider"
).addEventListener(
    "change",
    updateFrame
);

document.addEventListener(
    "keydown",
    event => {
        if (
            event.target.tagName === "INPUT" ||
            event.target.tagName === "SELECT"
        ) return;

        if (event.key === "ArrowRight") step(1);
        if (event.key === "ArrowLeft") step(-1);
    }
);

window.addEventListener("resize", resizeAll);

async function initialize() {
    try {
        const urls = [
            "/api/info",
            "/api/spatial",
            "/api/registration",
            "/api/cad"
        ];

        const responses = await Promise.all(
            urls.map(url => fetch(url))
        );

        for (const response of responses) {
            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status}`
                );
            }
        }

        const [
            info,
            spatial,
            registration,
            cadData
        ] = await Promise.all(
            responses.map(response => response.json())
        );

        frameCount = info.frames;
        trajectory = spatial.trajectory;
        framePositions = spatial.frames;
        candidates = registration.candidates;
        cad = cadData;

        document.getElementById(
            "totalFrames"
        ).textContent = info.frames;

        document.getElementById(
            "matchedFrames"
        ).textContent = info.matched_frames;

        document.getElementById(
            "trajectoryCount"
        ).textContent = info.trajectory_records;

        document.getElementById(
            "candidateCount"
        ).textContent = candidates.length;

        document.getElementById(
            "frameSlider"
        ).max = Math.max(0, frameCount - 1);

        const selector = document.getElementById(
            "candidateSelect"
        );

        for (const candidate of candidates) {
            const option = document.createElement(
                "option"
            );

            option.value = candidate.rank;
            option.textContent =
                `Candidate #${candidate.rank} · ` +
                `${candidate.angle_deg.toFixed(1)}°`;

            selector.appendChild(option);
        }

        // Default to candidate #1 for exploration,
        // never as an assertion of correctness.
        selector.value = candidates.length
            ? String(candidates[0].rank)
            : "none";

        resizeAll();
        updateRegistration();
        updateFrame();

    } catch (error) {
        console.error(error);

        document.getElementById(
            "frameLabel"
        ).textContent =
            "Failed to load spatial data";

        document.getElementById(
            "cadStatus"
        ).textContent =
            "Failed to load CAD geometry or registration data.";
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
            f"No Hilti frames found at {FRAMES_DIR}"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    if not MODEL_CARD.exists():
        raise RuntimeError(
            f"Missing model card: {MODEL_CARD}"
        )

    print(f"Camera frames: {len(frames)}")
    print(
        "Matched frames:",
        sum(p["matched"] for p in frame_positions),
    )
    print(f"Trajectory records: {len(trajectory)}")
    print(f"CAD segments: {len(cad_geometry['segments'])}")
    print(f"Registration candidates: {len(candidates)}")
    print("Registration status: UNVERIFIED")
    print("Open http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )
