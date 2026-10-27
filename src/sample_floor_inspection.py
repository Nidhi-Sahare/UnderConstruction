
"""
Under-construction — Hilti Multi-Floor Inspection

Floor 1:
    Original DXF plan
    270 real walkthrough frames
    Timestamp-synchronized reference trajectory

Floor 2:
    Original dashboard inspection remains intact

Floor 3:
    Original DXF plan
    237 real walkthrough frames
    Timestamp-synchronized reference trajectory

Important:
    Reference trajectories are NOT registered to CAD.
    Camera extrinsics are NOT applied.
    Construction progress is NOT inferred.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

import ezdxf
from flask import Blueprint, Response, jsonify, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "data" / "floorplans"
GALLERY = ROOT / "sample_gallery"

floor_inspection_bp = Blueprint(
    "sample_floor_inspection",
    __name__,
)

COLORS = {
    "010 Aussenwände": "#b89aff",
    "015 Innenwände": "#aab1c9",
    "016 Trennwände": "#f0784b",
    "019 Vormauerungen": "#80cbb0",
    "010 Fenster": "#79b7ef",
    "010 Türen": "#e3b977",
    "011 Fassadensystem": "#b89aff",
    "012 Vorhangfassade": "#b89aff",
    "020 Stützen": "#d5d2df",
}


def extract_segments(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        a = entity.dxf.start
        b = entity.dxf.end
        return [((a.x, a.y), (b.x, b.y))]

    if kind == "LWPOLYLINE":
        points = list(entity.get_points("xyb"))

        if any(abs(float(p[2])) > 1e-9 for p in points):
            return []

        xy = [
            (float(p[0]), float(p[1]))
            for p in points
        ]

        if entity.closed and len(xy) > 2:
            xy.append(xy[0])

        return list(zip(xy, xy[1:]))

    if kind == "POLYLINE" and entity.is_2d_polyline:
        vertices = list(entity.vertices)

        if any(
            abs(float(v.dxf.get("bulge", 0))) > 1e-9
            for v in vertices
        ):
            return []

        xy = [
            (
                float(v.dxf.location.x),
                float(v.dxf.location.y),
            )
            for v in vertices
        ]

        if entity.is_closed and len(xy) > 2:
            xy.append(xy[0])

        return list(zip(xy, xy[1:]))

    return []


@lru_cache(maxsize=3)
def build_floor_svg(number):
    if number not in (1, 2, 3):
        raise ValueError("Invalid floor")

    path = PLANS / f"floor_{number}.dxf"
    doc = ezdxf.readfile(str(path))

    if doc.header.get("$INSUNITS", 0) != 6:
        raise ValueError(
            "Expected DXF coordinates in metres"
        )

    segments = []

    for entity in doc.modelspace():
        layer = entity.dxf.get("layer", "")

        if layer not in COLORS:
            continue

        for a, b in extract_segments(entity):
            if not all(
                math.isfinite(v)
                for v in (*a, *b)
            ):
                continue

            if math.dist(a, b) < 0.005:
                continue

            segments.append((a, b, layer))

    if not segments:
        raise ValueError(
            "No supported CAD segments found"
        )

    xs = [
        value
        for a, b, _ in segments
        for value in (a[0], b[0])
    ]

    ys = [
        value
        for a, b, _ in segments
        for value in (a[1], b[1])
    ]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = max(xmax - xmin, 0.01)
    height = max(ymax - ymin, 0.01)
    margin = 2.0

    lines = []

    for a, b, layer in segments:
        color = COLORS[layer]

        x1 = a[0] - xmin + margin
        y1 = ymax - a[1] + margin
        x2 = b[0] - xmin + margin
        y2 = ymax - b[1] + margin

        lines.append(
            f'<line x1="{x1:.5f}" y1="{y1:.5f}" '
            f'x2="{x2:.5f}" y2="{y2:.5f}" '
            f'stroke="{color}" stroke-width=".055" '
            f'stroke-linecap="round"/>'
        )

    total_width = width + 2 * margin
    total_height = height + 2 * margin

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_width:.5f} '
        f'{total_height:.5f}" '
        'preserveAspectRatio="xMidYMid meet">'
        '<rect width="100%" height="100%" '
        'fill="#14151d"/>'
        '<g>'
        + "".join(lines)
        + "</g>"
        + f'<text x="{margin}" y="{margin * .65}" '
        'fill="#aaa5b9" font-size=".65">'
        + escape(
            f"FLOOR {number} / ORIGINAL CAD PLAN"
        )
        + "</text></svg>"
    )


@floor_inspection_bp.get(
    "/api/sample/floor/<int:number>/plan.svg"
)
def floor_plan_svg(number):
    if number not in (1, 2, 3):
        return jsonify({"error": "Unknown floor"}), 404

    try:
        return Response(
            build_floor_svg(number),
            mimetype="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600"
            },
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


def walkthrough_directory(number):
    if number not in (1, 3):
        raise ValueError("Unsupported walkthrough floor")

    return GALLERY / f"real_floor_{number}"


@floor_inspection_bp.get(
    "/api/sample/floor/<int:number>/walkthrough"
)
def walkthrough_data(number):
    if number not in (1, 3):
        return jsonify({"error": "Unknown floor"}), 404

    directory = walkthrough_directory(number)

    try:
        manifest = json.loads(
            (directory / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        trajectory = json.loads(
            (directory / "trajectory.json").read_text(
                encoding="utf-8"
            )
        )

        frames = manifest["frames"]
        samples = trajectory["samples"]

        if len(frames) != len(samples):
            raise ValueError(
                "Frame and trajectory counts disagree"
            )

        if (
            manifest.get("floor") != number
            or trajectory.get("floor") != number
        ):
            raise ValueError(
                "Floor metadata does not match"
            )

        combined = []

        for frame, sample in zip(frames, samples):
            if frame["index"] != sample["frame_index"]:
                raise ValueError(
                    "Frame and pose indices disagree"
                )

            filename = frame["filename"]

            if not (
                isinstance(filename, str)
                and filename.startswith("frame_")
                and filename.endswith(".jpg")
                and Path(filename).name == filename
            ):
                raise ValueError(
                    "Invalid frame filename"
                )

            combined.append({
                "index": frame["index"],
                "filename": filename,
                "time_seconds": frame["time_seconds"],
                "image_url": (
                    f"/api/sample/floor/{number}/"
                    f"frame/{filename}"
                ),
                "pose_available": sample["pose_available"],
                "position": sample["position"],
                "orientation_xyzw": (
                    sample["orientation_xyzw"]
                ),
                "time_error_seconds": (
                    sample["time_error_seconds"]
                ),
            })

        return jsonify({
            "floor": number,
            "frame_count": len(combined),
            "matched_pose_count": trajectory["matched_frames"],
            "duration_seconds": (
                manifest["duration_seconds_sampled"]
            ),
            "camera": manifest["camera"],
            "coordinate_system": trajectory["coordinate_system"],
            "cad_registered": False,
            "camera_extrinsics_applied": False,
            "frames": combined,
        })

    except (OSError, ValueError, KeyError, TypeError) as exc:
        return jsonify({
            "error": str(exc)
        }), 503


@floor_inspection_bp.get(
    "/api/sample/floor/<int:number>/frame/<filename>"
)
def walkthrough_frame(number, filename):
    if number not in (1, 3):
        return jsonify({"error": "Unknown floor"}), 404

    if not (
        filename.startswith("frame_")
        and filename.endswith(".jpg")
        and filename[6:-4].isdigit()
        and Path(filename).name == filename
    ):
        return jsonify({"error": "Invalid filename"}), 400

    directory = walkthrough_directory(number) / "frames"

    return send_from_directory(
        str(directory),
        filename,
        mimetype="image/jpeg",
        max_age=3600,
    )


CSS = """
<style id="uc-floor-inspection-style">
.uc-inspection-selectbar {
  margin:18px 0 14px;
  padding:17px 19px;
  background:#191a22;
  border:1px solid #343542;
  border-radius:11px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:16px;
  flex-wrap:wrap;
}
.uc-inspection-selectbar strong {
  font-size:13px;
}
.uc-inspection-selectbar p {
  margin:6px 0 0;
  color:#9898aa;
  font-size:11px;
  line-height:1.6;
}
.uc-inspection-selectbar select {
  background:#292a35;
  color:#f0eef7;
  border:1px solid #625576;
  border-radius:8px;
  padding:11px 35px 11px 13px;
  min-width:185px;
  font-size:12px;
}
.uc-floor-plan-overlay {
  position:absolute;
  inset:0;
  z-index:3;
  background:#14151d;
}
.uc-floor-plan-overlay img {
  width:100%;
  height:100%;
  display:block;
  object-fit:contain;
}
.uc-floor-video-overlay {
  position:absolute;
  inset:0;
  z-index:4;
  background:#14151d;
  display:flex;
  flex-direction:column;
  align-items:stretch;
  justify-content:center;
  overflow:hidden;
}
.uc-floor-video-overlay img {
  width:100%;
  height:100%;
  object-fit:contain;
  min-height:0;
}
.uc-inspection-hidden {
  display:none!important;
}
.uc-floor-playback {
  padding:12px;
  background:#1a1b25;
  border-top:1px solid #343542;
  display:flex;
  flex-direction:column;
  gap:10px;
}
.uc-floor-playback-row {
  display:flex;
  gap:8px;
  align-items:center;
  flex-wrap:wrap;
}
.uc-floor-playback button {
  border:1px solid #555568;
  border-radius:6px;
  background:#292b38;
  color:#f3f1fa;
  padding:7px 12px;
  cursor:pointer;
}
.uc-floor-playback button:hover {
  background:#393b4b;
}
.uc-floor-playback input[type=range] {
  flex:1;
  min-width:80px;
  accent-color:#f0784b;
}
.uc-floor-playback small {
  color:#aaa5b9;
  font:11px Consolas,monospace;
}
.uc-floor-trajectory {
  margin-top:10px;
  padding:13px;
  border:1px solid #343542;
  border-radius:9px;
  background:#171821;
}
.uc-floor-trajectory canvas {
  width:100%;
  height:190px;
  display:block;
}
.uc-floor-trajectory-title {
  color:#e9e5f1;
  font-size:12px;
  font-weight:600;
  margin-bottom:8px;
}
.uc-floor-trajectory-note {
  color:#a7a5b6;
  font-size:11px;
  line-height:1.7;
}
.uc-floor-status {
  padding:12px 15px;
  border-top:1px solid #343542;
  color:#aaa5b9;
  font:11px Consolas,monospace;
  line-height:1.7;
}
.uc-floor-status strong {
  color:#e7b67e;
}
.uc-floor-error {
  padding:25px;
  color:#e7b67e;
  font-size:12px;
  text-align:center;
}
</style>
"""


SCRIPT = """
<script id="uc-floor-inspection-script">
(function () {
  "use strict";

  const byId = id => document.getElementById(id);

  let initialized = false;
  let currentFloor = 2;
  let currentIndex = 0;
  let playbackTimer = null;
  let loadedFloor = null;
  let walkthrough = null;
  let requestVersion = 0;

  const hiddenElements = new Map();

  function hideTemporarily(element) {
    if (!element || hiddenElements.has(element)) return;

    hiddenElements.set(element, element.style.display);

    element.style.setProperty(
      "display", "none", "important"
    );
  }

  function restoreHidden() {
    for (const [element, value] of hiddenElements) {
      if (value) {
        element.style.display = value;
      } else {
        element.style.removeProperty("display");
      }
    }

    hiddenElements.clear();
  }

  function stopPlayback() {
    if (playbackTimer !== null) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }

    const button = byId("uc-floor-play");

    if (button) button.textContent = "Play";
  }

  function makeOverlay(className, parent, id) {
    const element = document.createElement("div");
    element.className = className;
    element.id = id;
    parent.append(element);
    return element;
  }

  function findViewer(stage) {
    return stage?.closest(".viewer") || null;
  }

  function hideOriginalVideoFurniture(videoStage) {
    const viewer = findViewer(videoStage);
    if (!viewer) return;

    function visit(container) {
      for (const child of Array.from(container.children)) {
        if (child === videoStage) continue;

        if (child.contains(videoStage)) {
          visit(child);
          continue;
        }

       if (
  child.classList.contains("viewer-head") ||
  child.id === "uc-floor-status" ||
  child.id === "uc-floor-trajectory"
) {
  continue;
}

        hideTemporarily(child);
      }
    }

    visit(viewer);

    for (const id of [
      "frame-controls",
      "uc-align-controls"
    ]) {
      hideTemporarily(byId(id));
    }
  }

  function updateHeading(stage, text) {
    const heading = findViewer(stage)
      ?.querySelector(".viewer-head strong");

    if (heading) heading.textContent = text;
  }

  function trajectoryPoints() {
    if (!walkthrough) return [];

    return walkthrough.frames
      .map((frame, index) => ({
        index,
        position: frame.pose_available
          ? frame.position
          : null
      }))
      .filter(item =>
        item.position &&
        Number.isFinite(item.position.x) &&
        Number.isFinite(item.position.y)
      );
  }

  function drawTrajectory() {
    const canvas = byId("uc-floor-path");
    if (!canvas || !walkthrough) return;

    const rect = canvas.getBoundingClientRect();
    const width = Math.max(200, Math.round(rect.width));
    const height = 190;

    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, width, height);

    const points = trajectoryPoints();

    if (!points.length) {
      ctx.fillStyle = "#aaa5b9";
      ctx.fillText("No synchronized poses", 15, 30);
      return;
    }

    const xs = points.map(p => p.position.x);
    const ys = points.map(p => p.position.y);

    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const ymin = Math.min(...ys);
    const ymax = Math.max(...ys);

    const padding = 20;
    const usableWidth = width - padding * 2;
    const usableHeight = height - padding * 2;

    const rangeX = Math.max(xmax - xmin, 0.001);
    const rangeY = Math.max(ymax - ymin, 0.001);

    const scale = Math.min(
      usableWidth / rangeX,
      usableHeight / rangeY
    );

    const offsetX = (
      width - rangeX * scale
    ) / 2;

    const offsetY = (
      height - rangeY * scale
    ) / 2;

    function project(position) {
      return {
        x: offsetX + (position.x - xmin) * scale,
        y: height - (
          offsetY + (position.y - ymin) * scale
        )
      };
    }

    ctx.strokeStyle = "#505467";
    ctx.lineWidth = 1;

    ctx.strokeRect(
      0.5, 0.5, width - 1, height - 1
    );

    ctx.strokeStyle = "#9b9db7";
    ctx.lineWidth = 1.8;
    ctx.beginPath();

    let previousIndex = -2;

    for (const point of points) {
      const p = project(point.position);

      if (point.index !== previousIndex + 1) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }

      previousIndex = point.index;
    }

    ctx.stroke();

    const current = walkthrough.frames[currentIndex];

    if (current && current.pose_available) {
      const p = project(current.position);

      ctx.fillStyle = "#f0784b";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.fillStyle = "#9c9bad";
    ctx.font = "10px Consolas, monospace";
    ctx.fillText("REFERENCE XY", 9, 15);
  }

  function showFrame(index) {
    if (!walkthrough) return;

    const frames = walkthrough.frames;

    currentIndex = Math.max(
      0,
      Math.min(frames.length - 1, index)
    );

    const frame = frames[currentIndex];

    byId("uc-floor-real-image").src =
      frame.image_url;

    byId("uc-floor-range").value =
      String(currentIndex);

    byId("uc-floor-time").textContent =
      `${frame.time_seconds.toFixed(1)}s / ` +
      `${walkthrough.duration_seconds.toFixed(1)}s`;

    byId("uc-floor-counter").textContent =
      `${currentIndex + 1} / ${frames.length}`;

    const poseText = byId("uc-floor-pose");

    if (frame.pose_available && frame.position) {
      const p = frame.position;

      poseText.textContent =
        `Reference pose: ` +
        `X ${p.x.toFixed(2)}, ` +
        `Y ${p.y.toFixed(2)}, ` +
        `Z ${p.z.toFixed(2)} ` +
        `(not CAD-aligned)`;
    } else {
      poseText.textContent =
        "Reference pose unavailable for this frame";
    }

    drawTrajectory();
  }

  function schedulePlayback() {
    if (!walkthrough || playbackTimer === null) {
      return;
    }

    if (
      currentIndex >= walkthrough.frames.length - 1
    ) {
      stopPlayback();
      return;
    }

    const current =
      walkthrough.frames[currentIndex];

    const next =
      walkthrough.frames[currentIndex + 1];

    const delay = Math.max(
      50,
      Math.min(
        2000,
        (next.time_seconds - current.time_seconds)
        * 1000
      )
    );

    playbackTimer = setTimeout(() => {
      if (currentFloor !== loadedFloor) {
        stopPlayback();
        return;
      }

      showFrame(currentIndex + 1);
      schedulePlayback();
    }, delay);
  }

  function togglePlayback() {
    if (!walkthrough) return;

    if (playbackTimer !== null) {
      stopPlayback();
      return;
    }

    if (
      currentIndex >= walkthrough.frames.length - 1
    ) {
      showFrame(0);
    }

    playbackTimer = -1;
    byId("uc-floor-play").textContent = "Pause";

    schedulePlayback();
  }

  async function loadWalkthrough(number) {
    const version = ++requestVersion;

    stopPlayback();

    loadedFloor = null;
    walkthrough = null;

    byId("uc-floor-playback").classList.add(
      "uc-inspection-hidden"
    );

    byId("uc-floor-real-image").removeAttribute("src");

    byId("uc-floor-error").textContent =
      `Loading Floor ${number} real walkthrough...`;

    byId("uc-floor-error").classList.remove(
      "uc-inspection-hidden"
    );

    try {
      const response = await fetch(
        `/api/sample/floor/${number}/walkthrough`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Walkthrough unavailable"
        );
      }

      if (
        version !== requestVersion ||
        currentFloor !== number
      ) {
        return;
      }

      if (!data.frames?.length) {
        throw new Error("No frames in walkthrough");
      }

      walkthrough = data;
      loadedFloor = number;

      const slider = byId("uc-floor-range");
      slider.min = "0";
      slider.max = String(data.frames.length - 1);
      slider.step = "1";

      byId("uc-floor-error").classList.add(
        "uc-inspection-hidden"
      );

      byId("uc-floor-playback").classList.remove(
        "uc-inspection-hidden"
      );

      showFrame(0);

      const status = byId("uc-floor-status");

      status.textContent =
        `FLOOR ${number} / ${data.frame_count} REAL FRAMES / ` +
        `${data.matched_pose_count} MATCHED POSES / ` +
        "CAD REGISTRATION UNVERIFIED";

    } catch (error) {
      if (
        version !== requestVersion ||
        currentFloor !== number
      ) {
        return;
      }

      byId("uc-floor-error").textContent =
        `Could not load Floor ${number}: ${error.message}`;
    }
  }

  function selectFloor(number) {
    if (![1, 2, 3].includes(number)) return;

    stopPlayback();
    ++requestVersion;

    currentFloor = number;

    const selector = byId("uc-inspection-floor");
    if (selector) selector.value = String(number);

    const cadStage = byId("cad-stage");
    const videoStage = byId("frame-image")
      ?.closest(".video-stage");

    if (!cadStage || !videoStage) return;

    const isFloor2 = number === 2;

    restoreHidden();

    byId("uc-floor-plan-overlay").classList.toggle(
      "uc-inspection-hidden",
      isFloor2
    );

    byId("uc-floor-video-overlay").classList.toggle(
      "uc-inspection-hidden",
      isFloor2
    );

    byId("uc-floor-trajectory").classList.toggle(
      "uc-inspection-hidden",
      isFloor2
    );

    byId("uc-floor-status").classList.toggle(
      "uc-inspection-hidden",
      isFloor2
    );

    if (!isFloor2) {
      byId("uc-floor-plan-image").src =
        `/api/sample/floor/${number}/plan.svg`;

      hideOriginalVideoFurniture(videoStage);
      loadWalkthrough(number);
    }

    for (const id of [
      "zoom-in",
      "zoom-out",
      "reset-cad"
    ]) {
      const control = byId(id);
      if (control) control.disabled = !isFloor2;
    }

    updateHeading(
      cadStage,
      `01 / Floor ${number} · Architectural plan`
    );

    updateHeading(
      videoStage,
      `02 / Floor ${number} · Real walkthrough`
    );

    const footer = byId("cad-footer");

    if (footer) {
      if (!footer.dataset.ucOriginalText) {
        footer.dataset.ucOriginalText =
          footer.textContent;
      }

      footer.textContent = isFloor2
        ? footer.dataset.ucOriginalText
        : `FLOOR ${number} / REAL HILTI DXF / ` +
          "CONSTRUCTION STATUS UNVERIFIED";
    }

    for (const id of [
      "uc-3d-panel",
      "uc-3d-workspace"
    ]) {
      const element = byId(id);

      if (element) {
        element.classList.toggle(
          "uc-inspection-hidden",
          !isFloor2
        );
      }
    }

    document.dispatchEvent(
      new CustomEvent("uc:inspection-floor-change", {
        detail: {
          floor: number,
          walkthroughConnected: true,
          cadRegistered: false
        }
      })
    );
  }

  function initialize() {
    if (initialized) return;

    const viewer = byId("viewer-view");
    const grid = viewer?.querySelector(".viewer-grid");
    const cadStage = byId("cad-stage");
    const videoStage = byId("frame-image")
      ?.closest(".video-stage");

    if (!viewer || !grid || !cadStage || !videoStage) {
      console.warn(
        "[Floor Inspection] Viewer not found"
      );
      return;
    }

    initialized = true;

    const toolbar = document.createElement("div");
    toolbar.className = "uc-inspection-selectbar";
    toolbar.id = "uc-inspection-selectbar";

    toolbar.innerHTML = `
      <div>
        <strong>02 / FLOOR INSPECTION</strong>
        <p>
          Real architectural plans and recorded walkthroughs
          for all three floors.
        </p>
      </div>
      <select id="uc-inspection-floor"
              aria-label="Select floor">
        <option value="1">Floor 1</option>
        <option value="2" selected>Floor 2</option>
        <option value="3">Floor 3</option>
      </select>
    `;

    grid.parentElement.insertBefore(toolbar, grid);

    const planOverlay = makeOverlay(
      "uc-floor-plan-overlay uc-inspection-hidden",
      cadStage,
      "uc-floor-plan-overlay"
    );

    const planImage = document.createElement("img");
    planImage.id = "uc-floor-plan-image";
    planImage.alt = "Original Hilti architectural plan";
    planOverlay.append(planImage);

    const videoOverlay = makeOverlay(
      "uc-floor-video-overlay uc-inspection-hidden",
      videoStage,
      "uc-floor-video-overlay"
    );

    videoOverlay.innerHTML = `
      <div id="uc-floor-error"
           class="uc-floor-error">
        Loading real walkthrough...
      </div>
      <img id="uc-floor-real-image"
           alt="Real Hilti walkthrough camera frame">
      <div id="uc-floor-playback"
           class="uc-floor-playback uc-inspection-hidden">
        <div class="uc-floor-playback-row">
          <button id="uc-floor-prev" type="button">Previous</button>
          <button id="uc-floor-play" type="button">Play</button>
          <button id="uc-floor-next" type="button">Next</button>
          <small id="uc-floor-counter">0 / 0</small>
        </div>
        <div class="uc-floor-playback-row">
          <input id="uc-floor-range"
                 type="range"
                 min="0" max="0" value="0">
          <small id="uc-floor-time">0s</small>
        </div>
      </div>
    `;

    const videoViewer = findViewer(videoStage);

    if (videoViewer) {
      const trajectory = document.createElement("div");

      trajectory.id = "uc-floor-trajectory";
      trajectory.className =
        "uc-floor-trajectory uc-inspection-hidden";

      trajectory.innerHTML = `
        <div class="uc-floor-trajectory-title">
          Recorded trajectory · Reference XY
        </div>
        <canvas id="uc-floor-path"></canvas>
        <div id="uc-floor-pose"
             class="uc-floor-trajectory-note"></div>
        <div class="uc-floor-trajectory-note">
          This path uses reference trajectory coordinates.
          It is not registered to the CAD floor plan.
          Camera extrinsics have not been applied.
          Gaps indicate missing synchronized poses.
        </div>
      `;

      videoViewer.append(trajectory);

      const status = document.createElement("div");
      status.id = "uc-floor-status";
      status.className =
        "uc-floor-status uc-inspection-hidden";

      videoViewer.append(status);
    }

    byId("uc-inspection-floor").addEventListener(
      "change",
      event => {
        selectFloor(Number(event.target.value));
      }
    );

    byId("uc-floor-prev").addEventListener(
      "click",
      () => {
        stopPlayback();
        showFrame(currentIndex - 1);
      }
    );

    byId("uc-floor-next").addEventListener(
      "click",
      () => {
        stopPlayback();
        showFrame(currentIndex + 1);
      }
    );

    byId("uc-floor-play").addEventListener(
      "click",
      togglePlayback
    );

    byId("uc-floor-range").addEventListener(
      "input",
      event => {
        stopPlayback();
        showFrame(Number(event.target.value));
      }
    );

    window.addEventListener("resize", () => {
      if (currentFloor !== 2 && walkthrough) {
        drawTrajectory();
      }
    });

    selectFloor(2);

    console.info(
      "[Floor Inspection] Multi-floor walkthrough ready"
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener(
      "DOMContentLoaded",
      initialize,
      {once: true}
    );
  } else {
    initialize();
  }
})();
</script>
"""


def enhance_floor_inspection(html: str) -> str:
    if 'id="uc-floor-inspection-script"' in html:
        return html

    if "</head>" not in html or "</body>" not in html:
        raise ValueError("Incomplete dashboard HTML")

    html = html.replace(
        "</head>",
        CSS + "\n</head>",
        1,
    )

    html = html.replace(
        "</body>",
        SCRIPT + "\n</body>",
        1,
    )

    return html
