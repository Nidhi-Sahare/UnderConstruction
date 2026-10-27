
from __future__ import annotations

import bisect
import json
import math
import re
from pathlib import Path

from flask import Blueprint, jsonify, send_file, abort

ROOT = Path(__file__).resolve().parents[1]
bp = Blueprint("floor_walkthrough_upgrade", __name__)

FLOORS = (1, 3)


def read_trajectory(floor):
    path = ROOT / "data" / "hilti_groundtruth" / f"floor_{floor}.txt"
    if not path.is_file():
        return []

    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            values = [float(v) for v in line.split()[:8]]
            if len(values) != 8 or not all(map(math.isfinite, values)):
                continue
            result.append({
                "t": values[0],
                "x": values[1],
                "y": values[2],
            })
        except ValueError:
            continue

    return sorted(result, key=lambda p: p["t"])


def gallery(floor):
    return ROOT / "sample_gallery" / f"real_floor_{floor}"


def read_frames(floor):
    directory = gallery(floor)
    frames_dir = directory / "frames"

    if not frames_dir.is_dir():
        return []

    index_file = directory / "frame_index.json"
    indexed = {}

    if index_file.is_file():
        try:
            payload = json.loads(index_file.read_text(encoding="utf-8"))
            for item in payload.get("frames", []):
                name = Path(str(item.get("filename", ""))).name
                if name:
                    indexed[name] = item
        except (ValueError, OSError):
            pass

    images = sorted(
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    result = []

    for path in images:
        metadata = indexed.get(path.name, {})
        timestamp = None

        for key in ("timestamp", "timestamp_s", "time_s"):
            try:
                candidate = float(metadata[key])
                if math.isfinite(candidate):
                    timestamp = candidate
                    break
            except (KeyError, TypeError, ValueError):
                pass

        result.append({
            "filename": path.name,
            "url": (
                f"/api/floor-walkthrough/{floor}/frame/"
                f"{path.name}"
            ),
            "timestamp": timestamp,
        })

    return result


def interpolate(poses, timestamp):
    if not poses or timestamp is None:
        return None

    times = [p["t"] for p in poses]
    i = bisect.bisect_left(times, timestamp)

    if i == 0 or i >= len(poses):
        return None

    a, b = poses[i - 1], poses[i]
    dt = b["t"] - a["t"]

    if dt <= 0 or dt > 0.25:
        return None

    alpha = (timestamp - a["t"]) / dt

    return [
        a["x"] + alpha * (b["x"] - a["x"]),
        a["y"] + alpha * (b["y"] - a["y"]),
    ]


@bp.get("/api/floor-walkthrough/<int:floor>")
def walkthrough(floor):
    if floor not in FLOORS:
        abort(404)

    frames = read_frames(floor)
    poses = read_trajectory(floor)

    if not frames:
        return jsonify({
            "error": f"Floor {floor} gallery frames not found"
        }), 404

    # The gallery contains sampled frames. If timestamps are
    # unavailable, DO NOT invent exact camera-to-frame matching.
    matched = 0

    for frame in frames:
        frame["position"] = interpolate(
            poses, frame["timestamp"]
        )
        if frame["position"] is not None:
            matched += 1

    return jsonify({
        "floor": floor,
        "frames": frames,
        "trajectory": [
            [p["x"], p["y"]] for p in poses[::5]
        ],
        "trajectory_start": poses[0]["t"] if poses else None,
        "trajectory_end": poses[-1]["t"] if poses else None,
        "matched_frames": matched,
        "timing_verified": matched > 0,
        "cad_registration_verified": False,
    })


@bp.get("/api/floor-walkthrough/<int:floor>/frame/<filename>")
def walkthrough_frame(floor, filename):
    if floor not in FLOORS:
        abort(404)

    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(jpg|jpeg|png)", filename, re.I):
        abort(404)

    path = gallery(floor) / "frames" / filename

    if not path.is_file():
        abort(404)

    return send_file(path, conditional=True)


CSS = """
<style>
#uc-floor-video-overlay.uc-fw-active {
    display:flex!important;
    align-items:stretch!important;
    justify-content:stretch!important;
    padding:0!important;
    background:#101117!important;
    z-index:30!important;
}
.uc-fw-player {
    width:100%;
    height:100%;
    display:flex;
    flex-direction:column;
    position:relative;
    background:#101117;
}
.uc-fw-image {
    width:100%;
    min-height:0;
    flex:1;
    object-fit:contain;
}
.uc-fw-controls {
    display:flex;
    align-items:center;
    gap:12px;
    padding:12px;
    background:#20212b;
    border-top:1px solid #343542;
}
.uc-fw-controls button {
    background:#343040;
    color:#fff;
    border:1px solid #625576;
    border-radius:7px;
    padding:9px 14px;
}
.uc-fw-controls input {
    flex:1;
    min-width:40px;
    accent-color:#b89aff;
}
.uc-fw-controls span {
    font:11px Consolas,monospace;
    color:#b9b5c8;
    white-space:nowrap;
}
.uc-fw-route {
    position:absolute;
    right:12px;
    bottom:66px;
    width:165px;
    height:165px;
    background:#11131deb;
    border:1px solid #5b526d;
    border-radius:12px;
    overflow:hidden;
    padding:10px;
    box-shadow:0 10px 30px #0009;
}
.uc-fw-route svg {
    width:100%;
    height:125px;
}
.uc-fw-route small {
    font:9px Consolas,monospace;
    color:#b9b5c8;
}
.uc-fw-caption {
    position:absolute;
    top:12px;
    left:12px;
    background:#101117df;
    padding:8px 12px;
    border-radius:7px;
    color:#eee;
    font:11px Consolas,monospace;
}
</style>
"""

SCRIPT = r"""
<script>
(function () {
  "use strict";

  let floor = 2;
  let frames = [];
  let frameIndex = 0;
  let playing = false;
  let interval = null;
  let requestId = 0;
  let trajectory = [];
  let matched = 0;

  const byId = id => document.getElementById(id);

  function stop() {
    playing = false;
    if (interval) clearInterval(interval);
    interval = null;
    const button = byId("uc-fw-play");
    if (button) button.textContent = "▶ Play";
  }

  function fitPoints(points) {
    if (!points.length) return [];

    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const span = Math.max(
      maxX - minX,
      maxY - minY,
      0.001
    );

    return points.map(p => [
      7 + 86 * (p[0] - minX) / span,
      93 - 86 * (p[1] - minY) / span
    ]);
  }

  function drawRoute() {
    const svg = byId("uc-fw-svg");
    if (!svg) return;

    const projected = fitPoints(trajectory);
    const path = projected.map(
      p => p.map(v => v.toFixed(2)).join(",")
    ).join(" ");

    byId("uc-fw-track").setAttribute("points", path);

    const current = frames[frameIndex];
    const marker = byId("uc-fw-marker");

    if (!current || !current.position || !trajectory.length) {
      marker.setAttribute("visibility", "hidden");
      return;
    }

    // Use the same bounds for trajectory and marker.
    const xs = trajectory.map(p => p[0]);
    const ys = trajectory.map(p => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const span = Math.max(
      Math.max(...xs) - minX,
      Math.max(...ys) - minY,
      0.001
    );

    const x = 7 + 86 * (current.position[0] - minX) / span;
    const y = 93 - 86 * (current.position[1] - minY) / span;

    marker.setAttribute("cx", x);
    marker.setAttribute("cy", y);
    marker.setAttribute("visibility", "visible");
  }

  function render() {
    if (!frames.length) return;

    frameIndex = Math.max(
      0,
      Math.min(frameIndex, frames.length - 1)
    );

    byId("uc-fw-image").src = frames[frameIndex].url;
    byId("uc-fw-slider").max = frames.length - 1;
    byId("uc-fw-slider").value = frameIndex;

    byId("uc-fw-count").textContent =
      `${frameIndex + 1} / ${frames.length}`;

    drawRoute();
  }

  function play() {
    if (playing) {
      stop();
      return;
    }

    if (!frames.length) return;

    if (frameIndex >= frames.length - 1) {
      frameIndex = 0;
    }

    playing = true;
    byId("uc-fw-play").textContent = "Ⅱ Pause";

    // Gallery playback is illustrative when frame timestamps
    // are unavailable; it is not claimed to be real-time.
    interval = setInterval(() => {
      if (frameIndex >= frames.length - 1) {
        stop();
        return;
      }
      frameIndex++;
      render();
    }, 180);

    render();
  }

  function createPlayer(overlay) {
    overlay.innerHTML = `
      <div class="uc-fw-player">
        <div class="uc-fw-caption" id="uc-fw-caption">
          REAL SITE FOOTAGE
        </div>
        <img class="uc-fw-image" id="uc-fw-image"
             alt="Recorded construction walkthrough">
        <div class="uc-fw-route">
          <small>RECORDED CAMERA ROUTE</small>
          <svg id="uc-fw-svg" viewBox="0 0 100 100">
            <polyline id="uc-fw-track"
              fill="none" stroke="#b89aff"
              stroke-width="1.5"
              stroke-linecap="round"
              stroke-linejoin="round"/>
            <circle id="uc-fw-marker" r="2.7"
              fill="#f0784b" stroke="white"
              stroke-width="1" visibility="hidden"/>
          </svg>
        </div>
        <div class="uc-fw-controls">
          <button id="uc-fw-play">▶ Play</button>
          <input id="uc-fw-slider" type="range"
                 min="0" max="0" value="0">
          <span id="uc-fw-count">0 / 0</span>
        </div>
      </div>
    `;

    byId("uc-fw-play").addEventListener("click", play);

    byId("uc-fw-slider").addEventListener("input", e => {
      frameIndex = Number(e.target.value);
      render();
    });
  }

  async function load(number) {
    const token = ++requestId;
    floor = number;
    stop();

    const overlay = byId("uc-floor-video-overlay");
    if (!overlay) return;

    if (number === 2) {
      overlay.classList.remove("uc-fw-active");
      return;
    }

    overlay.classList.add("uc-fw-active");
    createPlayer(overlay);

    byId("uc-fw-caption").textContent =
      `FLOOR ${number} · LOADING REAL FOOTAGE`;

    try {
      const response = await fetch(
        `/api/floor-walkthrough/${number}`
      );

      const result = await response.json();

      if (token !== requestId || floor !== number) return;

      if (!response.ok) {
        throw Error(result.error || "Walkthrough unavailable");
      }

      frames = result.frames;
      trajectory = result.trajectory;
      matched = result.matched_frames;

      byId("uc-fw-caption").textContent =
        `FLOOR ${number} · REAL SITE FOOTAGE`;

      const routeLabel = overlay.querySelector(
        ".uc-fw-route small"
      );

      routeLabel.textContent = matched
        ? "TIMESTAMP-MATCHED ROUTE"
        : "RECORDED ROUTE · FRAME SYNC UNVERIFIED";

      render();

    } catch (error) {
      if (token !== requestId) return;

      byId("uc-fw-caption").textContent =
        `FLOOR ${number} · ${error.message}`;
    }
  }

  function init() {
    const selector = byId("uc-inspection-floor");
    if (!selector) return;

    // Run after the original floor-selector handler.
    selector.addEventListener("change", e => {
      const number = Number(e.target.value);
      setTimeout(() => load(number), 0);
    });

    load(Number(selector.value));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
"""


def enhance_floor_walkthrough(html):
    if "uc-fw-player" in html:
        return html

    return (
        html.replace("</head>", CSS + "</head>", 1)
            .replace("</body>", SCRIPT + "</body>", 1)
    )
