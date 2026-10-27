
"""
SiteWitness — Immersive Walkthrough Explorer

Runs inside the existing BuildProgress Flask application.

Route:
    /sitewitness/experience

Floors 1 and 3:
    Real ROS bag frames synchronized to ground-truth trajectories.

Floor 2:
    Existing walkthrough gallery. Route is omitted until its
    camera-pose format is validated.

Dependencies:
    Flask (already used by BuildProgress)

No separate server. No manual annotation.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from pathlib import Path

from flask import Blueprint, Response, jsonify, send_file


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "sitewitness" / "immersive"
CACHE.mkdir(parents=True, exist_ok=True)

bp = Blueprint("sitewitness_immersive", __name__)

FLOORS = (1, 2, 3)
FRAME_INTERVAL = 0.5  # seconds


def trajectory_for(floor):
    path = ROOT / f"data/hilti_groundtruth/floor_{floor}.txt"

    if not path.exists():
        return []

    poses = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 8:
            continue

        try:
            values = [float(x) for x in parts[:8]]
        except ValueError:
            continue

        if not all(math.isfinite(v) for v in values):
            continue

        t, x, y, z, qx, qy, qz, qw = values

        poses.append({
            "t": t,
            "x": x,
            "y": y,
            "z": z,
        })

    poses.sort(key=lambda p: p["t"])

    return poses


def interpolate_position(poses, timestamp):
    if not poses:
        return None

    times = [p["t"] for p in poses]

    index = bisect.bisect_left(times, timestamp)

    if index == 0:
        if abs(timestamp - times[0]) > 0.1:
            return None
        p = poses[0]
        return [p["x"], p["y"]]

    if index >= len(poses):
        if abs(timestamp - times[-1]) > 0.1:
            return None
        p = poses[-1]
        return [p["x"], p["y"]]

    a = poses[index - 1]
    b = poses[index]

    gap = b["t"] - a["t"]

    if gap <= 0 or gap > 0.25:
        return None

    alpha = (timestamp - a["t"]) / gap

    return [
        a["x"] + alpha * (b["x"] - a["x"]),
        a["y"] + alpha * (b["y"] - a["y"]),
    ]


def extract_image(data):
    """
    Extract JPEG/PNG payload from a serialized ROS2
    CompressedImage message.

    Image markers are validated by image format signatures.
    """

    start = data.find(b"\xff\xd8\xff")

    if start >= 0:
        end = data.find(b"\xff\xd9", start + 3)

        if end >= 0:
            return data[start:end + 2], ".jpg"

    png = b"\x89PNG\r\n\x1a\n"
    start = data.find(png)

    if start >= 0:
        end = data.find(b"IEND", start + len(png))

        if end >= 0:
            return data[start:end + 8], ".png"

    return None, None


def build_floor(floor):
    target = CACHE / f"floor_{floor}"
    manifest_path = target / "manifest.json"

    if manifest_path.exists():
        try:
            saved = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            if saved.get("frames"):
                first = saved["frames"][0]["image"]
                last = saved["frames"][-1]["image"]

                if (
                    (target / first).exists()
                    and (target / last).exists()
                ):
                    return saved
        except (OSError, ValueError, KeyError):
            pass

    target.mkdir(parents=True, exist_ok=True)

    poses = trajectory_for(floor)

    bag = (
        ROOT
        / f"data/hilti_recordings/floor_{floor}/rosbag.db3"
    )

    if not bag.exists():
        return {
            "floor": floor,
            "frames": [],
            "route": [],
            "status": "Recording unavailable",
        }

    frames = []
    next_sample = None

    with sqlite3.connect(str(bag)) as db:
        cursor = db.execute(
            """
            SELECT m.timestamp, m.data
            FROM messages m
            JOIN topics t ON t.id = m.topic_id
            WHERE t.name = ?
            ORDER BY m.timestamp
            """,
            ("/cam0/image_raw/compressed",),
        )

        for timestamp_ns, data in cursor:
            timestamp = timestamp_ns / 1e9

            # Only use frames with trajectory coverage.
            if poses and (
                timestamp < poses[0]["t"]
                or timestamp > poses[-1]["t"]
            ):
                continue

            if (
                next_sample is not None
                and timestamp < next_sample
            ):
                continue

            payload, extension = extract_image(data)

            if payload is None:
                continue

            position = interpolate_position(
                poses, timestamp
            )

            name = f"frame_{len(frames):05d}{extension}"

            (target / name).write_bytes(payload)

            frames.append({
                "image": name,
                "timestamp": timestamp,
                "position": position,
            })

            next_sample = timestamp + FRAME_INTERVAL

    route = [
        [p["x"], p["y"]]
        for p in poses
    ]

    manifest = {
        "floor": floor,
        "frames": frames,
        "route": route,
        "status": (
            "Trajectory synchronized"
            if frames and route
            else "Insufficient synchronized evidence"
        ),
        "source": "Hilti ROS bag + ground-truth trajectory",
        "coordinate_system": "Ground-truth XY; CAD alignment unverified",
    }

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return manifest


def gallery_floor_two():
    directory = (
        ROOT
        / "sample_gallery"
        / "real_floor_2"
        / "frames"
    )

    images = sorted(directory.glob("*.jpg"))

    return {
        "floor": 2,
        "frames": [
            {
                "image": path.name,
                "timestamp": None,
                "position": None,
            }
            for path in images
        ],
        "route": [],
        "status": "Walkthrough gallery available",
        "source": "Existing Floor 2 gallery",
        "coordinate_system": (
            "Gallery frame timing and camera route "
            "not verified"
        ),
    }


@bp.route("/sitewitness/api/floor/<int:floor>")
def floor_data(floor):
    if floor not in FLOORS:
        return jsonify({"error": "Unknown floor"}), 404

    if floor == 2:
        return jsonify(gallery_floor_two())

    try:
        return jsonify(build_floor(floor))
    except Exception as exc:
        return jsonify({
            "error": f"{type(exc).__name__}: {exc}"
        }), 500


@bp.route(
    "/sitewitness/api/frame/<int:floor>/<path:filename>"
)
def floor_frame(floor, filename):
    if floor not in FLOORS:
        return Response(status=404)

    # Prevent arbitrary file traversal.
    if Path(filename).name != filename:
        return Response(status=404)

    if floor == 2:
        directory = (
            ROOT
            / "sample_gallery"
            / "real_floor_2"
            / "frames"
        )
    else:
        directory = CACHE / f"floor_{floor}"

    path = directory / filename

    if not path.is_file():
        return Response(status=404)

    return send_file(path)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>SiteWitness — Explore</title>

<style>
:root {
    color-scheme: dark;
    --bg: #101116;
    --panel: #1b1d25;
    --border: #343641;
    --text: #f2f0ed;
    --muted: #a3a3ad;
    --accent: #e8a86e;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", sans-serif;
}

header {
    height: 76px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 36px;
    border-bottom: 1px solid var(--border);
}

.brand {
    font-size: 23px;
    font-weight: 750;
    letter-spacing: -.8px;
}

.brand span { color: var(--accent); }

header a {
    color: var(--muted);
    text-decoration: none;
    font-size: 13px;
}

main {
    max-width: 1550px;
    margin: auto;
    padding: 32px;
}

.eyebrow {
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--accent);
    font-weight: 700;
}

h1 {
    font-size: clamp(30px, 4vw, 54px);
    letter-spacing: -2px;
    margin: 12px 0;
}

.intro {
    color: var(--muted);
    margin-bottom: 26px;
    line-height: 1.6;
}

.toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}

.floors {
    display: flex;
    gap: 8px;
}

button {
    border: 1px solid var(--border);
    color: var(--text);
    background: var(--panel);
    padding: 12px 17px;
    border-radius: 9px;
    cursor: pointer;
    font: inherit;
    font-size: 13px;
}

button:hover {
    border-color: var(--accent);
}

button.active {
    background: var(--accent);
    color: #171411;
    border-color: var(--accent);
    font-weight: 700;
}

.status {
    color: var(--muted);
    font-size: 12px;
}

.workspace {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(300px, 1fr);
    gap: 16px;
}

.viewer, .map-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

.panel-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    font-size: 13px;
    font-weight: 650;
}

.panel-top small {
    color: var(--muted);
    font-weight: 400;
}

.frame-wrap {
    background: #090a0d;
    height: min(56vh, 560px);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.frame-wrap img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: none;
}

.empty {
    position: absolute;
    color: var(--muted);
    text-align: center;
    padding: 25px;
    line-height: 1.7;
}

.corner {
    position: absolute;
    left: 16px;
    top: 16px;
    background: rgba(0,0,0,.7);
    border: 1px solid #ffffff30;
    padding: 8px 12px;
    border-radius: 7px;
    font-size: 11px;
    letter-spacing: .7px;
    z-index: 2;
}

.controls {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 18px;
}

#play {
    background: var(--accent);
    color: #16120f;
    border: 0;
    min-width: 82px;
    font-weight: 700;
}

input[type=range] {
    flex: 1;
    accent-color: var(--accent);
    min-width: 40px;
    cursor: pointer;
}

#time {
    color: var(--muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    min-width: 90px;
    text-align: right;
}

.map-wrap {
    height: min(56vh, 560px);
    background:
      linear-gradient(#ffffff06 1px, transparent 1px),
      linear-gradient(90deg, #ffffff06 1px, transparent 1px),
      #161820;
    background-size: 30px 30px;
    position: relative;
}

svg {
    width: 100%;
    height: 100%;
}

#route {
    fill: none;
    stroke: #888a9a;
    stroke-width: 1.5;
    opacity: .8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

#travelled {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
}

#camera {
    fill: var(--accent);
    stroke: #fff;
    stroke-width: 1.5;
    filter: drop-shadow(0 0 7px #e8a86e88);
}

.map-caption {
    padding: 17px 20px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
    border-top: 1px solid var(--border);
}

.footer {
    margin-top: 20px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
}

@media(max-width: 900px) {
    header { padding: 0 20px; }
    main { padding: 20px; }
    .workspace { grid-template-columns: 1fr; }
    .frame-wrap, .map-wrap { height: 45vh; }
}
</style>
</head>

<body>

<header>
    <div class="brand">Site<span>Witness</span></div>
    <a href="/">← Back to BuildProgress</a>
</header>

<main>
    <div class="eyebrow">AUTONOMOUS SITE EXPLORATION</div>
    <h1>Walk through your building.</h1>

    <p class="intro">
        Real site footage. Recorded camera movement.
        Explore each floor without leaving your desk.
    </p>

    <div class="toolbar">
        <div class="floors">
            <button data-floor="1" class="active">
                Floor 01
            </button>
            <button data-floor="2">
                Floor 02
            </button>
            <button data-floor="3">
                Floor 03
            </button>
        </div>

        <div class="status" id="status">
            Preparing walkthrough...
        </div>
    </div>

    <div class="workspace">

        <section class="viewer">
            <div class="panel-top">
                <span>On-site perspective</span>
                <small>REAL CAMERA FOOTAGE</small>
            </div>

            <div class="frame-wrap">
                <div class="corner" id="frame-label">
                    FLOOR 01 · CAM 0
                </div>

                <img id="frame" alt="Actual site walkthrough">

                <div class="empty" id="empty">
                    Preparing synchronized footage...
                </div>
            </div>

            <div class="controls">
                <button id="play">▶ Play</button>

                <input
                    id="scrub"
                    type="range"
                    min="0"
                    max="100"
                    value="0"
                    step="1"
                >

                <span id="time">00:00 / 00:00</span>
            </div>
        </section>

        <section class="map-panel">
            <div class="panel-top">
                <span>Where the camera travelled</span>
                <small>RECORDED TRAJECTORY</small>
            </div>

            <div class="map-wrap">
                <svg id="map" viewBox="0 0 100 100"
                     preserveAspectRatio="xMidYMid meet">
                    <polyline id="route" points=""></polyline>
                    <polyline id="travelled" points=""></polyline>
                    <circle id="camera" cx="-10" cy="-10" r="1.4"></circle>
                </svg>

                <div class="empty" id="map-empty">
                    Loading camera trajectory...
                </div>
            </div>

            <div class="map-caption" id="map-caption">
                The marker follows recorded camera positions.
                CAD alignment is not assumed.
            </div>
        </section>

    </div>

    <div class="footer" id="footer">
        SiteWitness displays measured evidence only.
        Reconstructed geometry and construction discrepancies
        are not marked as verified without CAD registration.
    </div>
</main>

<script>
let floor = 1;
let data = null;
let index = 0;
let playing = false;
let lastTick = 0;
let loadingToken = 0;
let routePoints = [];
let projectedFrames = [];

const $ = id => document.getElementById(id);

const frame = $("frame");
const scrub = $("scrub");
const playButton = $("play");

function formatTime(seconds) {
    seconds = Math.max(0, Math.floor(seconds || 0));

    const m = Math.floor(seconds / 60);
    const s = seconds % 60;

    return String(m).padStart(2, "0") +
        ":" + String(s).padStart(2, "0");
}

function stop() {
    playing = false;
    playButton.textContent = "▶ Play";
}

function projectRoute(route, frames) {
    const coords = route.filter(p =>
        Array.isArray(p) &&
        p.length >= 2 &&
        Number.isFinite(p[0]) &&
        Number.isFinite(p[1])
    );

    if (!coords.length) {
        routePoints = [];
        projectedFrames = [];
        return;
    }

    const xs = coords.map(p => p[0]);
    const ys = coords.map(p => p[1]);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const span = Math.max(
        maxX - minX,
        maxY - minY,
        0.001
    );

    const convert = p => [
        8 + ((p[0] - minX) / span) * 84,
        92 - ((p[1] - minY) / span) * 84
    ];

    routePoints = coords.map(convert);

    projectedFrames = frames.map(f =>
        f.position ? convert(f.position) : null
    );
}

function pointString(points) {
    return points.map(p =>
        p[0].toFixed(3) + "," + p[1].toFixed(3)
    ).join(" ");
}

function drawMap() {
    $("route").setAttribute(
        "points", pointString(routePoints)
    );

    const position = projectedFrames[index];

    if (!position) {
        $("camera").setAttribute("cx", -10);
        $("camera").setAttribute("cy", -10);
        $("travelled").setAttribute("points", "");
        return;
    }

    $("camera").setAttribute("cx", position[0]);
    $("camera").setAttribute("cy", position[1]);

    const valid = projectedFrames
        .slice(0, index + 1)
        .filter(Boolean);

    // This trace joins sampled positions. It is not
    // a verified continuous physical path through gaps.
    $("travelled").setAttribute(
        "points", pointString(valid)
    );
}

function showFrame() {
    if (!data || !data.frames.length) return;

    index = Math.max(
        0,
        Math.min(index, data.frames.length - 1)
    );

    const current = data.frames[index];

    frame.src =
        "/sitewitness/api/frame/" +
        floor + "/" +
        encodeURIComponent(current.image);

    frame.style.display = "block";
    $("empty").style.display = "none";

    scrub.max = Math.max(0, data.frames.length - 1);
    scrub.value = index;

    let elapsed;
    let duration;

    if (
        current.timestamp != null &&
        data.frames[0].timestamp != null
    ) {
        elapsed = current.timestamp -
            data.frames[0].timestamp;

        duration = data.frames.at(-1).timestamp -
            data.frames[0].timestamp;
    } else {
        elapsed = index;
        duration = Math.max(0, data.frames.length - 1);
    }

    $("time").textContent =
        formatTime(elapsed) + " / " +
        formatTime(duration);

    $("frame-label").textContent =
        "FLOOR " + String(floor).padStart(2, "0") +
        " · CAM 0";

    drawMap();

    // Preload the next image.
    if (index + 1 < data.frames.length) {
        const next = new Image();
        next.src =
            "/sitewitness/api/frame/" +
            floor + "/" +
            encodeURIComponent(data.frames[index + 1].image);
    }
}

async function loadFloor(number) {
    const token = ++loadingToken;

    stop();

    floor = number;
    data = null;
    index = 0;

    frame.removeAttribute("src");
    frame.style.display = "none";

    $("empty").style.display = "block";
    $("empty").textContent = "Preparing real walkthrough...";

    $("status").textContent =
        "Loading Floor " + floor + " evidence...";

    $("map-empty").style.display = "block";
    $("route").setAttribute("points", "");
    $("travelled").setAttribute("points", "");
    $("camera").setAttribute("cx", -10);
    $("camera").setAttribute("cy", -10);

    document.querySelectorAll("[data-floor]").forEach(
        button => {
            button.classList.toggle(
                "active",
                Number(button.dataset.floor) === floor
            );
        }
    );

    try {
        const response = await fetch(
            "/sitewitness/api/floor/" + floor
        );

        const result = await response.json();

        if (token !== loadingToken) return;

        if (!response.ok) {
            throw new Error(result.error || "Load failed");
        }

        data = result;

        if (!data.frames.length) {
            throw new Error(
                "No usable walkthrough frames were found"
            );
        }

        projectRoute(data.route || [], data.frames);

        $("map-empty").style.display =
            routePoints.length ? "none" : "block";

        $("map-empty").textContent =
            "Camera route not yet available for this floor.";

        $("map-caption").textContent =
            routePoints.length
            ? "Camera path from timestamped ground truth. " +
              "Shown in trajectory coordinates; " +
              "not yet aligned to the CAD floor plan."
            : "Real walkthrough imagery is available. " +
              "Camera position is not yet verified.";

        $("status").textContent =
            data.status + " · " +
            data.frames.length + " frames";

        showFrame();

    } catch (error) {
        if (token !== loadingToken) return;

        $("status").textContent = "Evidence unavailable";

        $("empty").textContent =
            "Unable to load this walkthrough: " +
            error.message;
    }
}

function animate(now) {
    if (playing && data && data.frames.length) {
        if (!lastTick) lastTick = now;

        const current = data.frames[index];
        const next = data.frames[index + 1];

        if (!next) {
            stop();
        } else {
            let delay = 500;

            if (
                current.timestamp != null &&
                next.timestamp != null
            ) {
                delay = Math.max(
                    30,
                    (next.timestamp - current.timestamp) * 1000
                );
            }

            if (now - lastTick >= delay) {
                lastTick = now;
                index++;
                showFrame();
            }
        }
    } else {
        lastTick = 0;
    }

    requestAnimationFrame(animate);
}

playButton.addEventListener("click", () => {
    if (!data || !data.frames.length) return;

    if (index >= data.frames.length - 1) {
        index = 0;
        showFrame();
    }

    playing = !playing;
    lastTick = 0;

    playButton.textContent =
        playing ? "Ⅱ Pause" : "▶ Play";
});

scrub.addEventListener("input", () => {
    if (!data) return;

    index = Number(scrub.value);
    lastTick = 0;
    showFrame();
});

document.querySelectorAll("[data-floor]").forEach(
    button => {
        button.addEventListener("click", () => {
            loadFloor(Number(button.dataset.floor));
        });
    }
);

requestAnimationFrame(animate);
loadFloor(1);
</script>

</body>
</html>
"""


@bp.route("/sitewitness/experience")
def experience():
    return Response(PAGE, mimetype="text/html")


def register_sitewitness_immersive(app):
    if "sitewitness_immersive" not in app.blueprints:
        app.register_blueprint(bp)
