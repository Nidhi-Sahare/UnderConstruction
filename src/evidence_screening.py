
"""
Under-construction — Automatic Visual Evidence Screening

Registers:
    GET /screening
    GET /api/screening/element/<element_id>?candidate=1&limit=6

Integration:
    Add `import evidence_screening` near the bottom of
    src/underconstruction_integrated.py, before app.run().

The output describes image-level material detections only.
It does NOT verify element identity or construction progress.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from pathlib import Path

from flask import jsonify, render_template_string, request

from underconstruction_app import app, infer


ROOT = Path(__file__).resolve().parents[1]

ASSOCIATION_DIR = (
    ROOT / "outputs" / "automatic_element_evidence"
)

MODEL_FILE = ROOT / "models" / "underconstruction_seg_v0.onnx"

CACHE = {}
CACHE_LOCK = threading.RLock()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_associations(candidate):
    if candidate not in range(1, 6):
        raise ValueError("Candidate must be between 1 and 5")

    path = (
        ASSOCIATION_DIR
        / f"candidate_{candidate}_associations.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Generate associations first."
        )

    data = read_json(path)

    if data.get("registration_candidate_rank") != candidate:
        raise ValueError("Registration candidate mismatch")

    return data


def analyze_frame(frame_index):
    """
    Cached frame-level inference.

    The model detects construction-material candidates
    anywhere in the image, not on a particular CAD element.
    """
    with CACHE_LOCK:
        if frame_index in CACHE:
            return CACHE[frame_index]

        _, detections = infer(
            frame_index,
            "rectified",
            0.45,
        )

        records = []

        for detection in detections:
            score = float(detection.score)

            if not math.isfinite(score):
                continue

            records.append({
                "class_name": str(detection.class_name),
                "score": round(score, 4),
                "bbox": [
                    float(value)
                    for value in detection.bbox
                ],
            })

        counts = Counter(
            record["class_name"]
            for record in records
        )

        result = {
            "frame_index": frame_index,
            "detections": records,
            "material_counts": dict(counts),
            "detection_count": len(records),
            "screening_status": (
                "MODEL_DETECTIONS_PRESENT"
                if records
                else "NO_MODEL_DETECTIONS"
            ),
            "element_identity_verified": False,
        }

        CACHE[frame_index] = result

        return result


@app.route("/api/screening/element/<element_id>")
def api_screen_element(element_id):
    try:
        candidate = int(request.args.get("candidate", 1))
        limit = int(request.args.get("limit", 6))

        if not 1 <= limit <= 12:
            raise ValueError("Limit must be between 1 and 12")

        data = load_associations(candidate)

        element = next(
            (
                item
                for item in data.get("elements", [])
                if item.get("element_id") == element_id
            ),
            None,
        )

        if element is None:
            return jsonify({
                "error": "Element not found"
            }), 404

        candidate_frames = element.get(
            "candidate_frames", []
        )[:limit]

        results = []

        for frame in candidate_frames:
            frame_index = int(frame["frame_index"])

            try:
                screening = analyze_frame(frame_index)

                results.append({
                    **screening,
                    "distance_m": frame.get("distance_m"),
                    "filename": frame.get("filename"),
                })

            except Exception:
                app.logger.exception(
                    "Screening failed for frame %s",
                    frame_index,
                )

                results.append({
                    "frame_index": frame_index,
                    "filename": frame.get("filename"),
                    "distance_m": frame.get("distance_m"),
                    "screening_status": "INFERENCE_FAILED",
                    "detections": [],
                    "material_counts": {},
                    "detection_count": 0,
                    "element_identity_verified": False,
                })

        aggregate = Counter()

        for frame in results:
            aggregate.update(frame["material_counts"])

        return jsonify({
            "element_id": element_id,
            "category": element.get("category"),
            "registration_candidate": candidate,
            "registration_verified": False,
            "element_identity_verified": False,
            "construction_status": "UNKNOWN",
            "completion_percentage": None,
            "model_available": MODEL_FILE.is_file(),
            "screened_frames": len(results),
            "aggregate_image_detections": dict(aggregate),
            "frames": results,
            "interpretation": (
                "Material detections describe entire photographs. "
                "They are not assigned to the selected CAD element. "
                "Construction completion cannot be inferred."
            ),
        })

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/screening")
def screening_page():
    return render_template_string(PAGE)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>Under-construction — Visual Screening</title>

<style>
:root {
    --bg:#101317;
    --panel:#1b2026;
    --border:#343b43;
    --text:#f2f2f2;
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
    flex-wrap:wrap;
    gap:16px;
    padding:20px 28px;
    border-bottom:1px solid var(--border);
}
.brand { font-size:23px; font-weight:750; }
.brand span { color:var(--accent); }
nav { display:flex; gap:20px; }
a { color:var(--accent); text-decoration:none; }
main {
    max-width:1450px;
    margin:auto;
    padding:28px;
}
h1 { margin:0 0 8px; font-size:28px; }
p { color:var(--muted); line-height:1.6; }
.controls {
    display:grid;
    grid-template-columns:minmax(220px,1fr) 170px 150px;
    gap:12px;
    align-items:end;
    margin:24px 0;
}
label {
    display:block;
    margin-bottom:8px;
    color:var(--muted);
    font-size:13px;
}
select,button {
    padding:12px;
    border-radius:8px;
    font-size:14px;
}
select {
    width:100%;
    background:var(--panel);
    color:var(--text);
    border:1px solid var(--border);
}
button {
    background:var(--accent);
    color:#171717;
    border:0;
    cursor:pointer;
    font-weight:700;
}
button:disabled {
    opacity:.5;
    cursor:wait;
}
.warning {
    border:1px solid #725039;
    background:#30261f;
    padding:14px;
    border-radius:10px;
    color:#f0c8a8;
    font-size:13px;
    line-height:1.6;
}
.summary {
    margin-top:20px;
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:10px;
    padding:18px;
}
.grid {
    display:grid;
    grid-template-columns:
        repeat(auto-fill,minmax(260px,1fr));
    gap:15px;
    margin-top:20px;
}
.card {
    border:1px solid var(--border);
    background:var(--panel);
    border-radius:10px;
    overflow:hidden;
}
.card img {
    width:100%;
    height:230px;
    object-fit:contain;
    background:#090b0e;
}
.card-body { padding:15px; }
.card h3 { margin:0 0 9px; font-size:16px; }
.meta {
    font-size:12px;
    color:var(--muted);
    margin:7px 0;
}
.chip {
    display:inline-block;
    padding:5px 9px;
    margin:4px 5px 0 0;
    border:1px solid var(--border);
    border-radius:6px;
    font-size:12px;
}
@media(max-width:750px) {
    .controls { grid-template-columns:1fr; }
}
</style>
</head>

<body>
<header>
    <div class="brand">Under<span>-construction</span></div>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/evidence">Evidence Explorer</a>
        <a href="/screening">Visual Screening</a>
    </nav>
</header>

<main>
    <h1>Automatic Visual Screening</h1>
    <p>
        Examine candidate walkthrough frames using the existing
        construction-material segmentation model.
    </p>

    <div class="controls">
        <div>
            <label for="element">Building element</label>
            <select id="element"></select>
        </div>

        <div>
            <label for="candidate">Registration</label>
            <select id="candidate">
                <option value="1">Candidate #1</option>
                <option value="2">Candidate #2</option>
                <option value="3">Candidate #3</option>
                <option value="4">Candidate #4</option>
                <option value="5">Candidate #5</option>
            </select>
        </div>

        <button id="run">Analyze frames</button>
    </div>

    <div class="warning">
        This is image-level screening, not wall-level verification.
        A detected material may belong to any visible surface.
        Camera-to-CAD registration is unverified, so construction
        status remains UNKNOWN.
    </div>

    <div id="status" class="summary">
        Select an element and click Analyze frames.
    </div>

    <div id="gallery" class="grid"></div>
</main>

<script>
const $ = id => document.getElementById(id);

async function getJSON(url) {
    const response = await fetch(url);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }
    return data;
}

function text(tag, content, className="") {
    const element = document.createElement(tag);
    element.textContent = content;
    if (className) element.className = className;
    return element;
}

async function loadElements() {
    $("element").replaceChildren();
    $("element").appendChild(
        text("option", "Loading elements…")
    );

    try {
        const candidate = $("candidate").value;

        const data = await getJSON(
            `/api/evidence/elements?candidate=${candidate}`
        );

        $("element").replaceChildren();

        const eligible = data.elements.filter(
            item => item.candidate_count > 0
        );

        for (const item of eligible) {
            const option = document.createElement("option");
            option.value = item.element_id;
            option.textContent =
                `${item.category} · ${item.element_id.slice(-8)} ` +
                `(${item.candidate_count} candidates)`;
            $("element").appendChild(option);
        }

        $("status").textContent =
            `${eligible.length} elements available for screening.`;

    } catch (error) {
        $("status").textContent = error.message;
    }
}

async function runScreening() {
    const elementId = $("element").value;

    if (!elementId) return;

    $("run").disabled = true;
    $("status").textContent =
        "Analyzing candidate frames. CPU inference may take time…";
    $("gallery").replaceChildren();

    try {
        const candidate = $("candidate").value;

        const data = await getJSON(
            `/api/screening/element/` +
            `${encodeURIComponent(elementId)}` +
            `?candidate=${candidate}&limit=6`
        );

        $("status").replaceChildren();

        $("status").appendChild(
            text("h3", `Element: ${data.element_id}`)
        );

        $("status").appendChild(
            text(
                "p",
                `Frames screened: ${data.screened_frames} · ` +
                `Construction status: UNKNOWN`
            )
        );

        $("status").appendChild(
            text(
                "p",
                "Image-level detections: " +
                JSON.stringify(data.aggregate_image_detections)
            )
        );

        for (const frame of data.frames) {
            const card = document.createElement("div");
            card.className = "card";

            const image = document.createElement("img");
            image.loading = "lazy";
            image.alt = `Frame ${frame.frame_index}`;
            image.src =
                `/api/frame/${frame.frame_index}` +
                `?mode=rectified&overlay=1`;

            const body = document.createElement("div");
            body.className = "card-body";

            body.appendChild(
                text("h3", `Frame ${frame.frame_index}`)
            );

            body.appendChild(
                text(
                    "div",
                    `Candidate distance: ${frame.distance_m} m`,
                    "meta"
                )
            );

            body.appendChild(
                text(
                    "div",
                    frame.screening_status,
                    "meta"
                )
            );

            const counts = Object.entries(
                frame.material_counts || {}
            );

            if (!counts.length) {
                body.appendChild(
                    text(
                        "div",
                        "No material detections reported",
                        "meta"
                    )
                );
            }

            for (const [name, count] of counts) {
                body.appendChild(
                    text("span", `${name}: ${count}`, "chip")
                );
            }

            card.append(image, body);
            $("gallery").appendChild(card);
        }

    } catch (error) {
        $("status").textContent = error.message;
    } finally {
        $("run").disabled = false;
    }
}

$("candidate").addEventListener("change", loadElements);
$("run").addEventListener("click", runScreening);

loadElements();
</script>
</body>
</html>
"""
