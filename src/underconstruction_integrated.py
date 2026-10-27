
"""
Under-construction: integrated dashboard + evidence explorer.

Run:
    python src/underconstruction_integrated.py

Main dashboard:
    http://127.0.0.1:5000/

Evidence explorer:
    http://127.0.0.1:5000/evidence
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import jsonify, render_template_string, request


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from underconstruction_app import app  # noqa: E402


ASSOCIATION_DIR = (
    ROOT / "outputs"
    / "automatic_element_evidence"
)

TWIN_FILE = (
    ROOT / "data"
    / "digital_twin"
    / "building_elements.json"
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_associations(candidate: int):
    if candidate not in range(1, 6):
        raise ValueError("Candidate must be between 1 and 5")

    path = (
        ASSOCIATION_DIR
        / f"candidate_{candidate}_associations.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Association file not found: {path.name}. "
            "Run associate_element_evidence.py first."
        )

    data = read_json(path)

    if data.get("registration_candidate_rank") != candidate:
        raise ValueError("Association candidate mismatch")

    return data


def element_metadata():
    if not TWIN_FILE.is_file():
        return {}

    result = {}

    for element in read_json(TWIN_FILE).get("elements", []):
        element_id = element.get("element_id")

        if not element_id:
            continue

        result[element_id] = {
            "category": element.get("category", "unknown"),
            "length_m": element.get("length_m"),
            "start": element.get("start"),
            "end": element.get("end"),
            "progress": element.get("progress", {}),
        }

    return result


@app.route("/api/evidence/summary")
def evidence_summary():
    try:
        candidate = int(request.args.get("candidate", "1"))
        data = get_associations(candidate)

        elements = data.get("elements", [])

        with_candidates = sum(
            bool(item.get("candidate_frames"))
            for item in elements
        )

        return jsonify({
            "candidate": candidate,
            "total_elements": len(elements),
            "with_candidates": with_candidates,
            "without_candidates": len(elements) - with_candidates,
            "registration_verified": False,
            "visibility_verified": False,
            "progress_inferred": False,
            "ranking_method": data.get("ranking_method"),
            "search_radius_m": data.get("search_radius_m"),
        })

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/evidence/elements")
def evidence_elements():
    try:
        candidate = int(request.args.get("candidate", "1"))
        data = get_associations(candidate)
        metadata = element_metadata()

        elements = []

        for item in data.get("elements", []):
            element_id = item["element_id"]
            meta = metadata.get(element_id, {})
            frames = item.get("candidate_frames", [])

            elements.append({
                "element_id": element_id,
                "category": item.get("category", "unknown"),
                "length_m": meta.get("length_m"),
                "candidate_count": item.get(
                    "candidate_count_within_radius", 0
                ),
                "nearest_distance_m": (
                    frames[0].get("distance_m")
                    if frames else None
                ),
                "association_status": item.get(
                    "association_status"
                ),
                "construction_status": "UNKNOWN",
            })

        return jsonify({
            "candidate": candidate,
            "elements": elements,
        })

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/evidence/element/<element_id>")
def evidence_element(element_id):
    try:
        candidate = int(request.args.get("candidate", "1"))
        data = get_associations(candidate)
        metadata = element_metadata()

        matching = next(
            (
                item
                for item in data.get("elements", [])
                if item.get("element_id") == element_id
            ),
            None,
        )

        if matching is None:
            return jsonify({"error": "Element not found"}), 404

        return jsonify({
            **matching,
            "geometry": metadata.get(element_id, {}),
            "registration_verified": False,
            "visibility_verified": False,
            "construction_status": "UNKNOWN",
            "warning": (
                "These photographs are spatial candidates only. "
                "The selected element has not been visually verified."
            ),
        })

    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Under-construction — Evidence Explorer</title>

<style>
:root {
    --bg: #101317;
    --panel: #1a1f25;
    --panel2: #222830;
    --border: #343b43;
    --text: #f2f2f2;
    --muted: #a7b0ba;
    --accent: #ff8a43;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, Arial, sans-serif;
}
header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}
.brand { font-size: 23px; font-weight: 750; }
.brand span { color: var(--accent); }
a { color: var(--accent); text-decoration: none; }
nav { display: flex; gap: 20px; }
main { padding: 24px; max-width: 1700px; margin: auto; }
h1 { margin: 0 0 8px; font-size: 27px; }
p { color: var(--muted); line-height: 1.55; }
.stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 20px 0;
}
.stat, .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
}
.stat strong { font-size: 27px; display: block; margin-top: 7px; }
.small { color: var(--muted); font-size: 12px; }
.layout {
    display: grid;
    grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
    gap: 16px;
}
.controls { display: flex; gap: 10px; flex-wrap: wrap; }
select, input {
    width: 100%;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 11px;
    font-size: 14px;
}
label { display: block; margin: 14px 0 7px; color: var(--muted); }
.element-list {
    margin-top: 15px;
    max-height: 60vh;
    overflow-y: auto;
}
.element {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    overflow-wrap: anywhere;
}
.element:hover { background: var(--panel2); }
.element.selected {
    border-left: 3px solid var(--accent);
    background: var(--panel2);
}
.element strong { font-size: 13px; }
.element .small { margin-top: 5px; }
.gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 18px;
}
.frame {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
}
.frame img {
    width: 100%;
    height: 190px;
    object-fit: contain;
    background: #080a0c;
    display: block;
}
.frame .details { padding: 11px; }
.frame strong { font-size: 13px; }
.warning {
    padding: 12px 14px;
    background: #30261f;
    border: 1px solid #71503a;
    border-radius: 8px;
    color: #f3cba8;
    margin-top: 14px;
    font-size: 13px;
    line-height: 1.5;
}
#viewer {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.92);
    z-index: 20;
    padding: 24px;
    text-align: center;
}
#viewer img {
    max-width: 95vw;
    max-height: 83vh;
    object-fit: contain;
}
#viewer button {
    display: block;
    margin: 0 0 15px auto;
    padding: 10px 16px;
    cursor: pointer;
}
@media(max-width: 850px) {
    .layout { grid-template-columns: 1fr; }
    .stats { grid-template-columns: 1fr; }
}
</style>
</head>

<body>
<header>
    <div class="brand">Under<span>-construction</span></div>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/evidence">Evidence Explorer</a>
    </nav>
</header>

<main>
    <h1>Automatic Evidence Explorer</h1>
    <p>
        Explore spatially shortlisted walkthrough photographs for
        individual digital-twin elements.
    </p>

    <div class="stats">
        <div class="stat">
            <div class="small">Building elements</div>
            <strong id="total">—</strong>
        </div>
        <div class="stat">
            <div class="small">With spatial candidates</div>
            <strong id="matched">—</strong>
        </div>
        <div class="stat">
            <div class="small">Without spatial candidates</div>
            <strong id="unmatched">—</strong>
        </div>
    </div>

    <div class="layout">
        <aside class="panel">
            <h3>Building elements</h3>

            <label for="candidate">Registration hypothesis</label>
            <select id="candidate">
                <option value="1">Candidate #1</option>
                <option value="2">Candidate #2</option>
                <option value="3">Candidate #3</option>
                <option value="4">Candidate #4</option>
                <option value="5">Candidate #5</option>
            </select>

            <label for="search">Search element</label>
            <input id="search" placeholder="Wall ID or category">

            <label for="filter">Evidence filter</label>
            <select id="filter">
                <option value="all">All elements</option>
                <option value="matched">Has spatial candidates</option>
                <option value="unmatched">No spatial candidates</option>
            </select>

            <div id="elementCount" class="small" style="margin-top:14px"></div>
            <div id="elements" class="element-list"></div>
        </aside>

        <section class="panel">
            <h3 id="elementTitle">Select a building element</h3>
            <div id="elementDetails" class="small">
                Choose an element on the left to view candidate images.
            </div>

            <div class="warning">
                UNVERIFIED ALIGNMENT — Nearby photographs are not proof
                that a wall is visible or constructed. Camera orientation,
                occlusion, and element identity remain unresolved.
            </div>

            <div id="gallery" class="gallery"></div>
        </section>
    </div>
</main>

<div id="viewer" onclick="closeViewer()">
    <button onclick="closeViewer()">Close ✕</button>
    <img id="viewerImage" alt="Walkthrough frame">
    <p id="viewerCaption"></p>
</div>

<script>
let elements = [];
let selectedId = null;
let generation = 0;

const byId = id => document.getElementById(id);

function safeNumber(value, digits=2) {
    return Number.isFinite(Number(value)) && value !== null
        ? Number(value).toFixed(digits)
        : "—";
}

async function getJSON(url) {
    const response = await fetch(url);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }
    return data;
}

function selectedCandidate() {
    return Number(byId("candidate").value);
}

function createText(tag, content, className="") {
    const element = document.createElement(tag);
    element.textContent = content;
    if (className) element.className = className;
    return element;
}

function renderElements() {
    const query = byId("search").value.toLowerCase().trim();
    const filter = byId("filter").value;
    const container = byId("elements");
    container.replaceChildren();

    const filtered = elements.filter(element => {
        const hasCandidates = element.candidate_count > 0;

        if (filter === "matched" && !hasCandidates) return false;
        if (filter === "unmatched" && hasCandidates) return false;

        return (
            element.element_id.toLowerCase().includes(query) ||
            element.category.toLowerCase().includes(query)
        );
    });

    byId("elementCount").textContent =
        `${filtered.length} elements shown`;

    for (const element of filtered) {
        const item = document.createElement("div");
        item.className = "element" +
            (selectedId === element.element_id ? " selected" : "");

        item.appendChild(createText(
            "strong",
            `${element.category} · ${element.element_id.slice(-8)}`
        ));

        item.appendChild(createText(
            "div",
            `${element.candidate_count} nearby frames · ` +
            `nearest ${safeNumber(element.nearest_distance_m)} m`,
            "small"
        ));

        item.addEventListener("click", () => {
            loadElement(element.element_id);
        });

        container.appendChild(item);
    }
}

async function loadAll() {
    const token = ++generation;
    selectedId = null;

    byId("elementTitle").textContent = "Select a building element";
    byId("elementDetails").textContent = "";
    byId("gallery").replaceChildren();
    byId("elements").textContent = "Loading…";

    const candidate = selectedCandidate();

    try {
        const [summary, listing] = await Promise.all([
            getJSON(`/api/evidence/summary?candidate=${candidate}`),
            getJSON(`/api/evidence/elements?candidate=${candidate}`)
        ]);

        if (token !== generation) return;

        elements = listing.elements;
        byId("total").textContent = summary.total_elements;
        byId("matched").textContent = summary.with_candidates;
        byId("unmatched").textContent = summary.without_candidates;

        renderElements();

    } catch (error) {
        if (token !== generation) return;
        elements = [];
        byId("elements").textContent = error.message;
        byId("total").textContent = "—";
        byId("matched").textContent = "—";
        byId("unmatched").textContent = "—";
    }
}

async function loadElement(elementId) {
    const token = ++generation;
    selectedId = elementId;
    renderElements();

    byId("elementTitle").textContent = elementId;
    byId("elementDetails").textContent = "Loading candidates…";
    byId("gallery").replaceChildren();

    try {
        const data = await getJSON(
            `/api/evidence/element/${encodeURIComponent(elementId)}` +
            `?candidate=${selectedCandidate()}`
        );

        if (token !== generation) return;

        const geometry = data.geometry || {};
        byId("elementDetails").textContent =
            `Category: ${data.category} · ` +
            `Length: ${safeNumber(geometry.length_m)} m · ` +
            `Candidates within radius: ${data.candidate_count_within_radius} · ` +
            `Construction status: UNKNOWN`;

        const gallery = byId("gallery");

        if (!data.candidate_frames.length) {
            gallery.appendChild(createText(
                "p", "No spatial candidates for this element."
            ));
            return;
        }

        for (const frame of data.candidate_frames) {
            const card = document.createElement("div");
            card.className = "frame";

            const image = document.createElement("img");
            image.loading = "lazy";
            image.alt = `Walkthrough frame ${frame.frame_index}`;
            image.src = `/api/frame/${frame.frame_index}?mode=raw`;

            const details = document.createElement("div");
            details.className = "details";

            details.appendChild(createText(
                "strong", `Frame ${frame.frame_index}`
            ));

            details.appendChild(createText(
                "div",
                `Distance: ${safeNumber(frame.distance_m)} m`,
                "small"
            ));

            details.appendChild(createText(
                "div",
                "Geometric candidate · Not verified",
                "small"
            ));

            card.append(image, details);

            card.addEventListener("click", () => {
                openViewer(frame);
            });

            gallery.appendChild(card);
        }

    } catch (error) {
        if (token !== generation) return;
        byId("elementDetails").textContent = error.message;
    }
}

function openViewer(frame) {
    byId("viewerImage").src =
        `/api/frame/${frame.frame_index}?mode=raw`;

    byId("viewerCaption").textContent =
        `Frame ${frame.frame_index} · ` +
        `${safeNumber(frame.distance_m)} m · ` +
        `Spatial candidate only`;

    byId("viewer").style.display = "block";
}

function closeViewer() {
    byId("viewer").style.display = "none";
    byId("viewerImage").removeAttribute("src");
}

byId("candidate").addEventListener("change", loadAll);
byId("search").addEventListener("input", renderElements);
byId("filter").addEventListener("change", renderElements);

loadAll();
</script>
</body>
</html>
"""


@app.route("/evidence")
def evidence_explorer():
    return render_template_string(HTML)


import evidence_screening
import digital_twin_3d
import wall_geometry_cleanup
import architectural_geometry

from window_geometry_api import window_geometry_bp
from registration_review import review_bp

app.register_blueprint(window_geometry_bp)
app.register_blueprint(review_bp)

from spatial_evidence_browser import spatial_evidence_bp
app.register_blueprint(spatial_evidence_bp)

if __name__ == "__main__":
    print("\nUNDER-CONSTRUCTION — INTEGRATED DASHBOARD")
    print("Dashboard: http://127.0.0.1:5000/")
    print("Evidence: http://127.0.0.1:5000/evidence")
    print("Screening: http://127.0.0.1:5000/screening")
    print("Registration Review: http://127.0.0.1:5000/registration-review")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )