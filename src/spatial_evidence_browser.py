
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    jsonify,
    render_template_string,
    request,
    send_file,
)


ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    ROOT
    / "outputs"
    / "spatial_evidence"
    / "spatial_evidence_index.json"
)


# SPATIAL_QUALITY_INTEGRATION_V1
QUALITY_PATH = (
    ROOT / "outputs" / "visual_quality"
    / "walkthrough_quality.json"
)

_quality_cache = {"mtime": None, "frames": {}}


def load_quality():
    if not QUALITY_PATH.is_file():
        return {}

    mtime = QUALITY_PATH.stat().st_mtime_ns

    if _quality_cache["mtime"] != mtime:
        data = json.loads(
            QUALITY_PATH.read_text(encoding="utf-8")
        )
        _quality_cache["frames"] = {
            int(frame["frame_index"]): frame
            for frame in data.get("frames", [])
        }
        _quality_cache["mtime"] = mtime

    return _quality_cache["frames"]


spatial_evidence_bp = Blueprint(
    "spatial_evidence_browser",
    __name__,
)


@lru_cache(maxsize=1)
def load_index_cached():
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"Missing spatial evidence index: {INDEX_PATH}"
        )

    return json.loads(
        INDEX_PATH.read_text(encoding="utf-8")
    )


def load_index():
    # Reload automatically if the index changes.
    modified = INDEX_PATH.stat().st_mtime_ns

    if getattr(load_index, "_modified", None) != modified:
        load_index_cached.cache_clear()
        load_index._modified = modified

    return load_index_cached()


def safe_frame_path(relative_path):
    if not isinstance(relative_path, str):
        abort(404)

    path = (ROOT / relative_path).resolve()

    # Never allow arbitrary filesystem access.
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        abort(404)

    if not path.is_file():
        abort(404)

    if path.suffix.lower() not in (
        ".jpg",
        ".jpeg",
        ".png",
    ):
        abort(404)

    return path


@spatial_evidence_bp.get("/api/spatial-evidence/summary")
def spatial_summary():
    data = load_index()

    return jsonify({
        "sequence": data.get("sequence"),
        "summary": data.get("summary", {}),
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
    })


@spatial_evidence_bp.get("/api/spatial-evidence/frames")
def spatial_frames():
    data = load_index()

    classification = request.args.get(
        "classification",
        "ALL",
    ).upper()

    allowed = {
        "ALL",
        "SPATIAL_CONSENSUS",
        "CANDIDATE_DEPENDENT",
        "UNRESOLVED",
    }

    if classification not in allowed:
        return jsonify({
            "error": "Invalid classification",
        }), 400

    try:
        limit = max(
            1,
            min(171, int(request.args.get("limit", 171))),
        )
    except ValueError:
        return jsonify({
            "error": "Invalid limit",
        }), 400

    frames = data.get("frames", [])

    if classification != "ALL":
        frames = [
            frame
            for frame in frames
            if frame.get("classification") == classification
        ]

    results = []
    quality = load_quality()

    for frame in frames[:limit]:
        results.append({
            "frame_index": frame["frame_index"],
            "filename": frame["filename"],
            "quality_class": quality.get(
                int(frame["frame_index"]), {}
            ).get("quality_class", "NOT_SCREENED"),
            "classification": frame["classification"],
            "candidate_max_separation_m": frame.get(
                "candidate_max_separation_m"
            ),
            "consensus_element_count": len(
                frame.get("consensus_elements", [])
            ),
            "image_url": (
                "/api/spatial-evidence/image/"
                + str(frame["frame_index"])
            ),
        })

    return jsonify({
        "total": len(frames),
        "frames": results,
    })


@spatial_evidence_bp.get(
    "/api/spatial-evidence/frame/<int:frame_index>"
)
def spatial_frame_detail(frame_index):
    data = load_index()

    frame = next(
        (
            item
            for item in data.get("frames", [])
            if item["frame_index"] == frame_index
        ),
        None,
    )

    if frame is None:
        abort(404)

    result = dict(frame)
    result["visual_quality"] = load_quality().get(
        frame_index
    )
    return jsonify(result)


@spatial_evidence_bp.get(
    "/api/spatial-evidence/image/<int:frame_index>"
)
def spatial_frame_image(frame_index):
    data = load_index()

    frame = next(
        (
            item
            for item in data.get("frames", [])
            if item["frame_index"] == frame_index
        ),
        None,
    )

    if frame is None:
        abort(404)

    path = safe_frame_path(frame["image_path"])

    return send_file(
        path,
        mimetype="image/jpeg",
        conditional=True,
    )


@spatial_evidence_bp.get(
    "/api/spatial-evidence/element/<element_id>"
)
def spatial_element_detail(element_id):
    data = load_index()

    element = data.get("elements", {}).get(element_id)

    if element is None:
        abort(404)

    return jsonify(element)


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>Under-construction | Spatial Evidence</title>

<style>
:root {
    color-scheme: dark;
    font-family: Inter, Arial, sans-serif;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: #10151e;
    color: #edf2f7;
}

header {
    padding: 24px 32px;
    border-bottom: 1px solid #303949;
}

header h1 {
    margin: 0 0 8px;
    font-size: 25px;
}

header p {
    margin: 0;
    color: #aab7c9;
    line-height: 1.5;
}

nav {
    display: flex;
    gap: 16px;
    margin-top: 16px;
}

a { color: #83caff; }

main {
    padding: 24px 32px;
    max-width: 1600px;
    margin: auto;
}

.summary {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 22px;
}

.metric {
    background: #1b2532;
    border: 1px solid #303949;
    border-radius: 12px;
    padding: 16px;
    min-width: 160px;
}

.metric strong {
    display: block;
    font-size: 27px;
    margin-bottom: 5px;
}

.metric span {
    color: #aab7c9;
    font-size: 13px;
}

.controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 22px;
}

select, input {
    background: #1b2532;
    color: white;
    border: 1px solid #455368;
    border-radius: 8px;
    padding: 10px 12px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(260px, 420px) 1fr;
    gap: 22px;
}

.frame-list {
    max-height: 75vh;
    overflow-y: auto;
    display: grid;
    gap: 10px;
}

.frame-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1b2532;
    border: 1px solid #303949;
    border-radius: 10px;
    padding: 9px;
    cursor: pointer;
    color: inherit;
    text-align: left;
    width: 100%;
}

.frame-card:hover,
.frame-card.active {
    border-color: #76c7ff;
}

.frame-card img {
    width: 96px;
    height: 75px;
    object-fit: cover;
    border-radius: 6px;
}

.frame-card small {
    display: block;
    color: #aab7c9;
    margin-top: 6px;
}

.panel {
    background: #1b2532;
    border: 1px solid #303949;
    border-radius: 12px;
    padding: 18px;
    min-width: 0;
}

.preview {
    display: block;
    width: 100%;
    max-height: 490px;
    object-fit: contain;
    background: #090d13;
    border-radius: 8px;
}

.notice {
    margin: 15px 0;
    padding: 12px;
    border: 1px solid #806d45;
    border-radius: 8px;
    color: #e5d3a3;
    line-height: 1.5;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
    font-size: 13px;
}

td, th {
    text-align: left;
    padding: 10px 8px;
    border-bottom: 1px solid #344052;
}

th { color: #aab7c9; }

.empty {
    color: #aab7c9;
    padding: 20px;
}

@media (max-width: 850px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .frame-list {
        max-height: 350px;
    }

    main, header {
        padding: 18px;
    }
}
</style>
</head>

<body>
<header>
    <h1>Under-construction / Spatial Evidence</h1>
    <p>
        Explore walkthrough images and provisional CAD proximity
        associations. No image visibility or construction progress
        has been verified.
    </p>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/twin">3D Twin</a>
        <a href="/evidence">Evidence</a>
    </nav>
</header>

<main>
    <div id="summary" class="summary"></div>

    <div class="controls">
        <label for="filter">Association:</label>
        <select id="filter">
            <option value="ALL">All frames</option>
            <option value="SPATIAL_CONSENSUS">
                Proximity consensus
            </option>
            <option value="CANDIDATE_DEPENDENT">
                Candidate-dependent
            </option>
            <option value="UNRESOLVED">
                Unresolved
            </option>
        </select>

        <label for="qualityFilter">Image quality:</label>
        <select id="qualityFilter">
            <option value="ALL">All quality levels</option>
            <option value="USABLE_CANDIDATE">
                Usable candidate
            </option>
            <option value="LOW_LIGHT">Low light</option>
            <option value="LOW_TEXTURE">Low texture</option>
            <option value="LOW_DETAIL_OR_BLUR">
                Low detail / blur
            </option>
            <option value="OVEREXPOSED">Overexposed</option>
            <option value="NOT_SCREENED">Not screened</option>
        </select>

        <input
            id="search"
            type="search"
            placeholder="Search frame number"
        >
    </div>

    <div class="layout">
        <section>
            <div id="frameList" class="frame-list"></div>
        </section>

        <section id="detail" class="panel">
            <p class="empty">Select a walkthrough frame.</p>
        </section>
    </div>
</main>

<script>
let allFrames = [];
let selectedFrame = null;

const $ = id => document.getElementById(id);

function escapeHtml(value) {
    return String(value ?? "").replace(
        /[&<>"']/g,
        char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        })[char]
    );
}

async function getJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(
            "Request failed: " + response.status
        );
    }

    return response.json();
}

function metric(value, label) {
    return `
        <div class="metric">
            <strong>${escapeHtml(value)}</strong>
            <span>${escapeHtml(label)}</span>
        </div>
    `;
}

async function initialize() {
    try {
        const [summary, frames] = await Promise.all([
            getJSON("/api/spatial-evidence/summary"),
            getJSON("/api/spatial-evidence/frames")
        ]);

        const s = summary.summary;
        const counts = s.classification_counts || {};

        $("summary").innerHTML =
            metric(s.matched_frames, "Matched frames") +
            metric(
                counts.SPATIAL_CONSENSUS || 0,
                "Proximity consensus"
            ) +
            metric(
                counts.CANDIDATE_DEPENDENT || 0,
                "Candidate-dependent"
            ) +
            metric(
                counts.UNRESOLVED || 0,
                "Unresolved"
            ) +
            metric(
                s.consensus_source_entities || 0,
                "Linked CAD source entities"
            );

        allFrames = frames.frames;
        renderFrames();

        if (allFrames.length) {
            await selectFrame(allFrames[0].frame_index);
        }
    } catch (error) {
        $("detail").textContent = error.message;
    }
}

function renderFrames() {
    const filter = $("filter").value;
    const search = $("search").value.trim();
    const qualityFilter = $("qualityFilter").value;

    const visible = allFrames.filter(frame => {
        const matchesFilter =
            filter === "ALL" ||
            frame.classification === filter;

        const matchesSearch =
            !search ||
            String(frame.frame_index).includes(search);

        const matchesQuality =
            qualityFilter === "ALL" ||
            frame.quality_class === qualityFilter;

        return matchesFilter && matchesSearch && matchesQuality;
    });

    const container = $("frameList");
    container.replaceChildren();

    if (!visible.length) {
        container.innerHTML =
            '<p class="empty">No matching frames.</p>';
        return;
    }

    for (const frame of visible) {
        const button = document.createElement("button");
        button.className =
            "frame-card" +
            (
                frame.frame_index === selectedFrame
                    ? " active"
                    : ""
            );

        const img = document.createElement("img");
        img.src = frame.image_url;
        img.loading = "lazy";
        img.alt = "Walkthrough frame";

        const info = document.createElement("div");
        info.innerHTML = `
            <strong>Frame ${frame.frame_index}</strong>
            <small>${escapeHtml(frame.classification)}</small>
            <small>
                Quality: ${escapeHtml(
                    frame.quality_class || "NOT_SCREENED"
                )}
            </small>
            <small>
                Candidate spread:
                ${escapeHtml(
                    frame.candidate_max_separation_m
                )} m
            </small>
        `;

        button.append(img, info);
        button.addEventListener(
            "click",
            () => selectFrame(frame.frame_index)
        );

        container.appendChild(button);
    }
}

async function selectFrame(frameIndex) {
    selectedFrame = frameIndex;
    renderFrames();

    $("detail").innerHTML =
        '<p class="empty">Loading frame...</p>';

    try {
        const frame = await getJSON(
            "/api/spatial-evidence/frame/" + frameIndex
        );

        const quality = frame.visual_quality || null;
        const metrics = quality?.metrics || {};

        const qualityHtml = quality
            ? `
                <h3>Visual quality screening</h3>
                <p>
                    <strong>${escapeHtml(
                        quality.quality_class
                    )}</strong>
                </p>
                <table>
                    <tbody>
                        <tr>
                            <th>Mean brightness</th>
                            <td>${escapeHtml(
                                metrics.mean_brightness
                            )}</td>
                        </tr>
                        <tr>
                            <th>Contrast (std)</th>
                            <td>${escapeHtml(
                                metrics.contrast_std
                            )}</td>
                        </tr>
                        <tr>
                            <th>Laplacian variance</th>
                            <td>${escapeHtml(
                                metrics.laplacian_variance
                            )}</td>
                        </tr>
                        <tr>
                            <th>Edge fraction</th>
                            <td>${escapeHtml(
                                metrics.edge_fraction
                            )}</td>
                        </tr>
                    </tbody>
                </table>
                <p class="empty">
                    Heuristic image-quality screening only.
                    This does not verify CAD alignment,
                    element visibility, or construction progress.
                </p>
              `
            : `<p class="empty">
                   No image-quality screening available.
               </p>`;

        const candidates = frame.candidate_results || [];
        const consensus = frame.consensus_elements || [];

        const candidateRows = candidates.map(item => `
            <tr>
                <td>#${escapeHtml(item.candidate_rank)}</td>
                <td>
                    ${escapeHtml(item.cad_position.x)},
                    ${escapeHtml(item.cad_position.y)}
                </td>
                <td>${item.nearby_elements.length}</td>
            </tr>
        `).join("");

        const elementRows = consensus.map(item => `
            <tr>
                <td>${escapeHtml(item.element_id)}</td>
                <td>${escapeHtml(item.category)}</td>
                <td>
                    ${escapeHtml(item.supporting_candidates)}
                    / ${escapeHtml(item.candidate_count)}
                </td>
                <td>${escapeHtml(item.distance_m)} m</td>
            </tr>
        `).join("");

        $("detail").innerHTML = `
            <h2>Frame ${frame.frame_index}</h2>
            <p>${escapeHtml(frame.classification)}</p>

            <img
                class="preview"
                src="/api/spatial-evidence/image/${frameIndex}"
                alt="Original walkthrough photograph"
            >

            <div class="notice">
                This is an unverified spatial hypothesis.
                Proximity does not prove that the CAD element
                appears in this image. Construction progress:
                UNKNOWN.
            </div>

            <p>
                Maximum candidate separation:
                <strong>
                    ${escapeHtml(
                        frame.candidate_max_separation_m
                    )} m
                </strong>
            </p>

            ${qualityHtml}

            <h3>Registration candidates</h3>
            <table>
                <thead>
                    <tr>
                        <th>Candidate</th>
                        <th>CAD position (m)</th>
                        <th>Nearby entities</th>
                    </tr>
                </thead>
                <tbody>${candidateRows}</tbody>
            </table>

            <h3>Shared nearby CAD source entities</h3>

            ${
                consensus.length
                    ? `<table>
                        <thead>
                            <tr>
                                <th>CAD ID</th>
                                <th>Category</th>
                                <th>Support</th>
                                <th>Min. distance</th>
                            </tr>
                        </thead>
                        <tbody>${elementRows}</tbody>
                       </table>`
                    : `<p class="empty">
                        No shared nearby CAD source entities
                        under the current indexing criteria.
                       </p>`
            }
        `;
    } catch (error) {
        $("detail").textContent = error.message;
    }
}

$("filter").addEventListener("change", renderFrames);
$("qualityFilter").addEventListener("change", renderFrames);
$("search").addEventListener("input", renderFrames);

async function initializeFromURL() {
    await initialize();

    const params = new URLSearchParams(
        window.location.search
    );

    const requestedFrame = params.get("frame");

    if (requestedFrame === null) {
        return;
    }

    const index = Number(requestedFrame);

    if (
        !Number.isInteger(index) ||
        !allFrames.some(
            frame => frame.frame_index === index
        )
    ) {
        return;
    }

    await selectFrame(index);
}

initializeFromURL();
</script>
</body>
</html>
"""

@spatial_evidence_bp.get(
    "/api/spatial-evidence/element-search"
)
def spatial_element_search():
    data = load_index()

    query = request.args.get(
        "q", ""
    ).strip().lower()

    category = request.args.get(
        "category", ""
    ).strip().upper()

    try:
        limit = max(
            1,
            min(
                100,
                int(request.args.get("limit", 25)),
            ),
        )
    except ValueError:
        return jsonify({
            "error": "Invalid limit"
        }), 400

    results = []

    for element_id, element in data.get(
        "elements", {}
    ).items():

        if query and query not in element_id.lower():
            continue

        if category and (
            element.get("category", "").upper()
            != category
        ):
            continue

        frames = element.get("frames", [])

        results.append({
            "element_id": element_id,
            "category": element.get("category"),
            "source_layer": element.get("source_layer"),
            "candidate_frame_count": len(frames),
            "registration_verified": False,
            "visibility_verified": False,
            "construction_progress": "UNKNOWN",
            "frames": frames[:5],
        })

    results.sort(
        key=lambda item: (
            -item["candidate_frame_count"],
            item["element_id"],
        )
    )

    return jsonify({
        "total": len(results),
        "results": results[:limit],
    })


@spatial_evidence_bp.get("/spatial-evidence")
def spatial_evidence_page():
    return render_template_string(PAGE)
