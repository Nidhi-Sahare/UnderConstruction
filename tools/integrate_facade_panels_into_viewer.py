
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VIEWER_FILE = ROOT / "src/digital_twin_3d.py"
BACKUP_FILE = ROOT / "src/digital_twin_3d.before_facade_panels.py"

RUNS_FILE = ROOT / "outputs/cad_facade/facade_runs.json"

START_MARKER = "UNDER_CONSTRUCTION_FACADE_PANELS_START"
END_MARKER = "UNDER_CONSTRUCTION_FACADE_PANELS_END"

EXPECTED_METHOD = "CURTAIN_WALL_COLLINEAR_SOURCE_SEGMENTS"


def validate_inputs():
    if not VIEWER_FILE.exists():
        raise FileNotFoundError(
            f"Missing 3D viewer: {VIEWER_FILE}"
        )

    if not RUNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing façade runs: {RUNS_FILE}"
        )

    data = json.loads(
        RUNS_FILE.read_text(encoding="utf-8")
    )

    if data.get("method") != EXPECTED_METHOD:
        raise RuntimeError(
            "The façade runs are not curtain-wall-driven. "
            "Run tools/reconstruct_facade_runs.py first."
        )

    runs = data.get("runs")

    if not isinstance(runs, list) or not runs:
        raise RuntimeError(
            "No CAD-supported façade runs were found."
        )

    return data


def build_panel_segments(data):
    """
    Extract only original CAD-supported source segments.

    No unverified gaps are filled.
    No physical façade dimensions are inferred.
    """
    segments = []
    seen = set()

    for run in data["runs"]:
        for segment in run.get("source_segments", []):
            start = segment.get("start")
            end = segment.get("end")

            if (
                not isinstance(start, list)
                or not isinstance(end, list)
                or len(start) < 2
                or len(end) < 2
            ):
                continue

            try:
                x1 = float(start[0])
                y1 = float(start[1])
                x2 = float(end[0])
                y2 = float(end[1])
            except (TypeError, ValueError):
                continue

            if not all(
                abs(value) < 1e7
                for value in (x1, y1, x2, y2)
            ):
                continue

            length = (
                (x2 - x1) ** 2
                + (y2 - y1) ** 2
            ) ** 0.5

            if length < 0.20:
                continue

            # Treat opposite directions as the same
            # geometric edge for duplicate screening.
            a = (round(x1, 5), round(y1, 5))
            b = (round(x2, 5), round(y2, 5))
            key = tuple(sorted((a, b)))

            if key in seen:
                continue

            seen.add(key)

            segments.append({
                "start": [x1, y1],
                "end": [x2, y2],
                "length_m": round(length, 6),
                "source_segment_id": segment.get(
                    "segment_id"
                ),
                "source_handle": segment.get(
                    "source_handle"
                ),
            })

    return segments


def build_js():
    """
    Install an isolated overlay.

    Existing façade lines remain available.
    The new panel layer can be toggled separately.
    """
    return r"""
// UNDER_CONSTRUCTION_FACADE_PANELS_START

const provisionalFacadePanels = new THREE.Group();
provisionalFacadePanels.name = "Provisional CAD Facade Panels";
scene.add(provisionalFacadePanels);

const FACADE_PANEL_HEIGHT_ASSUMED_M = 3.0;
const FACADE_PANEL_THICKNESS_VISUAL_M = 0.035;

const facadePanelMaterial = new THREE.MeshBasicMaterial({
    color: 0x67d6d0,
    transparent: true,
    opacity: 0.23,
    side: THREE.DoubleSide,
    depthWrite: false
});

const facadePanelEdgesMaterial = new THREE.LineBasicMaterial({
    color: 0x17a8a4,
    transparent: true,
    opacity: 0.75
});

function buildProvisionalFacadePanel(segment) {
    if (!segment || !segment.start || !segment.end) {
        return null;
    }

    const x1 = Number(segment.start[0]);
    const z1 = Number(segment.start[1]);
    const x2 = Number(segment.end[0]);
    const z2 = Number(segment.end[1]);

    if (![x1, z1, x2, z2].every(Number.isFinite)) {
        return null;
    }

    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.hypot(dx, dz);

    if (length < 0.20) {
        return null;
    }

    const geometry = new THREE.BoxGeometry(
        length,
        FACADE_PANEL_HEIGHT_ASSUMED_M,
        FACADE_PANEL_THICKNESS_VISUAL_M
    );

    const mesh = new THREE.Mesh(
        geometry,
        facadePanelMaterial
    );

    mesh.position.set(
        (x1 + x2) / 2 - modelCenter.x,
        FACADE_PANEL_HEIGHT_ASSUMED_M / 2,
        (z1 + z2) / 2 - modelCenter.y
    );

    mesh.rotation.y = -Math.atan2(dz, dx);

    mesh.userData = {
        type: "PROVISIONAL_FACADE_PANEL",
        sourceSegmentId: segment.source_segment_id,
        sourceHandle: segment.source_handle,
        constructionStatus: "UNKNOWN",
        heightVerified: false,
        openingsVerified: false
    };

    const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        facadePanelEdgesMaterial
    );

    mesh.add(edges);

    return mesh;
}

async function loadProvisionalFacadePanels() {
    try {
        const response = await fetch(
            "/api/twin/facade/panels"
        );

        if (!response.ok) {
            throw new Error(
                "Façade panel API returned " + response.status
            );
        }

        const data = await response.json();

        if (!Array.isArray(data.segments)) {
            throw new Error(
                "Façade panel response has no segments"
            );
        }

        for (const segment of data.segments) {
            const panel = buildProvisionalFacadePanel(
                segment
            );

            if (panel) {
                provisionalFacadePanels.add(panel);
            }
        }

        console.log(
            "Provisional CAD façade panels:",
            provisionalFacadePanels.children.length
        );

    } catch (error) {
        console.warn(
            "Unable to load provisional façade panels:",
            error
        );
    }
}

function installFacadePanelToggle() {
    const existing = document.getElementById(
        "showFacadePanels"
    );

    if (existing) {
        return;
    }

    const label = document.createElement("label");

    label.style.display = "block";
    label.style.marginTop = "8px";
    label.style.cursor = "pointer";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "showFacadePanels";
    checkbox.checked = true;

    checkbox.addEventListener("change", () => {
        provisionalFacadePanels.visible =
            checkbox.checked;
    });

    label.appendChild(checkbox);
    label.appendChild(
        document.createTextNode(
            " Provisional façade panels (assumed 3 m)"
        )
    );

    const windowsToggle = document.getElementById(
        "showWindows"
    );

    if (windowsToggle && windowsToggle.parentElement) {
        windowsToggle.parentElement.insertAdjacentElement(
            "afterend",
            label
        );
    } else {
        document.body.appendChild(label);
    }
}

function initializeProvisionalFacadePanels() {
    if (
        typeof scene === "undefined"
        || typeof modelCenter === "undefined"
    ) {
        console.warn(
            "Viewer geometry is not ready for façade panels"
        );
        return;
    }

    installFacadePanelToggle();
    loadProvisionalFacadePanels();
}

initializeProvisionalFacadePanels();

// UNDER_CONSTRUCTION_FACADE_PANELS_END
"""


def build_api_code():
    return '''
# UNDER_CONSTRUCTION_FACADE_PANELS_START

@app.route("/api/twin/facade/panels")
def twin_facade_panels_api():
    """CAD-supported provisional façade panel segments."""
    from flask import jsonify

    import json
    import math

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/cad_facade/facade_runs.json"

    if not source.exists():
        return jsonify({
            "error": "FACADE_RUNS_NOT_FOUND",
            "segments": [],
        }), 404

    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    if data.get("method") != (
        "CURTAIN_WALL_COLLINEAR_SOURCE_SEGMENTS"
    ):
        return jsonify({
            "error": "UNEXPECTED_FACADE_RUN_METHOD",
            "segments": [],
        }), 409

    segments = []
    seen = set()

    for run in data.get("runs", []):
        for item in run.get("source_segments", []):
            a = item.get("start")
            b = item.get("end")

            if not (
                isinstance(a, list)
                and isinstance(b, list)
                and len(a) >= 2
                and len(b) >= 2
            ):
                continue

            try:
                x1, y1 = float(a[0]), float(a[1])
                x2, y2 = float(b[0]), float(b[1])
            except (ValueError, TypeError):
                continue

            if not all(
                math.isfinite(v)
                for v in (x1, y1, x2, y2)
            ):
                continue

            length = math.hypot(x2 - x1, y2 - y1)

            if length < 0.20:
                continue

            key = tuple(sorted((
                (round(x1, 5), round(y1, 5)),
                (round(x2, 5), round(y2, 5)),
            )))

            if key in seen:
                continue

            seen.add(key)

            segments.append({
                "start": [x1, y1],
                "end": [x2, y2],
                "length_m": round(length, 6),
                "source_segment_id": item.get(
                    "segment_id"
                ),
                "source_handle": item.get(
                    "source_handle"
                ),
            })

    return jsonify({
        "representation":
            "PROVISIONAL_CAD_SUPPORTED_FACADE_PANELS",
        "coordinate_system": "DXF_METERS",
        "assumed_panel_height_m": 3.0,
        "height_verified": False,
        "physical_facade_verified": False,
        "wall_cutouts_generated": False,
        "construction_status": "UNKNOWN",
        "count": len(segments),
        "segments": segments,
    })

# UNDER_CONSTRUCTION_FACADE_PANELS_END
'''


def remove_previous_block(source, start, end):
    """
    Allow rerunning the integration without
    duplicating the installed code.
    """
    pattern = re.compile(
        r"\n?[^\n]*"
        + re.escape(start)
        + r".*?"
        + re.escape(end)
        + r"[^\n]*\n?",
        flags=re.DOTALL,
    )

    return pattern.sub("\n", source)


def insert_api(source):
    code = build_api_code()

    # Route is installed before the module's launch block,
    # or appended when there is no direct launch block.
    launch = re.search(
        r"(?m)^if __name__\s*==\s*['\"]__main__['\"]\s*:",
        source,
    )

    if launch:
        return (
            source[:launch.start()]
            + code
            + "\n"
            + source[launch.start():]
        )

    return source.rstrip() + "\n\n" + code


def insert_javascript(source):
    """
    Inject into the final inline script containing
    the existing Three.js scene and modelCenter.

    Does not modify existing mesh-generation logic.
    """
    script_pattern = re.compile(
        r"(<script(?:\s[^>]*)?>)(.*?)(</script>)",
        flags=re.DOTALL | re.IGNORECASE,
    )

    matches = list(script_pattern.finditer(source))

    candidates = []

    for match in matches:
        body = match.group(2)

        if (
            "THREE" in body
            and "modelCenter" in body
            and "scene" in body
        ):
            candidates.append(match)

    if not candidates:
        raise RuntimeError(
            "Could not locate the Three.js viewer script. "
            "The original viewer file was not changed."
        )

    match = max(
        candidates,
        key=lambda item: len(item.group(2)),
    )

    javascript = build_js()

    # Insert near the end of the existing script.
    return (
        source[:match.end(2)]
        + "\n"
        + javascript
        + "\n"
        + source[match.end(2):]
    )


def main():
    data = validate_inputs()

    original = VIEWER_FILE.read_text(
        encoding="utf-8"
    )

    if START_MARKER in original:
        raise RuntimeError(
            "Façade panels are already installed. "
            "No changes were made."
        )

    # Confirm this is the expected viewer architecture.
    required = [
        "THREE",
        "modelCenter",
        "scene",
    ]

    missing = [
        token for token in required
        if token not in original
    ]

    if missing:
        raise RuntimeError(
            "Unexpected viewer architecture. "
            f"Missing: {missing}"
        )

    updated = insert_api(original)
    updated = insert_javascript(updated)

    try:
        ast.parse(updated)
    except SyntaxError as error:
        raise RuntimeError(
            "Integration failed Python syntax validation. "
            f"No changes were made: {error}"
        ) from error

    # Save backup only after validation succeeds.
    shutil.copy2(VIEWER_FILE, BACKUP_FILE)

    VIEWER_FILE.write_text(
        updated,
        encoding="utf-8"
    )

    source_count = sum(
        len(run.get("source_segments", []))
        for run in data["runs"]
    )

    print("\n=== FACADE PANEL INTEGRATION COMPLETE ===")
    print("Viewer:", VIEWER_FILE.relative_to(ROOT))
    print("Backup:", BACKUP_FILE.relative_to(ROOT))
    print("CAD-supported source edges:", source_count)
    print("API: /api/twin/facade/panels")
    print("Viewer: /twin")
    print("Panel height: ASSUMED 3.0 m")
    print("Physical façade verified: NO")
    print("Construction progress verified: NO")


if __name__ == "__main__":
    main()
