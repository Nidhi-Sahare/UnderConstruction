
from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

VIEWER = ROOT / "src" / "digital_twin_3d.py"
BACKUP = ROOT / "src" / "digital_twin_3d.before_facade.py"

MARKER = "UNDER_CONSTRUCTION_FACADE_INTEGRATION_V2"


API_CODE = r'''

# UNDER_CONSTRUCTION_FACADE_INTEGRATION_V2
@app.route("/api/twin/facade")
def twin_facade_api():
    import json
    from pathlib import Path
    from flask import jsonify

    project_root = Path(__file__).resolve().parents[1]

    source_path = (
        project_root
        / "outputs"
        / "cad_facade"
        / "facade_geometry.json"
    )

    if not source_path.exists():
        return jsonify({
            "error": "Facade geometry has not been generated",
            "count": 0,
            "paths": [],
        }), 404

    data = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    paths = []

    for item in data.get("paths", []):
        if item.get("category") != "CURTAIN_WALL":
            continue

        points = item.get("points", [])

        if len(points) < 2:
            continue

        paths.append({
            "id": item.get("id"),
            "source_handle": item.get("source_handle"),
            "source_layer": item.get("source_layer"),
            "category": item.get("category"),
            "closed": item.get("closed", False),
            "points": points,
            "construction_status": "UNKNOWN",
            "vertical_dimensions_verified": False,
            "physical_component_verified": False,
        })

    return jsonify({
        "coordinate_system": "DXF_METERS",
        "representation": "PROVISIONAL_PLAN_FRAMING",
        "count": len(paths),
        "paths": paths,
        "physical_components_verified": 0,
        "vertical_dimensions_verified": False,
        "construction_progress_verified": False,
    })


'''


FACADE_JS = r'''

// UNDER_CONSTRUCTION_FACADE_INTEGRATION_V2

const facadeGroup = new THREE.Group();
facadeGroup.name = "Provisional curtain-wall framing";
scene.add(facadeGroup);

let facadePaths = [];
let facadeVisible = true;
let facadeLoadStarted = false;


function buildFacadeControl() {
    if (document.getElementById("showFacade")) {
        return;
    }

    const windowToggle = document.getElementById(
        "showWindows"
    );

    if (!windowToggle) {
        console.warn(
            "Facade toggle unavailable: window toggle missing."
        );
        return;
    }

    const parent = windowToggle.closest("label")
        || windowToggle.parentElement;

    if (!parent || !parent.parentElement) {
        console.warn(
            "Facade toggle unavailable: no suitable parent."
        );
        return;
    }

    const control = document.createElement("label");

    control.style.display = "flex";
    control.style.alignItems = "center";
    control.style.gap = "8px";
    control.style.marginTop = "8px";
    control.style.cursor = "pointer";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "showFacade";
    checkbox.checked = true;

    const label = document.createElement("span");
    label.textContent = "Facade framing (provisional)";

    control.appendChild(checkbox);
    control.appendChild(label);

    parent.parentElement.insertBefore(
        control,
        parent.nextSibling
    );

    checkbox.addEventListener("change", () => {
        facadeVisible = checkbox.checked;
        facadeGroup.visible = facadeVisible;
    });
}


function clearFacadeGeometry() {
    const materials = new Set();

    while (facadeGroup.children.length > 0) {
        const child = facadeGroup.children[0];

        facadeGroup.remove(child);

        if (child.geometry) {
            child.geometry.dispose();
        }

        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(
                    material => materials.add(material)
                );
            } else {
                materials.add(child.material);
            }
        }
    }

    materials.forEach(material => material.dispose());
}


function buildFacadeGeometry(paths) {
    clearFacadeGeometry();

    facadePaths = paths;

    // This is an assumed visualization elevation.
    // It is NOT a verified CAD height.
    const visualizationHeight = 2.4;

    const material = new THREE.LineBasicMaterial({
        color: 0x9b87f5,
        transparent: true,
        opacity: 0.9,
        depthTest: true
    });

    let rendered = 0;

    for (const path of paths) {
        if (
            !Array.isArray(path.points)
            || path.points.length < 2
        ) {
            continue;
        }

        const points = [];

        for (const point of path.points) {
            if (
                !Array.isArray(point)
                || point.length < 2
            ) {
                continue;
            }

            const x = Number(point[0]);
            const y = Number(point[1]);

            if (
                !Number.isFinite(x)
                || !Number.isFinite(y)
            ) {
                continue;
            }

            points.push(
                new THREE.Vector3(
                    x - modelCenter.x,
                    visualizationHeight,
                    y - modelCenter.y
                )
            );
        }

        if (points.length < 2) {
            continue;
        }

        const geometry = new THREE.BufferGeometry()
            .setFromPoints(points);

        const line = new THREE.Line(
            geometry,
            material
        );

        line.userData = {
            id: path.id,
            type: "PROVISIONAL_FACADE",
            sourceHandle: path.source_handle,
            sourceLayer: path.source_layer,
            constructionStatus: "UNKNOWN",
            verticalDimensionsVerified: false,
            physicalComponentVerified: false
        };

        facadeGroup.add(line);
        rendered++;
    }

    facadeGroup.visible = facadeVisible;

    console.log(
        "Provisional facade paths rendered:",
        rendered
    );
}


async function loadFacadeGeometry() {
    if (facadeLoadStarted) {
        return;
    }

    facadeLoadStarted = true;

    try {
        const response = await fetch(
            "/api/twin/facade"
        );

        if (!response.ok) {
            throw new Error(
                "Facade API returned HTTP "
                + response.status
            );
        }

        const data = await response.json();

        buildFacadeGeometry(
            data.paths || []
        );

        console.log(
            "Facade API path count:",
            data.count
        );

    } catch (error) {
        facadeLoadStarted = false;

        console.warn(
            "Facade visualization unavailable:",
            error
        );
    }
}


function waitForFacadeViewerReady() {
    buildFacadeControl();

    let attempts = 0;
    const maxAttempts = 100;

    const interval = setInterval(() => {
        attempts++;

        // The wall model establishes modelCenter.
        // Checking that walls have been rendered avoids
        // loading facade coordinates prematurely.
        const centerReady = (
            typeof modelCenter !== "undefined"
            && modelCenter !== null
            && Number.isFinite(modelCenter.x)
            && Number.isFinite(modelCenter.y)
        );

        const wallsReady = (
            typeof wallMeshes !== "undefined"
            && Array.isArray(wallMeshes)
            && wallMeshes.length > 0
        );

        if (centerReady && wallsReady) {
            clearInterval(interval);
            loadFacadeGeometry();
            return;
        }

        if (attempts >= maxAttempts) {
            clearInterval(interval);

            console.warn(
                "Facade loading skipped: "
                + "viewer geometry was not ready."
            );
        }
    }, 100);
}


window.addEventListener(
    "load",
    waitForFacadeViewerReady
);

'''


def locate_main_script(source: str):
    """
    Find the inline HTML <script> block that contains
    the viewer's Three.js initialization.

    This deliberately does not assume that there is
    only one script tag.
    """
    pattern = re.compile(
        r"<script\b([^>]*)>(.*?)</script\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates = []

    for match in pattern.finditer(source):
        attributes = match.group(1)
        body = match.group(2)

        if re.search(
            r"\bsrc\s*=",
            attributes,
            flags=re.IGNORECASE,
        ):
            continue

        indicators = [
            "THREE",
            "modelCenter",
            "windowGroup",
            "showWindows",
        ]

        score = sum(
            indicator in body
            for indicator in indicators
        )

        if score >= 3:
            candidates.append(
                (score, len(body), match)
            )

    if not candidates:
        raise RuntimeError(
            "Could not identify the main viewer JavaScript "
            "block. No changes were saved."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    if (
        len(candidates) > 1
        and candidates[0][0] == candidates[1][0]
        and candidates[0][1] == candidates[1][1]
    ):
        raise RuntimeError(
            "Multiple equally likely viewer scripts found. "
            "No changes were saved."
        )

    return candidates[0][2]


def insert_facade_javascript(source: str):
    match = locate_main_script(source)

    # Insert before the selected script's closing tag.
    insertion_position = match.end(2)

    return (
        source[:insertion_position]
        + "\n"
        + FACADE_JS
        + "\n"
        + source[insertion_position:]
    )


def insert_facade_api(source: str):
    route_pattern = re.compile(
        r"(?m)^[ \t]*@app\.route\("
    )

    match = route_pattern.search(source)

    if not match:
        raise RuntimeError(
            "Could not locate the Flask route definitions. "
            "No changes were saved."
        )

    return (
        source[:match.start()]
        + API_CODE
        + source[match.start():]
    )


def validate_existing_viewer(source: str):
    required = [
        "windowGroup",
        "windowMeshes",
        "showWindows",
        "/api/twin/windows",
    ]

    missing = [
        name
        for name in required
        if name not in source
    ]

    if missing:
        raise RuntimeError(
            "This integration requires the existing "
            "window-enabled viewer.\n"
            "Missing: "
            + ", ".join(missing)
        )

    if MARKER in source:
        print(
            "Facade integration V2 is already installed."
        )
        return False

    if "UNDER_CONSTRUCTION_FACADE_INTEGRATION_V1" in source:
        raise RuntimeError(
            "An older facade integration is already present. "
            "Restore the pre-facade backup before running V2."
        )

    return True


def main():
    if not VIEWER.exists():
        raise FileNotFoundError(VIEWER)

    original = VIEWER.read_text(
        encoding="utf-8"
    )

    if not validate_existing_viewer(original):
        return

    # Verify that the original Python file is valid.
    ast.parse(
        original,
        filename=str(VIEWER),
    )

    # Insert JavaScript first, while the original
    # viewer's script blocks are easy to identify.
    updated = insert_facade_javascript(original)

    # Then register the Flask API route.
    updated = insert_facade_api(updated)

    # Verify the resulting Python file before saving.
    ast.parse(
        updated,
        filename=str(VIEWER),
    )

    if updated.count(MARKER) != 2:
        raise RuntimeError(
            "Integration marker validation failed. "
            "No changes were saved."
        )

    # Backup is created only after validation passes.
    if not BACKUP.exists():
        BACKUP.write_text(
            original,
            encoding="utf-8",
        )

    VIEWER.write_text(
        updated,
        encoding="utf-8",
    )

    print()
    print("=== FACADE INTEGRATION COMPLETE ===")
    print(
        "Backup:",
        BACKUP.relative_to(ROOT),
    )
    print(
        "Updated:",
        VIEWER.relative_to(ROOT),
    )
    print("API: /api/twin/facade")
    print("Viewer: /twin")
    print()
    print(
        "Facade geometry is provisional."
    )
    print(
        "No physical openings or construction "
        "progress have been verified."
    )


if __name__ == "__main__":
    main()
