
"""
Under-construction — Step 107
CAD-first construction progress API.

Purpose
-------
Create a usable application backend without falsely claiming
that the camera walkthrough has been registered to the CAD plan.

Provides:
    GET /api/health
    GET /api/project
    GET /api/cad/elements
    GET /api/cad/elements/<element_id>
    GET /api/walkthrough/frames
    GET /api/walkthrough/frame/<filename>
    GET /api/progress/summary
    GET /api/diagnostics

Also writes:
    outputs/cad_progress_api/project_snapshot.json

Requirements:
    pip install flask ezdxf

Run:
    .\\.venv\\Scripts\\python.exe src\\cad_progress_api.py

Important:
    - All CAD elements are PLANNED, not confirmed built.
    - All construction progress statuses are UNVERIFIED.
    - Walkthrough frames are NOT linked to CAD elements.
    - No geometry registration or progress estimation occurs.
    - No AI or LLMs are used.
"""

import csv
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ezdxf
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import NotFound


ROOT = Path(__file__).resolve().parents[1]

CAD_PATH = (
    ROOT / "data" / "floorplans" / "floor_2.dxf"
)

CAM_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST_PATH = CAM_ROOT / "frame_manifest.csv"
FRAME_DIR = CAM_ROOT / "frames"

OUTPUT_DIR = ROOT / "outputs" / "cad_progress_api"

PROJECT_ID = "underconstruction_floor_2"
PROJECT_NAME = "Under-construction — Floor 2"

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

# This backend intentionally does not assume that one DXF
# drawing unit equals one meter. The original CAD coordinate
# values are preserved.
#
# Do not calculate wall areas, completion percentages, or
# physical distances until CAD units and scale are confirmed.

MAX_FRAME_RESULTS = 250
DEFAULT_FRAME_RESULTS = 30

HOST = "127.0.0.1"
PORT = int(os.environ.get("CAD_PROGRESS_PORT", "5051"))

app = Flask(__name__)

PROJECT_CACHE = {
    "loaded": False,
    "elements": [],
    "element_by_id": {},
    "frames": [],
    "frame_by_name": {},
    "cad_metadata": {},
    "diagnostics": {},
}


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value):
        return None

    return value


def xy_point(value):
    try:
        x = finite_number(value[0])
        y = finite_number(value[1])
    except (IndexError, TypeError, KeyError):
        return None

    if x is None or y is None:
        return None

    return [x, y]


def unique_points(points):
    cleaned = []

    for point in points:
        if point is None:
            continue

        if (
            cleaned
            and point == cleaned[-1]
        ):
            continue

        cleaned.append(point)

    return cleaned


def polyline_length(points, closed=False):
    if len(points) < 2:
        return 0.0

    pairs = list(
        zip(points[:-1], points[1:])
    )

    if closed and points[0] != points[-1]:
        pairs.append(
            (points[-1], points[0])
        )

    total = 0.0

    for a, b in pairs:
        total += math.hypot(
            b[0] - a[0],
            b[1] - a[1],
        )

    return float(total)


def make_element(
    element_id,
    handle,
    layer,
    source_type,
    points,
    closed=False,
    geometry_supported=True,
    warning=None,
):
    points = unique_points(points)

    length = (
        polyline_length(points, closed)
        if geometry_supported
        else None
    )

    return {
        "element_id": element_id,
        "cad_handle": handle,
        "layer": layer,
        "source_entity_type": source_type,
        "category": "PLANNED_WALL_GEOMETRY",
        "geometry": {
            "type": (
                "polyline"
                if geometry_supported
                else "unsupported"
            ),
            "coordinates": (
                points if geometry_supported else []
            ),
            "closed": (
                bool(closed)
                if geometry_supported
                else None
            ),
            "length_in_drawing_units": length,
            "coordinate_system": "DXF_DRAWING",
        },
        "planning_status": "PLANNED",
        "construction_status": "UNVERIFIED",
        "verification": {
            "cad_element_exists": True,
            "physical_correspondence_verified": False,
            "video_registration_verified": False,
            "construction_progress_verified": False,
            "evidence_frame_ids": [],
            "reason": (
                "No validated CAD-to-video correspondence "
                "or physical construction evidence is available."
            ),
        },
        "geometry_warning": warning,
    }


def extract_line(entity, element_id):
    start = xy_point(
        entity.dxf.start
    )

    end = xy_point(
        entity.dxf.end
    )

    if start is None or end is None:
        return None

    return make_element(
        element_id=element_id,
        handle=entity.dxf.handle,
        layer=entity.dxf.layer,
        source_type="LINE",
        points=[start, end],
    )


def extract_lwpolyline(entity, element_id):
    # A bulge represents a curved segment. Drawing it as
    # a straight line would misrepresent the CAD geometry.
    vertices = list(
        entity.get_points("xyb")
    )

    if not vertices:
        return None

    has_bulge = any(
        abs(float(vertex[2])) > 1e-10
        for vertex in vertices
    )

    if has_bulge:
        return make_element(
            element_id=element_id,
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            source_type="LWPOLYLINE",
            points=[],
            geometry_supported=False,
            warning=(
                "Curved LWPOLYLINE with bulge values; "
                "not approximated by straight edges."
            ),
        )

    points = [
        xy_point(vertex)
        for vertex in vertices
    ]

    return make_element(
        element_id=element_id,
        handle=entity.dxf.handle,
        layer=entity.dxf.layer,
        source_type="LWPOLYLINE",
        points=points,
        closed=bool(entity.closed),
    )


def extract_polyline(entity, element_id):
    if not entity.is_2d_polyline:
        return make_element(
            element_id=element_id,
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            source_type="POLYLINE",
            points=[],
            geometry_supported=False,
            warning=(
                "Non-2D POLYLINE not converted "
                "into planar wall geometry."
            ),
        )

    vertices = list(entity.vertices)

    has_bulge = any(
        abs(
            float(
                vertex.dxf.get(
                    "bulge", 0.0
                )
            )
        ) > 1e-10
        for vertex in vertices
    )

    if has_bulge:
        return make_element(
            element_id=element_id,
            handle=entity.dxf.handle,
            layer=entity.dxf.layer,
            source_type="POLYLINE",
            points=[],
            geometry_supported=False,
            warning=(
                "POLYLINE contains curved bulge "
                "segments; geometry withheld."
            ),
        )

    points = [
        xy_point(
            vertex.dxf.location
        )
        for vertex in vertices
    ]

    return make_element(
        element_id=element_id,
        handle=entity.dxf.handle,
        layer=entity.dxf.layer,
        source_type="POLYLINE",
        points=points,
        closed=bool(entity.is_closed),
    )


def extract_unsupported(entity, element_id):
    source_type = entity.dxftype()

    return make_element(
        element_id=element_id,
        handle=entity.dxf.handle,
        layer=entity.dxf.layer,
        source_type=source_type,
        points=[],
        geometry_supported=False,
        warning=(
            f"{source_type} on a wall layer "
            "is not expanded into line geometry."
        ),
    )


def extract_entity(entity, element_id):
    kind = entity.dxftype()

    if kind == "LINE":
        return extract_line(
            entity, element_id
        )

    if kind == "LWPOLYLINE":
        return extract_lwpolyline(
            entity, element_id
        )

    if kind == "POLYLINE":
        return extract_polyline(
            entity, element_id
        )

    return extract_unsupported(
        entity, element_id
    )


def calculate_bounds(elements):
    coordinates = []

    for element in elements:
        geometry = element["geometry"]

        if geometry["type"] != "polyline":
            continue

        coordinates.extend(
            geometry["coordinates"]
        )

    if not coordinates:
        return None

    xs = [
        point[0]
        for point in coordinates
    ]

    ys = [
        point[1]
        for point in coordinates
    ]

    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
        "width_in_drawing_units": (
            max(xs) - min(xs)
        ),
        "height_in_drawing_units": (
            max(ys) - min(ys)
        ),
    }


def load_cad():
    if not CAD_PATH.is_file():
        raise FileNotFoundError(
            f"CAD file not found: {CAD_PATH}"
        )

    document = ezdxf.readfile(
        str(CAD_PATH)
    )

    modelspace = document.modelspace()

    elements = []
    source_counts = Counter()

    for entity in modelspace:
        layer = entity.dxf.get(
            "layer", ""
        )

        if layer not in WALL_LAYERS:
            continue

        source_type = entity.dxftype()
        source_counts[source_type] += 1

        handle = str(
            entity.dxf.get(
                "handle", ""
            )
        )

        # Handles normally identify DXF entities uniquely.
        # Prefixing with the project ID makes IDs stable
        # within this CAD file.
        element_id = (
            f"{PROJECT_ID}_{handle}"
        )

        extracted = extract_entity(
            entity,
            element_id,
        )

        if extracted is not None:
            elements.append(
                extracted
            )

    elements.sort(
        key=lambda item: item["element_id"]
    )

    supported = sum(
        element["geometry"]["type"] == "polyline"
        for element in elements
    )

    unsupported = (
        len(elements) - supported
    )

    # DXF $INSUNITS can be absent or set to 0.
    # We expose its raw value but do not silently convert.
    raw_insunits = document.header.get(
        "$INSUNITS", 0
    )

    metadata = {
        "source_file": str(
            CAD_PATH.relative_to(ROOT)
        ),
        "wall_layers": sorted(
            WALL_LAYERS
        ),
        "dxf_insunits_code": int(
            raw_insunits
        ),
        "drawing_unit_to_meter_conversion_applied": False,
        "modelspace_only": True,
        "source_entity_counts": dict(
            source_counts
        ),
        "element_count": len(
            elements
        ),
        "supported_geometry_count": supported,
        "unsupported_geometry_count": unsupported,
        "bounds": calculate_bounds(
            elements
        ),
        "cad_to_camera_transform": None,
        "cad_registration_verified": False,
    }

    return elements, metadata


def load_frames():
    frames = []

    if not MANIFEST_PATH.is_file():
        return frames

    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for index, row in enumerate(
            reader
        ):
            raw_name = row.get(
                "filename", ""
            )

            filename = Path(
                raw_name
            ).name

            if not filename:
                continue

            image_path = (
                FRAME_DIR / filename
            )

            if not image_path.is_file():
                continue

            frames.append(
                {
                    "frame_id": filename,
                    "sequence_index": index,
                    "image_url": (
                        "/api/walkthrough/frame/"
                        + filename
                    ),
                    "camera_pose_available": all(
                        finite_number(
                            row.get(key)
                        ) is not None
                        for key in (
                            "tx", "ty", "tz",
                            "qx", "qy",
                            "qz", "qw",
                        )
                    ),
                    "cad_registered": False,
                    "linked_cad_element_ids": [],
                    "evidence_status": (
                        "UNREGISTERED_WALKTHROUGH_FRAME"
                    ),
                }
            )

    return frames


def progress_summary(elements):
    counts = Counter(
        element["construction_status"]
        for element in elements
    )

    return {
        "total_planned_elements": len(
            elements
        ),
        "status_counts": dict(
            counts
        ),
        "verified_completed_elements": 0,
        "verified_incomplete_elements": 0,
        "unverified_elements": counts.get(
            "UNVERIFIED", 0
        ),
        "completion_percentage": None,
        "completion_percentage_available": False,
        "reason": (
            "No CAD-to-video registration or "
            "construction-state verification has "
            "been established. A completion "
            "percentage would be misleading."
        ),
    }


def initialize_project():
    elements, cad_metadata = (
        load_cad()
    )

    frames = load_frames()

    element_by_id = {
        element["element_id"]: element
        for element in elements
    }

    frame_by_name = {
        frame["frame_id"]: frame
        for frame in frames
    }

    diagnostics = {
        "cad_file_exists": (
            CAD_PATH.is_file()
        ),
        "manifest_exists": (
            MANIFEST_PATH.is_file()
        ),
        "walkthrough_frame_count": len(
            frames
        ),
        "cad_elements_loaded": len(
            elements
        ),
        "cad_geometry_supported": (
            cad_metadata[
                "supported_geometry_count"
            ]
        ),
        "cad_geometry_unsupported": (
            cad_metadata[
                "unsupported_geometry_count"
            ]
        ),
        "image_to_cad_correspondence_available": False,
        "physical_element_identity_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
    }

    PROJECT_CACHE.update(
        {
            "loaded": True,
            "elements": elements,
            "element_by_id": element_by_id,
            "frames": frames,
            "frame_by_name": frame_by_name,
            "cad_metadata": cad_metadata,
            "diagnostics": diagnostics,
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = {
        "schema_version": "1.0",
        "project_id": PROJECT_ID,
        "project_name": PROJECT_NAME,
        "generated_at": utc_now(),
        "cad": cad_metadata,
        "walkthrough": {
            "frame_count": len(
                frames
            ),
            "cad_registered": False,
        },
        "progress": progress_summary(
            elements
        ),
        "diagnostics": diagnostics,
    }

    snapshot_path = (
        OUTPUT_DIR
        / "project_snapshot.json"
    )

    snapshot_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return snapshot


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "cad_progress_api",
            "project_loaded": (
                PROJECT_CACHE["loaded"]
            ),
        }
    )


@app.get("/api/project")
def project():
    return jsonify(
        {
            "project_id": PROJECT_ID,
            "name": PROJECT_NAME,
            "mode": (
                "CAD_FIRST_UNVERIFIED"
            ),
            "cad": PROJECT_CACHE[
                "cad_metadata"
            ],
            "walkthrough_frame_count": len(
                PROJECT_CACHE["frames"]
            ),
            "progress": progress_summary(
                PROJECT_CACHE["elements"]
            ),
        }
    )


@app.get("/api/cad/elements")
def cad_elements():
    layer = request.args.get(
        "layer"
    )

    elements = PROJECT_CACHE[
        "elements"
    ]

    if layer:
        elements = [
            element
            for element in elements
            if element["layer"] == layer
        ]

    return jsonify(
        {
            "count": len(
                elements
            ),
            "elements": elements,
            "coordinate_system": (
                "DXF_DRAWING"
            ),
            "cad_registered_to_video": False,
        }
    )


@app.get("/api/cad/elements/<element_id>")
def cad_element_detail(element_id):
    element = PROJECT_CACHE[
        "element_by_id"
    ].get(element_id)

    if element is None:
        raise NotFound(
            "Unknown CAD element"
        )

    return jsonify(
        element
    )


@app.get("/api/walkthrough/frames")
def walkthrough_frames():
    try:
        limit = int(
            request.args.get(
                "limit",
                DEFAULT_FRAME_RESULTS,
            )
        )

        offset = int(
            request.args.get(
                "offset", 0
            )
        )

    except ValueError:
        return jsonify(
            {
                "error": (
                    "limit and offset "
                    "must be integers"
                )
            }
        ), 400

    if limit < 1 or offset < 0:
        return jsonify(
            {
                "error": (
                    "limit must be positive "
                    "and offset nonnegative"
                )
            }
        ), 400

    limit = min(
        limit,
        MAX_FRAME_RESULTS,
    )

    frames = PROJECT_CACHE[
        "frames"
    ]

    selected = frames[
        offset:offset + limit
    ]

    return jsonify(
        {
            "total": len(
                frames
            ),
            "offset": offset,
            "limit": limit,
            "frames": selected,
            "cad_registered": False,
            "element_correspondences_verified": False,
        }
    )


@app.get("/api/walkthrough/frame/<filename>")
def walkthrough_frame_image(filename):
    # The frame must be present in the parsed manifest.
    # Arbitrary paths cannot be requested.
    if filename not in PROJECT_CACHE[
        "frame_by_name"
    ]:
        raise NotFound(
            "Unknown walkthrough frame"
        )

    image_path = (
        FRAME_DIR / filename
    )

    if not image_path.is_file():
        raise NotFound(
            "Image file unavailable"
        )

    return send_file(
        image_path,
        mimetype="image/jpeg",
        conditional=True,
    )


@app.get("/api/progress/summary")
def get_progress_summary():
    return jsonify(
        progress_summary(
            PROJECT_CACHE["elements"]
        )
    )


@app.get("/api/diagnostics")
def get_diagnostics():
    return jsonify(
        PROJECT_CACHE[
            "diagnostics"
        ]
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify(
        {
            "error": "NOT_FOUND",
            "message": str(
                error.description
            ),
        }
    ), 404


def main():
    snapshot = initialize_project()

    print("\n=== STEP 107 ===")
    print(
        "Project:",
        PROJECT_NAME,
    )
    print(
        "CAD elements:",
        snapshot["cad"][
            "element_count"
        ],
    )
    print(
        "Supported geometries:",
        snapshot["cad"][
            "supported_geometry_count"
        ],
    )
    print(
        "Unsupported geometries:",
        snapshot["cad"][
            "unsupported_geometry_count"
        ],
    )
    print(
        "Walkthrough frames:",
        snapshot["walkthrough"][
            "frame_count"
        ],
    )
    print(
        "Completion percentage:",
        "UNAVAILABLE",
    )
    print(
        "CAD registration:",
        "UNVERIFIED",
    )
    print(
        "Snapshot:",
        OUTPUT_DIR
        / "project_snapshot.json",
    )

    print(
        "\nAPI:",
        f"http://{HOST}:{PORT}/api/project",
    )

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
