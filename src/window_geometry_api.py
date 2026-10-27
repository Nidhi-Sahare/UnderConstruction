
import json
import math
from pathlib import Path

from flask import Blueprint, jsonify


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WINDOW_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "cad_openings"
    / "windows"
    / "window_assemblies.json"
)

window_geometry_bp = Blueprint("window_geometry", __name__)


def finite_point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    try:
        point = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(number) for number in point):
        return None

    return point


def load_window_geometry():
    if not WINDOW_REPORT.is_file():
        raise FileNotFoundError(
            f"Window assembly report not found: {WINDOW_REPORT}"
        )

    with WINDOW_REPORT.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("coordinate_system") != "DXF_METERS":
        raise ValueError(
            "Window geometry must use DXF_METERS coordinates."
        )

    assemblies = report.get("assemblies", [])

    if not isinstance(assemblies, list):
        raise ValueError("Invalid window assembly list.")

    windows = []
    rejected = 0
    invalid = 0

    for assembly in assemblies:
        status = assembly.get("status")

        if status != "PROVISIONAL_STRAIGHT_ASSEMBLY":
            rejected += 1
            continue

        start = finite_point(assembly.get("start"))
        end = finite_point(assembly.get("end"))

        if start is None or end is None:
            invalid += 1
            continue

        length = math.dist(start, end)

        if length <= 0.01:
            invalid += 1
            continue

        windows.append(
            {
                "id": assembly.get("assembly_id"),
                "type": "window",
                "status": "PROVISIONAL_GEOMETRY",
                "construction_status": "UNKNOWN",
                "start": start,
                "end": end,
                "center": [
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                ],
                "span_m": round(length, 6),
                "source_layer": assembly.get("source_layer"),
                "source_handles": assembly.get(
                    "source_handles", []
                ),
                "source_entity_count": assembly.get(
                    "source_entity_count", 0
                ),
                "representation": assembly.get(
                    "representation"
                ),
                "wall_association_verified": False,
                "window_opening_verified": False,
                "vertical_dimensions_verified": False,
                "safe_to_cut": False,
            }
        )

    return {
        "schema_version": "1.0",
        "coordinate_system": "DXF_METERS",
        "source": str(
            WINDOW_REPORT.relative_to(PROJECT_ROOT)
        ),
        "count": len(windows),
        "rejected_count": rejected,
        "invalid_count": invalid,
        "verified_openings": 0,
        "wall_cutouts_generated": False,
        "geometry_role": "PLAN_VIEW_REFERENCE",
        "vertical_geometry_available": False,
        "windows": windows,
    }


@window_geometry_bp.get("/api/twin/windows")
def get_twin_windows():
    try:
        return jsonify(load_window_geometry())

    except FileNotFoundError as error:
        return jsonify(
            {
                "error": "WINDOW_REPORT_NOT_FOUND",
                "detail": str(error),
            }
        ), 404

    except (ValueError, json.JSONDecodeError) as error:
        return jsonify(
            {
                "error": "INVALID_WINDOW_REPORT",
                "detail": str(error),
            }
        ), 422
