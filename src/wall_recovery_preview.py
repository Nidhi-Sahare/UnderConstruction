
import json
import math
from pathlib import Path

from flask import Blueprint, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOVERY_FILE = PROJECT_ROOT / "floor_2_wall_recovery.json"

wall_recovery_bp = Blueprint("wall_recovery", __name__)


def valid_outline(points):
    if not isinstance(points, list) or len(points) < 3:
        return False

    for point in points:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in point
            )
        ):
            return False

    return True


@wall_recovery_bp.get("/api/twin/recovered-walls")
def recovered_walls():
    if not RECOVERY_FILE.is_file():
        return jsonify({
            "error": "Recovery JSON not found",
            "path": str(RECOVERY_FILE),
        }), 404

    try:
        with RECOVERY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 500

    candidates = data.get("candidate_walls", [])
    walls = []

    for candidate in candidates:
        outline = candidate.get("outline_xy", [])

        if not valid_outline(outline):
            continue

        if candidate.get("automatic_integration_approved") is True:
            continue

        walls.append({
            "id": candidate.get("id"),
            "source_handle": candidate.get("source_handle"),
            "source_edge_id": candidate.get("source_edge_id"),
            "layer": candidate.get("layer"),
            "outline_xy": outline,
            "width_m": candidate.get("width_m"),
            "status": "INFERRED_FROM_UNPAIRED_CAD_EDGE",
            "boundary_side_unverified": True,
            "safe_to_cut": False,
            "preview_only": True,
        })

    return jsonify({
        "units": "metres",
        "mode": "INFERRED_PREVIEW_ONLY",
        "candidate_count": len(walls),
        "existing_mesh_count": data.get(
            "summary", {}
        ).get("existing_mesh_count"),
        "wall_height_m": None,
        "walls": walls,
        "warning": (
            "Recovered wall outlines are hypotheses. "
            "They are not verified wall geometry, "
            "must not replace existing meshes, "
            "and must not authorize opening cuts."
        ),
    })

