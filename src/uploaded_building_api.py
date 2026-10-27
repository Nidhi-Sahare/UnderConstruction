
"""
Under-construction — User Building Upload API

Reusable DXF-to-3D reconstruction for arbitrary buildings.

Does not modify the existing Hilti sample, dashboard, or viewer.
Does not infer construction progress or camera registration.

Register the blueprint in the existing Flask application:

    from uploaded_building_api import upload_blueprint
    app.register_blueprint(upload_blueprint)

Dependencies:
    flask, ezdxf, shapely
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

import ezdxf
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from architectural_geometry import reconstruct


ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = ROOT / "data" / "user_projects"

upload_blueprint = Blueprint("uploaded_buildings", __name__)

MAX_FLOORS = 20
MAX_FILE_BYTES = 100 * 1024 * 1024

# DXF INSUNITS -> metres per drawing unit.
UNIT_FACTORS = {
    1: 0.0254,       # inches
    2: 0.3048,       # feet
    4: 0.001,        # millimetres
    5: 0.01,         # centimetres
    6: 1.0,          # metres
    7: 1000.0,       # kilometres
    10: 0.9144,      # yards
    14: 0.1,         # decimetres
}

CATEGORIES = {
    "exterior",
    "interior",
    "partition",
    "lining",
}


def error(message, status=400):
    return jsonify({"error": message}), status


def positive_number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be numeric")

    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")

    return result


def finite_number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be numeric")

    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")

    return result


def project_directory(project_id):
    if not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise ValueError("Invalid project ID")

    return PROJECTS_DIR / project_id


def load_dxf(path):
    try:
        return ezdxf.readfile(str(path))
    except Exception as exc:
        raise ValueError(
            f"Could not read DXF: {exc}"
        ) from exc


def get_unit_factor(doc, override=None):
    if override is not None:
        return positive_number(
            override, "metres_per_unit"
        )

    code = int(doc.header.get("$INSUNITS", 0))

    if code not in UNIT_FACTORS:
        raise ValueError(
            f"DXF units code {code} is unsupported or "
            "unspecified. Supply metres_per_unit."
        )

    return UNIT_FACTORS[code]


def inspect_document(doc):
    counts = Counter()
    types = Counter()

    for entity in doc.modelspace():
        counts[entity.dxf.layer] += 1
        types[entity.dxftype()] += 1

    unit_code = int(doc.header.get("$INSUNITS", 0))

    return {
        "unit_code": unit_code,
        "metres_per_unit": UNIT_FACTORS.get(unit_code),
        "units_need_confirmation":
            unit_code not in UNIT_FACTORS,
        "layers": [
            {
                "name": layer,
                "entity_count": count,
            }
            for layer, count in sorted(counts.items())
        ],
        "entity_types": dict(types),
    }


def line_segments(entity):
    """
    Extract straight segments only.

    Curved/bulged polyline sections are deliberately
    excluded rather than converted to inaccurate walls.
    """
    kind = entity.dxftype()

    if kind == "LINE":
        yield (
            entity.dxf.start.x,
            entity.dxf.start.y,
            entity.dxf.end.x,
            entity.dxf.end.y,
        )

    elif kind == "LWPOLYLINE":
        points = list(entity.get_points("xyb"))

        if len(points) < 2:
            return

        closed = bool(entity.closed)
        count = len(points) if closed else len(points) - 1

        for index in range(count):
            a = points[index]
            b = points[(index + 1) % len(points)]

            if abs(float(a[2])) > 1e-9:
                continue

            yield (a[0], a[1], b[0], b[1])

    elif kind == "POLYLINE":
        if not entity.is_2d_polyline:
            return

        vertices = list(entity.vertices)

        if len(vertices) < 2:
            return

        closed = bool(entity.is_closed)
        count = len(vertices) if closed else len(vertices) - 1

        for index in range(count):
            a = vertices[index]
            b = vertices[(index + 1) % len(vertices)]

            if abs(float(a.dxf.bulge or 0)) > 1e-9:
                continue

            yield (
                a.dxf.location.x,
                a.dxf.location.y,
                b.dxf.location.x,
                b.dxf.location.y,
            )


def parse_floor(doc, floor_config, floor_index):
    """
    Convert selected CAD centreline layers to the
    element schema expected by architectural_geometry.

    Layer selection is explicit. This prevents arbitrary
    annotations or furniture lines becoming walls.
    """
    factor = get_unit_factor(
        doc, floor_config.get("metres_per_unit")
    )

    wall_layers = floor_config.get("wall_layers")

    if not isinstance(wall_layers, dict) or not wall_layers:
        raise ValueError(
            f"Floor {floor_index + 1}: wall_layers must "
            "map CAD layer names to wall properties"
        )

    elevation = finite_number(
        floor_config.get("elevation_m", 0),
        "elevation_m",
    )

    default_height = positive_number(
        floor_config.get("height_m", 3.0),
        "height_m",
    )

    default_thickness = positive_number(
        floor_config.get("thickness_m", 0.2),
        "thickness_m",
    )

    elements = []
    selected_count = 0
    unsupported_count = 0

    for entity in doc.modelspace():
        layer = entity.dxf.layer

        if layer not in wall_layers:
            continue

        selected_count += 1

        config = wall_layers[layer]

        if not isinstance(config, dict):
            raise ValueError(
                f"Layer {layer}: properties must be an object"
            )

        category = config.get("category", "interior")

        if category not in CATEGORIES:
            raise ValueError(
                f"Layer {layer}: invalid category {category}"
            )

        height = positive_number(
            config.get("height_m", default_height),
            "height_m",
        )

        thickness = positive_number(
            config.get("thickness_m", default_thickness),
            "thickness_m",
        )

        segments = list(line_segments(entity))

        if not segments:
            unsupported_count += 1

        for segment_index, (x1, y1, x2, y2) in enumerate(
            segments
        ):
            coordinates = [
                float(x1) * factor,
                float(y1) * factor,
                float(x2) * factor,
                float(y2) * factor,
            ]

            if not all(map(math.isfinite, coordinates)):
                continue

            source_id = (
                f"floor_{floor_index}_"
                f"{entity.dxf.handle}_{segment_index}"
            )

            elements.append({
                "id": source_id,
                "source_element_ids": [source_id],
                "start": coordinates[:2],
                "end": coordinates[2:],
                "height_m": height,
                "thickness_m": thickness,
                "category": category,
            })

    if not elements:
        raise ValueError(
            f"Floor {floor_index + 1}: no usable straight "
            "wall segments found in the selected layers"
        )

    geometry = reconstruct(elements)

    # The existing reconstruction engine produces 2D
    # footprints with height. Elevation is stored separately
    # so the frontend can extrude each floor at its level.
    geometry["floor_index"] = floor_index
    geometry["floor_name"] = floor_config.get(
        "name", f"Floor {floor_index + 1}"
    )
    geometry["elevation_m"] = elevation
    geometry["metres_per_unit"] = factor
    geometry["selected_cad_entity_count"] = selected_count
    geometry["unsupported_entity_count"] = unsupported_count
    geometry["source_plan"] = "USER_UPLOADED_DXF"
    geometry["height_source"] = "USER_CONFIGURED"
    geometry["thickness_source"] = "USER_CONFIGURED"
    geometry["camera_registration_verified"] = False
    geometry["construction_progress_verified"] = False

    return geometry


def save_upload(upload, destination, allowed_extensions):
    filename = secure_filename(upload.filename or "")

    if not filename:
        raise ValueError("Uploaded file has no filename")

    if Path(filename).suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type: {filename}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Read with a strict size bound.
    data = upload.stream.read(MAX_FILE_BYTES + 1)

    if len(data) > MAX_FILE_BYTES:
        raise ValueError(
            f"{filename} exceeds the 100 MB upload limit"
        )

    destination.write_bytes(data)

    return filename


@upload_blueprint.route(
    "/api/projects/inspect-dxf",
    methods=["POST"],
)
def inspect_dxf():
    upload = request.files.get("file")

    if upload is None:
        return error("Upload a DXF file using field 'file'")

    temporary = (
        PROJECTS_DIR /
        "_inspection" /
        f"{uuid.uuid4().hex}.dxf"
    )

    try:
        save_upload(upload, temporary, {".dxf"})
        doc = load_dxf(temporary)

        return jsonify(inspect_document(doc))

    except ValueError as exc:
        return error(str(exc))

    finally:
        temporary.unlink(missing_ok=True)


@upload_blueprint.route(
    "/api/projects/create",
    methods=["POST"],
)
def create_project():
    """
    Multipart form:

    config: JSON object
    floorplans: repeated DXF files, in floor order
    walkthroughs: optional repeated video files

    Example config:

    {
      "name": "My Building",
      "floors": [
        {
          "name": "Ground Floor",
          "elevation_m": 0,
          "height_m": 3,
          "thickness_m": 0.2,
          "wall_layers": {
            "WALLS": {
              "category": "interior"
            }
          }
        }
      ]
    }

    Floor plans are architectural geometry.
    Walkthroughs are stored as unregistered evidence.
    """
    try:
        config = json.loads(
            request.form.get("config", "{}")
        )

        if not isinstance(config, dict):
            return error("config must be a JSON object")

        floor_configs = config.get("floors")
        floor_uploads = request.files.getlist("floorplans")
        videos = request.files.getlist("walkthroughs")

        if not isinstance(floor_configs, list):
            return error("config.floors must be an array")

        if not 1 <= len(floor_configs) <= MAX_FLOORS:
            return error("Provide between 1 and 20 floors")

        if len(floor_uploads) != len(floor_configs):
            return error(
                "Number of floorplans must match config.floors"
            )

        project_id = uuid.uuid4().hex
        directory = project_directory(project_id)

        floor_results = []
        video_results = []

        try:
            for index, (upload, floor_config) in enumerate(
                zip(floor_uploads, floor_configs)
            ):
                if not isinstance(floor_config, dict):
                    raise ValueError(
                        f"Floor {index + 1} config is invalid"
                    )

                path = (
                    directory /
                    "floorplans" /
                    f"floor_{index + 1}.dxf"
                )

                original_name = save_upload(
                    upload, path, {".dxf"}
                )

                doc = load_dxf(path)

                geometry = parse_floor(
                    doc, floor_config, index
                )

                geometry["original_filename"] = original_name
                floor_results.append(geometry)

            for index, video in enumerate(videos):
                extension = Path(
                    secure_filename(video.filename or "")
                ).suffix.lower()

                if extension not in {".mp4", ".mov"}:
                    raise ValueError(
                        "Walkthroughs must be MP4 or MOV"
                    )

                path = (
                    directory /
                    "walkthroughs" /
                    f"walkthrough_{index + 1}{extension}"
                )

                original_name = save_upload(
                    video, path, {".mp4", ".mov"}
                )

                video_results.append({
                    "filename": original_name,
                    "stored_filename": path.name,
                    "registration_status": "NOT_REGISTERED",
                    "progress_verified": False,
                })

            result = {
                "project_id": project_id,
                "name": str(
                    config.get("name", "Untitled Building")
                )[:120],
                "floor_count": len(floor_results),
                "floors": floor_results,
                "walkthroughs": video_results,
                "coordinate_system": "CAD_METERS",
                "inter_floor_alignment_verified": False,
                "camera_registration_verified": False,
                "construction_progress_verified": False,
                "geometry_representation":
                    "CAD_DERIVED_WALL_FOOTPRINTS",
            }

            directory.mkdir(parents=True, exist_ok=True)

            (directory / "project.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

            return jsonify({
                "project_id": project_id,
                "name": result["name"],
                "floor_count": len(floor_results),
                "surface_count": sum(
                    floor["surface_count"]
                    for floor in floor_results
                ),
                "geometry_url":
                    f"/api/projects/{project_id}/geometry",
                "walkthrough_count": len(video_results),
                "status": "RECONSTRUCTED",
            }), 201

        except Exception:
            # Remove incomplete projects.
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
            raise

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return error(str(exc))

    except Exception:
        upload_blueprint.logger.exception(
            "Project creation failed"
        ) if hasattr(upload_blueprint, "logger") else None

        return error(
            "Project creation failed. Check server logs.",
            500,
        )


@upload_blueprint.route(
    "/api/projects/<project_id>/geometry",
    methods=["GET"],
)
def project_geometry(project_id):
    try:
        directory = project_directory(project_id)
        path = directory / "project.json"

        if not path.is_file():
            return error("Project not found", 404)

        return jsonify(json.loads(
            path.read_text(encoding="utf-8")
        ))

    except ValueError as exc:
        return error(str(exc))
