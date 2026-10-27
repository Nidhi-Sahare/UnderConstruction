
"""
Under-construction — Architectural Footprint Reconstruction

Builds continuous 2D wall footprints from the existing CAD-derived
wall centerlines.

Routes:
    GET /api/twin/footprints
    GET /api/twin/footprints?category=exterior

Does NOT:
    - modify building_elements.json
    - infer construction completion
    - invent door/window openings
    - assume nearby video frames prove visibility

Output:
    Polygon footprints with holes, original CAD element IDs,
    category, assumed height, and source provenance.

The existing /twin viewer is left unchanged until its polygon
renderer is connected in the next step.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from flask import jsonify, request
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from underconstruction_app import app
from digital_twin_3d import get_elements


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Conservative geometric tolerances in CAD metres.
MIN_SEGMENT_LENGTH = 0.01
MIN_POLYGON_AREA = 0.0001
COORDINATE_PRECISION = 6

# Do not bridge gaps. Buffering creates the footprint implied
# by the assumed wall thickness, but does not add artificial
# extensions to line endpoints.
BUFFER_CAP_STYLE = "flat"
BUFFER_JOIN_STYLE = "mitre"

VALID_CATEGORIES = {
    "exterior",
    "interior",
    "partition",
    "lining",
}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def rounded_coordinates(coordinates):
    return [
        [
            round(float(x), COORDINATE_PRECISION),
            round(float(y), COORDINATE_PRECISION),
        ]
        for x, y in coordinates
    ]


def iter_polygons(geometry):
    if geometry.is_empty:
        return

    if geometry.geom_type == "Polygon":
        yield geometry

    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_polygons(part)


def make_footprint(element):
    start = element.get("start")
    end = element.get("end")

    if not start or not end:
        return None

    coordinates = [
        finite(start[0]),
        finite(start[1]),
        finite(end[0]),
        finite(end[1]),
    ]

    thickness = finite(element.get("thickness_m"))
    height = finite(element.get("height_m"))

    if any(value is None for value in coordinates):
        return None

    if thickness is None or height is None:
        return None

    if thickness <= 0 or height <= 0:
        return None

    x1, y1, x2, y2 = coordinates

    if math.hypot(x2 - x1, y2 - y1) < MIN_SEGMENT_LENGTH:
        return None

    line = LineString([(x1, y1), (x2, y2)])

    polygon = line.buffer(
        thickness / 2.0,
        cap_style=BUFFER_CAP_STYLE,
        join_style=BUFFER_JOIN_STYLE,
    )

    if polygon.is_empty or polygon.area < MIN_POLYGON_AREA:
        return None

    return {
        "element": element,
        "line": line,
        "polygon": polygon,
        "height_m": height,
        "thickness_m": thickness,
        "category": str(element.get("category", "unknown")),
    }


def geometry_groups(records):
    """
    Keep different wall categories and dimensions separate.

    This prevents an interior wall from silently absorbing an
    exterior wall's identity or height assumption.
    """
    groups = defaultdict(list)

    for record in records:
        key = (
            record["category"],
            round(record["height_m"], 6),
            round(record["thickness_m"], 6),
        )

        groups[key].append(record)

    return groups


def polygon_to_json(polygon):
    if not isinstance(polygon, Polygon):
        raise TypeError("Expected a Shapely Polygon")

    return {
        "exterior": rounded_coordinates(
            polygon.exterior.coords
        ),
        "holes": [
            rounded_coordinates(interior.coords)
            for interior in polygon.interiors
        ],
    }


def source_ids_for_polygon(polygon, records):
    """
    Record source IDs whose buffered wall footprint has
    positive-area overlap with the resulting union polygon.

    These are geometric contributors, not proof of
    construction or photographic visibility.
    """
    source_ids = []

    for record in records:
        overlap = polygon.intersection(record["polygon"])

        if overlap.area > 1e-8:
            source_ids.extend(
                record["element"].get(
                    "source_element_ids",
                    [record["element"]["id"]],
                )
            )

    return list(dict.fromkeys(source_ids))


def reconstruct(elements):
    records = []

    rejected_ids = []

    for element in elements:
        footprint = make_footprint(element)

        if footprint is None:
            rejected_ids.append(element.get("id"))
            continue

        records.append(footprint)

    groups = geometry_groups(records)

    surfaces = []
    category_stats = Counter()

    for group_key, group_records in sorted(groups.items()):
        category, height, thickness = group_key

        unioned = unary_union([
            record["polygon"]
            for record in group_records
        ])

        polygons = [
            polygon
            for polygon in iter_polygons(unioned)
            if polygon.area >= MIN_POLYGON_AREA
        ]

        polygons.sort(
            key=lambda polygon: (
                -polygon.area,
                polygon.centroid.x,
                polygon.centroid.y,
            )
        )

        for polygon in polygons:
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            for valid_polygon in iter_polygons(polygon):
                if valid_polygon.area < MIN_POLYGON_AREA:
                    continue

                source_ids = source_ids_for_polygon(
                    valid_polygon,
                    group_records,
                )

                surface_id = f"surface_{len(surfaces):05d}"

                surfaces.append({
                    "surface_id": surface_id,
                    "category": category,
                    "height_m": height,
                    "thickness_m": thickness,
                    "area_m2": round(
                        valid_polygon.area,
                        6,
                    ),
                    "source_element_ids": source_ids,
                    "source_element_count": len(source_ids),
                    "footprint": polygon_to_json(
                        valid_polygon
                    ),
                    "construction_status": "UNKNOWN",
                    "progress_verified": False,
                })

                category_stats[category] += 1

    return {
        "surfaces": surfaces,
        "original_element_count": len(elements),
        "valid_element_count": len(records),
        "rejected_element_count": len(rejected_ids),
        "rejected_element_ids": rejected_ids,
        "surface_count": len(surfaces),
        "surfaces_by_category": dict(category_stats),
        "representation": "UNIONED_WALL_FOOTPRINTS",
        "coordinate_system": "CAD_METERS",
        "geometry_source": "CAD_CENTERLINES",
        "height_source": "ASSUMED",
        "thickness_source": "ASSUMED",
        "door_openings_verified": False,
        "window_openings_verified": False,
        "construction_progress_verified": False,
        "camera_registration_verified": False,
        "notes": [
            "Footprints are based on buffered CAD centerlines.",
            "Overlapping footprints are unioned within matching categories and dimensions.",
            "No intentional gap bridging is performed.",
            "Source IDs indicate geometric contribution only.",
            "Door and window cutouts have not been reconstructed.",
        ],
    }


@lru_cache(maxsize=1)
def get_reconstruction():
    elements = get_elements()
    return reconstruct(elements)


@app.route("/api/twin/footprints")
def api_twin_footprints():
    try:
        data = get_reconstruction()

        category = request.args.get(
            "category",
            "",
        ).strip().lower()

        if category:
            if category not in VALID_CATEGORIES:
                return jsonify({
                    "error": "Invalid category",
                    "valid_categories": sorted(
                        VALID_CATEGORIES
                    ),
                }), 400

            selected = [
                surface
                for surface in data["surfaces"]
                if surface["category"] == category
            ]

            return jsonify({
                **data,
                "surfaces": selected,
                "surface_count": len(selected),
                "category_filter": category,
            })

        return jsonify(data)

    except Exception:
        app.logger.exception(
            "Architectural footprint reconstruction failed"
        )

        return jsonify({
            "error": "Architectural reconstruction failed"
        }), 500
