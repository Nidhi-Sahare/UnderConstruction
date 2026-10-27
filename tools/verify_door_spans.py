
"""
Under-construction — Door Span Geometry Verification

Inputs:
    data/floorplans/floor_2.dxf
    outputs/cad_openings/spans/door_spans.json

Optional:
    --footprints path/to/footprints.json

Outputs:
    outputs/cad_openings/verification/door_span_verification.json
    outputs/cad_openings/verification/door_span_verification.png

Requirements:
    ezdxf, shapely, matplotlib

This is evidence screening, NOT approval to cut walls.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    shape,
)
from shapely.ops import unary_union
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"
SPANS_PATH = (
    ROOT / "outputs/cad_openings/spans/door_spans.json"
)
OUTPUT_DIR = (
    ROOT / "outputs/cad_openings/verification"
)

WALL_LAYERS = {
    "010 Aussenwände",
    "015 Innenwände",
    "016 Trennwände",
    "019 Vormauerungen",
}

# Diagnostic tolerances in DXF drawing units.
# These are hypotheses, not verified construction standards.
ENDPOINT_NEAR = 0.18
WALL_NEAR = 0.12
ALIGNMENT_ANGLE = 15.0
MIN_WALL_LENGTH = 0.025
SPAN_SAMPLE_COUNT = 21
ARC_SAME_LOCATION = 0.15


def xy(value):
    return (float(value[0]), float(value[1]))


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def segment_angle(a, b):
    return math.atan2(
        b[1] - a[1],
        b[0] - a[0],
    )


def parallel_error(a, b):
    difference = abs(a - b) % math.pi
    return min(difference, math.pi - difference)


def flatten_entity(entity):
    """
    Convert supported DXF wall entities into LINE geometries.

    LWPOLYLINE bulges are preserved through virtual_entities().
    Unsupported entities are reported, not silently invented.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        return [
            LineString([
                xy(entity.dxf.start),
                xy(entity.dxf.end),
            ])
        ]

    if kind == "ARC":
        center = xy(entity.dxf.center)
        radius = float(entity.dxf.radius)
        start = float(entity.dxf.start_angle)
        end = float(entity.dxf.end_angle)
        sweep = (end - start) % 360.0

        count = max(8, int(math.ceil(sweep / 3)) + 1)

        coords = []

        for i in range(count):
            angle = math.radians(
                start + sweep * i / (count - 1)
            )

            coords.append((
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            ))

        return [LineString(coords)]

    if kind in {"LWPOLYLINE", "POLYLINE"}:
        if (
            kind == "POLYLINE"
            and not entity.is_2d_polyline
        ):
            return []

        result = []

        for child in entity.virtual_entities():
            result.extend(flatten_entity(child))

        return result

    return []


def read_walls(dxf_path):
    document = ezdxf.readfile(dxf_path)

    walls = []
    skipped = Counter()

    for entity in document.modelspace():
        if entity.dxf.layer not in WALL_LAYERS:
            continue

        geometries = flatten_entity(entity)

        if not geometries:
            skipped[entity.dxftype()] += 1

        for geometry in geometries:
            if geometry.length < MIN_WALL_LENGTH:
                continue

            walls.append({
                "handle": str(entity.dxf.handle),
                "layer": entity.dxf.layer,
                "geometry": geometry,
            })

    return document, walls, dict(skipped)


def load_optional_footprints(path):
    if path is None:
        return None, "NOT_PROVIDED"

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Accept GeoJSON FeatureCollection or a list of
    # explicitly supplied GeoJSON polygon geometries.
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in data.get("features", [])
            if feature.get("geometry")
        ]
    elif isinstance(data, list):
        geometries = [
            shape(item["geometry"] if "geometry" in item else item)
            for item in data
        ]
    else:
        raise ValueError(
            "Unsupported footprint JSON format. "
            "Expected GeoJSON FeatureCollection or list."
        )

    polygons = [
        geom
        for geom in geometries
        if isinstance(geom, Polygon)
        or geom.geom_type == "MultiPolygon"
    ]

    if not polygons:
        return None, "NO_POLYGON_GEOMETRY"

    return unary_union(polygons), "LOADED"


def build_index(walls):
    geometries = [
        wall["geometry"]
        for wall in walls
    ]

    return STRtree(geometries)


def local_wall_indices(tree, geometry, radius):
    return [
        int(index)
        for index in tree.query(
            geometry.buffer(radius)
        )
    ]


def nearest_endpoints(point, walls, indices):
    observations = []

    for index in indices:
        wall = walls[index]
        coords = list(wall["geometry"].coords)

        for endpoint in (coords[0], coords[-1]):
            d = point.distance(Point(endpoint))

            if d <= ENDPOINT_NEAR:
                observations.append({
                    "wall_handle": wall["handle"],
                    "wall_layer": wall["layer"],
                    "point": list(endpoint),
                    "distance": round(d, 6),
                })

    observations.sort(
        key=lambda item: item["distance"]
    )

    return observations


def sample_wall_proximity(span, walls, indices):
    """
    Fraction of interior sample positions close to a
    wall boundary.

    This is NOT a solid-wall intersection test.
    CAD wall boundaries can flank an actual opening.
    """

    samples = [
        span.interpolate(i / (SPAN_SAMPLE_COUNT - 1),
                         normalized=True)
        for i in range(1, SPAN_SAMPLE_COUNT - 1)
    ]

    if not indices:
        return 0.0

    nearby = 0

    for point in samples:
        if min(
            point.distance(walls[i]["geometry"])
            for i in indices
        ) <= WALL_NEAR:
            nearby += 1

    return nearby / len(samples)


def aligned_wall_evidence(span, walls, indices):
    """
    Count wall line primitives locally parallel to the
    candidate span. This is a weak alignment indicator,
    not proof of a wall opening.
    """

    coords = list(span.coords)
    target_angle = segment_angle(coords[0], coords[-1])

    matches = []

    for index in indices:
        wall = walls[index]
        points = list(wall["geometry"].coords)

        if len(points) < 2:
            continue

        if wall["geometry"].distance(span) > WALL_NEAR:
            continue

        # Evaluate individual polyline segments.
        for a, b in zip(points, points[1:]):
            if distance(a, b) < MIN_WALL_LENGTH:
                continue

            angle = segment_angle(a, b)
            error = math.degrees(
                parallel_error(target_angle, angle)
            )

            if error <= ALIGNMENT_ANGLE:
                matches.append({
                    "handle": wall["handle"],
                    "layer": wall["layer"],
                    "angle_error_degrees": round(
                        error, 3
                    ),
                })
                break

    return matches


def footprint_observation(span, footprints):
    if footprints is None:
        return None

    # A small buffer measures nearby polygon occupancy,
    # not whether the CAD intended a door cutout.
    probe = span.buffer(0.04, cap_style=2)

    if probe.area <= 0:
        return None

    intersection = probe.intersection(footprints)

    return {
        "probe_area": round(probe.area, 6),
        "overlap_area": round(
            intersection.area, 6
        ),
        "overlap_fraction": round(
            intersection.area / probe.area, 6
        ),
        "interpretation":
            "SIMPLIFIED_FOOTPRINT_OCCUPANCY_ONLY",
    }


def evaluate_span(record, walls, tree, footprints):
    a = tuple(record["start"])
    b = tuple(record["end"])

    span = LineString([a, b])

    indices = local_wall_indices(
        tree, span, max(ENDPOINT_NEAR, WALL_NEAR)
    )

    endpoint_a = nearest_endpoints(
        Point(a), walls, indices
    )
    endpoint_b = nearest_endpoints(
        Point(b), walls, indices
    )

    wall_fraction = sample_wall_proximity(
        span, walls, indices
    )

    alignment = aligned_wall_evidence(
        span, walls, indices
    )

    both_endpoints = bool(endpoint_a and endpoint_b)

    # This describes evidence, not physical verification.
    if both_endpoints and wall_fraction <= 0.30:
        status = "GEOMETRICALLY_SUPPORTED_CANDIDATE"

    elif (
        endpoint_a
        or endpoint_b
        or alignment
        or wall_fraction > 0
    ):
        status = "AMBIGUOUS_GEOMETRY"

    else:
        status = "INSUFFICIENT_WALL_EVIDENCE"

    return {
        "span_id": record["span_id"],
        "start": record["start"],
        "end": record["end"],
        "length": record["length"],
        "source_arc_handles":
            record["source_arc_handles"],
        "source_line_handles":
            record["source_line_handles"],
        "status": status,
        "evidence": {
            "endpoint_a_candidates": endpoint_a[:15],
            "endpoint_b_candidates": endpoint_b[:15],
            "both_endpoints_near_wall_terminations":
                both_endpoints,
            "interior_wall_boundary_proximity_fraction":
                round(wall_fraction, 4),
            "aligned_wall_primitives": alignment[:20],
            "nearby_wall_primitive_count": len(indices),
            "footprint": footprint_observation(
                span, footprints
            ),
        },
        "opening_verified": False,
        "safe_to_cut": False,
    }


def same_location(a, b):
    direct = max(
        distance(a["start"], b["start"]),
        distance(a["end"], b["end"]),
    )

    reverse = max(
        distance(a["start"], b["end"]),
        distance(a["end"], b["start"]),
    )

    return min(direct, reverse) <= ARC_SAME_LOCATION


def report_duplicate_locations(results):
    """
    Identify spans representing approximately the same
    location. This does not merge or delete them.
    """

    groups = []
    used = set()

    for i, candidate in enumerate(results):
        if i in used:
            continue

        group = [candidate["span_id"]]
        used.add(i)

        for j in range(i + 1, len(results)):
            if j in used:
                continue

            if same_location(candidate, results[j]):
                group.append(results[j]["span_id"])
                used.add(j)

        if len(group) > 1:
            groups.append(group)

    return groups


def plot_results(results, walls, output):
    fig, ax = plt.subplots(figsize=(16, 11))

    for wall in walls:
        x, y = wall["geometry"].xy

        ax.plot(
            x, y,
            color="#cbd5e1",
            linewidth=0.55,
            alpha=0.6,
        )

    colors = {
        "GEOMETRICALLY_SUPPORTED_CANDIDATE":
            "#16a34a",
        "AMBIGUOUS_GEOMETRY":
            "#f59e0b",
        "INSUFFICIENT_WALL_EVIDENCE":
            "#dc2626",
    }

    for result in results:
        a = result["start"]
        b = result["end"]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=colors[result["status"]],
            linewidth=2.3,
            alpha=0.9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)

    ax.set_title(
        "Door span evidence screening\n"
        "Green: geometrically supported | "
        "Orange: ambiguous | Red: insufficient evidence\n"
        "No openings verified"
    )

    ax.set_xlabel("DXF X")
    ax.set_ylabel("DXF Y")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dxf",
        type=Path,
        default=DXF_PATH,
    )

    parser.add_argument(
        "--spans",
        type=Path,
        default=SPANS_PATH,
    )

    parser.add_argument(
        "--footprints",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    document, walls, skipped = read_walls(
        args.dxf
    )

    with args.spans.open(
        "r", encoding="utf-8"
    ) as file:
        source = json.load(file)

    if source.get("schema_version") != "1.0":
        raise ValueError(
            "Expected door span report schema 1.0"
        )

    footprints, footprint_status = (
        load_optional_footprints(args.footprints)
    )

    tree = build_index(walls)

    results = [
        evaluate_span(
            span, walls, tree, footprints
        )
        for span in source["spans"]
    ]

    counts = dict(Counter(
        item["status"]
        for item in results
    ))

    duplicates = report_duplicate_locations(
        results
    )

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "schema_version": "1.0",
        "source_dxf": str(args.dxf),
        "source_spans": str(args.spans),
        "drawing_units_code":
            document.header.get("$INSUNITS"),
        "wall_primitive_count": len(walls),
        "skipped_wall_entities": skipped,
        "footprint_status": footprint_status,
        "span_count": len(results),
        "status_counts": counts,
        "same_location_span_groups": duplicates,
        "results": results,
        "limitations": [
            "Both radial alternatives may be present.",
            "Wall endpoint proximity is not proof of a jamb.",
            "Wall boundary proximity is not proof of solid "
            "wall occupancy or an opening.",
            "The supported label indicates screening evidence "
            "only, not physical verification.",
            "No automatic door-leaf orientation is selected.",
            "Simplified wall footprints may fill real openings.",
            "No geometry or progress state was modified.",
        ],
        "verified_opening_count": 0,
        "wall_cutouts_generated": False,
    }

    output_json = (
        args.out / "door_span_verification.json"
    )

    with output_json.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    output_image = (
        args.out / "door_span_verification.png"
    )

    plot_results(
        results, walls, output_image
    )

    print("\n=== DOOR SPAN GEOMETRY VERIFICATION ===")
    print("Wall primitives:", len(walls))
    print("Provisional spans:", len(results))
    print("Screening statuses:", counts)
    print("Same-location groups:", len(duplicates))
    print("Footprint input:", footprint_status)
    print("Skipped wall entities:", skipped)

    print("\nJSON:", output_json)
    print("Preview:", output_image)
    print("Verified openings: 0")
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
