
"""
BuildProgress - CAD Geometry Normalizer

Creates a reusable, source-traceable planar line network
from architectural DXF wall layers.

Outputs:
    data/geometry/normalized_segments.json
    data/geometry/normalization_report.json
    data/geometry/normalized_geometry.png

Does NOT:
    - close gaps
    - infer wall thickness
    - reconstruct wall polygons
    - estimate construction progress

Run:
    python cad_geometry_normalizer.py

Dependencies:
    ezdxf, shapely, matplotlib
"""

import json
import math
from collections import Counter
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ezdxf.path import make_path
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()
DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

OUTPUT_DIR = ROOT / "data" / "geometry"

SEGMENTS_PATH = OUTPUT_DIR / "normalized_segments.json"
REPORT_PATH = OUTPUT_DIR / "normalization_report.json"
IMAGE_PATH = OUTPUT_DIR / "normalized_geometry.png"

WALL_LAYERS = {
    "010 AussenwÃ¤nde": "exterior",
    "015 InnenwÃ¤nde": "interior",
    "016 TrennwÃ¤nde": "partition",
    "019 Vormauerungen": "lining",
}

COLORS = {
    "exterior": "#2563EB",
    "interior": "#E59A37",
    "partition": "#D65D8A",
    "lining": "#A16207",
}

CURVE_TOLERANCE_M = 0.01
CONNECTIVITY_TOLERANCE_M = 0.001
MIN_SEGMENT_LENGTH_M = 0.00001
ROUND_DIGITS = 8


def xy(point):
    return float(point[0]), float(point[1])


def flatten_entity(entity):
    """
    Convert supported DXF entities to 2D line segments.

    INSERT entities are handled by the recursive collector,
    not by this function.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        return [(xy(entity.dxf.start), xy(entity.dxf.end))]

    if kind in ("LWPOLYLINE", "POLYLINE"):
        result = []

        for child in entity.virtual_entities():
            result.extend(flatten_entity(child))

        return result

    if kind in ("ARC", "CIRCLE"):
        path = make_path(entity)

        vertices = [
            xy(point)
            for point in path.flattening(
                distance=CURVE_TOLERANCE_M
            )
        ]

        return list(zip(vertices, vertices[1:]))

    return []


def collect_segments(
    entity,
    inherited_layer=None,
    parent_handle=None,
    depth=0,
):
    """
    Recursively expand supported block references.

    ezdxf.virtual_entities() applies INSERT transforms.

    The original INSERT handle is retained as provenance.
    """

    if depth > 10:
        raise RuntimeError(
            "Maximum nested block depth exceeded."
        )

    entity_layer = entity.dxf.get("layer", "0")

    if entity_layer == "0" and inherited_layer is not None:
        effective_layer = inherited_layer
    else:
        effective_layer = entity_layer

    handle = str(entity.dxf.get("handle", ""))

    if entity.dxftype() == "INSERT":
        original_parent = parent_handle or handle
        result = []

        for child in entity.virtual_entities():
            result.extend(
                collect_segments(
                    child,
                    inherited_layer=effective_layer,
                    parent_handle=original_parent,
                    depth=depth + 1,
                )
            )

        return result

    if effective_layer not in WALL_LAYERS:
        return []

    result = []

    for start, end in flatten_entity(entity):
        length = math.dist(start, end)

        if length < MIN_SEGMENT_LENGTH_M:
            continue

        result.append({
            "start": start,
            "end": end,
            "layer": effective_layer,
            "category": WALL_LAYERS[effective_layer],
            "source_handle": handle,
            "parent_insert_handle": parent_handle,
            "source_entity_type": entity.dxftype(),
        })

    return result


def load_geometry():
    document = ezdxf.readfile(DXF_PATH)

    if document.header.get("$INSUNITS", 0) != 6:
        raise ValueError(
            "Expected DXF coordinates in meters."
        )

    segments = []
    top_level_insert_count = 0

    for entity in document.modelspace():
        layer = entity.dxf.get("layer", "0")

        if layer not in WALL_LAYERS:
            continue

        if entity.dxftype() == "INSERT":
            top_level_insert_count += 1

        segments.extend(
            collect_segments(entity)
        )

    return segments, top_level_insert_count


def analyze_connections(segments):
    """
    Detect geometric connections within a tolerance.

    Endpoint-to-endpoint and endpoint-to-segment
    connections are counted separately.

    No coordinates are modified.
    """

    if not segments:
        return {
            "endpoint_count": 0,
            "endpoint_endpoint_connected": 0,
            "endpoint_segment_connected": 0,
            "unmatched_endpoints": 0,
        }, []

    lines = [
        LineString([s["start"], s["end"]])
        for s in segments
    ]

    tree = STRtree(lines)

    endpoints = []

    for segment_index, segment in enumerate(segments):
        endpoints.append(
            (segment_index, segment["start"])
        )
        endpoints.append(
            (segment_index, segment["end"])
        )

    endpoint_connected = 0
    segment_connected = 0
    unmatched = []

    for source_index, coordinate in endpoints:
        point = Point(coordinate)

        nearby = tree.query(
            point.buffer(CONNECTIVITY_TOLERANCE_M)
        )

        found_endpoint = False
        found_segment = False

        for candidate_index in nearby:
            candidate_index = int(candidate_index)

            if candidate_index == source_index:
                continue

            other = segments[candidate_index]

            if (
                math.dist(coordinate, other["start"])
                <= CONNECTIVITY_TOLERANCE_M
                or math.dist(coordinate, other["end"])
                <= CONNECTIVITY_TOLERANCE_M
            ):
                found_endpoint = True
                break

            if (
                point.distance(lines[candidate_index])
                <= CONNECTIVITY_TOLERANCE_M
            ):
                found_segment = True

        if found_endpoint:
            endpoint_connected += 1
        elif found_segment:
            segment_connected += 1
        else:
            unmatched.append(coordinate)

    statistics = {
        "endpoint_count": len(endpoints),
        "endpoint_endpoint_connected": endpoint_connected,
        "endpoint_segment_connected": segment_connected,
        "unmatched_endpoints": len(unmatched),
    }

    return statistics, unmatched


def build_noded_network(segments):
    """
    Split linework at exact geometric intersections.

    unary_union nodes the network but does not
    bridge gaps between nearby, disconnected lines.

    This network is diagnostic and may lose the
    one-to-one mapping to original CAD handles.
    """

    if not segments:
        return []

    lines = [
        LineString([s["start"], s["end"]])
        for s in segments
    ]

    network = unary_union(lines)

    result = []

    def visit(geometry):
        if geometry.geom_type == "LineString":
            coordinates = list(geometry.coords)

            for a, b in zip(
                coordinates,
                coordinates[1:],
            ):
                if math.dist(a, b) >= MIN_SEGMENT_LENGTH_M:
                    result.append([
                        list(a),
                        list(b),
                    ])

        elif hasattr(geometry, "geoms"):
            for child in geometry.geoms:
                visit(child)

    visit(network)

    return result


def rounded_point(point):
    return [
        round(float(point[0]), ROUND_DIGITS),
        round(float(point[1]), ROUND_DIGITS),
    ]


def save_visualization(segments, unmatched):
    fig, ax = plt.subplots(
        figsize=(14, 10),
        dpi=150,
    )

    for category in WALL_LAYERS.values():
        lines = [
            [s["start"], s["end"]]
            for s in segments
            if s["category"] == category
        ]

        if lines:
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=COLORS[category],
                    linewidths=0.9,
                    label=category,
                )
            )

    if unmatched:
        ax.scatter(
            [p[0] for p in unmatched],
            [p[1] for p in unmatched],
            s=7,
            color="#DC2626",
            label="Unmatched endpoints",
            zorder=3,
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(
        "BuildProgress | Floor 2 - Normalized CAD Geometry"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.12)

    fig.savefig(
        IMAGE_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DXF_PATH.exists():
        raise FileNotFoundError(
            f"Missing DXF: {DXF_PATH}"
        )

    print("Loading and normalizing CAD geometry...")

    segments, insert_count = load_geometry()

    connection_stats, unmatched = analyze_connections(
        segments
    )

    noded_segments = build_noded_network(segments)

    exported_segments = []

    for index, segment in enumerate(segments, start=1):
        exported_segments.append({
            "id": f"CAD_SEG_{index:06d}",
            "start": rounded_point(segment["start"]),
            "end": rounded_point(segment["end"]),
            "layer": segment["layer"],
            "category": segment["category"],
            "source_handle": segment["source_handle"],
            "parent_insert_handle":
                segment["parent_insert_handle"],
            "source_entity_type":
                segment["source_entity_type"],
        })

    segment_report = {
        "schema_version": "cad-segments-1.0",
        "source_dxf": str(DXF_PATH),
        "units": "meters",
        "segments": exported_segments,
        "noded_network": [
            [
                rounded_point(a),
                rounded_point(b),
            ]
            for a, b in noded_segments
        ],
    }

    SEGMENTS_PATH.write_text(
        json.dumps(segment_report, indent=2),
        encoding="utf-8",
    )

    report = {
        "schema_version": "cad-normalization-1.0",
        "source_dxf": str(DXF_PATH),
        "units": "meters",
        "top_level_insert_count": insert_count,
        "normalized_segment_count": len(segments),
        "noded_segment_count": len(noded_segments),
        "connectivity": connection_stats,
        "category_counts": dict(
            Counter(
                s["category"]
                for s in segments
            )
        ),
        "limitations": [
            "No gap closing performed",
            "No wall thickness inferred",
            "Noded network lacks direct per-edge provenance",
            "Near-coincident endpoints are not snapped",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    save_visualization(segments, unmatched)

    print("\n" + "=" * 65)
    print("BUILDPROGRESS - CAD NORMALIZATION")
    print("=" * 65)

    print(f"Expanded top-level INSERTs: {insert_count}")
    print(f"Normalized segments: {len(segments)}")
    print(f"Noded network segments: {len(noded_segments)}")

    print("\nCONNECTIVITY")
    for key, value in connection_stats.items():
        print(f"{key}: {value}")

    print("\nCATEGORY COUNTS")
    for key, value in report["category_counts"].items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(SEGMENTS_PATH)
    print(REPORT_PATH)
    print(IMAGE_PATH)

    print(
        "\nNormalization complete. "
        "No gaps were closed or wall regions inferred."
    )


if __name__ == "__main__":
    main()

