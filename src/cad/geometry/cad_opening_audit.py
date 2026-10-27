
"""
BuildProgress - Step 7: CAD Block and Opening Audit

Audits:
    1. INSERT references and their virtual geometry
    2. Door and window geometry
    3. Unmatched wall endpoints
    4. Proximity between wall endpoints and openings

Does not modify the DXF or infer wall polygons.

Run:
    python cad_opening_audit.py

Outputs:
    data/geometry/block_audit.json
    data/geometry/opening_audit.json
    data/geometry/opening_audit.png
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

BLOCK_REPORT_PATH = OUTPUT_DIR / "block_audit.json"
OPENING_REPORT_PATH = OUTPUT_DIR / "opening_audit.json"
IMAGE_PATH = OUTPUT_DIR / "opening_audit.png"

WALL_LAYERS = {
    "010 AussenwÃ¤nde": "exterior",
    "015 InnenwÃ¤nde": "interior",
    "016 TrennwÃ¤nde": "partition",
    "019 Vormauerungen": "lining",
}

OPENING_LAYERS = {
    "010 TÃ¼ren": "door",
    "010 Fenster": "window",
}

CURVE_TOLERANCE_M = 0.01
CONNECTIVITY_TOLERANCE_M = 0.001
OPENING_SEARCH_RADIUS_M = 0.30
MIN_SEGMENT_LENGTH_M = 0.00001
MAX_BLOCK_DEPTH = 12

COLORS = {
    "exterior": "#2563EB",
    "interior": "#E59A37",
    "partition": "#D65D8A",
    "lining": "#A16207",
    "door": "#16A34A",
    "window": "#7C3AED",
}


def coordinates(point):
    return (float(point[0]), float(point[1]))


def as_json_point(point):
    return [
        round(float(point[0]), 7),
        round(float(point[1]), 7),
    ]


def entity_segments(entity):
    """
    Convert supported entities to planar line segments.
    Block expansion is handled separately.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        return [
            (
                coordinates(entity.dxf.start),
                coordinates(entity.dxf.end),
            )
        ]

    if kind in ("LWPOLYLINE", "POLYLINE"):
        result = []

        try:
            children = list(entity.virtual_entities())
        except Exception:
            return []

        for child in children:
            result.extend(entity_segments(child))

        return result

    if kind in (
        "ARC",
        "CIRCLE",
        "ELLIPSE",
        "SPLINE",
    ):
        try:
            path = make_path(entity)
            points = [
                coordinates(p)
                for p in path.flattening(
                    distance=CURVE_TOLERANCE_M
                )
            ]
        except Exception:
            return []

        return list(zip(points, points[1:]))

    return []


def walk_entity(
    entity,
    inherited_layer=None,
    root_handle=None,
    block_path=None,
    depth=0,
):
    """
    Expand INSERTs recursively while tracking provenance.

    Child entities on layer '0' inherit the INSERT layer.
    """

    if block_path is None:
        block_path = []

    entity_type = entity.dxftype()
    raw_layer = entity.dxf.get("layer", "0")

    if raw_layer == "0" and inherited_layer:
        effective_layer = inherited_layer
    else:
        effective_layer = raw_layer

    handle = str(entity.dxf.get("handle", ""))

    if root_handle is None:
        root_handle = handle

    if entity_type == "INSERT":
        name = str(entity.dxf.name)

        event = {
            "block_name": name,
            "insert_handle": handle,
            "root_handle": root_handle,
            "effective_layer": effective_layer,
            "depth": depth,
            "block_path": block_path + [name],
            "expanded_children": 0,
            "error": None,
        }

        if depth >= MAX_BLOCK_DEPTH:
            event["error"] = "Maximum block depth exceeded"
            return [], [event], []

        try:
            children = list(entity.virtual_entities())
        except Exception as exc:
            event["error"] = str(exc)
            return [], [event], []

        event["expanded_children"] = len(children)

        all_segments = []
        all_events = [event]
        all_entities = []

        for child in children:
            segments, events, entities = walk_entity(
                child,
                inherited_layer=effective_layer,
                root_handle=root_handle,
                block_path=block_path + [name],
                depth=depth + 1,
            )

            all_segments.extend(segments)
            all_events.extend(events)
            all_entities.extend(entities)

        return all_segments, all_events, all_entities

    entity_record = {
        "type": entity_type,
        "raw_layer": raw_layer,
        "effective_layer": effective_layer,
        "root_handle": root_handle,
        "entity_handle": handle,
        "block_path": block_path,
    }

    category = (
        WALL_LAYERS.get(effective_layer)
        or OPENING_LAYERS.get(effective_layer)
    )

    segments = []

    if category:
        for start, end in entity_segments(entity):
            if math.dist(start, end) < MIN_SEGMENT_LENGTH_M:
                continue

            segments.append({
                "start": start,
                "end": end,
                "category": category,
                "layer": effective_layer,
                "root_handle": root_handle,
                "entity_handle": handle,
                "block_path": block_path,
            })

    return segments, [], [entity_record]


def load_dxf():
    if not DXF_PATH.exists():
        raise FileNotFoundError(
            f"DXF not found: {DXF_PATH}"
        )

    document = ezdxf.readfile(DXF_PATH)

    units = document.header.get("$INSUNITS", 0)

    if units != 6:
        raise ValueError(
            f"Expected meters (DXF units code 6), got {units}"
        )

    all_segments = []
    all_block_events = []
    all_entity_records = []

    relevant_layers = (
        set(WALL_LAYERS)
        | set(OPENING_LAYERS)
    )

    for entity in document.modelspace():
        layer = entity.dxf.get("layer", "0")

        if layer not in relevant_layers:
            continue

        segments, events, records = walk_entity(entity)

        all_segments.extend(segments)
        all_block_events.extend(events)
        all_entity_records.extend(records)

    return (
        all_segments,
        all_block_events,
        all_entity_records,
    )


def find_unmatched_wall_endpoints(wall_segments):
    """
    Reproduce the Step 6 endpoint-connectivity test.

    Each endpoint is compared with other wall segments.
    """

    if not wall_segments:
        return []

    lines = [
        LineString([s["start"], s["end"]])
        for s in wall_segments
    ]

    tree = STRtree(lines)
    unmatched = []

    for index, segment in enumerate(wall_segments):
        for endpoint_name in ("start", "end"):
            point_xy = segment[endpoint_name]
            point = Point(point_xy)

            candidates = tree.query(
                point.buffer(CONNECTIVITY_TOLERANCE_M)
            )

            connected = False

            for candidate in candidates:
                other_index = int(candidate)

                if other_index == index:
                    continue

                if (
                    point.distance(lines[other_index])
                    <= CONNECTIVITY_TOLERANCE_M
                ):
                    connected = True
                    break

            if not connected:
                unmatched.append({
                    "point": point_xy,
                    "segment_index": index,
                    "endpoint": endpoint_name,
                    "category": segment["category"],
                    "root_handle": segment["root_handle"],
                })

    return unmatched


def classify_opening_proximity(
    unmatched,
    opening_segments,
):
    """
    Classify proximity, not semantic certainty.

    The nearest door/window line is used only as
    an audit signal.
    """

    if not opening_segments:
        return [
            {
                **endpoint,
                "nearest_opening": None,
                "distance_m": None,
                "classification": "unexplained",
            }
            for endpoint in unmatched
        ]

    opening_lines = [
        LineString([s["start"], s["end"]])
        for s in opening_segments
    ]

    tree = STRtree(opening_lines)
    results = []

    for endpoint in unmatched:
        point = Point(endpoint["point"])

        candidate_indices = tree.query(
            point.buffer(OPENING_SEARCH_RADIUS_M)
        )

        nearest_index = None
        nearest_distance = math.inf

        for candidate in candidate_indices:
            index = int(candidate)
            distance = point.distance(opening_lines[index])

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        if nearest_index is None:
            results.append({
                **endpoint,
                "nearest_opening": None,
                "distance_m": None,
                "classification": "unexplained",
            })
            continue

        nearest = opening_segments[nearest_index]

        results.append({
            **endpoint,
            "nearest_opening": nearest["category"],
            "distance_m": round(nearest_distance, 5),
            "opening_source_handle":
                nearest["root_handle"],
            "classification":
                "near_opening_geometry",
        })

    return results


def save_image(
    segments,
    classified_endpoints,
):
    fig, ax = plt.subplots(
        figsize=(15, 11),
        dpi=150,
    )

    for category in COLORS:
        lines = [
            [s["start"], s["end"]]
            for s in segments
            if s["category"] == category
        ]

        if not lines:
            continue

        ax.add_collection(
            LineCollection(
                lines,
                colors=COLORS[category],
                linewidths=(
                    0.9 if category in WALL_LAYERS.values()
                    else 0.65
                ),
                alpha=(
                    1.0 if category in WALL_LAYERS.values()
                    else 0.65
                ),
                label=category,
            )
        )

    near_opening = [
        e["point"]
        for e in classified_endpoints
        if e["classification"] == "near_opening_geometry"
    ]

    unexplained = [
        e["point"]
        for e in classified_endpoints
        if e["classification"] == "unexplained"
    ]

    if near_opening:
        ax.scatter(
            [p[0] for p in near_opening],
            [p[1] for p in near_opening],
            s=10,
            c="#16A34A",
            label="Endpoint near opening",
            zorder=4,
        )

    if unexplained:
        ax.scatter(
            [p[0] for p in unexplained],
            [p[1] for p in unexplained],
            s=10,
            c="#DC2626",
            label="Unexplained endpoint",
            zorder=4,
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    ax.set_title(
        "BuildProgress | Floor 2 - Opening and Block Audit"
    )

    ax.legend(fontsize=8, loc="best")

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

    print("Loading Floor 2 DXF...")

    segments, block_events, entity_records = load_dxf()

    wall_categories = set(WALL_LAYERS.values())
    opening_categories = set(OPENING_LAYERS.values())

    wall_segments = [
        s for s in segments
        if s["category"] in wall_categories
    ]

    opening_segments = [
        s for s in segments
        if s["category"] in opening_categories
    ]

    unmatched = find_unmatched_wall_endpoints(
        wall_segments
    )

    classified = classify_opening_proximity(
        unmatched,
        opening_segments,
    )

    block_report = {
        "schema_version": "cad-block-audit-1.0",
        "source": str(DXF_PATH),
        "block_reference_count": len(block_events),
        "block_events": block_events,
        "expanded_entity_types": dict(
            Counter(
                record["type"]
                for record in entity_records
                if record["block_path"]
            )
        ),
        "expanded_effective_layers": dict(
            Counter(
                record["effective_layer"]
                for record in entity_records
                if record["block_path"]
            )
        ),
        "expanded_entities_on_wall_layers": sum(
            record["effective_layer"] in WALL_LAYERS
            for record in entity_records
            if record["block_path"]
        ),
        "expanded_entities_on_opening_layers": sum(
            record["effective_layer"] in OPENING_LAYERS
            for record in entity_records
            if record["block_path"]
        ),
    }

    opening_report = {
        "schema_version": "cad-opening-audit-1.0",
        "source": str(DXF_PATH),
        "wall_segment_count": len(wall_segments),
        "opening_segment_count": len(opening_segments),
        "opening_category_counts": dict(
            Counter(
                s["category"]
                for s in opening_segments
            )
        ),
        "unmatched_wall_endpoint_count": len(unmatched),
        "endpoint_classifications": dict(
            Counter(
                e["classification"]
                for e in classified
            )
        ),
        "opening_search_radius_m":
            OPENING_SEARCH_RADIUS_M,
        "endpoints": [
            {
                **endpoint,
                "point": as_json_point(endpoint["point"]),
            }
            for endpoint in classified
        ],
        "interpretation": (
            "Proximity to door/window geometry is only "
            "an audit signal, not a verified opening."
        ),
    }

    BLOCK_REPORT_PATH.write_text(
        json.dumps(block_report, indent=2),
        encoding="utf-8",
    )

    OPENING_REPORT_PATH.write_text(
        json.dumps(opening_report, indent=2),
        encoding="utf-8",
    )

    save_image(segments, classified)

    print("\n" + "=" * 60)
    print("BUILDPROGRESS - BLOCK AND OPENING AUDIT")
    print("=" * 60)

    print(f"Block references visited: {len(block_events)}")
    print(f"Wall segments: {len(wall_segments)}")
    print(f"Opening segments: {len(opening_segments)}")
    print(f"Unmatched wall endpoints: {len(unmatched)}")

    print("\nOPENING CATEGORIES")

    for category, count in Counter(
        s["category"]
        for s in opening_segments
    ).items():
        print(f"  {category}: {count}")

    print("\nENDPOINT CLASSIFICATION")

    for category, count in Counter(
        e["classification"]
        for e in classified
    ).items():
        print(f"  {category}: {count}")

    print("\nSaved:")
    print(BLOCK_REPORT_PATH)
    print(OPENING_REPORT_PATH)
    print(IMAGE_PATH)

    print("\nAudit complete. No CAD geometry modified.")


if __name__ == "__main__":
    main()

